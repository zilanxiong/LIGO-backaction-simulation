"""
Core channel + QFI machinery for the back-action state sweep.

Conventions (documented in README; unit-tested in test_sweep.py):
  x = a + a†,  p = -i(a - a†),  [x, p] = 2i,  vacuum Var(x) = Var(p) = 1.
  Signal   S = exp(-i eps x)          (GW displaces the phase quadrature)
  Back-action  B = exp(-i g x^2) with g = K/4, so that for vacuum input the
  output obeys KLMTV's input-output relation  p_out = p_in - K x_in  with the
  SAME numerical Kimble factor K(Omega) as rp_kappa:
      K(Omega) = 2 X gamma^4 / (Omega^2 (gamma^2 + Omega^2)),
      X = I0/I_SQL = 4 P w0 / (m L gamma^3 c).
  Loss placements: 'inj' (before S+B), 'det' (after S+B), 'conc'
  (Lindblad D[a] running during the same evolution as S+B — the physical
  arm-loss case), 'none'.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
from numpy.linalg import eigh
from scipy.integrate import solve_ivp

PREC_DIFF = 1e-5
PREC_EIG = 1e-10
SOLVER_OPTS = {"atol": 1e-12, "rtol": 1e-12, "nsteps": 200_000}

_C = 299_792_458.0


# ---------------------------------------------------------------------------
# Kimble factor (mirrors quantum-sensing rp_kappa; O4 LIGO defaults)
# ---------------------------------------------------------------------------

def rp_power_ratio(m=40.0, L=3995.0, lambda0=1064e-9, P_circ=360e3,
                   gamma=2 * np.pi * 450.0):
    w0 = 2 * np.pi * _C / lambda0
    return 4 * P_circ * w0 / (m * L * gamma**3 * _C)


def kimble_K(Omega, gamma=2 * np.pi * 450.0, **kw):
    X = rp_power_ratio(gamma=gamma, **kw)
    return 2 * X * gamma**4 / (Omega**2 * (gamma**2 + Omega**2))


# ---------------------------------------------------------------------------
# Fock-space channel
# ---------------------------------------------------------------------------

def x_op(N):
    a = qt.destroy(N)
    return a + a.dag()


def apply_channel(rho0, eps, K, N, placement="none", eta=1.0):
    """rho0 (dm) -> output dm. Signal + back-action always act together
    (verified: their mutual order is irrelevant); `placement` says where the
    loss D[a] sits relative to that block."""
    a = qt.destroy(N)
    x = x_op(N)
    H_sb = eps * x + (K / 4.0) * x * x

    def lossy(rho, eta_):
        if eta_ >= 1.0:
            return rho
        lam = -np.log(eta_)
        res = qt.mesolve(0 * a.dag() * a, rho, [0.0, 1.0],
                         [np.sqrt(lam) * a], options=SOLVER_OPTS)
        return res.states[-1]

    def unitary_sb(rho):
        U = (-1j * H_sb).expm()
        return U * rho * U.dag()

    if placement == "none" or eta >= 1.0:
        return unitary_sb(rho0)
    if placement == "inj":
        return unitary_sb(lossy(rho0, eta))
    if placement == "det":
        return lossy(unitary_sb(rho0), eta)
    if placement == "conc":
        lam = -np.log(eta)
        res = qt.mesolve(H_sb, rho0, [0.0, 1.0], [np.sqrt(lam) * a],
                         options=SOLVER_OPTS)
        return res.states[-1]
    raise ValueError(placement)


def qfi_fock(rho0, K, N, placement="none", eta=1.0, prec=PREC_DIFF):
    """QFI for eps at eps = 0 via central difference + eigendecomposition."""
    rhos = [apply_channel(rho0, e, K, N, placement, eta)
            for e in (+prec, -prec, 0.0)]
    drho = (rhos[0].full() - rhos[1].full()) / (2 * prec)
    rho = rhos[2].full()
    evals, evecs = eigh(rho)
    drot = evecs.conj().T @ drho @ evecs
    denom = evals[:, None] + evals[None, :]
    mask = np.abs(denom) > PREC_EIG
    return float(np.real(np.sum(2.0 * np.abs(drot[mask]) ** 2 / denom[mask])))


# ---------------------------------------------------------------------------
# Exact Gaussian channel (any placement, incl. concurrent) — used both as
# the primary method for Gaussian probes and as the convergence guard.
# ---------------------------------------------------------------------------

def gaussian_qfi(V0, K, placement="none", eta=1.0):
    """QFI for eps at eps=0, probe with zero mean and covariance V0
    (2x2, vacuum = identity). Exact for Gaussian probes.

    Sequential blocks: shear Sm = [[1,0],[-K,1]]; loss V -> eta V + (1-eta)I,
    u -> sqrt(eta) u; signal contributes |du/deps| = (0, 2).
    Concurrent: integrate dV/dt = A V + V A^T + lam I, du/dt = A u + b,
    A = [[-lam/2, 0], [-K, -lam/2]], b = (0, -2) (sign irrelevant), t=1.
    """
    V0 = np.asarray(V0, float)
    Sm = np.array([[1.0, 0.0], [-K, 1.0]])
    s = np.array([0.0, 2.0])
    I2 = np.eye(2)

    if placement == "none" or eta >= 1.0:
        # the shear leaves the p-displacement derivative (0,2) unchanged
        V, u = Sm @ V0 @ Sm.T, s
    elif placement == "inj":
        V = eta * V0 + (1 - eta) * I2
        V = Sm @ V @ Sm.T
        u = s
    elif placement == "det":
        V = Sm @ V0 @ Sm.T
        V = eta * V + (1 - eta) * I2
        u = np.sqrt(eta) * s
    elif placement == "conc":
        lam = -np.log(eta)
        A = np.array([[-lam / 2, 0.0], [-K, -lam / 2]])

        def rhs(t, y):
            V = y[:4].reshape(2, 2)
            u = y[4:]
            dV = A @ V + V @ A.T + lam * I2
            du = A @ u + np.array([0.0, -2.0])
            return np.concatenate([dV.ravel(), du])

        sol = solve_ivp(rhs, (0.0, 1.0), np.concatenate([V0.ravel(),
                                                         np.zeros(2)]),
                        rtol=1e-12, atol=1e-12, dense_output=False)
        y = sol.y[:, -1]
        V, u = y[:4].reshape(2, 2), y[4:]
    else:
        raise ValueError(placement)

    return float(u @ np.linalg.solve(V, u))


def sqz_cov(r, theta):
    """Covariance of S(r e^{i theta})|0>: squeezed axis rotated by theta/2."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([np.exp(2 * r), np.exp(-2 * r)]) @ R.T


def best_sqz_angle(n_target, K, placement, eta, n_grid=181):
    """Optimal squeeze orientation at fixed <n> via the exact Gaussian QFI."""
    r = np.arcsinh(np.sqrt(n_target))
    thetas = np.linspace(0, np.pi, n_grid)
    q = [gaussian_qfi(sqz_cov(r, t), K, placement, eta) for t in thetas]
    j = int(np.argmax(q))
    return float(thetas[j]), float(q[j])
