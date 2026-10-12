"""QFI state optimization on the loss + phase-noise (+ back-action) channel.

Reproduces the setting of Maliakal et al., arXiv:2606.19649: a probe state is
dephased during preparation (Gaussian phase noise, rms sigma_phi), then senses
an infinitesimal displacement epsilon via H = epsilon (a + a^dag) while photon
loss (power transmission eta) acts concurrently.  Optionally a radiation-
pressure shear H_BA = ba_to_g(kappa_ba) x^2 acts during sensing as well.

The channel is state-independent, so it is precomputed once as dense
superoperators; after that a QFI evaluation is three matrix-vector products
and one eigendecomposition, fast enough to sit inside an optimizer loop.
QFI conventions match the rest of the package (and the paper): for a pure
lossless probe F_Q = 8 Var(x), e.g. 88.0 for squeezed vacuum with nbar = 5.
"""

import numpy as np
from numpy.linalg import eigh
from scipy.linalg import expm
from scipy.optimize import minimize
import qutip as qt

from .conversions import loss_to_kappa, phirms_to_chi, ba_to_g

EIG_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class DisplacementChannel:
    """Dephasing (prep) -> {displacement signal + loss + back-action} (sensing).

    Precomputes exp(L) at epsilon = 0, +delta, -delta so that
    rho_out and drho_out/depsilon|_0 come from plain matvecs.
    """

    def __init__(self, N, *, eta=1.0, sigma_phi=0.0, kappa_ba=0.0, delta=1e-4):
        self.N = int(N)
        self.eta = float(eta)
        self.sigma_phi = float(sigma_phi)
        self.kappa_ba = float(kappa_ba)
        self.delta = float(delta)

        a = qt.destroy(self.N)
        G = a + a.dag()                       # signal generator sqrt(2) x
        x2 = (a + a.dag()) ** 2 / 2           # x^2
        c_ops = []
        if self.eta < 1.0:
            c_ops.append(np.sqrt(loss_to_kappa(1 - self.eta)) * a)
        H_ba = ba_to_g(self.kappa_ba) * x2 if self.kappa_ba != 0.0 else 0 * G

        def propagator(eps):
            L = qt.liouvillian(eps * G + H_ba, c_ops)
            return expm(L.full())

        self._S0 = propagator(0.0)
        self._Sp = propagator(+self.delta)
        self._Sm = propagator(-self.delta)
        self._Sd = (self._Sp - self._Sm) / (2 * self.delta)

        m = np.arange(self.N)
        chi = phirms_to_chi(self.sigma_phi)
        self._deph = np.exp(-chi * (m[:, None] - m[None, :]) ** 2 / 2.0)

    def _apply(self, S, rho):
        v = S @ rho.reshape(-1, order="F")
        return v.reshape(self.N, self.N, order="F")

    def output(self, rho_in):
        """(rho_out, drho_out/deps at eps=0) for a dense input density matrix."""
        rho_p = rho_in * self._deph
        rho0 = self._apply(self._S0, rho_p)
        drho = (self._apply(self._Sp, rho_p) - self._apply(self._Sm, rho_p)) \
            / (2 * self.delta)
        return rho0, drho

    def qfi(self, state):
        """QFI of the epsilon-family for a ket (1-D) or density matrix input."""
        rho_in = np.outer(state, state.conj()) if state.ndim == 1 else state
        rho0, drho = self.output(rho_in)
        return qfi_from_rho_drho(rho0, drho)

    def qfi_and_grad(self, psi):
        """(F, dF/dpsi*) for a normalized ket, via the SLD identity
        dF = 2 Tr[L drho'] - Tr[L^2 drho] pulled back through the channel
        adjoint.  The gradient is with respect to the ket entries treating
        psi as unconstrained (normalization handled by the caller)."""
        N = self.N
        rho_in = np.outer(psi, psi.conj())
        rho0, drho = self.output(rho_in)

        p, U = eigh(rho0)
        M = U.conj().T @ drho @ U
        denom = p[:, None] + p[None, :]
        mask = denom > EIG_FLOOR
        L_rot = np.where(mask, 2.0 * M / np.where(mask, denom, 1.0), 0.0)
        F = float(np.real(np.sum(L_rot * M.conj())))
        L = U @ L_rot @ U.conj().T

        # dF/drho_in: adjoint of (dephase -> S) applied to (2L for the
        # derivative branch, -L^2 for the state branch).
        A = self._S0.conj().T @ (-(L @ L)).reshape(-1, order="F") \
            + self._Sd.conj().T @ (2.0 * L).reshape(-1, order="F")
        G = A.reshape(N, N, order="F") * self._deph
        # rho_in = psi psi^dag: dF/dpsi* = G psi (G is Hermitian up to
        # numerical noise; symmetrize for safety)
        G = (G + G.conj().T) / 2.0
        return F, G @ psi


def qfi_from_rho_drho(rho, drho, floor=EIG_FLOOR):
    """F_Q = sum_ij 2 |<i|drho|j>|^2 / (p_i + p_j) over p_i + p_j > floor."""
    p, U = eigh(rho)
    M = U.conj().T @ drho @ U
    denom = p[:, None] + p[None, :]
    mask = denom > floor
    return float(np.sum(2.0 * np.abs(M[mask]) ** 2 / denom[mask]))


# ---------------------------------------------------------------------------
# Analytic references (paper conventions)
# ---------------------------------------------------------------------------

def qfi_bounds(eta, sigma_phi, nbar):
    """Convexity, variance, and combined-channel upper bounds on the QFI
    (Maliakal et al.), with A(x) = x + e^{-2 sigma_phi^2} sqrt(x(x+1))."""
    def A(x):
        return x + np.exp(-2 * sigma_phi**2) * np.sqrt(x * (x + 1))

    F_conv = 4 * (2 * eta * A(nbar) + 1) / (1 + 4 * eta * (1 - eta) * nbar)
    F_var = 4 * (1 + 2 * A(eta * nbar))
    F_comb = 4 * (1 + 2 * A(nbar)) / (1 + 2 * (1 - eta) * A(nbar))
    return {"convexity": F_conv, "variance": F_var, "combined": F_comb,
            "min": min(F_conv, F_var, F_comb)}


# ---------------------------------------------------------------------------
# State constructors
# ---------------------------------------------------------------------------

def gaussian_ket(N, r, theta_r, alpha):
    """D(alpha) S(r e^{i theta_r}) |0> as a dense vector."""
    psi = qt.displace(N, alpha) * qt.squeeze(N, r * np.exp(1j * theta_r)) \
        * qt.fock(N, 0)
    return psi.full().ravel()


def mean_n(state):
    if state.ndim == 1:
        return float(np.sum(np.arange(len(state)) * np.abs(state) ** 2))
    return float(np.real(np.sum(np.arange(state.shape[0]) * np.diag(state))))


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def optimize_gaussian(channel, nbar, n_starts=24, seed=0):
    """Best Gaussian state D(alpha) S(r e^{i theta_r})|0> with <n> <= nbar.

    Energy split is parametrized exactly on the constraint surface:
    sinh^2 r = s * f * nbar, |alpha|^2 = s * (1-f) * nbar with s, f in [0,1],
    plus the squeeze orientation theta_r and displacement phase theta_a.
    """
    N = channel.N
    rng = np.random.default_rng(seed)

    def build(p):
        s, f, theta_r, theta_a = p
        s, f = np.clip(s, 0, 1), np.clip(f, 0, 1)
        r = np.arcsinh(np.sqrt(s * f * nbar))
        amp = np.sqrt(s * (1 - f) * nbar)
        return gaussian_ket(N, r, theta_r, amp * np.exp(1j * theta_a))

    def cost(p):
        return -channel.qfi(build(p))

    best = None
    for _ in range(n_starts):
        p0 = np.array([rng.uniform(0.5, 1.0), rng.uniform(0, 1),
                       rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)])
        res = minimize(cost, p0, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    psi = build(best.x)
    return {"qfi": -best.fun, "state": psi, "params": best.x,
            "nbar": mean_n(psi)}


def optimize_fock_superposition(channel, nbar, K=None, n_starts=12, seed=0,
                                penalty=2000.0):
    """Optimize complex Fock coefficients c_0..c_K under <n> <= nbar.

    <n> <= nbar is enforced by a smooth quadratic penalty (as in the paper);
    the reported state is checked against the budget by the caller.
    """
    N = channel.N
    if K is None:
        K = min(3 * int(nbar) + 1, N - 1)
    n_idx = np.arange(K + 1)
    rng = np.random.default_rng(seed)

    nvec = np.zeros(N)
    nvec[:K + 1] = n_idx

    def build(p):
        c = p[:K + 1] + 1j * p[K + 1:]
        norm = np.linalg.norm(c)
        psi = np.zeros(N, dtype=complex)
        psi[:K + 1] = c / norm
        return psi, norm

    def cost_and_grad(p):
        psi, norm = build(p)
        F, gF = channel.qfi_and_grad(psi)
        nb = float(np.sum(nvec * np.abs(psi) ** 2))
        over = max(0.0, nb - nbar)
        # d(cost)/dpsi* before projecting onto the unit sphere
        gt = -gF + 2.0 * penalty * over * (nvec * psi)
        # chain rule through psi = c/||c||
        gt = (gt - psi * np.real(np.vdot(psi, gt))) / norm
        grad = np.concatenate([2 * np.real(gt[:K + 1]), 2 * np.imag(gt[:K + 1])])
        return -F + penalty * over ** 2, grad

    best = None
    for _ in range(n_starts):
        p0 = rng.normal(size=2 * (K + 1))
        # bias the start toward the photon budget
        w = np.exp(-((n_idx - nbar) / (nbar + 1)) ** 2)
        p0[:K + 1] *= w
        p0[K + 1:] *= 0.1 * w
        res = minimize(cost_and_grad, p0, method="L-BFGS-B", jac=True,
                       options={"maxiter": 1500, "ftol": 1e-13, "gtol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    psi, _ = build(best.x)
    return {"qfi": channel.qfi(psi), "state": psi, "nbar": mean_n(psi)}
