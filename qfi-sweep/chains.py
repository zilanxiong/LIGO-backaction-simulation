"""
Noise-chain channel in the sheared (co-moving) frame.

Physical channel: ordered stages around the signal+back-action block SB
(S = exp(-i eps x), B = exp(-i (K/4) x^2); S and B commute, verified).
Noise stages are loss D[a] (strength eta) and dephasing D[n] (strength chi),
placed before ('pre'), after ('post'), or concurrent ('conc') with SB.

Frame trick (validated in test_chains.py): the QFI is invariant under the
final unitary, so B is never applied to the state. Instead every noise
operator acting after a shear of accumulated strength k is conjugated:
    U_B(k)† a U_B(k) = a - i (k/2) x = (1 - i k/2) a - (i k/2) a†
    U_B(k)† n U_B(k) = ã† ã  with ã as above.
Pre-SB noise sees k = 0 (bare operators); post-SB noise sees k = K;
concurrent noise is Trotterized into slices with k growing linearly.
This keeps the simulated state compact even at K ~ 10 (10 Hz), where the
direct sheared state would need a Fock basis in the thousands.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
from numpy.linalg import eigh

PREC_DIFF = 1e-5
PREC_EIG = 1e-10
SOLVER_OPTS = {"atol": 1e-12, "rtol": 1e-12, "nsteps": 200_000}
N_TROTTER = 24


def _a_sheared(N, k):
    """U_B(k)† a U_B(k) in the Fock basis."""
    a = qt.destroy(N)
    return (1 - 0.5j * k) * a - 0.5j * k * a.dag()


def _noise_ops(N, k, kind):
    if kind == "loss":
        return _a_sheared(N, k)
    if kind == "pn":
        at = _a_sheared(N, k)
        return at.dag() * at
    raise ValueError(kind)


def _lindblad_step(rho, ops_rates, N, t=1.0, H=None):
    if H is None:
        H = 0 * qt.qeye(N)
    c_ops = [np.sqrt(r) * o for o, r in ops_rates if r > 0]
    if not c_ops and H.norm() == 0:
        return rho
    res = qt.mesolve(H, rho, [0.0, t], c_ops, options=SOLVER_OPTS)
    return res.states[-1]


def apply_chain(rho0, eps, K, N, stages):
    """Evolve rho0 through `stages`, a list of
        ('loss'|'pn', strength, 'pre'|'post'|'conc')
    around the SB block, in the sheared frame (B never applied; S applied
    at its correct slot — it commutes with B so only its position relative
    to the noise stages matters).

    Strengths: loss -> eta (transmission), pn -> chi (rms^2 rad^2).
    Returns the output density matrix in the sheared frame (QFI-equivalent
    to the lab frame).
    """
    x = qt.destroy(N) + qt.destroy(N).dag()

    pre = [s for s in stages if s[2] == "pre"]
    conc = [s for s in stages if s[2] == "conc"]
    post = [s for s in stages if s[2] == "post"]

    rho = rho0
    # --- pre-SB noise: bare operators (k = 0) ---
    for kind, strength, _ in pre:
        rate = -np.log(strength) if kind == "loss" else strength
        rho = _lindblad_step(rho, [(_noise_ops(N, 0.0, kind), rate)], N)

    # --- SB block ---
    U_s = (-1j * eps * x).expm()          # signal (B dropped: final-unitary)
    if not conc:
        rho = U_s * rho * U_s.dag()
    else:
        # concurrent: H = eps x in the co-moving frame; noise operators
        # conjugated by the accumulated shear k(t) = K t, Trotterized.
        dt = 1.0 / N_TROTTER
        H = eps * x
        for j in range(N_TROTTER):
            k_mid = K * (j + 0.5) * dt
            ops = []
            for kind, strength, _ in conc:
                rate = -np.log(strength) if kind == "loss" else strength
                ops.append((_noise_ops(N, k_mid, kind), rate))
            rho = _lindblad_step(rho, ops, N, t=dt, H=H)

    # --- post-SB noise: fully sheared operators (k = K) ---
    for kind, strength, _ in post:
        rate = -np.log(strength) if kind == "loss" else strength
        rho = _lindblad_step(rho, [(_noise_ops(N, K, kind), rate)], N)

    return rho


def qfi_chain(rho0, K, N, stages, prec=PREC_DIFF):
    outs = [apply_chain(rho0, e, K, N, stages) for e in (prec, -prec, 0.0)]
    drho = (outs[0].full() - outs[1].full()) / (2 * prec)
    rho = outs[2].full()
    evals, evecs = eigh(rho)
    drot = evecs.conj().T @ drho @ evecs
    denom = evals[:, None] + evals[None, :]
    mask = np.abs(denom) > PREC_EIG
    return float(np.real(np.sum(2.0 * np.abs(drot[mask]) ** 2 / denom[mask])))


def basis_size_chain(n_target, var_x_est, stages, K, base=50, cap=340):
    """In the sheared frame the state is never stretched by K directly, but
    post/conc noise conjugated by k ~ K injects ~ (1-eta) K^2 (or chi K^2)
    worth of quanta; size the basis for that."""
    inject = 0.0
    for kind, strength, place in stages:
        k = K if place in ("post", "conc") else 0.0
        s = (1 - strength) if kind == "loss" else strength
        inject += s * (1 + k**2) * max(var_x_est, 1.0) * 0.5
    need = base + 14 * n_target + int(4 * inject)
    return int(min(cap, need))
