"""
Composable channel stages and BA1/BA2/BA3 orderings for the RP QFI study.

Each stage is a pure function rho -> rho on a fixed cutoff.  Unitary stages
(signal displacement, ponderomotive shear) are applied as exact matrix
exponentials — no ODE solver, no solver noise in the QFI finite difference.
Dissipative stages (loss, dephasing) reuse the mesolve machinery and rate
conventions of dynamics.py.

Orderings (t_final = 1 conventions of dynamics.py throughout):
    BA1  loss_in -> shear -> signal -> loss_out   (RP, then displacement)
    BA2  loss_in -> signal -> shear -> loss_out   (displacement, then RP)
    BA3  loss_in -> [signal + shear simultaneous] -> loss_out
         (the physical case; single Hamiltonian, matches
          dynamics.get_state_single_mode_rp)

For epsilon_a sensing the signal generator x commutes with H_BA ~ x^2, so
BA1 = BA2 = BA3 exactly when no loss sits between the stages; they split for
epsilon_p sensing or once loss/dephasing intervenes.  Injection loss
(eta_in, before the shear) vs detection loss (eta_out, after) is the
loss-ordering physics and differs at kappa_ba > 0.

Gaussian ground truth: gaussian.gaussian_rp_channel with
signal_order = "after" (BA1), "before" (BA2), "simultaneous" (BA3).
"""

import numpy as np
import qutip as qt

from .conversions import ba_to_g, loss_to_kappa, phirms_to_chi
from .dynamics import SOLVER_OPTIONS, _ensure_dm


# ---------------------------------------------------------------------------
# Stage ops: each returns a function rho -> rho for a given N_basis
# ---------------------------------------------------------------------------

def _apply_unitary(U, rho):
    return U * rho * U.dag()


def op_signal(N_basis, epsilon_a=0.0, epsilon_p=0.0, t_final=1.0):
    """Exact integrated signal displacement:
    U = exp(-i t [epsilon_a (a†+a) + i epsilon_p (a†-a)]),
    i.e. D(beta) with beta = (epsilon_p - i epsilon_a) t."""
    a = qt.destroy(N_basis)
    H = epsilon_a * (a.dag() + a) + 1j * epsilon_p * (a.dag() - a)
    U = (-1j * t_final * H).expm()
    return lambda rho: _apply_unitary(U, rho)


def op_shear(N_basis, kappa_ba, t_final=1.0):
    """Exact ponderomotive shear U = exp(-i g t x^2), g = ba_to_g(kappa_ba,
    t_final): (x, p) -> (x, p - kappa_ba x)."""
    a = qt.destroy(N_basis)
    x2 = (a + a.dag()) ** 2 / 2
    U = (-1j * ba_to_g(kappa_ba, t_final) * t_final * x2).expm()
    return lambda rho: _apply_unitary(U, rho)


def op_loss_kraus(N_basis, eta, trace_tol=1e-14):
    """Exact bosonic loss channel (transmissivity eta) via Kraus operators
    E_k = sqrt((1-eta)^k / k!) eta^{n/2} a^k, built iteratively.  Equivalent
    to the mesolve Lindblad loss (kappa = -ln eta over unit time) but exact
    and much faster at large cutoffs.  The Kraus sum is truncated on the
    state: since the channel is trace preserving, terms are added until the
    accumulated trace reaches 1 - trace_tol (the number of terms needed
    scales with (1-eta) <n> of the state, not with the cutoff)."""
    a = qt.destroy(N_basis)
    E0 = (0.5 * np.log(eta) * qt.num(N_basis)).expm()  # eta^{n/2}

    def _apply(rho):
        E = E0
        out = None
        cum = 0.0
        target = rho.tr().real
        for k in range(N_basis):
            if k > 0:
                E = E * a * np.sqrt((1 - eta) / k)
            term = E * rho * E.dag()
            out = term if out is None else out + term
            cum += term.tr().real
            if cum >= target - trace_tol:
                break
        return out
    return _apply


def op_loss_dephase(N_basis, eta=1.0, pn=0.0):
    """Loss (transmission eta) and/or dephasing (rms pn).  Pure loss uses
    the exact Kraus channel; any dephasing falls back to mesolve with the
    rate conventions of dynamics._apply_noise_stage."""
    if pn == 0.0:
        if eta >= 1.0:
            return lambda rho: rho
        return op_loss_kraus(N_basis, eta)

    a = qt.destroy(N_basis)
    n_op = a.dag() * a
    c_ops = [np.sqrt(phirms_to_chi(pn)) * n_op]
    if eta < 1.0:
        c_ops.append(np.sqrt(loss_to_kappa(1 - eta)) * a)

    def _apply(rho):
        res = qt.mesolve(0 * n_op, rho, [0.0, 1.0], c_ops,
                         options=dict(SOLVER_OPTIONS))
        return res.states[-1]
    return _apply


def op_signal_shear_simultaneous(N_basis, epsilon_a=0.0, epsilon_p=0.0,
                                 kappa_ba=0.0, t_final=1.0, eta_ch=1.0):
    """Joint evolution under H_sig + H_BA (single Hamiltonian, BA3).
    Unitary (exact expm) when eta_ch = 1; with channel loss eta_ch < 1 the
    loss runs simultaneously with the Hamiltonian via mesolve (the same
    stage-2 physics as dynamics.get_state_single_mode with eta_ch)."""
    a = qt.destroy(N_basis)
    x2 = (a + a.dag()) ** 2 / 2
    H = (epsilon_a * (a.dag() + a) + 1j * epsilon_p * (a.dag() - a)
         + ba_to_g(kappa_ba, t_final) * x2)
    if eta_ch >= 1.0:
        U = (-1j * t_final * H).expm()
        return lambda rho: _apply_unitary(U, rho)

    c_ops = [np.sqrt(loss_to_kappa(1 - eta_ch)) * a]

    def _apply(rho):
        res = qt.mesolve(H, rho, [0.0, t_final], c_ops,
                         options=dict(SOLVER_OPTIONS))
        return res.states[-1]
    return _apply


def apply_chain(rho, ops):
    """Apply stage ops in sequence (kets are promoted to density matrices)."""
    rho = _ensure_dm(rho)
    for op in ops:
        rho = op(rho)
    return rho


# ---------------------------------------------------------------------------
# BA1 / BA2 / BA3 channels (calculate_qfi-compatible signatures)
# ---------------------------------------------------------------------------

def _get_state_ordered(order, *, epsilon_a=0.0, epsilon_p=0.0, kappa_ba=0.0,
                       eta_in=1.0, eta_mid=1.0, eta_out=1.0, eta_ch=1.0,
                       pn_in=0.0, pn_mid=0.0, pn_out=0.0,
                       t_final=1.0, N_basis=20, rho=None):
    if rho is None:
        rho = qt.ket2dm(qt.coherent(N_basis, 1.0))

    stage = {
        "loss_in": lambda: op_loss_dephase(N_basis, eta_in, pn_in),
        "loss_mid": lambda: op_loss_dephase(N_basis, eta_mid, pn_mid),
        "loss_out": lambda: op_loss_dephase(N_basis, eta_out, pn_out),
        "sig": lambda: op_signal(N_basis, epsilon_a, epsilon_p, t_final),
        "shear": lambda: op_shear(N_basis, kappa_ba, t_final),
        "sig+shear": lambda: op_signal_shear_simultaneous(
            N_basis, epsilon_a, epsilon_p, kappa_ba, t_final, eta_ch),
    }
    return apply_chain(rho, [stage[name]() for name in order])


def get_state_ba1(**kwargs):
    """BA1: radiation pressure, then displacement sensing
    (loss_in -> shear -> loss_mid -> signal -> loss_out).
    eta_mid/pn_mid sit BETWEEN the back-action and the signal."""
    return _get_state_ordered(
        ("loss_in", "shear", "loss_mid", "sig", "loss_out"), **kwargs)


def get_state_ba2(**kwargs):
    """BA2: displacement sensing, then radiation pressure
    (loss_in -> signal -> loss_mid -> shear -> loss_out).
    eta_mid/pn_mid sit BETWEEN the signal and the back-action."""
    return _get_state_ordered(
        ("loss_in", "sig", "loss_mid", "shear", "loss_out"), **kwargs)


def get_state_ba3(**kwargs):
    """BA3: simultaneous signal + back-action, single Hamiltonian
    (loss_in -> [signal+shear] -> loss_out).  The physical case; BA1/BA2
    bound it.  eta_ch < 1 runs channel loss simultaneously with the
    Hamiltonian (the BA3 analog of eta_mid); eta_mid is not accepted."""
    if kwargs.get("eta_mid", 1.0) < 1.0 or kwargs.get("pn_mid", 0.0) > 0.0:
        raise ValueError("BA3 has no 'between' stage: use eta_ch for loss "
                         "concurrent with the signal+shear evolution")
    return _get_state_ordered(("loss_in", "sig+shear", "loss_out"),
                              **kwargs)


BA_CHANNELS = {"ba1": get_state_ba1, "ba2": get_state_ba2,
               "ba3": get_state_ba3}
