"""
Fixed-mean-photon-number probe states for the RP QFI study.

Every factory takes a target <n> and returns a callable N_basis -> ket, the
form the convergence harness (convergence.converged_qfi) consumes.  The
amplitude of each family is solved so that <n> hits the target exactly
(numerically, in a large reference basis), so QFI comparisons across
families are at equal energy.

Families: coherent, squeezed vacuum, even cat, squeezed cat, Fock, and the
paper's optimized states loaded from states.h5.
"""

import numpy as np
import qutip as qt
from scipy.optimize import brentq

from .states import load_state_params, reconstruct_state

_N_REF = 160  # reference cutoff for solving amplitudes


def mean_n(psi):
    N = psi.shape[0]
    return float(qt.expect(qt.num(N), psi).real)


def _solve_scale(build, n_target, lo=1e-6, hi=None):
    """Find s such that <n> of build(s, N_REF) equals n_target."""
    if hi is None:
        hi = 2 * np.sqrt(n_target) + 3.0
    f = lambda s: mean_n(build(s, _N_REF)) - n_target
    while f(hi) < 0:
        hi *= 1.5
    return brentq(f, lo, hi, xtol=1e-10)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def coherent(n_target):
    alpha = np.sqrt(n_target)
    return lambda N: qt.coherent(N, alpha)


def squeezed_vacuum(n_target):
    """p-squeezed vacuum (r < 0 in the qutip convention): oriented so the
    epsilon_a signal, which displaces p, is squeezed-quadrature-aligned —
    the orientation that maximizes the lossless QFI 8 Var(x) = 4 e^{2r}."""
    r = np.arcsinh(np.sqrt(n_target))
    return lambda N: qt.squeeze(N, -r) * qt.fock(N, 0)


def fock(n_target):
    if abs(n_target - round(n_target)) > 1e-9:
        raise ValueError("Fock probe needs integer <n>")
    n = int(round(n_target))
    return lambda N: qt.fock(N, n)


def _cat(alpha, N):
    return (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()


def cat(n_target):
    """Even cat |alpha> + |-alpha>, alpha solved for <n> = n_target."""
    alpha = _solve_scale(_cat, n_target)
    return lambda N: _cat(alpha, N)


def squeezed_cat(n_target, sqz_fraction=0.5):
    """S(r) (|alpha> + |-alpha>), with r fixed by putting sqz_fraction of the
    photon budget into squeezing (r = arcsinh(sqrt(f n))) and alpha solved
    numerically for the total <n>."""
    r = np.arcsinh(np.sqrt(sqz_fraction * n_target))
    build = lambda a, N: (qt.squeeze(N, -r) * _cat(a, N)).unit()
    alpha = _solve_scale(build, n_target)
    return lambda N: build(alpha, N)


def optimized(n_target, state_type="fock_sup", loss_config="loss_ch",
              eta=1.0, pn=0.0):
    """Best-QFI optimized state from states.h5 at N_target = n_target
    (states optimized WITHOUT back-action, from the previous campaign).
    Requires set_data_dir to point at the states.h5 directory."""
    entries = [e for e in load_state_params(state_type, loss_config, eta, pn)
               if abs(e["N_target"] - n_target) < 1e-6]
    if not entries:
        raise ValueError(f"no optimized {state_type} entry at "
                         f"N_target={n_target}, eta={eta}, pn={pn}")
    best = max(entries, key=lambda e: e["qfi"])

    # The stored states were optimized for epsilon_p sensing (x displacement,
    # generator p); the RP study senses epsilon_a (generator x).  Orientation
    # in phase space is a gauge choice, so rotate by pi/2 if that raises the
    # lossless QFI 8 Var(x).
    psi = reconstruct_state(best, N_basis=_N_REF).unit()
    a = qt.destroy(_N_REF)
    x = (a + a.dag()) / np.sqrt(2)
    p = (a - a.dag()) / (1j * np.sqrt(2))
    var = lambda op: (qt.expect(op * op, psi) - qt.expect(op, psi) ** 2).real
    rotate = var(p) > var(x)

    def factory(N):
        s = reconstruct_state(best, N_basis=N).unit()
        if rotate:
            s = (1j * (np.pi / 2) * qt.num(N)).expm() * s
        return s
    return factory


PROBE_FAMILIES = {
    "coherent": coherent,
    "sqz_vac": squeezed_vacuum,
    "cat": cat,
    "sqz_cat": squeezed_cat,
    "fock": fock,
}
