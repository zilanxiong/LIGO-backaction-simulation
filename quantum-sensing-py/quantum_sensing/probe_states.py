"""
Probe states at fixed mean photon number.

Every constructor takes a target :math:`\\bar n = \\langle \\hat n\\rangle` and
returns a normalised QuTiP ket in the truncated Fock space, with the free
parameter solved for numerically so the *realised* :math:`\\bar n` matches the
target.  Comparing QFI across state families is only meaningful at equal photon
number, so this is the entry point for every state scan.

Orientation
-----------
The gravitational-wave signal is the ``epsilon_a`` term of the sensing
Hamiltonian, whose generator is :math:`\\sqrt2\\,\\hat x`; the QFI of a pure
state is therefore proportional to :math:`\\mathrm{Var}(x)`.  All states are
oriented so their *extended* quadrature is :math:`x` -- squeezed states
anti-squeeze :math:`x`, cats lie along :math:`x`.  That is the orientation which
maximises the QFI in the absence of back-action, and also the one that suffers
most from radiation pressure once loss is present.

Convention note
---------------
Squeezing uses the textbook relation :math:`\\bar n = \\sinh^2 r`, matching
:func:`quantum_sensing.conversions.n_to_r`.  It deliberately does **not** use
:func:`quantum_sensing.conversions.r_to_var`, which returns :math:`e^{-r^2}`
rather than :math:`e^{-2r}` -- that function documents itself as following the
optimisation code's convention, and it is inconsistent with ``n_to_r`` in the
same module.  Mixing the two silently corrupts any squeezing comparison.
"""

import numpy as np
import qutip as qt
from scipy.optimize import brentq

__all__ = [
    "STATE_FAMILIES",
    "STATE_LABELS",
    "make_state",
    "mean_photon_number",
    "coherent_state",
    "squeezed_vacuum",
    "fock_state",
    "cat_state",
    "squeezed_cat",
    "optimal_no_backaction",
]


def mean_photon_number(state):
    """Mean photon number of a ket or density matrix."""
    N_basis = state.shape[0]
    n_op = qt.num(N_basis)
    if state.isket:
        return float(np.real(qt.expect(n_op, state)))
    return float(np.real((n_op * state).tr()))


def _vacuum(N_basis):
    return qt.basis(N_basis, 0)


def _coherent_ket(N_basis, alpha):
    """Coherent state from its analytic Fock amplitudes, renormalised on the cutoff.

    ``qt.coherent(N, alpha)`` defaults to exponentiating a dense displacement
    operator; the closed form is used here because the cat constructors call it
    inside a root-finding loop at cutoffs of several hundred.
    """
    n = np.arange(N_basis)
    if alpha == 0:
        return _vacuum(N_basis)
    from scipy.special import gammaln

    log_amp = -0.5 * abs(alpha) ** 2 + n * np.log(complex(alpha)) - 0.5 * gammaln(n + 1.0)
    amp = np.exp(log_amp)
    return qt.Qobj((amp / np.linalg.norm(amp)).reshape(-1, 1))


def coherent_state(N_basis, nbar):
    r""":math:`|\alpha\rangle` with :math:`|\alpha|^2 = \bar n`, aligned along :math:`x`."""
    return _coherent_ket(N_basis, np.sqrt(max(nbar, 0.0)))


def squeezed_vacuum(N_basis, nbar):
    r"""Squeezed vacuum with :math:`\sinh^2 r = \bar n`, anti-squeezed in :math:`x`.

    ``qt.squeeze(N, z) = exp[(z* a^2 - z a^{dag 2})/2]`` squeezes :math:`x` for
    real ``z > 0``, so ``z = -r`` gives :math:`\mathrm{Var}(x) = e^{2r}/2`.
    """
    r = np.arcsinh(np.sqrt(max(nbar, 0.0)))
    return qt.squeeze(N_basis, -r) * _vacuum(N_basis)


def fock_state(N_basis, nbar):
    r""":math:`|n\rangle` with ``n = round(nbar)``; the realised :math:`\bar n` is an integer."""
    n = int(round(nbar))
    if n >= N_basis:
        raise ValueError(f"Fock state |{n}> does not fit in a space of dimension {N_basis}")
    return qt.basis(N_basis, n)


def _cat_ket(N_basis, alpha, parity="even"):
    sign = 1.0 if parity == "even" else -1.0
    psi = _coherent_ket(N_basis, alpha) + sign * _coherent_ket(N_basis, -alpha)
    return psi.unit()


def cat_state(N_basis, nbar, parity="even"):
    r"""Even/odd cat :math:`\propto |\alpha\rangle \pm |-\alpha\rangle` with real :math:`\alpha`.

    :math:`\bar n = |\alpha|^2\tanh|\alpha|^2` (even) or
    :math:`|\alpha|^2\coth|\alpha|^2` (odd).  ``alpha`` is found by root finding
    on the numerically evaluated :math:`\bar n`, so the cutoff is accounted for.
    """
    if nbar <= 0:
        return _vacuum(N_basis)
    if parity == "odd" and nbar < 1.0:
        raise ValueError("an odd cat has nbar >= 1")

    def f(alpha):
        return mean_photon_number(_cat_ket(N_basis, alpha, parity)) - nbar

    lo, hi = 1e-6, 1.0
    while f(hi) < 0:
        hi *= 1.5
        if hi > np.sqrt(N_basis):
            raise ValueError(f"cannot reach nbar={nbar} with cutoff N_basis={N_basis}")
    if parity == "odd" and f(lo) > 0:
        return _cat_ket(N_basis, lo, parity)
    return _cat_ket(N_basis, brentq(f, lo, hi, xtol=1e-12, rtol=1e-13), parity)


def squeezed_cat(N_basis, nbar, squeeze_fraction=0.5, parity="even"):
    r"""Squeezed cat :math:`S(-r)(|\alpha\rangle + |-\alpha\rangle)`.

    ``squeeze_fraction`` :math:`f \in [0,1)` sets how much of the photon budget
    goes into squeezing (:math:`\sinh^2 r = f\,\bar n`); :math:`\alpha` is then
    solved for so the total realised :math:`\bar n` hits the target.
    """
    if not 0.0 <= squeeze_fraction < 1.0:
        raise ValueError("squeeze_fraction must lie in [0, 1)")
    if nbar <= 0:
        return _vacuum(N_basis)
    r = np.arcsinh(np.sqrt(squeeze_fraction * nbar))
    S = qt.squeeze(N_basis, -r)

    def ket(alpha):
        return (S * _cat_ket(N_basis, alpha, parity)).unit()

    def f(alpha):
        return mean_photon_number(ket(alpha)) - nbar

    lo, hi = 1e-8, 1.0
    if f(lo) > 0:
        return ket(lo)
    while f(hi) < 0:
        hi *= 1.5
        if hi > np.sqrt(N_basis):
            raise ValueError(f"cannot reach nbar={nbar} with cutoff N_basis={N_basis}")
    return ket(brentq(f, lo, hi, xtol=1e-12, rtol=1e-13))


def optimal_no_backaction(N_basis, nbar):
    r"""State maximising the no-back-action QFI at fixed :math:`\bar n`.

    Without radiation pressure or loss the QFI for the ``epsilon_a`` displacement
    is proportional to :math:`\mathrm{Var}(x)`, so the optimum solves

    .. math:: \max_\psi \langle \hat x^2 - \mu \hat n \rangle

    with the multiplier :math:`\mu` tuned to meet the photon-number constraint.
    This is a two-term Lagrangian eigenproblem, so it is solved exactly by a
    one-dimensional root find rather than by gradient descent.  The answer is
    the squeezed vacuum with :math:`\sinh^2 r = \bar n`.

    This is the analytic stand-in for "states optimised without back-action from
    previous code".  The saved optimisation results in
    :mod:`quantum_sensing.states` (``get_best_qfi_envelope``,
    ``reconstruct_state``) are the real thing and need a data directory; see
    :func:`quantum_sensing.states.set_data_dir`.
    """
    if nbar <= 0:
        return _vacuum(N_basis)

    from .radiation_pressure import x_quadrature

    x2 = (x_quadrature(N_basis) ** 2).full().real
    n_diag = np.arange(N_basis, dtype=float)

    def top_eigvec(mu):
        w, v = np.linalg.eigh(x2 - mu * np.diag(n_diag))
        return v[:, -1]

    def f(mu):
        v = top_eigvec(mu)
        return float(np.sum(n_diag * np.abs(v) ** 2)) - nbar

    lo, hi = 1e-6, 1.0
    while f(hi) > 0 and hi < 200.0:
        hi *= 2.0
    while f(lo) < 0 and lo > 1e-12:
        lo /= 4.0
    v = top_eigvec(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))
    return qt.Qobj(v.reshape(-1, 1)).unit()


STATE_FAMILIES = {
    "vacuum": lambda N, nbar: _vacuum(N),
    "coherent": coherent_state,
    "squeezed": squeezed_vacuum,
    "fock": fock_state,
    "cat_even": lambda N, nbar: cat_state(N, nbar, "even"),
    "cat_odd": lambda N, nbar: cat_state(N, nbar, "odd"),
    "squeezed_cat": lambda N, nbar: squeezed_cat(N, nbar, 0.5),
    "opt_no_ba": optimal_no_backaction,
}

STATE_LABELS = {
    "vacuum": "vacuum",
    "coherent": "coherent",
    "squeezed": "squeezed",
    "fock": "Fock",
    "cat_even": "even cat",
    "cat_odd": "odd cat",
    "squeezed_cat": "squeezed cat",
    "opt_no_ba": "opt. (no BA)",
}


def make_state(name, N_basis, nbar):
    """Build a probe state by family name (see :data:`STATE_FAMILIES`)."""
    try:
        builder = STATE_FAMILIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown state family {name!r}; choose from {sorted(STATE_FAMILIES)}"
        ) from exc
    return builder(N_basis, nbar)
