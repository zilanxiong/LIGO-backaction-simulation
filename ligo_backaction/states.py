r"""Probe states at fixed mean photon number.

Every constructor takes the target :math:`\bar n = \langle \hat n\rangle` and
returns a normalised ket in the truncated Fock space, with the free parameter
solved for numerically so that the realised :math:`\bar n` matches the target
(this is what makes QFI comparisons between state families fair).

Orientation convention
----------------------
The gravitational-wave signal is a displacement along :math:`p`, whose
generator is :math:`\hat x`; the QFI of a pure state is therefore
:math:`4\,\mathrm{Var}(x)`.  All states are oriented so that their *large*
quadrature is :math:`x` -- i.e. squeezed states anti-squeeze :math:`x`, cats
are aligned along :math:`x`.  This is the orientation that maximises the QFI,
and it is also the one that suffers most from radiation pressure once loss is
present, which is the trade-off the study is about.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .operators import annihilation, coherent_ket, number, squeeze_op

__all__ = [
    "STATE_FAMILIES",
    "make_state",
    "coherent",
    "squeezed_vacuum",
    "fock",
    "cat",
    "squeezed_cat",
    "mean_photon_number",
    "normalize",
]


def normalize(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    return psi / np.linalg.norm(psi)


def mean_photon_number(psi: np.ndarray) -> float:
    psi = np.asarray(psi)
    if psi.ndim == 1:
        return float(np.real(np.sum(np.arange(psi.size) * np.abs(psi) ** 2)))
    return float(np.real(np.trace(number(psi.shape[0]) @ psi)))


def _basis(N: int, n: int) -> np.ndarray:
    v = np.zeros(N, dtype=complex)
    v[n] = 1.0
    return v


def coherent(N: int, nbar: float) -> np.ndarray:
    r""":math:`|\alpha\rangle` with :math:`|\alpha|^2 = \bar n`, aligned along :math:`x`."""
    alpha = np.sqrt(max(nbar, 0.0))
    return coherent_ket(N, alpha)


def squeezed_vacuum(N: int, nbar: float) -> np.ndarray:
    r"""Squeezed vacuum with :math:`\sinh^2 r = \bar n`, anti-squeezed in :math:`x`.

    ``squeeze_op(N, -r)`` gives :math:`\mathrm{Var}(x) = e^{2r}/2`.
    """
    r = np.arcsinh(np.sqrt(max(nbar, 0.0)))
    return normalize(squeeze_op(N, -r) @ _basis(N, 0))


def fock(N: int, nbar: float) -> np.ndarray:
    r""":math:`|n\rangle` with ``n = round(nbar)`` (the realised :math:`\bar n` is an integer)."""
    n = int(round(nbar))
    if n >= N:
        raise ValueError(f"Fock state |{n}> does not fit in a space of dimension {N}")
    return _basis(N, n)


def _cat_ket(N: int, alpha: float, parity: str = "even") -> np.ndarray:
    sign = 1.0 if parity == "even" else -1.0
    return normalize(coherent_ket(N, alpha) + sign * coherent_ket(N, -alpha))


def cat(N: int, nbar: float, parity: str = "even") -> np.ndarray:
    r"""Even/odd cat :math:`\propto |\alpha\rangle \pm |-\alpha\rangle` with real :math:`\alpha`.

    :math:`\bar n = |\alpha|^2\tanh|\alpha|^2` (even) or
    :math:`|\alpha|^2\coth|\alpha|^2` (odd); ``alpha`` is found by root finding
    on the numerically evaluated :math:`\bar n`, so truncation is accounted for.
    """
    if nbar <= 0:
        return _basis(N, 0)
    if parity == "odd" and nbar < 1.0:
        raise ValueError("an odd cat has nbar >= 1")

    def f(alpha):
        return mean_photon_number(_cat_ket(N, alpha, parity)) - nbar

    lo, hi = 1e-6, 1.0
    while f(hi) < 0:
        hi *= 1.5
        if hi > np.sqrt(N):
            raise ValueError(f"cannot reach nbar={nbar} with cutoff N={N}")
    if parity == "odd":
        lo = 1e-6
        if f(lo) > 0:  # odd cat already has nbar > target at alpha -> 0
            return _cat_ket(N, lo, parity)
    alpha = brentq(f, lo, hi, xtol=1e-12, rtol=1e-13)
    return _cat_ket(N, alpha, parity)


def squeezed_cat(N: int, nbar: float, squeeze_fraction: float = 0.5, parity: str = "even") -> np.ndarray:
    r"""Squeezed cat :math:`S(-r)(|\alpha\rangle + |-\alpha\rangle)`.

    ``squeeze_fraction`` :math:`f\in[0,1)` fixes how much of the photon budget
    goes into squeezing (:math:`\sinh^2 r = f\,\bar n`); :math:`\alpha` is then
    solved for so that the total realised :math:`\bar n` hits the target.
    """
    if not 0.0 <= squeeze_fraction < 1.0:
        raise ValueError("squeeze_fraction must lie in [0, 1)")
    if nbar <= 0:
        return _basis(N, 0)
    r = np.arcsinh(np.sqrt(squeeze_fraction * nbar))
    S = squeeze_op(N, -r)

    def ket(alpha):
        return normalize(S @ _cat_ket(N, alpha, parity))

    def f(alpha):
        return mean_photon_number(ket(alpha)) - nbar

    lo, hi = 1e-8, 1.0
    if f(lo) > 0:
        return ket(lo)
    while f(hi) < 0:
        hi *= 1.5
        if hi > np.sqrt(N):
            raise ValueError(f"cannot reach nbar={nbar} with cutoff N={N}")
    alpha = brentq(f, lo, hi, xtol=1e-12, rtol=1e-13)
    return ket(alpha)


STATE_FAMILIES = {
    "vacuum": lambda N, nbar: _basis(N, 0),
    "coherent": coherent,
    "squeezed": squeezed_vacuum,
    "fock": fock,
    "cat_even": lambda N, nbar: cat(N, nbar, "even"),
    "cat_odd": lambda N, nbar: cat(N, nbar, "odd"),
    "squeezed_cat": lambda N, nbar: squeezed_cat(N, nbar, 0.5),
}


def make_state(name: str, N: int, nbar: float) -> np.ndarray:
    """Build a probe state by family name (see :data:`STATE_FAMILIES`)."""
    try:
        builder = STATE_FAMILIES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown state family {name!r}; choose from {sorted(STATE_FAMILIES)}") from exc
    return builder(N, nbar)
