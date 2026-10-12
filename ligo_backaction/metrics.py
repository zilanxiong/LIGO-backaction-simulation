r"""Broadband cost metrics over an array of sideband frequencies.

Given per-frequency QFI values :math:`\mathcal{F}(\Omega)` (in strain units),
the three metrics suggested in the project plan are

(i)   :math:`C_w = \int w(\Omega)\,/\,\mathcal{F}(\Omega)\; d\Omega`
      -- the waveform-estimation QCRB with the signal spectrum as weight
      (lower is better);
(ii)  :math:`C_{\min} = \min_\Omega \mathcal{F}(\Omega)`
      -- worst case over the band, i.e. broadband flatness (higher is better);
(iii) :math:`C_\Sigma = \sum_\Omega \mathcal{F}(\Omega)`
      -- total information for a fixed frequency set (higher is better).

Caveat on additivity
--------------------
Summing per-frequency QFI is exact for Gaussian states in the linearised
two-photon formalism, where distinct sidebands factorise.  For non-Gaussian
states the frequency modes do not factorise and the sum is only an
approximation -- see :mod:`ligo_backaction.twomode`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "weighted_inverse_cost",
    "worst_case_qfi",
    "total_qfi",
    "power_law_weight",
    "normalize_weight",
]


def normalize_weight(freqs, weight):
    """Normalise ``weight`` to unit integral over ``freqs`` (trapezoidal)."""
    freqs = np.asarray(freqs, dtype=float)
    weight = np.asarray(weight, dtype=float)
    norm = np.trapezoid(weight, freqs)
    if norm <= 0:
        raise ValueError("weight must be positive with non-zero integral")
    return weight / norm


def power_law_weight(freqs, index: float = -4.0 / 3.0, normalize: bool = True):
    r"""Weight :math:`w(f)\propto f^{\,\rm index}`.

    The default :math:`f^{-4/3}` is the amplitude-spectrum scaling of an
    inspiral chirp, a convenient stand-in for a signal spectrum.
    """
    freqs = np.asarray(freqs, dtype=float)
    w = freqs**index
    return normalize_weight(freqs, w) if normalize else w


def weighted_inverse_cost(freqs, qfi, weight=None):
    r"""Metric (i): :math:`\int w/\mathcal{F}\,df` (lower is better)."""
    freqs = np.asarray(freqs, dtype=float)
    qfi = np.asarray(qfi, dtype=float)
    if weight is None:
        weight = power_law_weight(freqs)
    weight = normalize_weight(freqs, weight)
    if np.any(qfi <= 0):
        raise ValueError("QFI must be strictly positive to form the inverse cost")
    return float(np.trapezoid(weight / qfi, freqs))


def worst_case_qfi(qfi):
    """Metric (ii): worst-case QFI over the band (higher is better)."""
    return float(np.min(np.asarray(qfi, dtype=float)))


def total_qfi(qfi):
    """Metric (iii): summed QFI over a fixed frequency set (higher is better)."""
    return float(np.sum(np.asarray(qfi, dtype=float)))
