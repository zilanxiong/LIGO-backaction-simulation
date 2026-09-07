"""
Band cost metrics for QFI(Omega) curves.

Metric (i) is the waveform-estimation QCRB weighting: the integrated
variance bound is proportional to int w(Omega)/QFI(Omega) dOmega, so the
figure of merit reported is its normalized inverse — the w-weighted
harmonic mean of the QFI over the band (reduces to the plain harmonic mean
for flat weight).  Metric (ii) is the worst-case (min) QFI over the band,
metric (iii) the plain sum over the fixed frequency set.

Weights: flat_weight, and inspiral_weight w = |h(f)|^2 ~ f^{-7/3} for a
leading-order compact-binary inspiral (amplitude h(f) ~ f^{-7/6}).
"""

import numpy as np


def flat_weight(f_hz):
    return np.ones_like(np.asarray(f_hz, dtype=float))


def inspiral_weight(f_hz):
    """|h(f)|^2 for a leading-order inspiral: f^{-7/3} (unnormalized)."""
    return np.asarray(f_hz, dtype=float) ** (-7.0 / 3.0)


def weighted_harmonic_qfi(f_hz, qfi, weight=flat_weight):
    """Metric (i): [int w df / int (w/QFI) df]  via trapezoid on the given
    grid (any order; sorted internally)."""
    f = np.asarray(f_hz, dtype=float)
    q = np.asarray(qfi, dtype=float)
    idx = np.argsort(f)
    f, q = f[idx], q[idx]
    w = weight(f)
    if len(f) == 1:
        return float(q[0])
    return float(np.trapezoid(w, f) / np.trapezoid(w / q, f))


def band_metrics(f_hz, qfi, weight=flat_weight):
    """All three metrics for one QFI(f) curve."""
    q = np.asarray(qfi, dtype=float)
    return {
        "metric_i": weighted_harmonic_qfi(f_hz, qfi, weight),
        "metric_ii": float(q.min()),
        "metric_iii": float(q.sum()),
    }
