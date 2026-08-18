r"""Automated Fock-cutoff convergence.

Radiation pressure is a *shear*: it stretches the :math:`p` quadrature by
roughly :math:`\kappa` and therefore inflates the photon number of the output
state by a large factor.  A cutoff that is perfectly adequate for the input
state can be badly inadequate after the channel -- and non-Gaussian probes
(cats in particular) are the worst offenders.  Everything user-facing in this
package should therefore be run through :func:`converge`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .operators import as_density_matrix, number

__all__ = ["ConvergenceResult", "converge", "converged_qfi", "tail_population", "suggested_cutoff"]


@dataclass
class ConvergenceResult:
    value: float
    cutoff: int
    converged: bool
    rel_change: float
    history: list[tuple[int, float]] = field(default_factory=list)

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        flag = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"ConvergenceResult(value={self.value:.8g}, cutoff={self.cutoff}, "
            f"rel_change={self.rel_change:.2g}, {flag})"
        )


def converge(
    fn: Callable[[int], float],
    cutoffs: Sequence[int] | None = None,
    rtol: float = 1e-3,
    atol: float = 0.0,
    n_stable: int = 1,
) -> ConvergenceResult:
    """Evaluate ``fn(N)`` on an increasing ladder of cutoffs until it stops moving.

    Parameters
    ----------
    fn : callable taking the Fock cutoff and returning a scalar.
    cutoffs : ladder of cutoffs to try (default: geometric from 20 to 400).
    rtol, atol : convergence is declared when
        ``|v_new - v_old| <= atol + rtol * |v_new|``.
    n_stable : number of *consecutive* successful comparisons required.

    Returns
    -------
    ConvergenceResult
        ``converged`` is ``False`` if the ladder was exhausted first -- callers
        should treat that as a hard warning, not a rounding detail.
    """
    if cutoffs is None:
        cutoffs = _default_ladder()
    history: list[tuple[int, float]] = []
    stable = 0
    prev_val: float | None = None
    rel = np.inf
    for N in cutoffs:
        val = float(fn(N))
        history.append((N, val))
        if prev_val is not None:
            diff = abs(val - prev_val)
            rel = diff / max(abs(val), 1e-300)
            if diff <= atol + rtol * abs(val):
                stable += 1
                if stable >= n_stable:
                    return ConvergenceResult(val, N, True, rel, history)
            else:
                stable = 0
        prev_val = val
    return ConvergenceResult(history[-1][1], history[-1][0], False, rel, history)


def _default_ladder() -> list[int]:
    ladder, N = [], 20
    while N <= 400:
        ladder.append(N)
        N = int(np.ceil(N * 1.5))
    return ladder


def converged_qfi(state_builder: Callable[[int], np.ndarray], spec, **kwargs) -> ConvergenceResult:
    """Convergence-checked :func:`ligo_backaction.qfi.qfi_lambda`.

    ``state_builder(N)`` must return the probe state in dimension ``N``
    (e.g. ``lambda N: make_state("cat_even", N, 2.0)``).
    """
    from .qfi import qfi_lambda

    conv_kwargs = {k: kwargs.pop(k) for k in ("cutoffs", "rtol", "atol", "n_stable") if k in kwargs}
    return converge(lambda N: qfi_lambda(state_builder(N), spec, **kwargs), **conv_kwargs)


def tail_population(state: np.ndarray, n_levels: int = 5) -> float:
    """Population in the top ``n_levels`` Fock levels -- a cheap truncation alarm."""
    rho = as_density_matrix(state)
    return float(np.real(np.trace(rho[-n_levels:, -n_levels:])))


def suggested_cutoff(nbar: float, kappa: float, safety: float = 8.0) -> int:
    r"""Rule-of-thumb cutoff for a probe of mean photon number ``nbar`` under shear ``kappa``.

    The shear multiplies the :math:`p` variance by up to :math:`1+\kappa^2`, so
    the output photon number grows roughly like
    :math:`\bar n_{\rm out}\sim (1+\kappa^2)(\bar n + 1)`.  This is only a
    starting point for :func:`converge`, never a substitute for it.
    """
    return int(np.ceil(safety * (1.0 + kappa**2) * (nbar + 1.0)))
