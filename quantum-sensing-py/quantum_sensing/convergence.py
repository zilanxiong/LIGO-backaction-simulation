"""
Automated Fock-cutoff convergence.

Radiation pressure is a shear: it stretches the :math:`p` quadrature by roughly
:math:`\\kappa` and inflates the output photon number by about
:math:`(1+\\kappa^2)(\\bar n + 1)`.  A cutoff that is perfectly adequate for the
input state -- including the package default ``N_basis=20``, which is fine for
displacement and rotation -- can be badly inadequate after the channel, and
non-Gaussian probes are the worst offenders.  Every user-facing back-action
number should be produced through :func:`converge`.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConvergenceResult", "converge", "converged_qfi", "tail_population"]


@dataclass
class ConvergenceResult:
    """Result of a cutoff ladder.  ``converged=False`` is a hard warning, not a detail."""

    value: float
    cutoff: int
    converged: bool
    rel_change: float
    history: list = field(default_factory=list)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        flag = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"ConvergenceResult(value={self.value:.8g}, cutoff={self.cutoff}, "
            f"rel_change={self.rel_change:.2g}, {flag})"
        )


def _default_ladder(start=20, stop=600, factor=1.4):
    ladder, n = [], int(start)
    while n < stop:
        ladder.append(n)
        n = int(np.ceil(n * factor))
    ladder.append(int(stop))
    return ladder


def converge(fn, cutoffs=None, rtol=1e-3, atol=0.0, n_stable=1):
    """Evaluate ``fn(N_basis)`` on an increasing ladder of cutoffs until it stops moving.

    Parameters
    ----------
    fn : callable
        Takes the Fock cutoff, returns a scalar.
    cutoffs : sequence of int, optional
        Ladder to try.  Defaults to a geometric ladder from 20 to 600.
    rtol, atol : float
        Convergence when ``|v_new - v_old| <= atol + rtol * |v_new|``.
    n_stable : int
        Number of consecutive successful comparisons required.

    Returns
    -------
    ConvergenceResult
    """
    if cutoffs is None:
        cutoffs = _default_ladder()
    history, stable, prev, rel = [], 0, None, np.inf
    for N in cutoffs:
        val = float(fn(N))
        history.append((N, val))
        if prev is not None:
            diff = abs(val - prev)
            rel = diff / max(abs(val), 1e-300)
            if diff <= atol + rtol * abs(val):
                stable += 1
                if stable >= n_stable:
                    return ConvergenceResult(val, N, True, rel, history)
            else:
                stable = 0
        prev = val
    return ConvergenceResult(history[-1][1], history[-1][0], False, rel, history)


def converged_qfi(state_builder, channel, param_type="epsilon_a", param_value=0.0,
                  cutoffs=None, rtol=1e-3, n_stable=1, **channel_kwargs):
    """Convergence-checked QFI through a channel.

    Parameters
    ----------
    state_builder : callable
        ``state_builder(N_basis)`` returns the probe ket, e.g.
        ``lambda N: make_state("cat_even", N, 2.0)``.
    channel : callable
        A dynamics function such as
        :func:`quantum_sensing.radiation_pressure.get_state_single_mode_ba`.
    param_type, param_value : str, float
        Estimated parameter and the operating point, as in
        :func:`quantum_sensing.sld.calculate_qfi`.
    **channel_kwargs
        Forwarded to ``channel`` (``kappa``, ``ordering``, ``eta_out``, ...).

    Returns
    -------
    ConvergenceResult
    """
    from .sld import calculate_qfi

    def value(N_basis):
        return calculate_qfi(
            channel, param_value=param_value, param_type=param_type,
            rho=state_builder(N_basis), N_basis=N_basis, **channel_kwargs,
        )

    return converge(value, cutoffs=cutoffs, rtol=rtol, n_stable=n_stable)


def tail_population(state, n_levels=5):
    """Population in the top ``n_levels`` Fock levels -- a cheap truncation alarm."""
    rho = state.full() if not state.isket else (lambda v: v @ v.conj().T)(state.full())
    return float(np.real(np.trace(rho[-n_levels:, -n_levels:])))
