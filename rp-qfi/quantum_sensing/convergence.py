"""
Automated Fock-cutoff convergence for QFI calculations.

The ponderomotive shear H_BA ~ x^2 pumps population to higher Fock levels,
so a cutoff that comfortably holds the input state can still truncate the
output — silently, and worst for cat / squeezed-cat states.  This module
grows N_basis until the QFI is stable AND the output state has negligible
population near the cutoff.

Usage:
    res = converged_qfi(
        state_factory=lambda N: qt.coherent(N, 2.0),
        param_type="epsilon_a", kappa_ba=1.0)
    res["qfi"], res["N_basis"], res["converged"], res["history"]
"""

import numpy as np
import qutip as qt

from .dynamics import get_state_single_mode_rp
from .sld import calculate_qfi

# Fraction of the highest Fock levels treated as the "tail" for the leakage
# check, and the maximum population allowed there.
TAIL_FRACTION = 0.1
TAIL_TOL = 1e-6


def tail_population(rho, fraction=TAIL_FRACTION):
    """Population of the top `fraction` of Fock levels of a state."""
    rho = qt.ket2dm(rho) if rho.isket else rho
    N = rho.shape[0]
    pops = np.real(np.diag(rho.full()))
    n_tail = max(1, int(np.ceil(fraction * N)))
    return float(pops[-n_tail:].sum())


def converged_qfi(state_factory, *, dynamics=None, param_type="epsilon_a",
                  N_start=20, N_step=10, N_max=200, rtol=1e-4, n_agree=2,
                  tail_tol=TAIL_TOL, verbose=False, **channel_kwargs):
    """Compute QFI with an automated Fock-cutoff convergence check.

    Parameters
    ----------
    state_factory : callable N_basis -> Qobj
        Rebuilds the input state (ket or dm) at each cutoff.
    dynamics : callable, default get_state_single_mode_rp
        Channel; called with rho, N_basis, and **channel_kwargs.
    param_type : str
        Sensing parameter differentiated for the QFI.
    N_start, N_step, N_max : int
        Cutoff schedule (N_step also grows geometrically x1.5 after each
        failed check, so pathological cases don't crawl).
    rtol : float
        Relative QFI tolerance between successive cutoffs.
    n_agree : int
        Number of consecutive cutoff increases that must agree within rtol.
    tail_tol : float
        Maximum allowed population in the top TAIL_FRACTION of Fock levels
        of the *output* state at the accepted cutoff.

    Returns
    -------
    dict with qfi, N_basis, converged (bool), history (list of
    (N, qfi, tail_pop) tuples).
    """
    if dynamics is None:
        dynamics = get_state_single_mode_rp

    history = []
    qfi_prev = None
    agree = 0
    N = int(N_start)
    step = int(N_step)

    while True:
        psi = state_factory(N)
        qfi = calculate_qfi(dynamics, param_type=param_type,
                            rho=psi, N_basis=N, **channel_kwargs)
        rho_out = dynamics(rho=psi, N_basis=N, **channel_kwargs)
        tail = tail_population(rho_out)
        history.append((N, qfi, tail))
        if verbose:
            print(f"  N={N:4d}  qfi={qfi:.8g}  tail={tail:.2e}")

        if qfi_prev is not None:
            rel = abs(qfi - qfi_prev) / max(abs(qfi), 1e-300)
            agree = agree + 1 if rel < rtol else 0
        qfi_prev = qfi

        if agree >= n_agree and tail < tail_tol:
            return {"qfi": qfi, "N_basis": N, "converged": True,
                    "history": history}
        if N >= N_max:
            return {"qfi": qfi, "N_basis": N, "converged": False,
                    "history": history}
        N = min(int(N + step), int(N_max))
        step = int(np.ceil(step * 1.5))
