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
                  tail_tol=TAIL_TOL, check_output_tail=True, verbose=False,
                  **channel_kwargs):
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
        (input state always; output state if check_output_tail).
    check_output_tail : bool
        Set False when everything after the last dissipative stage is
        unitary (e.g. injection-loss configs, or lossless channels): the
        QFI is invariant under the (truncated, still unitary) final stage,
        so shear-pumped output population near the cutoff cannot bias it
        and would only inflate N.  The input-tail check still guards state
        representability.

    The accepted cutoff is the FIRST N of the run of n_agree+1 mutually
    agreeing evaluations whose tail checks pass — so warm-starting a
    subsequent call at that N does not ratchet the cutoff upward.

    Returns
    -------
    dict with qfi, N_basis, converged (bool), history (list of
    (N, qfi, tail_in, tail_out) tuples).
    """
    if dynamics is None:
        dynamics = get_state_single_mode_rp

    history = []
    agree = 0
    N = int(N_start)
    step = int(N_step)

    def _tails_ok(entry):
        _, _, tail_in, tail_out = entry
        return tail_in < tail_tol and \
            (not check_output_tail or tail_out < tail_tol)

    while True:
        psi = state_factory(N)
        tail_in = tail_population(psi)
        qfi = calculate_qfi(dynamics, param_type=param_type,
                            rho=psi, N_basis=N, **channel_kwargs)
        if check_output_tail:
            rho_out = dynamics(rho=psi, N_basis=N, **channel_kwargs)
            tail_out = tail_population(rho_out)
        else:
            tail_out = 0.0
        history.append((N, qfi, tail_in, tail_out))
        if verbose:
            print(f"  N={N:4d}  qfi={qfi:.8g}  tail_in={tail_in:.2e}  "
                  f"tail_out={tail_out:.2e}")

        if len(history) > 1:
            rel = abs(qfi - history[-2][1]) / max(abs(qfi), 1e-300)
            agree = agree + 1 if rel < rtol else 0

        if agree >= n_agree:
            # accept the earliest entry of the agreeing run that passes
            # the tail checks
            for entry in history[-(agree + 1):]:
                if _tails_ok(entry):
                    return {"qfi": entry[1], "N_basis": entry[0],
                            "converged": True, "history": history}
        if N >= N_max:
            return {"qfi": qfi, "N_basis": N, "converged": False,
                    "history": history}
        N = min(int(N + step), int(N_max))
        step = int(np.ceil(step * 1.5))
