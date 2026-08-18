r"""Quantum Fisher information of optical states under radiation-pressure back-action.

The package is organised as two parallel tracks over the same channel
specification:

* a **Fock-space track** (:mod:`~ligo_backaction.channels`,
  :mod:`~ligo_backaction.qfi`) that handles arbitrary non-Gaussian probe states,
  loss and phase noise, with automated cutoff convergence
  (:mod:`~ligo_backaction.convergence`);
* a **Gaussian track** (:mod:`~ligo_backaction.gaussian`) that is exact for
  Gaussian states and reproduces the textbook SQL and frequency-dependent
  squeezing curves; it doubles as the analytic unit test of the Fock code.

See :mod:`ligo_backaction.conventions` for quadrature and sign conventions.
"""

from .channels import ChannelSpec, ORDERINGS, run_channel, output_and_derivative
from .convergence import converge, converged_qfi, suggested_cutoff
from .gaussian import (
    GaussianState,
    fd_squeeze_angle,
    gaussian_qfi_lambda,
    homodyne_noise_spectrum,
    optimal_squeeze_angle,
    run_gaussian_channel,
    shear_to_squeeze_rotation,
    squeezed_vacuum_at_fixed_nbar,
    squeezed_vacuum_gaussian,
    vacuum,
)
from .ifo import ALIGO, IFOParams, kappa_of_omega, h_sql, qfi_h_from_qfi_lambda
from .metrics import total_qfi, weighted_inverse_cost, worst_case_qfi
from .optimize import optimal_no_backaction_state, optimize_state_for_channel
from .qfi import pure_state_qfi, qfi_from_rho_drho, qfi_lambda, qfi_strain
from .states import STATE_FAMILIES, make_state, mean_photon_number

__version__ = "0.1.0"

__all__ = [
    "ALIGO",
    "ChannelSpec",
    "GaussianState",
    "IFOParams",
    "ORDERINGS",
    "STATE_FAMILIES",
    "converge",
    "converged_qfi",
    "fd_squeeze_angle",
    "gaussian_qfi_lambda",
    "h_sql",
    "homodyne_noise_spectrum",
    "kappa_of_omega",
    "make_state",
    "mean_photon_number",
    "optimal_no_backaction_state",
    "optimal_squeeze_angle",
    "squeezed_vacuum_at_fixed_nbar",
    "optimize_state_for_channel",
    "output_and_derivative",
    "pure_state_qfi",
    "qfi_from_rho_drho",
    "qfi_h_from_qfi_lambda",
    "qfi_lambda",
    "qfi_strain",
    "run_channel",
    "run_gaussian_channel",
    "shear_to_squeeze_rotation",
    "squeezed_vacuum_gaussian",
    "suggested_cutoff",
    "total_qfi",
    "vacuum",
    "weighted_inverse_cost",
    "worst_case_qfi",
]
