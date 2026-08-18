r"""Quantum Fisher information.

For a family :math:`\rho(\theta)` the SLD quantum Fisher information is

.. math::
    \mathcal{F} = 2\sum_{j,k\,:\,p_j+p_k>0}
        \frac{|\langle j|\partial_\theta\rho|k\rangle|^2}{p_j + p_k},

with :math:`\rho = \sum_j p_j |j\rangle\langle j|`.  For a pure state under a
unitary :math:`e^{i\theta G}` this reduces to :math:`4\,\mathrm{Var}(G)`.
"""

from __future__ import annotations

import numpy as np

from .channels import ChannelSpec, effective_support, output_and_derivative, run_channel
from .ifo import qfi_h_from_qfi_lambda
from .operators import as_density_matrix, quad_op, variance

__all__ = [
    "qfi_from_rho_drho",
    "qfi_lambda",
    "qfi_strain",
    "pure_state_qfi",
    "purity",
]


def qfi_from_rho_drho(rho: np.ndarray, drho: np.ndarray, eig_tol: float = 1e-10) -> float:
    """SLD QFI from a density matrix and its derivative.

    ``eig_tol`` is a *relative* floor (in units of the largest eigenvalue) on
    :math:`p_j + p_k`; pairs below it are dropped.  Numerically-negative
    eigenvalues from round-off are clipped to zero.
    """
    rho = np.asarray(rho)
    drho = np.asarray(drho)
    # Work only on the occupied block: everything above it is numerically zero
    # in both rho and drho, and the eigendecomposition is the dominant cost.
    s = effective_support(rho, drho) + 1
    if s < rho.shape[0]:
        rho, drho = rho[:s, :s], drho[:s, :s]
    w, V = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    w = np.clip(w, 0.0, None)
    scale = max(w.max(), 1e-300)
    D = V.conj().T @ (0.5 * (drho + drho.conj().T)) @ V
    denom = w[:, None] + w[None, :]
    mask = denom > eig_tol * scale
    return float(2.0 * np.sum(np.abs(D[mask]) ** 2 / denom[mask]))


def pure_state_qfi(psi: np.ndarray, phi: float = 0.0) -> float:
    r""":math:`4\,\mathrm{Var}(\hat q_\varphi)` -- the lossless, noiseless QFI."""
    psi = np.asarray(psi)
    N = psi.shape[0]
    return 4.0 * variance(quad_op(N, phi), psi)


def qfi_lambda(state: np.ndarray, spec: ChannelSpec, eig_tol: float = 1e-10, **kwargs) -> float:
    r"""QFI for the dimensionless displacement :math:`\lambda` through ``spec``."""
    rho, drho = output_and_derivative(state, spec, **kwargs)
    return qfi_from_rho_drho(rho, drho, eig_tol=eig_tol)


def qfi_strain(state: np.ndarray, spec: ChannelSpec, h_sql_value: float, **kwargs) -> float:
    r"""QFI for the strain :math:`h`, :math:`\mathcal{F}_h = (\kappa/h_{\rm SQL}^2)\mathcal{F}_\lambda`."""
    return float(qfi_h_from_qfi_lambda(qfi_lambda(state, spec, **kwargs), spec.kappa, h_sql_value))


def purity(rho: np.ndarray) -> float:
    rho = as_density_matrix(rho)
    return float(np.real(np.trace(rho @ rho)))
