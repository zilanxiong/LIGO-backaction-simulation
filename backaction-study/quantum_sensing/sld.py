"""
Quantum Fisher Information, Symmetric Logarithmic Derivative, and wavefunction utilities.

Merged from sld_helper.py (numpy-based SLD/QFI) and qfi_helper.py (wavefunction tools).
"""

import numpy as np
from numpy.linalg import eigh

PREC_DIFF = 1e-5
PREC_EIG = 1e-8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_dense_dm(dynamics, params):
    """Call dynamics function and return a dense numpy matrix."""
    state = dynamics(**params)
    if state.isket:
        v = state.full()
        dm = v @ v.conj().T
    else:
        dm = state.full()
    return dm


# ---------------------------------------------------------------------------
# SLD calculation
# ---------------------------------------------------------------------------

def calculate_sld(dynamics, param_value=0.0, param_type="Delta",
                  prec=PREC_DIFF, prec_eig=PREC_EIG, **kwargs):
    """
    Calculate the Symmetric Logarithmic Derivative (SLD) operator.

    Parameters
    ----------
    dynamics : callable
        Function that returns a QuTiP Qobj state rho(theta).  Called as
        ``dynamics(**kwargs)`` with the sensing parameter overridden.
    param_value : float
        Parameter value theta at which to evaluate the SLD.
    param_type : str
        Name of the keyword argument to vary (e.g. "Delta").
    prec : float
        Step size for the central-difference derivative.
    prec_eig : float
        Threshold below which (p_i + p_j) is treated as zero.
    **kwargs
        Additional keyword arguments forwarded to *dynamics*.

    Returns
    -------
    dict with keys:
        L              - SLD matrix (numpy ndarray)
        eigenvalues_L  - eigenvalues of L
        eigenvectors_L - eigenvectors of L (columns)
        qfi            - Quantum Fisher Information  Tr[rho L^2]
        rho            - density matrix rho(theta)  (numpy ndarray)
        eigenvalues_rho  - eigenvalues of rho
        eigenvectors_rho - eigenvectors of rho (columns)
        drho           - derivative d_rho/d_theta  (numpy ndarray)
    """
    # Central difference
    params_plus = {**kwargs, param_type: param_value + prec}
    params_minus = {**kwargs, param_type: param_value - prec}
    params_center = {**kwargs, param_type: param_value}

    rho_plus = _get_dense_dm(dynamics, params_plus)
    rho_minus = _get_dense_dm(dynamics, params_minus)
    drho = (rho_plus - rho_minus) / (2 * prec)

    rho = _get_dense_dm(dynamics, params_center)

    # Eigendecomposition of rho (Hermitian)
    evals, evecs = eigh(rho)

    # Rotate derivative into eigenbasis
    drho_rot = evecs.conj().T @ drho @ evecs

    # Construct SLD in rotated basis
    n = len(evals)
    L_rot = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            denom = evals[i] + evals[j]
            if abs(denom) > prec_eig:
                L_rot[i, j] = 2.0 * drho_rot[i, j] / denom

    # Rotate back
    L = evecs @ L_rot @ evecs.conj().T

    # Enforce Hermiticity
    L = (L + L.conj().T) / 2.0

    # QFI = Tr[rho L^2]
    qfi = np.real(np.trace(rho @ L @ L))

    # Diagonalise L for optimal measurement basis
    L_evals, L_evecs = eigh(L)

    return {
        "L": L,
        "eigenvalues_L": L_evals,
        "eigenvectors_L": L_evecs,
        "qfi": qfi,
        "rho": rho,
        "eigenvalues_rho": evals,
        "eigenvectors_rho": evecs,
        "drho": drho,
    }


# ---------------------------------------------------------------------------
# QFI calculation (scalar, without full SLD)
# ---------------------------------------------------------------------------

def calculate_qfi(dynamics, param_value=0.0, param_type="Delta",
                  prec=PREC_DIFF, prec_eig=PREC_EIG, **kwargs):
    """
    Compute the Quantum Fisher Information (scalar) without building the full SLD.

    Uses the direct summation formula:
        F_Q = sum_{i,j}  2 |<i|d_rho/d_theta|j>|^2 / (p_i + p_j)

    Parameters are the same as :func:`calculate_sld`.

    Returns
    -------
    float
        Quantum Fisher Information F_Q(theta).
    """
    params_plus = {**kwargs, param_type: param_value + prec}
    params_minus = {**kwargs, param_type: param_value - prec}
    params_center = {**kwargs, param_type: param_value}

    rho_plus = _get_dense_dm(dynamics, params_plus)
    rho_minus = _get_dense_dm(dynamics, params_minus)
    drho = (rho_plus - rho_minus) / (2 * prec)

    rho = _get_dense_dm(dynamics, params_center)

    # Eigendecomposition
    evals, evecs = eigh(rho)

    # Rotate derivative
    drho_rot = evecs.conj().T @ drho @ evecs

    # QFI via direct sum
    qfi = 0.0
    for i in range(len(evals)):
        for j in range(len(evals)):
            denom = evals[i] + evals[j]
            if abs(denom) > prec_eig:
                qfi += 2.0 * abs(drho_rot[i, j]) ** 2 / denom

    return float(np.real(qfi))
