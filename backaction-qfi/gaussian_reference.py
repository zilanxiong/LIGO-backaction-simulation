"""
Independent Gaussian reference for the back-action channel.

A covariance-matrix implementation of the same channel, written from the
symplectic side rather than the Fock side.  It is *exact* for Gaussian states,
so it is the yardstick the Fock code is checked against in ``test_backaction.py``:
two implementations sharing no code that agree to five digits is much stronger
evidence than either alone.

Phase-space vector is ``(x, p)`` with the vacuum covariance ``I/2``
(so ``Var(x) = Var(p) = 1/2``), matching ``radiation_pressure``.

The estimated parameter here is the dimensionless displacement ``lambda``.  The
package's sensing parameter ``epsilon_a`` drives the generator ``sqrt(2) x`` over
``t_final``, so::

    F_epsilon_a = 2 * t_final**2 * F_lambda

Phase noise is deliberately absent: a random rotation with Gaussian-distributed
angle is *not* a Gaussian channel, so it has no covariance-matrix form.  Those
comparisons stay in the Fock track.
"""

import numpy as np

__all__ = [
    "EPS_A_PER_LAMBDA",
    "vacuum",
    "squeezed_vacuum_moments",
    "shear_matrix",
    "gaussian_channel",
    "gaussian_qfi_lambda",
]

#: F_epsilon_a = EPS_A_PER_LAMBDA * t_final**2 * F_lambda
EPS_A_PER_LAMBDA = 2.0

V_VAC = 0.5 * np.eye(2)


def vacuum():
    """``(cov, dmean)`` for the vacuum, with a zero parameter-derivative."""
    return V_VAC.copy(), np.zeros(2)


def squeezed_vacuum_moments(nbar):
    r"""Squeezed vacuum with :math:`\sinh^2 r = \bar n`, anti-squeezed in :math:`x`."""
    r = np.arcsinh(np.sqrt(max(nbar, 0.0)))
    return 0.5 * np.diag([np.exp(2 * r), np.exp(-2 * r)]), np.zeros(2)


def shear_matrix(kappa):
    r"""Ponderomotive shear :math:`S = \begin{pmatrix}1&0\\-\kappa&1\end{pmatrix}`."""
    return np.array([[1.0, 0.0], [-float(kappa), 1.0]])


def gaussian_channel(cov, dmean, kappa=0.0, ordering="BA3", eta_in=1.0, eta_out=1.0):
    """Apply the same pipeline as ``get_state_single_mode_ba`` to the moments.

    Only ``dmean = d(mean)/d(lambda)`` is tracked: the QFI for a parameter that
    enters through the first moments alone does not depend on where the mean
    itself sits.
    """
    cov, dmean = np.array(cov, dtype=float), np.array(dmean, dtype=float)

    def loss(c, dm, eta):
        return eta * c + (1.0 - eta) * V_VAC, np.sqrt(eta) * dm

    def shear(c, dm):
        S = shear_matrix(kappa)
        return S @ c @ S.T, S @ dm

    # The signal displaces p, so its derivative direction is (0, 1).
    signal = np.array([0.0, 1.0])

    if eta_in != 1.0:
        cov, dmean = loss(cov, dmean, eta_in)
    if ordering == "none" or kappa == 0.0:
        dmean = dmean + signal
    elif ordering == "BA1":  # back-action, then signal
        cov, dmean = shear(cov, dmean)
        dmean = dmean + signal
    elif ordering == "BA2":  # signal, then back-action
        dmean = dmean + signal
        cov, dmean = shear(cov, dmean)
    elif ordering == "BA3":  # simultaneous
        # exp(A) with A nilpotent of order 2, so the accumulated displacement is
        # int_0^1 e^{A(1-t)} dt = I + A/2 applied to the signal direction.
        A = np.array([[0.0, 0.0], [-float(kappa), 0.0]])
        cov, dmean = shear(cov, dmean)
        dmean = dmean + (np.eye(2) + 0.5 * A) @ signal
    else:
        raise ValueError(f"unknown ordering {ordering!r}")
    if eta_out != 1.0:
        cov, dmean = loss(cov, dmean, eta_out)
    return cov, dmean


def gaussian_qfi_lambda(cov, dmean, **kwargs):
    r"""QFI for a parameter entering only through the first moments.

    .. math:: \mathcal{F}_\lambda = (\partial_\lambda d)^{\mathsf T} V^{-1} (\partial_\lambda d)
    """
    cov, dmean = gaussian_channel(cov, dmean, **kwargs)
    return float(dmean @ np.linalg.solve(cov, dmean))
