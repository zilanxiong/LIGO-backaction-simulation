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

**Conditioning at large kappa.**  The shear multiplies the covariance
eigenvalues by ~kappa^2 each way, so ``cov`` has condition number ~kappa^4 and
double precision runs out around kappa ~ 1e4.  This is only a problem for
configurations where the shear is the *last* thing to touch the covariance --
injection loss, and BA2 under concurrent loss -- and in exactly those the QFI is
provably independent of kappa, so the caller should evaluate at ``kappa = 0``
instead (``run_study.study_frequency`` does).  Where the answer genuinely
depends on kappa, loss follows the shear and re-conditions it: detection loss at
kappa = 1e5 has condition number 4e11, and the QFI there agrees with an explicit
2x2 inverse to full precision.

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
    "gaussian_qfi_epsilon_a",
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


def _concurrent_gaussian(cov, dmean, kappa, eta_ch, t_final=1.0):
    r"""Exact Gaussian evolution with loss acting *during* the shear.

    Over ``[0, t_final]`` the drift and diffusion are

    .. math::
        A = -\tfrac{\Gamma}{2} I + K, \quad
        K = \begin{pmatrix}0&0\\-\kappa/t_f&0\end{pmatrix}, \quad
        D = \tfrac{\Gamma}{2} I,

    with :math:`\Gamma = -\ln \eta_{\rm ch} / t_f` so that the transmissivity
    over the stage is ``eta_ch``, and :math:`D` fixed by the requirement that
    loss alone relaxes to the vacuum :math:`I/2`.

    ``A`` is a scalar plus a nilpotent, so
    :math:`\Phi(\tau) = e^{A\tau} = e^{-\Gamma\tau/2}(I + K\tau)` exactly, and
    every integral below is elementary -- no matrix exponential and no
    integrator.  This is what lets the Gaussian track run at the ``kappa`` of a
    real interferometer, where the Fock cutoff cannot follow.

    The signal is a constant drive of the ``p`` direction over the whole stage,
    so its accumulated derivative is :math:`\int_0^{t_f}\Phi(\tau)\,d\tau`
    applied to the signal direction -- the continuous analogue of the
    ``(I + A/2)`` factor used in the lossless BA3 branch.
    """
    cov = np.array(cov, dtype=float)
    dmean = np.array(dmean, dtype=float)
    T = float(t_final)
    Gamma = -np.log(eta_ch) / T
    K = np.array([[0.0, 0.0], [-float(kappa) / T, 0.0]])
    signal = np.array([0.0, 1.0]) / T  # unit total displacement over the stage

    Phi = np.exp(-Gamma * T / 2.0) * (np.eye(2) + K * T)

    # int_0^T tau^n exp(-Gamma tau) dtau, for the diffusion integral.
    g = Gamma
    e = np.exp(-g * T)
    m0 = (1.0 - e) / g
    m1 = (1.0 - e * (1.0 + g * T)) / g**2
    m2 = (2.0 - e * (2.0 + 2.0 * g * T + (g * T) ** 2)) / g**3
    KKt = K @ K.T
    KpKt = K + K.T
    noise = (Gamma / 2.0) * (m0 * np.eye(2) + m1 * KpKt + m2 * KKt)

    # int_0^T tau^n exp(-Gamma tau / 2) dtau, for the accumulated signal.
    h = Gamma / 2.0
    eh = np.exp(-h * T)
    c0 = (1.0 - eh) / h
    c1 = (1.0 - eh * (1.0 + h * T)) / h**2
    accum = c0 * np.eye(2) + c1 * K

    return Phi @ cov @ Phi.T + noise, Phi @ dmean + accum @ signal


def gaussian_channel(cov, dmean, kappa=0.0, ordering="BA3", eta_in=1.0,
                     eta_ch=1.0, eta_out=1.0, t_final=1.0):
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
    if ordering not in ("none", "BA1", "BA2", "BA3"):
        raise ValueError(f"unknown ordering {ordering!r}")

    if eta_ch != 1.0:
        # Loss acts during the sensing stage.  BA1/BA2 still put the whole shear
        # outside it; only BA3 has the shear inside.
        ba_now = float(kappa) if ordering == "BA3" else 0.0
        if ordering == "BA1":
            cov, dmean = shear(cov, dmean)
        cov, dmean = _concurrent_gaussian(cov, dmean, ba_now, eta_ch, t_final)
        if ordering == "BA2":
            cov, dmean = shear(cov, dmean)
    elif ordering == "none" or kappa == 0.0:
        dmean = dmean + signal
    elif ordering == "BA1":  # back-action, then signal
        cov, dmean = shear(cov, dmean)
        dmean = dmean + signal
    elif ordering == "BA2":  # signal, then back-action
        dmean = dmean + signal
        cov, dmean = shear(cov, dmean)
    else:  # BA3, simultaneous
        # exp(A) with A nilpotent of order 2, so the accumulated displacement is
        # int_0^1 e^{A(1-t)} dt = I + A/2 applied to the signal direction.
        A = np.array([[0.0, 0.0], [-float(kappa), 0.0]])
        cov, dmean = shear(cov, dmean)
        dmean = dmean + (np.eye(2) + 0.5 * A) @ signal
    if eta_out != 1.0:
        cov, dmean = loss(cov, dmean, eta_out)
    return cov, dmean


def gaussian_qfi_lambda(cov, dmean, **kwargs):
    r"""QFI for a parameter entering only through the first moments.

    .. math:: \mathcal{F}_\lambda = (\partial_\lambda d)^{\mathsf T} V^{-1} (\partial_\lambda d)
    """
    cov, dmean = gaussian_channel(cov, dmean, **kwargs)
    return float(dmean @ np.linalg.solve(cov, dmean))


def gaussian_qfi_epsilon_a(cov, dmean, t_final=1.0, **kwargs):
    """``gaussian_qfi_lambda`` in the units the Fock track reports."""
    return EPS_A_PER_LAMBDA * t_final**2 * gaussian_qfi_lambda(
        cov, dmean, t_final=t_final, **kwargs)
