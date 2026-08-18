"""Analytic Gaussian benchmarks: SQL, variational readout, FD squeezing, shear algebra."""

import numpy as np
import pytest

from ligo_backaction.gaussian import (
    fd_squeeze_angle,
    gaussian_qfi_lambda,
    homodyne_noise_spectrum,
    rotation_matrix,
    shear_matrix,
    shear_to_squeeze_rotation,
    squeeze_matrix,
    squeezed_vacuum_gaussian,
    vacuum,
)
from ligo_backaction.channels import ChannelSpec

KAPPAS = np.array([0.1, 0.5, 1.0, 2.0, 5.0])


@pytest.mark.parametrize("kappa", KAPPAS)
def test_shear_decomposes_into_rotation_and_squeezing(kappa):
    r, pre, post = shear_to_squeeze_rotation(kappa)
    rebuilt = rotation_matrix(post) @ squeeze_matrix(r) @ rotation_matrix(pre)
    assert np.allclose(rebuilt, shear_matrix(kappa), atol=1e-12)
    # The squeeze factor obeys e^r - e^-r = kappa.
    assert np.exp(r) - np.exp(-r) == pytest.approx(kappa, rel=1e-12, abs=1e-12)


def test_shear_is_symplectic():
    for kappa in KAPPAS:
        S = shear_matrix(kappa)
        assert np.linalg.det(S) == pytest.approx(1.0)


def test_sql_curve():
    """Phase-quadrature readout of vacuum gives (1+kappa^2)/(2 kappa), minimum 1 at kappa=1."""
    s = homodyne_noise_spectrum(KAPPAS)
    assert np.allclose(s, (1 + KAPPAS**2) / (2 * KAPPAS))
    assert homodyne_noise_spectrum(1.0) == pytest.approx(1.0)
    grid = np.unique(np.concatenate([np.linspace(0.05, 20, 4001), [1.0]]))
    assert np.min(homodyne_noise_spectrum(grid)) == pytest.approx(1.0, rel=1e-12)


def test_variational_readout_reaches_the_qcrb():
    """tan(zeta) = kappa cancels back-action and saturates the vacuum QFI bound."""
    s = homodyne_noise_spectrum(KAPPAS, zeta=np.arctan(KAPPAS))
    assert np.allclose(s, 1.0 / (2 * KAPPAS))
    qfi = np.array([gaussian_qfi_lambda(vacuum(), ChannelSpec(kappa=k, ordering="BA1")) for k in KAPPAS])
    assert np.allclose(1.0 / (qfi * KAPPAS), 1.0 / (2 * KAPPAS))


@pytest.mark.parametrize("r", [0.0, 0.5, 1.15])
def test_frequency_dependent_squeezing_curve(r):
    """Filter-cavity squeezing scales the whole SQL curve by exp(-2r)."""
    s = homodyne_noise_spectrum(KAPPAS, r=r, squeeze_angle=fd_squeeze_angle(KAPPAS))
    assert np.allclose(s, np.exp(-2 * r) * (1 + KAPPAS**2) / (2 * KAPPAS))


def test_frozen_squeeze_angle_is_worse_than_fd_squeezing():
    """A frequency-independent squeeze angle cannot beat the FD-squeezed curve."""
    r = 1.0
    fd = homodyne_noise_spectrum(KAPPAS, r=r, squeeze_angle=fd_squeeze_angle(KAPPAS))
    for angle in np.linspace(-np.pi, np.pi, 41):
        assert np.all(homodyne_noise_spectrum(KAPPAS, r=r, squeeze_angle=angle) >= fd - 1e-12)


def test_vacuum_qfi_is_two():
    assert gaussian_qfi_lambda(vacuum(), ChannelSpec(kappa=0.0, ordering="none")) == pytest.approx(2.0)


@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("eta", [1.0, 0.9, 0.5])
def test_analytic_backaction_then_loss(kappa, eta):
    """Vacuum, BA -> signal -> detection loss:  F = 2 eta / (1 + eta(1-eta) kappa^2)."""
    spec = ChannelSpec(kappa=kappa, ordering="BA1", eta_out=eta)
    expected = 2 * eta / (1 + eta * (1 - eta) * kappa**2)
    assert gaussian_qfi_lambda(vacuum(), spec) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_backaction_alone_does_not_change_qfi(kappa):
    """Radiation pressure is unitary and commutes with the p-displacement generator."""
    st = squeezed_vacuum_gaussian(0.8)
    base = gaussian_qfi_lambda(st, ChannelSpec(kappa=0.0, ordering="none"))
    for ordering in ("BA1", "BA2", "BA3"):
        got = gaussian_qfi_lambda(st, ChannelSpec(kappa=kappa, ordering=ordering))
        assert got == pytest.approx(base, rel=1e-12)


@pytest.mark.parametrize("r", [0.0, 0.7, 1.4])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 3.0])
def test_injection_loss_is_blind_to_backaction(r, kappa):
    """Injection loss: the QFI is independent of kappa, F = 2/(eta e^{-2r} + 1 - eta).

    Note this is *not* 4 Var(x): the state is mixed after loss, and the QFI of a
    mixed Gaussian state is (d_lambda d)^T V^{-1} (d_lambda d) <= 4 Var(x).
    """
    eta = 0.7
    st = squeezed_vacuum_gaussian(r)
    got = gaussian_qfi_lambda(st, ChannelSpec(kappa=kappa, ordering="BA1", eta_in=eta))
    assert got == pytest.approx(2.0 / (eta * np.exp(-2 * r) + 1 - eta), rel=1e-12)


def test_detection_loss_is_the_one_that_couples_to_backaction():
    """Injection loss leaves the QFI kappa-independent; detection loss does not."""
    st = squeezed_vacuum_gaussian(1.0)
    inj = [gaussian_qfi_lambda(st, ChannelSpec(kappa=k, ordering="BA1", eta_in=0.8)) for k in KAPPAS]
    det = [gaussian_qfi_lambda(st, ChannelSpec(kappa=k, ordering="BA1", eta_out=0.8)) for k in KAPPAS]
    assert np.allclose(inj, inj[0], rtol=1e-12)
    assert np.all(np.diff(det) < 0)
