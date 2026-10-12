"""Two-mode additivity sandbox, cost metrics and interferometer calibration."""

import numpy as np
import pytest

from ligo_backaction.channels import ChannelSpec
from ligo_backaction.ifo import ALIGO, IFOParams, h_sql, kappa_of_omega, qfi_h_from_qfi_lambda
from ligo_backaction.metrics import (
    normalize_weight,
    power_law_weight,
    total_qfi,
    weighted_inverse_cost,
    worst_case_qfi,
)
from ligo_backaction.qfi import qfi_lambda, qfi_strain
from ligo_backaction.states import make_state
from ligo_backaction.twomode import (
    TwoModeSpec,
    additivity_defect,
    entangled_cat,
    joint_qfi_matrix,
    per_frequency_qfi,
    product_state,
    two_mode_squeezed_vacuum,
)

N2 = 14


def test_kappa_is_unity_at_the_cavity_pole():
    assert kappa_of_omega(ALIGO.gamma, ALIGO) == pytest.approx(1.0)
    assert ALIGO.kappa(500.0) == pytest.approx(1.0)


def test_kappa_and_hsql_scalings():
    f = np.array([50.0, 100.0, 200.0])
    assert np.allclose(ALIGO.h_sql(f) * f, ALIGO.h_sql(f[0]) * f[0])  # h_SQL ~ 1/Omega
    low = np.array([1.0, 2.0])  # Omega << gamma: kappa ~ 1/Omega^2
    assert kappa_of_omega(2 * np.pi * low) / kappa_of_omega(2 * np.pi * low[::-1]) == pytest.approx(
        (low[::-1] / low) ** 2, rel=1e-3
    )


def test_strain_qfi_conversion_reproduces_the_variational_bound():
    """Vacuum input: sigma_h >= h_SQL / sqrt(2 kappa)."""
    params = IFOParams()
    f = 500.0
    kappa, hs = float(params.kappa(f)), float(params.h_sql(f))
    spec = ChannelSpec(kappa=kappa, ordering="BA1")
    fh = qfi_strain(make_state("vacuum", 40, 0.0), spec, hs)
    assert 1.0 / np.sqrt(fh) == pytest.approx(hs / np.sqrt(2 * kappa), rel=1e-9)
    assert qfi_h_from_qfi_lambda(2.0, kappa, hs) == pytest.approx(2 * kappa / hs**2)


def test_metrics_basic_behaviour():
    f = np.linspace(20.0, 1000.0, 64)
    flat = np.full_like(f, 4.0)
    assert weighted_inverse_cost(f, flat) == pytest.approx(0.25)
    assert worst_case_qfi(flat) == 4.0
    assert total_qfi(flat) == pytest.approx(4.0 * f.size)
    assert np.trapezoid(power_law_weight(f), f) == pytest.approx(1.0)
    assert np.trapezoid(normalize_weight(f, np.ones_like(f)), f) == pytest.approx(1.0)


def test_metrics_rank_a_flat_probe_above_a_peaked_one_on_the_worst_case():
    f = np.linspace(20.0, 1000.0, 64)
    flat = np.full_like(f, 4.0)
    peaked = 1.0 + 40.0 * np.exp(-(((f - 500.0) / 150.0) ** 2))
    assert total_qfi(peaked) > total_qfi(flat)
    assert worst_case_qfi(flat) > worst_case_qfi(peaked)
    assert weighted_inverse_cost(f, flat) < weighted_inverse_cost(f, peaked)


def test_weighted_cost_rejects_nonpositive_qfi():
    f = np.linspace(20.0, 100.0, 8)
    with pytest.raises(ValueError):
        weighted_inverse_cost(f, np.zeros_like(f))


@pytest.mark.parametrize("family", ["squeezed", "cat_even", "coherent"])
def test_per_frequency_sum_is_exact_for_uncorrelated_modes(family):
    """No cross-frequency coupling and a product input => the joint QFI is additive,
    Gaussian or not."""
    psi = product_state(make_state(family, N2, 1.0), make_state(family, N2, 1.0))
    spec = TwoModeSpec(kappa1=1.0, kappa2=0.6, kappa_cross=0.0, eta=0.8)
    d = additivity_defect(psi, spec, N2)
    assert abs(d["rel_defect"]) < 1e-6
    assert d["max_offdiag"] < 1e-7


@pytest.mark.parametrize("family", ["squeezed", "cat_even"])
def test_cross_frequency_backaction_breaks_additivity(family):
    """Keeping the x1 x2 beat term the per-frequency picture drops changes the answer."""
    psi = product_state(make_state(family, N2, 1.0), make_state(family, N2, 1.0))
    spec = TwoModeSpec(kappa1=1.0, kappa2=1.0, kappa_cross=0.8, eta=0.8)
    d = additivity_defect(psi, spec, N2)
    assert abs(d["rel_defect"]) > 1e-3
    assert d["max_offdiag"] > 1e-3


def test_cross_frequency_backaction_is_harmless_without_loss():
    """Like the single-mode case, the joint shear is unitary and QFI-preserving."""
    psi = product_state(make_state("cat_even", N2, 1.0), make_state("cat_even", N2, 1.0))
    lossless = TwoModeSpec(kappa1=1.0, kappa2=1.0, kappa_cross=0.8, eta=1.0)
    reference = TwoModeSpec(kappa1=1.0, kappa2=1.0, kappa_cross=0.0, eta=1.0)
    assert np.allclose(
        joint_qfi_matrix(psi, lossless, N2), joint_qfi_matrix(psi, reference, N2), atol=1e-8
    )


@pytest.mark.parametrize(
    "state_builder", [lambda N: two_mode_squeezed_vacuum(N, 0.6), lambda N: entangled_cat(N, 1.0)]
)
def test_correlated_inputs_break_additivity(state_builder):
    """Frequency-multiplexed (entangled) preparations are not described by a
    per-frequency channel even when the dynamics factorise."""
    spec = TwoModeSpec(kappa1=1.0, kappa2=1.0, kappa_cross=0.0, eta=0.8)
    d = additivity_defect(state_builder(N2), spec, N2)
    assert abs(d["rel_defect"]) > 1e-2
    assert d["max_offdiag"] > 1e-2


def test_joint_qfi_matrix_is_symmetric_positive_semidefinite():
    psi = entangled_cat(N2, 1.0)
    F = joint_qfi_matrix(psi, TwoModeSpec(1.0, 1.0, 0.4, eta=0.9), N2)
    assert np.allclose(F, F.T, atol=1e-9)
    assert np.linalg.eigvalsh(F).min() > -1e-9


def test_per_frequency_qfi_matches_the_single_mode_code():
    psi = product_state(make_state("squeezed", N2, 1.0), make_state("coherent", N2, 1.0))
    spec = TwoModeSpec(kappa1=1.2, kappa2=1.2, kappa_cross=0.0, eta=0.85)
    per = per_frequency_qfi(psi, spec, N2)
    single = [
        qfi_lambda(make_state(f, N2, 1.0), ChannelSpec(kappa=1.2, ordering="BA1", eta_out=0.85))
        for f in ("squeezed", "coherent")
    ]
    assert np.allclose(per, single, rtol=1e-9)
