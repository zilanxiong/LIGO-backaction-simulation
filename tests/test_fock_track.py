"""Fock-space track: agreement with the Gaussian track, channel algebra, states."""

import numpy as np
import pytest

from ligo_backaction.channels import ChannelSpec, dephasing_map, loss_map, output_and_derivative, run_channel
from ligo_backaction.convergence import converge, converged_qfi, suggested_cutoff, tail_population
from ligo_backaction.gaussian import gaussian_qfi_lambda, squeezed_vacuum_gaussian, coherent_gaussian, vacuum
from ligo_backaction.operators import backaction_unitary, p_op, signal_unitary, x_op
from ligo_backaction.optimize import optimal_no_backaction_state
from ligo_backaction.qfi import pure_state_qfi, qfi_lambda, purity
from ligo_backaction.states import STATE_FAMILIES, make_state, mean_photon_number

N_BIG = 220


def _trunc(op, m):
    return op[:m, :m]


@pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0])
def test_backaction_unitary_implements_the_shear(kappa):
    """U^dag x U = x and U^dag p U = p - kappa x.

    The cutoff has to be generous: the shear inflates the p variance by
    1 + kappa^2, so verifying the relation even on the lowest 20 levels needs a
    space several times larger.
    """
    N, m = 300, 20
    U = backaction_unitary(N, kappa)
    x, p = x_op(N), p_op(N)
    assert np.allclose(_trunc(U.conj().T @ x @ U, m), _trunc(x, m), atol=1e-9)
    assert np.allclose(_trunc(U.conj().T @ p @ U, m), _trunc(p - kappa * x, m), atol=1e-9)


def test_signal_unitary_displaces_p():
    N, m, lam = 120, 25, 0.4
    U = signal_unitary(N, lam)
    p = p_op(N)
    assert np.allclose(_trunc(U.conj().T @ p @ U, m), _trunc(p + lam * np.eye(N), m), atol=1e-9)


def test_channels_are_trace_preserving():
    N = 60
    psi = make_state("cat_even", N, 2.0)
    spec = ChannelSpec(kappa=1.0, lam=0.3, ordering="BA3", eta_in=0.8, eta_out=0.7, sigma_in=0.1, sigma_out=0.2)
    rho = run_channel(psi, spec)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_loss_map_limits():
    N = 30
    psi = make_state("coherent", N, 1.5)
    rho = np.outer(psi, psi.conj())
    assert np.allclose(loss_map(N, 1.0)(rho), rho)
    out = loss_map(N, 0.0)(rho)
    assert out[0, 0].real == pytest.approx(1.0)
    assert np.trace(out).real == pytest.approx(1.0)


def test_dephasing_kills_coherences_but_not_populations():
    N = 30
    psi = make_state("cat_even", N, 2.0)
    rho = np.outer(psi, psi.conj())
    out = dephasing_map(N, 2.0)(rho)
    assert np.allclose(np.diag(out), np.diag(rho))
    assert np.abs(out - np.diag(np.diag(out))).max() < np.abs(rho - np.diag(np.diag(rho))).max()


@pytest.mark.parametrize("family", sorted(STATE_FAMILIES))
@pytest.mark.parametrize("nbar", [1.0, 3.0])
def test_states_hit_their_target_photon_number(family, nbar):
    N = 160
    psi = make_state(family, N, nbar)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    expected = 0.0 if family == "vacuum" else nbar
    assert mean_photon_number(psi) == pytest.approx(expected, abs=1e-6)


def test_squeezed_vacuum_is_the_optimal_displacement_probe():
    """max 4Var(x) at fixed nbar equals 2(2n+1+2 sqrt(n(n+1))), attained by squeezed vacuum."""
    for nbar in (0.5, 1.0, 3.0):
        N = int(40 + 25 * nbar)
        opt = optimal_no_backaction_state(N, nbar)
        theory = 2 * (2 * nbar + 1 + 2 * np.sqrt(nbar * (nbar + 1)))
        assert pure_state_qfi(opt) == pytest.approx(theory, rel=1e-5)
        assert abs(np.vdot(opt, make_state("squeezed", N, nbar))) == pytest.approx(1.0, abs=1e-4)
        families = ["coherent", "fock", "cat_even", "squeezed_cat"] + (["cat_odd"] if nbar >= 1 else [])
        for family in families:
            assert pure_state_qfi(make_state(family, N, nbar)) <= theory + 1e-6


@pytest.mark.parametrize("kappa", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("ordering", ["none", "BA1", "BA2", "BA3"])
@pytest.mark.parametrize("eta_in,eta_out", [(1.0, 1.0), (0.7, 1.0), (1.0, 0.7), (0.85, 0.9)])
def test_fock_matches_gaussian_for_gaussian_states(kappa, ordering, eta_in, eta_out):
    nbar, r = 1.0, np.arcsinh(1.0)
    # The shear inflates the output photon number by ~(1 + kappa^2), so the
    # cutoff has to grow with kappa for the Fock answer to be trustworthy.
    N = N_BIG + 60 * int(kappa)
    spec = ChannelSpec(kappa=kappa, ordering=ordering, eta_in=eta_in, eta_out=eta_out)
    expected = gaussian_qfi_lambda(squeezed_vacuum_gaussian(r), spec)
    got = qfi_lambda(make_state("squeezed", N, nbar), spec)
    assert got == pytest.approx(expected, rel=2e-5)


@pytest.mark.parametrize("phi", [0.0, 0.4, np.pi / 2])
@pytest.mark.parametrize("ordering", ["none", "BA1", "BA2", "BA3"])
def test_fock_matches_gaussian_for_a_rotated_signal_quadrature(phi, ordering):
    spec = ChannelSpec(kappa=1.3, lam=0.25, phi_sig=phi, ordering=ordering, eta_out=0.8)
    expected = gaussian_qfi_lambda(coherent_gaussian(0.7), spec)
    got = qfi_lambda(make_state("coherent", N_BIG, 0.49), spec)
    assert got == pytest.approx(expected, rel=5e-5)


@pytest.mark.parametrize("family", ["squeezed", "cat_even", "fock", "squeezed_cat"])
def test_backaction_preserves_qfi_without_loss(family):
    """The ponderomotive shear commutes with the p-displacement generator."""
    N, nbar = 200, 2.0
    psi = make_state(family, N, nbar)
    base = pure_state_qfi(psi)
    for ordering in ("BA1", "BA2", "BA3"):
        got = qfi_lambda(psi, ChannelSpec(kappa=1.0, ordering=ordering))
        assert got == pytest.approx(base, rel=1e-6)


@pytest.mark.parametrize("family", ["squeezed", "cat_even", "coherent"])
def test_all_three_orderings_coincide_for_the_physical_signal_quadrature(family):
    """phi_sig = 0 makes [x^2, x] = 0, so BA1 = BA2 = BA3 exactly, even with loss and phase noise."""
    N = 200
    psi = make_state(family, N, 1.5)
    vals = [
        qfi_lambda(psi, ChannelSpec(kappa=1.4, lam=0.3, ordering=o, eta_in=0.9, eta_out=0.8, sigma_out=0.05))
        for o in ("BA1", "BA2", "BA3")
    ]
    assert np.allclose(vals, vals[0], rtol=1e-6)


def test_orderings_differ_when_the_generators_do_not_commute():
    N = 200
    psi = make_state("squeezed", N, 1.0)
    spec = dict(kappa=1.5, lam=0.3, phi_sig=np.pi / 2, eta_out=0.8)
    v1 = qfi_lambda(psi, ChannelSpec(ordering="BA1", **spec))
    v2 = qfi_lambda(psi, ChannelSpec(ordering="BA2", **spec))
    v3 = qfi_lambda(psi, ChannelSpec(ordering="BA3", **spec))
    assert abs(v1 - v2) > 1e-3
    assert min(v1, v2) - 1e-9 <= v3 <= max(v1, v2) + 1e-9


def test_exact_derivative_matches_finite_difference():
    N = 120
    psi = make_state("cat_even", N, 1.5)
    spec = ChannelSpec(kappa=0.9, lam=0.2, ordering="BA1", eta_out=0.85, sigma_out=0.1)
    _, drho = output_and_derivative(psi, spec)
    d = 1e-5
    plus = run_channel(psi, ChannelSpec(**{**spec.__dict__, "lam": spec.lam + d}))
    minus = run_channel(psi, ChannelSpec(**{**spec.__dict__, "lam": spec.lam - d}))
    assert np.allclose(drho, (plus - minus) / (2 * d), atol=1e-7)


def test_qfi_is_independent_of_the_operating_point_for_a_displacement():
    N = 150
    psi = make_state("squeezed", N, 1.0)
    vals = [qfi_lambda(psi, ChannelSpec(kappa=1.0, lam=lam, ordering="BA1", eta_out=0.8)) for lam in (0.0, 0.5, 1.0)]
    assert np.allclose(vals, vals[0], rtol=1e-6)


def test_loss_and_dephasing_never_increase_the_qfi():
    N = 180
    psi = make_state("cat_even", N, 2.0)
    base = qfi_lambda(psi, ChannelSpec(kappa=1.0, ordering="BA1"))
    for extra in (dict(eta_out=0.9), dict(eta_in=0.9), dict(sigma_out=0.2), dict(sigma_in=0.2)):
        assert qfi_lambda(psi, ChannelSpec(kappa=1.0, ordering="BA1", **extra)) <= base + 1e-8


def test_convergence_helper_flags_an_unconverged_ladder():
    good = converge(lambda N: 1.0 + 1.0 / N**2, cutoffs=[20, 40, 80], rtol=1e-2)
    assert good.converged and good.cutoff == 40
    bad = converge(lambda N: float(N), cutoffs=[20, 40, 80], rtol=1e-6)
    assert not bad.converged and bad.cutoff == 80


def test_converged_qfi_reproduces_the_gaussian_answer():
    spec = ChannelSpec(kappa=1.7, ordering="BA1", eta_out=0.7)
    expected = gaussian_qfi_lambda(squeezed_vacuum_gaussian(np.arcsinh(1.0)), spec)
    res = converged_qfi(lambda N: make_state("squeezed", N, 1.0), spec, rtol=1e-5, n_stable=2)
    assert res.converged
    assert res.value == pytest.approx(expected, rel=1e-3)


def test_suggested_cutoff_grows_with_kappa_and_nbar():
    assert suggested_cutoff(1.0, 0.0) < suggested_cutoff(1.0, 3.0)
    assert suggested_cutoff(1.0, 1.0) < suggested_cutoff(5.0, 1.0)


def test_tail_population_alarm():
    N = 40
    small = tail_population(make_state("coherent", N, 1.0))
    big = tail_population(run_channel(make_state("cat_even", N, 4.0), ChannelSpec(kappa=3.0, ordering="BA1")))
    assert small < 1e-12
    assert big > small


def test_purity_bounds():
    N = 60
    psi = make_state("squeezed", N, 1.0)
    assert purity(psi) == pytest.approx(1.0)
    assert purity(run_channel(psi, ChannelSpec(kappa=1.0, ordering="BA1", eta_out=0.6))) < 1.0
