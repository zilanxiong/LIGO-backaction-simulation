"""Back-action channel: algebra, orderings, loss placement, states.

Run with ``pytest backaction-qfi``.

Two independent references are used:

* closed-form Gaussian results (derived in the module docstrings), and
* :mod:`gaussian_reference`, a covariance-matrix implementation of the same
  channel that shares no code with the Fock track and is exact for Gaussian
  states.

The QFI here is always for ``epsilon_a`` -- the signal quadrature of the group's
sensing Hamiltonian.  Its relation to the dimensionless displacement ``lambda``
used by the Gaussian track is ``F_{epsilon_a} = 2 t_final^2 F_lambda`` (see
``EPS_A_PER_LAMBDA``), because ``epsilon_a`` drives the generator ``sqrt(2) x``.
"""

import numpy as np
import pytest
import qutip as qt

from convergence import converge, converged_qfi, tail_population
from dynamics import get_state_single_mode
from gaussian_reference import (
    EPS_A_PER_LAMBDA,
    gaussian_qfi_lambda,
    squeezed_vacuum_moments,
    vacuum,
)
from probe_states import (
    STATE_FAMILIES,
    make_state,
    mean_photon_number,
    optimal_no_backaction,
)
from radiation_pressure import (
    ORDERINGS,
    backaction_unitary,
    get_state_single_mode_ba,
    p_quadrature,
    shear_as_squeeze_rotation,
    suggested_cutoff,
    x_quadrature,
)
from sld import calculate_qfi, calculate_sld

KAPPAS = [0.0, 0.5, 1.0, 2.0, 3.0]


def qfi(state, **kwargs):
    """QFI for ``epsilon_a`` through the back-action channel."""
    kwargs.setdefault("N_basis", state.shape[0])
    return calculate_qfi(
        get_state_single_mode_ba, param_value=0.0, param_type="epsilon_a", rho=state, **kwargs
    )


# ---------------------------------------------------------------------------
# Operator algebra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kappa", [0.5, 2.0])
def test_backaction_unitary_implements_the_shear(kappa):
    """U^dag x U = x and U^dag p U = p - kappa x.

    The cutoff has to be generous: the shear inflates the p variance by
    1 + kappa^2, so checking even the lowest 20 levels needs several hundred.
    """
    N, m = 300, 20
    U = backaction_unitary(N, kappa)
    x, p = x_quadrature(N), p_quadrature(N)
    assert np.allclose((U.dag() * x * U).full()[:m, :m], x.full()[:m, :m], atol=1e-9)
    assert np.allclose((U.dag() * p * U).full()[:m, :m], (p - kappa * x).full()[:m, :m], atol=1e-9)
    assert np.allclose((U.dag() * U).full(), np.eye(N), atol=1e-10)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_shear_decomposes_into_squeezing_and_rotation(kappa):
    """S(kappa) = R(post) diag(e^r, e^-r) R(pre), with e^r - e^-r = kappa."""
    r, pre, post = shear_as_squeeze_rotation(kappa)

    def rot(t):
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    rebuilt = rot(post) @ np.diag([np.exp(r), np.exp(-r)]) @ rot(pre)
    assert np.allclose(rebuilt, np.array([[1.0, 0.0], [-kappa, 1.0]]), atol=1e-12)
    assert np.exp(r) - np.exp(-r) == pytest.approx(kappa, abs=1e-12)


def test_zero_kappa_reduces_to_the_existing_channel():
    """With kappa = 0 the new channel must be the package's own, bit for bit."""
    N = 25
    psi = make_state("cat_even", N, 1.5)
    common = dict(epsilon_a=0.2, Delta=0.1, eta_in=0.9, eta_out=0.85, pn_in=0.1, pn_out=0.2, N_basis=N)
    old = get_state_single_mode(rho=psi, **common)
    for ordering in ORDERINGS:
        new = get_state_single_mode_ba(kappa=0.0, ordering=ordering, rho=psi, solver="mesolve", **common)
        # Tolerance is set by the *old* channel: it runs mesolve at QuTiP's
        # defaults (atol 1e-8), while the back-action channel tightens them.
        assert np.allclose(new.full(), old.full(), atol=1e-7)


def test_exact_and_mesolve_solvers_agree():
    """The closed-form maps and the QuTiP integrator implement the same channel."""
    N = 40
    for family in ("coherent", "squeezed", "cat_even", "fock"):
        psi = make_state(family, N, 2.0)
        common = dict(kappa=1.3, eta_in=0.9, eta_out=0.85, pn_in=0.15, pn_out=0.1, N_basis=N)
        for ordering in ("BA1", "BA2", "BA3"):
            a = qfi(psi, ordering=ordering, solver="exact", **common)
            b = qfi(psi, ordering=ordering, solver="mesolve", **common)
            assert a == pytest.approx(b, rel=1e-7)


@pytest.mark.parametrize("noise", [
    dict(eta_ch=0.8),
    dict(pn_ch=0.3),
    dict(eta_ch=0.7, pn_ch=0.2),
    dict(eta_ch=0.8, eta_in=0.9, eta_out=0.85, pn_in=0.1),
])
@pytest.mark.parametrize("ordering", ["BA1", "BA2", "BA3"])
def test_concurrent_exact_matches_mesolve(noise, ordering):
    """Concurrent loss/dephasing: the Strang + Richardson path vs the integrator.

    This is the case where dissipation acts *during* the opto-mechanical
    interaction, so there is no single closed-form map; the split-step path is
    what makes it affordable at the cutoffs the shear needs.
    """
    N = 60
    for family in ("squeezed", "cat_even", "fock"):
        psi = make_state(family, N, 2.0)
        common = dict(kappa=1.5, ordering=ordering, N_basis=N, **noise)
        a = qfi(psi, solver="exact", **common)
        b = qfi(psi, solver="mesolve", **common)
        assert a == pytest.approx(b, rel=5e-6)


# ---------------------------------------------------------------------------
# Probe states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", sorted(STATE_FAMILIES))
@pytest.mark.parametrize("nbar", [1.0, 3.0])
def test_states_hit_their_target_photon_number(family, nbar):
    psi = make_state(family, 160, nbar)
    assert psi.norm() == pytest.approx(1.0)
    expected = 0.0 if family == "vacuum" else nbar
    assert mean_photon_number(psi) == pytest.approx(expected, abs=1e-6)


def test_optimal_no_backaction_state_is_squeezed_vacuum():
    """max Var(x) at fixed <n> gives F_lambda = 2(2n+1+2 sqrt(n(n+1))).

    ``optimal_no_backaction`` is not a reported probe (it is absent from
    STATE_FAMILIES); this pins the closed-form result it encodes, which is what
    makes squeezed vacuum the leader of every lossless ranking.
    """
    for nbar in (0.5, 1.0, 3.0):
        N = int(40 + 25 * nbar)
        opt = optimal_no_backaction(N, nbar)
        theory = EPS_A_PER_LAMBDA * 2 * (2 * nbar + 1 + 2 * np.sqrt(nbar * (nbar + 1)))
        assert qfi(opt, kappa=0.0, ordering="none") == pytest.approx(theory, rel=1e-5)
        overlap = abs(complex(make_state("squeezed", N, nbar).dag() * opt))
        assert overlap == pytest.approx(1.0, abs=1e-4)
        families = ["coherent", "fock", "cat_even", "squeezed_cat"] + (["cat_odd"] if nbar >= 1 else [])
        for family in families:
            assert qfi(make_state(family, N, nbar), kappa=0.0, ordering="none") <= theory + 1e-6


# ---------------------------------------------------------------------------
# BA1 / BA2 / BA3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["squeezed", "cat_even", "coherent", "fock"])
def test_orderings_coincide_for_the_physical_signal_quadrature(family):
    """epsilon_a and the back-action generator are both functions of x, so they
    commute and BA1 = BA2 = BA3 exactly.

    Loss and phase noise here are stage-separated (eta_in/eta_out, pn_in/pn_out),
    acting strictly before or after the interaction, so they cannot break it.
    In-channel dissipation does -- see
    ``test_orderings_differ_when_dissipation_acts_during_the_interaction``.
    """
    N = 200
    psi = make_state(family, N, 1.5)
    common = dict(kappa=1.4, eta_in=0.9, eta_out=0.8, pn_in=0.05, N_basis=N)
    vals = [qfi(psi, ordering=o, **common) for o in ("BA1", "BA2", "BA3")]
    assert np.allclose(vals, vals[0], rtol=1e-8)


@pytest.mark.parametrize("noise", [dict(eta_ch=0.8), dict(pn_ch=0.3)])
def test_orderings_differ_when_dissipation_acts_during_the_interaction(noise):
    """The orderings coincide only while the noise is stage-separated.

    eta_in/eta_out/pn_in/pn_out act before or after the opto-mechanical
    interaction, so they cannot spoil the commutation of the two generators.
    eta_ch and pn_ch act *during* it, and their Lindblad operators (a and n)
    do not commute with the shear -- so even in the physical signal quadrature
    the three orderings separate, with BA3 again bracketed by BA1 and BA2.
    """
    N = 150
    psi = make_state("cat_even", N, 2.0)
    v = {o: qfi(psi, kappa=1.5, ordering=o, **noise) for o in ("BA1", "BA2", "BA3")}
    assert abs(v["BA1"] - v["BA2"]) > 1e-2
    assert min(v["BA1"], v["BA2"]) - 1e-9 <= v["BA3"] <= max(v["BA1"], v["BA2"]) + 1e-9


def test_orderings_differ_and_ba3_is_bounded_for_a_non_commuting_signal():
    """For epsilon_p the generator is p, which does not commute with x^2:
    the orderings separate and BA3 lies between BA1 and BA2."""
    N = 200
    psi = make_state("squeezed", N, 1.0)
    common = dict(kappa=1.5, eta_out=0.8, N_basis=N)
    kw = dict(param_type="epsilon_p", param_value=0.0, rho=psi)
    v = {
        o: calculate_qfi(get_state_single_mode_ba, ordering=o, **common, **kw)
        for o in ("BA1", "BA2", "BA3")
    }
    assert abs(v["BA1"] - v["BA2"]) > 1e-3
    assert min(v["BA1"], v["BA2"]) - 1e-9 <= v["BA3"] <= max(v["BA1"], v["BA2"]) + 1e-9


@pytest.mark.parametrize("family", ["squeezed", "cat_even", "fock", "squeezed_cat"])
def test_backaction_alone_does_not_change_the_qfi(family):
    """Lossless: the shear is unitary and commutes with the signal generator."""
    N, nbar = 250, 2.0
    psi = make_state(family, N, nbar)
    base = qfi(psi, kappa=0.0, ordering="none")
    for kappa in (0.5, 1.0, 2.0):
        for ordering in ("BA1", "BA2", "BA3"):
            assert qfi(psi, kappa=kappa, ordering=ordering) == pytest.approx(base, rel=1e-6)


# ---------------------------------------------------------------------------
# Loss placement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("eta", [1.0, 0.9, 0.5])
def test_analytic_vacuum_backaction_then_detection_loss(kappa, eta):
    """Vacuum, BA -> signal -> detection loss:  F = 2 * 2 eta / (1 + eta(1-eta) kappa^2)."""
    N = 40 + int(60 * kappa)
    psi = make_state("vacuum", N, 0.0)
    expected = EPS_A_PER_LAMBDA * 2 * eta / (1 + eta * (1 - eta) * kappa**2)
    assert qfi(psi, kappa=kappa, ordering="BA1", eta_out=eta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("family", ["squeezed", "cat_even"])
def test_injection_loss_is_blind_to_backaction(family):
    """Loss *before* the interaction: the shear that follows is a QFI-preserving
    unitary, so the QFI does not depend on kappa at all."""
    N = 250
    psi = make_state(family, N, 2.0)
    vals = [qfi(psi, kappa=k, ordering="BA1", eta_in=0.8) for k in KAPPAS]
    assert np.allclose(vals, vals[0], rtol=1e-6)


@pytest.mark.parametrize("family", ["squeezed", "cat_even"])
def test_detection_loss_does_couple_to_backaction(family):
    """Loss *after* the interaction degrades monotonically with kappa."""
    N = 250
    psi = make_state(family, N, 2.0)
    vals = [qfi(psi, kappa=k, ordering="BA1", eta_out=0.8) for k in KAPPAS]
    assert np.all(np.diff(vals) < 0)


def test_noise_never_increases_the_qfi():
    N = 250
    psi = make_state("cat_even", N, 2.0)
    base = qfi(psi, kappa=1.0, ordering="BA3")
    for extra in (dict(eta_out=0.9), dict(eta_in=0.9), dict(pn_in=0.2), dict(pn_out=0.2)):
        assert qfi(psi, kappa=1.0, ordering="BA3", **extra) <= base + 1e-8


def test_phase_noise_then_backaction_then_loss_degrades_monotonically():
    N = 250
    psi = make_state("squeezed", N, 2.0)
    vals = [
        qfi(psi, kappa=1.0, ordering="BA3", pn_in=s, eta_out=0.9)
        for s in (0.0, 0.1, 0.2, 0.3, 0.4)
    ]
    assert np.all(np.diff(vals) < 0)


# ---------------------------------------------------------------------------
# Cross-check against the Gaussian covariance track
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kappa", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("ordering", ["none", "BA1", "BA2", "BA3"])
@pytest.mark.parametrize("eta_in,eta_out", [(1.0, 1.0), (0.7, 1.0), (1.0, 0.7), (0.85, 0.9)])
def test_matches_the_gaussian_reference(kappa, ordering, eta_in, eta_out):
    nbar = 1.0
    N = 220 + 60 * int(kappa)
    cov, dmean = squeezed_vacuum_moments(nbar)
    expected = EPS_A_PER_LAMBDA * gaussian_qfi_lambda(
        cov, dmean, kappa=kappa, ordering=ordering, eta_in=eta_in, eta_out=eta_out
    )
    got = qfi(make_state("squeezed", N, nbar), kappa=kappa, ordering=ordering,
              eta_in=eta_in, eta_out=eta_out, N_basis=N)
    assert got == pytest.approx(expected, rel=3e-5)


def test_vacuum_matches_the_gaussian_reference_under_backaction():
    for kappa in KAPPAS:
        N = 40 + int(60 * kappa)
        cov, dmean = vacuum()
        expected = EPS_A_PER_LAMBDA * gaussian_qfi_lambda(
            cov, dmean, kappa=kappa, ordering="BA1", eta_out=0.8
        )
        got = qfi(make_state("vacuum", N, 0.0), kappa=kappa, ordering="BA1", eta_out=0.8, N_basis=N)
        assert got == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# SLD and cutoff convergence
# ---------------------------------------------------------------------------

def test_sld_is_consistent_with_the_scalar_qfi():
    """calculate_sld's Tr[rho L^2] must agree with the direct QFI sum."""
    N = 120
    psi = make_state("squeezed", N, 1.0)
    res = calculate_sld(
        get_state_single_mode_ba, param_value=0.0, param_type="epsilon_a",
        kappa=1.0, ordering="BA3", eta_out=0.9, N_basis=N, rho=psi,
    )
    scalar = qfi(psi, kappa=1.0, ordering="BA3", eta_out=0.9)
    assert res["qfi"] == pytest.approx(scalar, rel=1e-6)
    assert np.allclose(res["L"], res["L"].conj().T, atol=1e-9)


def test_default_cutoff_is_not_trustworthy_under_backaction():
    """Regression guard for the trap: N_basis=20 is fine for the package's own
    channel but badly wrong once radiation pressure is on."""
    nbar, kappa = 2.0, 1.0
    coarse = qfi(make_state("squeezed", 20, nbar), kappa=kappa, ordering="BA3", eta_out=0.9)
    res = converged_qfi(
        lambda N: make_state("squeezed", N, nbar), get_state_single_mode_ba,
        kappa=kappa, ordering="BA3", eta_out=0.9, rtol=1e-5, n_stable=2,
    )
    assert res.converged
    assert abs(coarse - res.value) / res.value > 0.1


def test_convergence_helper_reports_failure():
    good = converge(lambda N: 1.0 + 1.0 / N**2, cutoffs=[20, 40, 80], rtol=1e-2)
    assert good.converged and good.cutoff == 40
    bad = converge(lambda N: float(N), cutoffs=[20, 40, 80], rtol=1e-6)
    assert not bad.converged and bad.cutoff == 80


def test_suggested_cutoff_grows_with_kappa_and_nbar():
    assert suggested_cutoff(1.0, 0.0) < suggested_cutoff(3.0, 0.0)
    assert suggested_cutoff(1.0, 1.0) < suggested_cutoff(1.0, 5.0)


def test_tail_population_alarm():
    small = tail_population(make_state("coherent", 60, 1.0))
    sheared = get_state_single_mode_ba(kappa=3.0, ordering="BA1", N_basis=40,
                                       rho=make_state("cat_even", 40, 4.0))
    assert small < 1e-12
    assert tail_population(sheared) > small
