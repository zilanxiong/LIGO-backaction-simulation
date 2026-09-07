"""Tests for band metrics and mid-channel loss placements."""

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quantum_sensing as qs
from quantum_sensing import gaussian as g
from quantum_sensing import metrics
from quantum_sensing.channels import get_state_ba1, get_state_ba2, get_state_ba3

N_BASIS = 60


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_flat_weight_is_trapezoid_harmonic_mean():
    f = np.linspace(10, 100, 7)
    q = np.array([1.0, 2, 4, 2, 1, 4, 2])
    got = metrics.weighted_harmonic_qfi(f, q)
    expected = (f[-1] - f[0]) / np.trapezoid(1 / q, f)
    assert got == pytest.approx(expected, rel=1e-12)


def test_constant_qfi_metrics():
    f = np.geomspace(20, 1000, 8)
    q = 4.0 * np.ones_like(f)
    for w in (metrics.flat_weight, metrics.inspiral_weight):
        m = metrics.band_metrics(f, q, w)
        assert m["metric_i"] == pytest.approx(4.0, rel=1e-12)
    assert m["metric_ii"] == 4.0
    assert m["metric_iii"] == pytest.approx(32.0)


def test_inspiral_weight_emphasizes_low_frequency():
    # QFI poor at low f: inspiral weighting must punish it harder than flat.
    f = np.geomspace(20, 1000, 8)
    q = np.where(f < 100, 1.0, 10.0)
    flat = metrics.weighted_harmonic_qfi(f, q, metrics.flat_weight)
    insp = metrics.weighted_harmonic_qfi(f, q, metrics.inspiral_weight)
    assert insp < flat


def test_order_invariance():
    f = np.geomspace(20, 1000, 8)
    q = np.linspace(1, 8, 8)
    a = metrics.weighted_harmonic_qfi(f, q)
    b = metrics.weighted_harmonic_qfi(f[::-1], q[::-1])
    assert a == pytest.approx(b, rel=1e-12)


# ---------------------------------------------------------------------------
# Mid-channel loss
# ---------------------------------------------------------------------------

def _sqz(r=0.6):
    return qt.squeeze(N_BASIS, -r) * qt.fock(N_BASIS, 0)


def test_mid_loss_breaks_ba1_ba2_equivalence_for_epsilon_a():
    # With loss between the stages, "shear -> loss -> sig" and
    # "sig -> loss -> shear" are different channels even for epsilon_a.
    psi = _sqz()
    kw = dict(rho=psi, N_basis=N_BASIS, param_type="epsilon_a",
              kappa_ba=1.5, eta_mid=0.9)
    F1 = qs.calculate_qfi(get_state_ba1, **kw)
    F2 = qs.calculate_qfi(get_state_ba2, **kw)
    assert abs(F1 - F2) / F2 > 1e-3


@pytest.mark.parametrize("ba,order", [(get_state_ba1, "after"),
                                      (get_state_ba2, "before")])
def test_mid_loss_matches_gaussian(ba, order):
    r, kappa, eta_mid = 0.6, 1.5, 0.85
    psi = _sqz(r)
    F_fock = qs.calculate_qfi(ba, rho=psi, N_basis=N_BASIS,
                              param_type="epsilon_a", kappa_ba=kappa,
                              eta_mid=eta_mid)
    V, d = g.squeezed_state(r, np.pi / 2)
    F_gauss = g.gaussian_qfi_rp(V, d, param_type="epsilon_a",
                                kappa_ba=kappa, eta_mid=eta_mid,
                                signal_order=order)
    assert F_fock == pytest.approx(F_gauss, rel=2e-3)


def test_ba3_channel_loss_matches_gaussian():
    r, kappa, eta_ch = 0.6, 1.5, 0.85
    psi = _sqz(r)
    F_fock = qs.calculate_qfi(get_state_ba3, rho=psi, N_basis=N_BASIS,
                              param_type="epsilon_a", kappa_ba=kappa,
                              eta_ch=eta_ch)
    V, d = g.squeezed_state(r, np.pi / 2)
    F_gauss = g.gaussian_qfi_rp(V, d, param_type="epsilon_a",
                                kappa_ba=kappa, eta_ch=eta_ch,
                                signal_order="simultaneous")
    assert F_fock == pytest.approx(F_gauss, rel=2e-3)


def test_ba3_rejects_eta_mid():
    with pytest.raises(ValueError):
        get_state_ba3(N_basis=20, eta_mid=0.9)
    V, d = g.vacuum_state()
    with pytest.raises(ValueError):
        g.gaussian_rp_channel(V, d, eta_mid=0.9,
                              signal_order="simultaneous")
