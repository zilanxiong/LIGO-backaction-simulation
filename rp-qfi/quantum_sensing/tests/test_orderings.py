"""
Tests for the BA1/BA2/BA3 channel orderings (channels.py): commutation
structure, agreement with the Gaussian track, and consistency with the
mesolve-based dynamics code.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quantum_sensing as qs
from quantum_sensing import gaussian as g
from quantum_sensing.channels import (
    get_state_ba1, get_state_ba2, get_state_ba3, BA_CHANNELS)

N_BASIS = 60
SIGNAL_ORDER = {"ba1": "after", "ba2": "before", "ba3": "simultaneous"}


def _sqz(N=N_BASIS, r=0.6):
    return qt.squeeze(N, r) * qt.fock(N, 0)


def _cat(N=N_BASIS, alpha=1.5):
    return (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()


def _qfi(channel, psi, **kw):
    return qs.calculate_qfi(channel, rho=psi, N_basis=N_BASIS, **kw)


# ---------------------------------------------------------------------------
# Commutation structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_state", [_sqz, _cat])
def test_orderings_coincide_for_epsilon_a_lossless(make_state):
    # x commutes with x^2: all three orderings identical, kappa-irrelevant.
    psi = make_state()
    F = {k: _qfi(ch, psi, param_type="epsilon_a", kappa_ba=1.5)
         for k, ch in BA_CHANNELS.items()}
    F0 = _qfi(get_state_ba1, psi, param_type="epsilon_a", kappa_ba=0.0)
    for k in F:
        assert F[k] == pytest.approx(F0, rel=1e-6), k


def test_orderings_split_for_epsilon_p():
    # p does not commute with x^2: BA1 != BA2 at kappa > 0, BA3 in between.
    psi = _sqz()
    F = {k: _qfi(ch, psi, param_type="epsilon_p", kappa_ba=1.5)
         for k, ch in BA_CHANNELS.items()}
    assert abs(F["ba1"] - F["ba2"]) / F["ba2"] > 1e-3
    assert min(F["ba1"], F["ba2"]) < F["ba3"] < max(F["ba1"], F["ba2"])


def test_orderings_split_with_intervening_loss():
    # Even for epsilon_a, loss between the stages breaks the equivalence:
    # detection loss after BA1's signal vs after BA2's shear act on
    # different states... but with only eta_in/eta_out the stage between
    # shear and signal is untouched, so instead probe injection vs
    # detection loss, which must differ at kappa > 0.
    kappa = 2.0
    psi = _sqz(r=qs.dB_to_r(10.0))
    F_inj = _qfi(get_state_ba3, psi, param_type="epsilon_a",
                 kappa_ba=kappa, eta_in=0.9)
    F_det = _qfi(get_state_ba3, psi, param_type="epsilon_a",
                 kappa_ba=kappa, eta_out=0.9)
    assert abs(F_inj - F_det) / F_det > 1e-3


# ---------------------------------------------------------------------------
# Agreement with the Gaussian covariance-matrix track
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ba", ["ba1", "ba2", "ba3"])
@pytest.mark.parametrize("param", ["epsilon_a", "epsilon_p"])
@pytest.mark.parametrize("eta_in,eta_out", [(1.0, 1.0), (0.9, 0.8)])
def test_fock_orderings_match_gaussian(ba, param, eta_in, eta_out):
    r, kappa = 0.6, 1.5
    psi = _sqz(r=r)
    F_fock = _qfi(BA_CHANNELS[ba], psi, param_type=param, kappa_ba=kappa,
                  eta_in=eta_in, eta_out=eta_out)
    V, d = g.squeezed_state(r, 0.0)
    F_gauss = g.gaussian_qfi_rp(V, d, param_type=param, kappa_ba=kappa,
                                eta_in=eta_in, eta_out=eta_out,
                                signal_order=SIGNAL_ORDER[ba])
    assert F_fock == pytest.approx(F_gauss, rel=2e-3)


# ---------------------------------------------------------------------------
# Consistency with the mesolve-based dynamics code
# ---------------------------------------------------------------------------

def test_ba3_matches_get_state_single_mode_rp():
    # Same channel, two implementations (exact expm vs mesolve): the output
    # states must agree, not just the QFI.
    psi = _sqz()
    kw = dict(epsilon_a=0.3, epsilon_p=0.1, kappa_ba=1.2)
    rho_a = get_state_ba3(rho=psi, N_basis=N_BASIS, **kw)
    rho_b = qs.get_state_single_mode_rp(rho=psi, N_basis=N_BASIS, **kw)
    assert (rho_a - rho_b).norm() < 1e-6
