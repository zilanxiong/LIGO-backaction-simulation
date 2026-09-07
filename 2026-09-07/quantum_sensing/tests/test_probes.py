"""Tests for fixed-<n> probe factories and the Kraus loss channel."""

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quantum_sensing as qs
from quantum_sensing import probes
from quantum_sensing.channels import op_loss_kraus


N_TARGET = 2.0


@pytest.mark.parametrize("family", list(probes.PROBE_FAMILIES))
def test_probe_hits_target_n(family):
    factory = probes.PROBE_FAMILIES[family](N_TARGET)
    psi = factory(160)
    assert probes.mean_n(psi) == pytest.approx(N_TARGET, abs=1e-6)
    assert psi.norm() == pytest.approx(1.0, rel=1e-9)


def test_probe_stable_across_cutoffs():
    factory = probes.cat(N_TARGET)
    p60, p120 = factory(60), factory(120)
    # same state, embedded in different cutoffs
    assert np.allclose(p60.full().ravel(), p120.full().ravel()[:60],
                       atol=1e-8)


def test_kraus_loss_matches_mesolve():
    N, eta = 40, 0.85
    psi = (qt.coherent(N, 1.5) + qt.coherent(N, -1.5)).unit()
    rho_kraus = op_loss_kraus(N, eta)(qt.ket2dm(psi))
    # mesolve reference with the dynamics.py conventions
    a = qt.destroy(N)
    kappa = qs.loss_to_kappa(1 - eta)
    res = qt.mesolve(0 * a.dag() * a, qt.ket2dm(psi), [0, 1.0],
                     [np.sqrt(kappa) * a],
                     options={"atol": 1e-12, "rtol": 1e-10})
    assert (rho_kraus - res.states[-1]).norm() < 1e-7
    assert rho_kraus.tr() == pytest.approx(1.0, abs=1e-10)


def test_sqz_vac_p_orientation():
    # p-squeezed vacuum: Var(x) = e^{2r}/2 (anti-squeezed along the
    # epsilon_a generator), Var(p) = e^{-2r}/2, <n> on target.
    n_target = 2.0
    r = np.arcsinh(np.sqrt(n_target))
    N = 200
    psi = probes.squeezed_vacuum_p(n_target)(N)
    a = qt.destroy(N)
    x = (a + a.dag()) / np.sqrt(2)
    p = (a - a.dag()) / (1j * np.sqrt(2))
    var_x = (qt.expect(x * x, psi) - qt.expect(x, psi) ** 2).real
    var_p = (qt.expect(p * p, psi) - qt.expect(p, psi) ** 2).real
    assert var_x == pytest.approx(np.exp(2 * r) / 2, rel=1e-6)
    assert var_p == pytest.approx(np.exp(-2 * r) / 2, rel=1e-5)
    assert probes.mean_n(psi) == pytest.approx(n_target, abs=1e-6)
