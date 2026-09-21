"""Tests for the automated Fock-cutoff convergence harness."""

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quantum_sensing as qs
from quantum_sensing.convergence import converged_qfi, tail_population


def _cat(N, alpha):
    psi = qt.coherent(N, alpha) + qt.coherent(N, -alpha)
    return psi.unit()


def test_vacuum_converges_immediately():
    res = converged_qfi(lambda N: qt.fock(N, 0), param_type="epsilon_a",
                        kappa_ba=1.0, N_start=10, N_step=5)
    assert res["converged"]
    assert res["qfi"] == pytest.approx(4.0, rel=1e-3)


def test_cat_converges_and_matches_lossless_analytic():
    # Lossless epsilon_a QFI = 8 Var(x), independent of kappa_ba — a strong
    # end-to-end check that the accepted cutoff is truly converged.
    alpha = 2.0
    res = converged_qfi(lambda N: _cat(N, alpha), param_type="epsilon_a",
                        kappa_ba=1.0, N_start=20, N_step=10, N_max=120)
    assert res["converged"]

    N = res["N_basis"]
    psi = _cat(N, alpha)
    a = qt.destroy(N)
    x = (a + a.dag()) / np.sqrt(2)
    var_x = qt.expect(x * x, psi) - qt.expect(x, psi) ** 2
    assert res["qfi"] == pytest.approx(8 * var_x, rel=1e-3)


def test_undersized_cutoff_is_rejected():
    # At N = 12 a cat with alpha = 2 under shear is badly truncated; the
    # harness must not report convergence there.
    res = converged_qfi(lambda N: _cat(N, 2.0), param_type="epsilon_a",
                        kappa_ba=1.0, N_start=12, N_step=4, N_max=16)
    assert not res["converged"]


def test_tail_population_flags_truncation():
    psi = qt.coherent(12, 2.5)  # heavily truncated coherent state
    assert tail_population(psi) > 1e-3
    psi_ok = qt.coherent(60, 2.5)
    assert tail_population(psi_ok) < 1e-10
