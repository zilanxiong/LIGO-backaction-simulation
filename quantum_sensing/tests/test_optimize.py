"""Tests for quantum_sensing.optimize (superoperator channel + QFI optimizers)."""

import numpy as np
import pytest
import qutip as qt

import quantum_sensing as qs
from quantum_sensing.optimize import (DisplacementChannel, qfi_bounds,
                                      qfi_from_rho_drho, gaussian_ket, mean_n,
                                      optimize_gaussian,
                                      optimize_fock_superposition)


def test_channel_matches_mesolve_pipeline():
    N = 30
    ch = DisplacementChannel(N, eta=0.9, sigma_phi=0.2)
    psi = (qt.squeeze(N, -1.2) * qt.fock(N, 0)).full().ravel()
    rho_in = np.outer(psi, psi.conj())
    rho0, _ = ch.output(rho_in)
    rho_ref = qs.get_state_single_mode(rho=qt.Qobj(rho_in), N_basis=N,
                                       eta_ch=0.9, pn_in=0.2).full()
    assert np.abs(rho0 - rho_ref).max() < 1e-9


def test_channel_qfi_matches_package_route():
    N = 30
    ch = DisplacementChannel(N, eta=0.9, sigma_phi=0.2)
    psi = (qt.squeeze(N, -1.2) * qt.fock(N, 0)).full().ravel()
    F_pkg = qs.calculate_qfi(lambda **kw: qs.get_state_single_mode(**kw),
                             param_type="epsilon_a", rho=qt.Qobj(psi),
                             N_basis=N, eta_ch=0.9, pn_in=0.2)
    assert ch.qfi(psi) == pytest.approx(F_pkg, rel=1e-5)


def test_channel_with_backaction_matches_rp_route():
    N = 40
    ch = DisplacementChannel(N, eta=0.9, sigma_phi=0.1, kappa_ba=1.0)
    psi = qt.coherent(N, np.sqrt(2)).full().ravel()
    F_pkg = qs.calculate_qfi(lambda **kw: qs.get_state_single_mode_rp(**kw),
                             param_type="epsilon_a", rho=qt.Qobj(psi),
                             N_basis=N, eta_ch=0.9, pn_in=0.1, kappa_ba=1.0)
    assert ch.qfi(psi) == pytest.approx(F_pkg, rel=1e-5)


def test_analytic_gradient_matches_finite_differences():
    N = 25
    ch = DisplacementChannel(N, eta=0.95, sigma_phi=0.2)
    rng = np.random.default_rng(1)
    c = (rng.normal(size=N) + 1j * rng.normal(size=N)) * np.exp(-np.arange(N) / 3)
    psi = c / np.linalg.norm(c)
    F, g = ch.qfi_and_grad(psi)
    assert F == pytest.approx(ch.qfi(psi), rel=1e-10)
    eps = 1e-6
    for k in (0, 3, 7):
        for direction in (1.0, 1j):
            dpsi = np.zeros(N, complex)
            dpsi[k] = direction * eps
            Fp = ch.qfi(np.outer(psi + dpsi, (psi + dpsi).conj()))
            Fm = ch.qfi(np.outer(psi - dpsi, (psi - dpsi).conj()))
            fd = (Fp - Fm) / (2 * eps)
            an = 2 * np.real(g[k]) if direction == 1.0 else 2 * np.imag(g[k])
            assert fd == pytest.approx(an, rel=1e-4, abs=1e-6)


def test_bounds_match_paper_lossless_value():
    # eta=1, sigma=0, nbar=5: all three bounds coincide at 4(1+2A(5)) ~ 87.8,
    # the paper's 88.0 squeezed-vacuum QFI.
    b = qfi_bounds(1.0, 0.0, 5.0)
    assert b["min"] == pytest.approx(4 * (1 + 2 * (5 + np.sqrt(30))), rel=1e-12)


def test_qfi_pure_state_formula():
    # pure lossless probe: F = 8 Var(x)
    N = 50
    ch = DisplacementChannel(N, eta=1.0, sigma_phi=0.0)
    assert ch.qfi(qt.fock(N, 3).full().ravel()) == pytest.approx(28.0, rel=1e-6)


def test_optimize_gaussian_recovers_squeezed_vacuum_lossless():
    # At eta=1, sigma=0 the best Gaussian is squeezed vacuum using the full
    # photon budget: F = 4 e^{2r}, sinh^2 r = nbar.
    N = 40
    nbar = 2.0
    ch = DisplacementChannel(N, eta=1.0, sigma_phi=0.0)
    g = optimize_gaussian(ch, nbar, n_starts=6, seed=2)
    r = np.arcsinh(np.sqrt(nbar))
    assert g["qfi"] == pytest.approx(4 * np.exp(2 * r), rel=2e-3)
    assert g["nbar"] <= nbar * 1.01


def test_optimize_fock_superposition_beats_gaussian_at_headline_point():
    # Small, fast version of the paper's eta=0.95, sigma=0.2 point at nbar=2.
    N = 30
    ch = DisplacementChannel(N, eta=0.95, sigma_phi=0.2)
    g = optimize_gaussian(ch, 2.0, n_starts=6, seed=3)
    f = optimize_fock_superposition(ch, 2.0, n_starts=4, seed=3)
    assert f["nbar"] <= 2.02
    assert f["qfi"] > g["qfi"]
