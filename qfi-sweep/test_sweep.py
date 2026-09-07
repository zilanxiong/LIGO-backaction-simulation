"""Correctness anchors for the sweep (run: pytest qfi-sweep)."""

import numpy as np
import pytest
import qutip as qt

from channel import (kimble_K, qfi_fock, gaussian_qfi, sqz_cov,
                     best_sqz_angle, apply_channel, x_op)
from probes import make_probe, var_x, mean_n, basis_size


def _cov(rho, N):
    a = qt.destroy(N)
    x, p = a + a.dag(), -1j * (a - a.dag())
    mx, mp = qt.expect(x, rho), qt.expect(p, rho)
    vxx = np.real(qt.expect(x * x, rho)) - mx**2
    vpp = np.real(qt.expect(p * p, rho)) - mp**2
    vxp = np.real(qt.expect((x * p + p * x) / 2, rho)) - mx * mp
    return np.array([[vxx, vxp], [vxp, vpp]])


def test_kimble_input_output_relation():
    """B with g = K/4 must reproduce KLMTV p_out = p_in - K x_in exactly:
    vacuum covariance -> [[1, -K], [-K, 1 + K^2]]."""
    K, N = 1.3, 60
    rho = qt.ket2dm(qt.basis(N, 0))
    out = apply_channel(rho, 0.0, K, N, "none")
    V = _cov(out, N)
    assert np.allclose(V, [[1, -K], [-K, 1 + K**2]], atol=1e-8)


def test_kimble_calibration_value():
    """K = 1 near 31 Hz for O4 defaults (X ~ 2.4e-3, gamma = 2pi*450)."""
    f = np.logspace(1, 2, 400)
    Kv = kimble_K(2 * np.pi * f)
    f1 = f[np.argmin(np.abs(Kv - 1.0))]
    assert 28 < f1 < 34


def test_lossless_qfi_is_4varx_and_K_independent():
    for kind in ("coherent", "cat"):
        psi = make_probe(kind, 2.0, 120)
        vals = [qfi_fock(qt.ket2dm(psi), K, 120, "none", 1.0)
                for K in (0.0, 0.5, 1.5)]
        assert np.allclose(vals, 4 * var_x(psi), rtol=1e-6)


def test_gaussian_vs_fock_all_placements():
    """Exact Gaussian covariance method vs Fock mesolve, incl. concurrent."""
    n_t, K, eta = 2.0, 1.0, 0.7
    r = np.arcsinh(np.sqrt(n_t))
    for placement in ("inj", "det", "conc"):
        qg = gaussian_qfi(sqz_cov(r, 0.0), K, placement, eta)
        psi = make_probe("sqz_vac", n_t, 240, theta=0.0)
        qf = qfi_fock(qt.ket2dm(psi), K, 240, placement, eta)
        assert abs(qf - qg) / qg < 2e-3, (placement, qf, qg)


def test_coherent_det_loss_closed_form():
    """Independent closed form for coherent + shear + detection loss."""
    K, eta = 1.7, 0.8
    V = eta * np.array([[1, -K], [-K, 1 + K**2]]) + (1 - eta) * np.eye(2)
    expect = 4 * eta * V[0, 0] / np.linalg.det(V)
    assert np.isclose(gaussian_qfi(np.eye(2), K, "det", eta), expect,
                      rtol=1e-10)


def test_bc_conventional_curve_is_upper_reference():
    """QFI eps_min (0.5, flat) must sit below the fixed-homodyne
    conventional-interferometer curve sqrt(1 + K^2)/2 (BC2001/KLMTV),
    merging with it as K -> 0."""
    for K in (0.0, 0.3, 1.0, 3.0):
        q = gaussian_qfi(np.eye(2), K, "none", 1.0)
        eps_qfi = 1 / np.sqrt(q)
        eps_hom = np.sqrt(1 + K**2) / 2
        assert eps_qfi <= eps_hom + 1e-12
    assert np.isclose(1 / np.sqrt(gaussian_qfi(np.eye(2), 1e-6, "none", 1.0)),
                      np.sqrt(1 + 1e-12) / 2, rtol=1e-6)


def test_detection_loss_ceiling():
    """Displacement QFI under detection loss is capped: for the sheared+lossy
    Gaussian family, F <= 4/(1-eta) however strong the squeezing."""
    eta = 0.9
    for n_t in (4.0, 16.0, 64.0):
        _, q = best_sqz_angle(n_t, 1.0, "det", eta)
        assert q < 4 / (1 - eta)
    # and it approaches the cap monotonically in n
    qs = [best_sqz_angle(n, 1.0, "det", eta)[1] for n in (4.0, 16.0, 64.0)]
    assert qs[0] < qs[1] < qs[2]


def test_loss_placement_ordering_matters():
    """inj vs det are genuinely different channels at K > 0, and conc lies
    between them for the x-anti-squeezed probe."""
    n_t, K, eta = 2.0, 1.0, 0.7
    r = np.arcsinh(np.sqrt(n_t))
    V0 = sqz_cov(r, 0.0)
    qi = gaussian_qfi(V0, K, "inj", eta)
    qc = gaussian_qfi(V0, K, "conc", eta)
    qd = gaussian_qfi(V0, K, "det", eta)
    assert qi != pytest.approx(qd, rel=1e-3)
    assert min(qi, qd) - 1e-9 <= qc <= max(qi, qd) + 1e-9


def test_probe_photon_budget():
    for kind in ("coherent", "sqz_vac", "cat", "fock"):
        psi = make_probe(kind, 2.0, 150)
        assert abs(mean_n(psi) - 2.0) < 1e-6
    psi = make_probe("sqz_cat", 2.0, 150, split=0.5)
    assert abs(mean_n(psi) - 2.0) < 1e-6
