"""
Unit tests for the Gaussian covariance-matrix track (gaussian.py):
analytic SQL / FD-squeezing curves, and agreement with the Fock-space
channel + QFI code for Gaussian inputs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quantum_sensing as qs
from quantum_sensing import gaussian as g


# ---------------------------------------------------------------------------
# Analytic facts, Gaussian side only
# ---------------------------------------------------------------------------

def test_vacuum_qfi_is_4_and_kappa_independent():
    V, d = g.vacuum_state()
    for kappa in [0.0, 0.5, 2.0, 10.0]:
        F = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa)
        assert F == pytest.approx(4.0, rel=1e-9)


def test_lossless_qfi_equals_8_var_x():
    # For epsilon_a the generator is x: pure-state QFI = 8 Var(x),
    # independent of kappa_ba.
    r, phi = 1.0, 0.7
    V, d = g.squeezed_state(r, phi)
    var_x = V[0, 0]
    for kappa in [0.0, 1.0, 5.0]:
        F = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa)
        assert F == pytest.approx(8 * var_x, rel=1e-9)


def test_sql_homodyne_curve():
    V, d = g.vacuum_state()
    for kappa in [0.0, 0.7, 3.0]:
        cfi = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa,
                                homodyne_angle=np.pi / 2)
        assert cfi == pytest.approx(g.sql_homodyne_cfi(kappa), rel=1e-9)


def test_fd_squeezing_restores_homodyne():
    r = qs.dB_to_r(10.0)
    for kappa in [0.0, 0.5, 2.0]:
        V, d = g.squeezed_state(r, g.fd_squeeze_angle(kappa))
        cfi = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa,
                                homodyne_angle=np.pi / 2)
        assert cfi == pytest.approx(g.fd_squeezed_homodyne_cfi(kappa, r),
                                    rel=1e-9)
        # Wrong (frequency-independent) angle is strictly worse at kappa > 0.
        if kappa > 0:
            V2, d2 = g.squeezed_state(r, g.fd_squeeze_angle(0.0))
            cfi_wrong = g.gaussian_qfi_rp(V2, d2, param_type="epsilon_a",
                                          kappa_ba=kappa,
                                          homodyne_angle=np.pi / 2)
            assert cfi_wrong < cfi


def test_signal_order_coincides_for_epsilon_a_lossless():
    V, d = g.squeezed_state(0.8, 0.3)
    F = [g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=1.3,
                           signal_order=so)
         for so in ("before", "after", "simultaneous")]
    assert F[0] == pytest.approx(F[1], rel=1e-9)
    assert F[0] == pytest.approx(F[2], rel=1e-9)


def test_injection_vs_detection_loss_differ_with_backaction():
    # loss -> BA vs BA -> loss must differ once kappa > 0 (this is the
    # BA/loss-ordering physics the study targets).
    r = qs.dB_to_r(10.0)
    kappa = 2.0
    V, d = g.squeezed_state(r, g.fd_squeeze_angle(kappa))
    F_inj = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa,
                              eta_in=0.9)
    F_det = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa,
                              eta_out=0.9)
    assert abs(F_inj - F_det) / F_det > 1e-3


def test_rp_kappa_anchor():
    gamma = 2 * np.pi * 450.0
    assert qs.rp_kappa(gamma) == pytest.approx(qs.rp_power_ratio(), rel=1e-12)


# ---------------------------------------------------------------------------
# Cross-validation against the Fock-space code
# ---------------------------------------------------------------------------

# 60 rather than 40: at kappa_ba = 2 the shear pumps enough population that
# N = 40 truncates the wider states (coherent, rotated-squeezed) at the
# few-1e-3 level — exactly the failure mode convergence.py exists to catch.
N_BASIS = 60


def _fock_state(kind, N=N_BASIS):
    if kind == "vacuum":
        return qt.fock(N, 0)
    if kind == "coherent":
        return qt.coherent(N, 1.2)
    if kind == "sqz_vac":
        return qt.squeeze(N, 0.6) * qt.fock(N, 0)
    if kind == "sqz_rot":
        r, phi = 0.6, 0.9
        # rotate the squeezed quadrature by phi (counter-clockwise in x-p)
        a = qt.destroy(N)
        rot = (1j * phi * a.dag() * a).expm()
        return rot * qt.squeeze(N, 0.6) * qt.fock(N, 0)
    raise ValueError(kind)


def _gauss_state(kind):
    if kind == "vacuum":
        return g.vacuum_state()
    if kind == "coherent":
        return g.coherent_state(1.2)
    if kind == "sqz_vac":
        return g.squeezed_state(0.6, 0.0)
    if kind == "sqz_rot":
        return g.squeezed_state(0.6, 0.9)
    raise ValueError(kind)


def test_squeeze_convention_matches_qutip():
    # qt.squeeze(N, r)|0> must have Var(x) = e^{-2r}/2 like squeezed_state.
    N, r = 40, 0.6
    psi = qt.squeeze(N, r) * qt.fock(N, 0)
    a = qt.destroy(N)
    x = (a + a.dag()) / np.sqrt(2)
    var_x = qt.expect(x * x, psi) - qt.expect(x, psi) ** 2
    assert var_x == pytest.approx(np.exp(-2 * r) / 2, rel=1e-8)


@pytest.mark.parametrize("kind", ["vacuum", "coherent", "sqz_vac", "sqz_rot"])
@pytest.mark.parametrize("kappa_ba,eta_in,eta_out", [
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.9, 1.0),
    (1.0, 1.0, 0.9),
    (2.0, 0.9, 0.8),
])
def test_fock_matches_gaussian(kind, kappa_ba, eta_in, eta_out):
    psi = _fock_state(kind)
    F_fock = qs.calculate_qfi(
        qs.get_state_single_mode_rp, param_type="epsilon_a",
        rho=psi, N_basis=N_BASIS, kappa_ba=kappa_ba,
        eta_in=eta_in, eta_out=eta_out)

    V, d = _gauss_state(kind)
    F_gauss = g.gaussian_qfi_rp(V, d, param_type="epsilon_a",
                                kappa_ba=kappa_ba,
                                eta_in=eta_in, eta_out=eta_out)
    assert F_fock == pytest.approx(F_gauss, rel=2e-3)


def test_fock_matches_gaussian_channel_loss():
    # eta_ch runs simultaneously with the shear — exercises the Lyapunov
    # (augmented-exponential) stage-2 path against mesolve.
    psi = qt.squeeze(N_BASIS, 0.6) * qt.fock(N_BASIS, 0)
    F_fock = qs.calculate_qfi(
        qs.get_state_single_mode_rp, param_type="epsilon_a",
        rho=psi, N_basis=N_BASIS, kappa_ba=1.0, eta_ch=0.85)
    V, d = g.squeezed_state(0.6, 0.0)
    F_gauss = g.gaussian_qfi_rp(V, d, param_type="epsilon_a",
                                kappa_ba=1.0, eta_ch=0.85)
    assert F_fock == pytest.approx(F_gauss, rel=2e-3)
