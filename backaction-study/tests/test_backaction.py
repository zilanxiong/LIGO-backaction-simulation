"""
Regression tests pinning the analytic anchors of the backaction QFI study.

Conventions under test (see quantum_sensing.backaction docstring):
  X_theta = (a e^{-i theta} + a† e^{i theta})/sqrt(2), vacuum Var = 1/2.
  Signal: U = exp(-i s X_theta_sig)  -> lossless pure-state QFI = 4 Var(X_theta_sig).
  Shear:  U = exp(-i (chi/2) X_0^2)  -> x -> x, p -> p - chi x, r = arcsinh(chi/2).
  Kerr:   U = exp(-i chi n^2).
"""

import numpy as np
import pytest
import qutip as qt

import quantum_sensing as qs


N = 60
QFI = lambda **kw: qs.calculate_qfi(qs.get_state_backaction, param_value=0.0,
                                    param_type="s", N_basis=N, **kw)


def covariance(state, N_basis):
    x = qs.quadrature(N_basis, 0.0)
    p = qs.quadrature(N_basis, np.pi / 2)
    ops = [x, p]
    C = np.zeros((2, 2))
    for i, A in enumerate(ops):
        for j, B in enumerate(ops):
            C[i, j] = (0.5 * qt.expect(A * B + B * A, state).real
                       - qt.expect(A, state).real * qt.expect(B, state).real)
    return C


# ---------------------------------------------------------------------------
# Generator and unitary conventions
# ---------------------------------------------------------------------------

def test_signal_shifts_orthogonal_quadrature():
    out = qs.get_state_backaction(s=0.7, chain=("sig",), N_basis=N,
                                  rho=qt.basis(N, 0))
    p = qs.quadrature(N, np.pi / 2)
    assert qt.expect(p, out).real == pytest.approx(-0.7, abs=1e-9)


def test_shear_maps_p_to_p_minus_chi_x():
    alpha = 1.5 / np.sqrt(2)  # <x> = 1.5
    out = qs.get_state_backaction(chi_ba=0.4, ba_type="shear", chain=("ba",),
                                  N_basis=N, rho=qt.coherent(N, alpha))
    x = qs.quadrature(N, 0.0)
    p = qs.quadrature(N, np.pi / 2)
    assert qt.expect(x, out).real == pytest.approx(1.5, abs=1e-8)
    assert qt.expect(p, out).real == pytest.approx(-0.6, abs=1e-8)


def test_shear_squeeze_parameter():
    chi = 1.3
    out = qs.get_state_backaction(chi_ba=chi, ba_type="shear", chain=("ba",),
                                  N_basis=N, rho=qt.basis(N, 0))
    ev = np.linalg.eigvalsh(covariance(out, N))
    r = qs.shear_squeeze_r(chi)
    assert ev[0] == pytest.approx(np.exp(-2 * r) / 2, rel=1e-8)
    assert ev[1] == pytest.approx(np.exp(+2 * r) / 2, rel=1e-8)


# ---------------------------------------------------------------------------
# Lossless QFI anchors
# ---------------------------------------------------------------------------

def test_vacuum_displacement_qfi_is_2():
    q = QFI(chain=("sig",), state="coherent", nbar=0.0)
    assert q == pytest.approx(2.0, rel=1e-6)


def test_qfi_equals_4_var_of_generator():
    # Squeezed input, both orientations; QFI = 4 Var(X_0) = 2 e^{±2r}.
    # (N_big keeps squeezed-tail truncation below the analytic tolerance.)
    N_big = 120
    for angle, sign in [(0.0, -1), (np.pi, +1)]:
        psi = qs.make_state("squeezed", 2.0, N_big, squeeze_angle=angle)
        var_x = covariance(psi, N_big)[0, 0]
        q = qs.calculate_qfi(qs.get_state_backaction, param_value=0.0,
                             param_type="s", N_basis=N_big,
                             chain=("sig",), rho=psi)
        assert q == pytest.approx(4 * var_x, rel=1e-5)
        r = np.arcsinh(np.sqrt(2.0))
        assert q == pytest.approx(2 * np.exp(2 * sign * r), rel=1e-3)


def test_ba_after_signal_never_changes_qfi():
    # BA2: any parameter-independent unitary after the signal is invisible
    # to QFI (unitary invariance).
    for ba_type, chi in [("shear", 2.0), ("kerr", 0.08)]:
        q0 = QFI(chain=("sig", "ba"), chi_ba=0.0, ba_type=ba_type,
                 state="cat", nbar=3.0)
        q1 = QFI(chain=("sig", "ba"), chi_ba=chi, ba_type=ba_type,
                 state="cat", nbar=3.0)
        assert q1 == pytest.approx(q0, rel=1e-6)


def test_shear_invisible_for_phase_quadrature_signal():
    # GW-like configuration: signal generator X_0 commutes with shear X_0^2,
    # so lossless QFI is chi-independent in BA1 and BA3 alike.
    for chain in [("ba", "sig"), ("basig",)]:
        q0 = QFI(chain=chain, chi_ba=0.0, ba_type="shear",
                 state="coherent", nbar=4.0)
        q1 = QFI(chain=chain, chi_ba=2.0, ba_type="shear",
                 state="coherent", nbar=4.0)
        assert q1 == pytest.approx(q0, rel=1e-6)


def test_shear_ba1_offaxis_analytic():
    # Signal generator p (theta_sig = pi/2) does not commute with the shear:
    # BA1 on vacuum gives QFI = 4 Var(p)_sheared = 2 (1 + chi^2).
    chi = 0.9
    q = QFI(chain=("ba", "sig"), chi_ba=chi, ba_type="shear",
            theta_sig=np.pi / 2, state="coherent", nbar=0.0)
    assert q == pytest.approx(2 * (1 + chi ** 2), rel=1e-6)


def test_kerr_ba1_changes_qfi():
    q0 = QFI(chain=("ba", "sig"), chi_ba=0.0, ba_type="kerr",
             state="coherent", nbar=4.0)
    q1 = QFI(chain=("ba", "sig"), chi_ba=0.05, ba_type="kerr",
             state="coherent", nbar=4.0)
    assert abs(q1 - q0) / q0 > 1e-3


# ---------------------------------------------------------------------------
# Loss placement (Gaussian closed forms)
# ---------------------------------------------------------------------------

def test_detection_loss_backaction_penalty():
    # Vacuum -> shear(chi) -> signal -> loss(eta). Post-signal loss damps the
    # signal mean by sqrt(eta) and mixes the shear correlations with vacuum:
    # Gaussian QFI = eta * Sigma'_xx / det(Sigma') = 2 eta / (1 + eta (1-eta) chi^2).
    # The chi-dependent factor is the backaction penalty; it vanishes at
    # eta = 1 (pure state) and eta = 0, peaking at eta = 1/2.
    chi, eta = 1.5, 0.6
    q = QFI(chain=("ba", "sig", "loss_out"), chi_ba=chi, ba_type="shear",
            eta_out=eta, state="coherent", nbar=0.0)
    assert q == pytest.approx(2.0 * eta / (1 + eta * (1 - eta) * chi ** 2),
                              rel=1e-4)


def test_injection_loss_no_backaction_penalty():
    # Vacuum -> loss -> shear -> signal: loss acts before the shear, and the
    # shear preserves Var(x), so QFI stays 2 for any chi.
    q = QFI(chain=("loss_in", "ba", "sig"), chi_ba=1.5, ba_type="shear",
            eta_in=0.6, state="coherent", nbar=0.0)
    assert q == pytest.approx(2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Closed-form noise channels vs independent mesolve evolution
# ---------------------------------------------------------------------------

def test_loss_channel_matches_mesolve():
    from quantum_sensing.backaction import _loss_channel
    Ns, eta = 15, 0.65
    psi = (qt.coherent(Ns, 1.2) + qt.basis(Ns, 3)).unit()
    rho0 = qt.ket2dm(psi)
    kappa = -np.log(eta)
    ref = qt.mesolve(0 * qt.num(Ns), rho0, [0.0, 1.0],
                     [np.sqrt(kappa) * qt.destroy(Ns)]).states[-1]
    out = _loss_channel(rho0, eta, Ns)
    assert (out - ref).norm() < 1e-6
    assert out.tr() == pytest.approx(1.0, abs=1e-10)


def test_loss_channel_no_overflow_at_large_cutoff():
    # Regression: the Kraus iterate a^k rho a†^k overflows float64 at large
    # N_basis unless (1-eta)^k / k! is folded in during the recursion. A
    # sheared anti-squeezed state at N=400 exercises exactly the failing case;
    # the Gaussian closed form gives the exact QFI for this configuration.
    kw = dict(chain=("ba", "sig", "loss_out"), chi_ba=2.0, ba_type="shear",
              eta_out=0.8, state="squeezed", nbar=4.0, squeeze_angle=np.pi)
    rho = qs.get_state_backaction(s=0.0, N_basis=400, **kw)
    assert np.isfinite(rho.full()).all()
    assert rho.tr().real == pytest.approx(1.0, abs=1e-8)
    q = qs.calculate_qfi(qs.get_state_backaction, param_value=0.0,
                         param_type="s", N_basis=400, **kw)
    r, chi, eta = np.arcsinh(2.0), 2.0, 0.8
    X = np.exp(2 * r) / 2
    Sxx = eta * X + (1 - eta) / 2
    Sxp = eta * (-chi * X)
    Spp = eta * (np.exp(-2 * r) / 2 + chi ** 2 * X) + (1 - eta) / 2
    q_ref = eta * Sxx / (Sxx * Spp - Sxp ** 2)
    assert q == pytest.approx(q_ref, rel=1e-3)


def test_dephasing_channel_matches_mesolve():
    from quantum_sensing.backaction import _dephasing_channel
    Ns, pn = 15, 0.4
    psi = (qt.coherent(Ns, 1.2) + qt.basis(Ns, 3)).unit()
    rho0 = qt.ket2dm(psi)
    chi = pn ** 2
    ref = qt.mesolve(0 * qt.num(Ns), rho0, [0.0, 1.0],
                     [np.sqrt(chi) * qt.num(Ns)]).states[-1]
    out = _dephasing_channel(rho0, pn, Ns)
    assert (out - ref).norm() < 1e-6


# ---------------------------------------------------------------------------
# dynamics.py loss/t_final regression
# ---------------------------------------------------------------------------

def test_single_mode_loss_transmission_matches_eta_for_any_tfinal():
    from quantum_sensing import get_state_single_mode
    eta, t_final = 0.7, 2.5
    rho0 = qt.coherent(20, 1.5)
    out = get_state_single_mode(eta_ch=eta, t_final=t_final, N_basis=20,
                                rho=rho0)
    nbar_out = qs.mean_photon_number(out)
    assert nbar_out == pytest.approx(eta * 2.25, rel=1e-5)


# ---------------------------------------------------------------------------
# State factory and convergence helper
# ---------------------------------------------------------------------------

def test_state_factory_hits_target_nbar():
    for fam in qs.STATE_FAMILIES:
        psi = qs.make_state(fam, 4.0, 100)
        assert qs.mean_photon_number(psi) == pytest.approx(4.0, abs=2e-3), fam


def test_converged_qfi_cat_kerr():
    q, N_used, hist = qs.calculate_qfi_converged(
        chain=("ba", "sig"), chi_ba=0.05, ba_type="kerr",
        state="cat", nbar=4.0, N_start=30, rtol=1e-4)
    assert len(hist) >= 2
    assert q > 0
