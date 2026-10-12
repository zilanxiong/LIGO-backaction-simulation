"""Step-1 validation of the quantum_sensing package (mentor's port).

Checks the radiation-pressure back-action channel `get_state_single_mode_rp`
and the QFI/CFI machinery against known physics:

1. QFI of the epsilon_a signal is *flat* in kappa_ba at eta = 1 -- the
   back-action generator x^2 commutes with the signal generator x, so the
   ponderomotive shear alone destroys no information.
2. Lossless QFI values match analytics: for a pure probe and signal
   H = epsilon_a (a + a^dag) = sqrt(2) epsilon_a x, F_Q = 8 Var(x).
3. With channel loss, kappa_ba *does* degrade the QFI (shear anti-squeezes
   the state, loss mixes the amplified quadrature back in).
4. Homodyne CFI at fixed phase readout (theta_LO = pi/2) for a coherent probe
   is exactly 4 / (1 + kappa_ba^2) -- ponderomotive noise -- while a
   variational readout angle recovers the full QFI of 4 (KLMTV back-action
   evasion).
5. CFI <= QFI always; the SLD-eigenbasis measurement saturates the QFI.
"""

import numpy as np
import pytest
import qutip as qt

import quantum_sensing as qs

N_BASIS = 60
R_SQZ = np.arcsinh(np.sqrt(2.0))  # nbar = 2 squeezed vacuum


def rp_dynamics(**kw):
    return qs.get_state_single_mode_rp(**kw)


def probe_states():
    return {
        "coherent": qt.coherent(N_BASIS, np.sqrt(2.0)),
        "fock2": qt.fock(N_BASIS, 2),
        # anti-squeezed in x: information about the p-displacement signal
        "squeezed": qt.squeeze(N_BASIS, -R_SQZ) * qt.fock(N_BASIS, 0),
        "even_cat": (qt.coherent(N_BASIS, 1.4) + qt.coherent(N_BASIS, -1.4)).unit(),
    }


def qfi(psi, **kw):
    return qs.calculate_qfi(rp_dynamics, param_type="epsilon_a",
                            rho=psi, N_basis=N_BASIS, **kw)


# ---------------------------------------------------------------------------
# 1. Back-action alone is free (commuting generators)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["coherent", "fock2", "squeezed", "even_cat"])
def test_qfi_flat_in_kappa_ba_lossless(name):
    psi = probe_states()[name]
    values = [qfi(psi, kappa_ba=k) for k in (0.0, 1.0, 2.0)]
    assert max(values) - min(values) < 1e-2 * max(values)


# ---------------------------------------------------------------------------
# 2. Lossless QFI analytics: F_Q = 8 Var(x)
# ---------------------------------------------------------------------------

def test_qfi_coherent_analytic():
    assert qfi(probe_states()["coherent"], kappa_ba=0.0) == pytest.approx(4.0, rel=1e-3)


def test_qfi_fock_analytic():
    # Fock |n>: Var(x) = (2n + 1)/2  ->  F_Q = 4 (2n + 1); n = 2 -> 20
    assert qfi(probe_states()["fock2"], kappa_ba=0.0) == pytest.approx(20.0, rel=1e-3)


def test_qfi_squeezed_analytic():
    # anti-squeezed x: Var(x) = e^{2r}/2 -> F_Q = 4 e^{2r}
    expected = 4.0 * np.exp(2 * R_SQZ)
    assert qfi(probe_states()["squeezed"], kappa_ba=0.0) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. Back-action x loss destroys information
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["coherent", "fock2", "even_cat"])
def test_qfi_decreases_with_kappa_under_loss(name):
    psi = probe_states()[name]
    values = [qfi(psi, kappa_ba=k, eta_ch=0.9) for k in (0.0, 1.0, 2.0)]
    assert values[0] > values[1] > values[2]


# ---------------------------------------------------------------------------
# 4. Homodyne: ponderomotive noise and variational recovery
# ---------------------------------------------------------------------------

def cfi_homodyne(psi, theta_LO, **kw):
    return qs.calculate_cfi(rp_dynamics, param_type="epsilon_a", rho=psi,
                            N_basis=N_BASIS, measurement="homodyne",
                            theta_LO=theta_LO, **kw)["cfi"]


@pytest.mark.parametrize("kappa_ba", [0.0, 0.5, 1.0, 2.0])
def test_homodyne_phase_readout_coherent(kappa_ba):
    psi = probe_states()["coherent"]
    expected = 4.0 / (1.0 + kappa_ba**2)
    assert cfi_homodyne(psi, np.pi / 2, kappa_ba=kappa_ba) == pytest.approx(expected, rel=1e-3)


def test_variational_readout_recovers_qfi():
    # KLMTV: theta with cot(theta) = kappa_ba evades back-action for a
    # coherent probe, restoring the lossless CFI of 4.
    psi = probe_states()["coherent"]
    kappa_ba = 1.0
    theta_var = np.arctan(1.0 / kappa_ba)
    assert cfi_homodyne(psi, theta_var, kappa_ba=kappa_ba) == pytest.approx(4.0, rel=1e-2)


# ---------------------------------------------------------------------------
# 5. CFI bounds
# ---------------------------------------------------------------------------

def test_cfi_bounded_by_qfi_and_sld_saturates():
    psi = probe_states()["even_cat"]
    kw = dict(kappa_ba=1.0, eta_ch=0.95, pn_in=0.1)
    F_Q = qfi(psi, **kw)
    for meas in ("photon_counting", "homodyne", "sld_optimal"):
        r = qs.calculate_cfi(rp_dynamics, param_type="epsilon_a", rho=psi,
                             N_basis=N_BASIS, measurement=meas, **kw)
        assert r["cfi"] <= F_Q * (1 + 1e-6)
        if meas == "sld_optimal":
            assert r["cfi"] == pytest.approx(F_Q, rel=1e-3)


# ---------------------------------------------------------------------------
# Truncation guard
# ---------------------------------------------------------------------------

def test_cutoff_convergence_at_large_shear():
    psi60 = qt.coherent(60, np.sqrt(2.0))
    psi80 = qt.coherent(80, np.sqrt(2.0))
    F60 = qs.calculate_qfi(rp_dynamics, param_type="epsilon_a",
                           rho=psi60, N_basis=60, kappa_ba=2.0, eta_ch=0.9)
    F80 = qs.calculate_qfi(rp_dynamics, param_type="epsilon_a",
                           rho=psi80, N_basis=80, kappa_ba=2.0, eta_ch=0.9)
    assert F60 == pytest.approx(F80, rel=1e-3)
