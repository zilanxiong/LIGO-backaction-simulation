"""
Unit conversions for quantum sensing — shared between dynamics, QFI, and SLD modules.

Canonical versions matching Julia DynamicsHelper.jl.
"""

import numpy as np


# --- Loss / dephasing rate conversions ---

def loss_to_kappa(eta, time=1.0):
    """Convert fractional loss (1-efficiency) to Lindblad decay rate."""
    return -np.log(1 - eta) / time


def kappa_to_loss(kappa, time=1.0):
    """Convert Lindblad decay rate to fractional loss."""
    return 1 - np.exp(-kappa * time)


def ba_to_g(kappa_ba, time=1.0):
    """Back-action Hamiltonian rate g in H_BA = g x^2 from the integrated
    shear kappa_ba (b2 = a2 - kappa_ba a1): [x^2, p] = 2i x gives 2 g t = kappa_ba.
    Matches Julia QuantumSensing.ba_to_g."""
    return kappa_ba / (2 * time)


_C_LIGHT = 299_792_458.0


def rp_power_ratio(*, m=40.0, L=3995.0, lambda0=1064e-9, P_circ=360e3,
                   gamma=2 * np.pi * 450.0):
    """X = I0/I_SQL = 4 P_circ omega0 / (m L gamma^3 c), omega0 = 2 pi c / lambda0.

    KLMTV (PRD 65, 022002) conventions: m is the mass of EACH mirror (the
    DARM reduced mass m/4 is already inside the KLMTV prefactors); gamma is
    the effective signal bandwidth [rad/s]. Current (O4) LIGO defaults
    (360 kW, 2*pi*450 Hz) give X ~ 2.4e-3; kappa ~ 1 near 30 Hz.
    """
    omega0 = 2 * np.pi * _C_LIGHT / lambda0
    return 4 * P_circ * omega0 / (m * L * gamma**3 * _C_LIGHT)


def rp_kappa(Omega, *, gamma=2 * np.pi * 450.0, **kwargs):
    """Dimensionless back-action gain kappa_ba(Omega) = 2 X gamma^4 /
    [Omega^2 (gamma^2 + Omega^2)] for get_state_single_mode_rp; free test
    mass, KLMTV Eq. (8). Anchor: rp_kappa(gamma) == rp_power_ratio().
    Pass your own cavity parameters (m, L, lambda0, P_circ, gamma) for other
    detectors."""
    X = rp_power_ratio(gamma=gamma, **kwargs)
    return 2 * X * gamma**4 / (Omega**2 * (gamma**2 + Omega**2))


def phirms_to_chi(phirms, time=1.0):
    """Convert phase-noise rms (rad) to dephasing Lindblad rate."""
    return phirms ** 2 / time


def chi_to_phirms(chi, time=1.0):
    """Convert dephasing Lindblad rate to phase-noise rms (rad)."""
    return np.sqrt(chi * time)


def qfi_to_phi_min(qfi):
    """Cramer-Rao bound: minimum detectable phase from QFI."""
    return 1.0 / np.sqrt(qfi)


# --- Squeezing conversions ---

def dB_to_r(dB):
    """Convert squeezing in dB to squeezing parameter r."""
    return dB / (20 * np.log10(np.e))


def r_to_dB(r):
    """Convert squeezing parameter r to dB."""
    return 20 * r * np.log10(np.e)


def var_to_dB(var):
    """Convert variance to dB relative to shot noise."""
    return 10 * np.log10(var)


def r_to_var(r):
    """Convert squeezing parameter r to squeezed quadrature variance.

    Matches Julia: exp(-r^2). Note: this is the convention used in the
    optimization code, not the textbook exp(-2r).
    """
    return np.exp(-(r ** 2))


# --- Photon number conversions ---

def dB_to_n(dB):
    """Convert squeezing dB to mean photon number."""
    return np.sinh(dB_to_r(dB)) ** 2


def n_to_dB(n):
    """Convert mean photon number to squeezing dB."""
    return r_to_dB(np.arcsinh(np.sqrt(n)))


def r_to_n(r):
    """Convert squeezing parameter to mean photon number."""
    return np.sinh(r) ** 2


def n_to_r(n):
    """Convert mean photon number to squeezing parameter."""
    return np.arcsinh(np.sqrt(n))


# --- Loss limit ---

def loss_lim_dB(loss):
    """Loss limit in dB."""
    return -10 * np.log10(loss)
