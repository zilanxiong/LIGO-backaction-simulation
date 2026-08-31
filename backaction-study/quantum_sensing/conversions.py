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


# --- Radiation-pressure (ponderomotive) coupling ---

def kimble_K(Omega, I_ratio=1.0, gamma=1.0):
    """
    Kimble factor K(Omega) of the tuned interferometer (KLMTV 2001):
        K = 2 (I0/I_SQL) gamma^4 / (Omega^2 (gamma^2 + Omega^2)).
    Equals 1 at Omega = gamma when I0 = I_SQL. Used as the frequency-
    dependent ponderomotive shear strength chi(Omega).
    """
    Omega = np.asarray(Omega, dtype=float)
    return 2.0 * I_ratio * gamma ** 4 / (Omega ** 2 * (gamma ** 2 + Omega ** 2))
