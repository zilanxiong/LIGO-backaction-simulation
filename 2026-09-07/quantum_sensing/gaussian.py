"""
Gaussian (covariance-matrix) track for the radiation-pressure QFI study.

Exact moment-level evolution of single-mode Gaussian states through the same
channel as dynamics.get_state_single_mode_rp: input loss -> (signal + RP
shear + channel loss, evolved simultaneously) -> output loss.  Serves as the
ground truth the Fock-space code is validated against, and reproduces the
SQL / frequency-dependent-squeezing curves analytically.

Conventions (match the Fock code with hbar = 1, t_final = 1):
    x = (a + a†)/sqrt(2),  p = (a - a†)/(i sqrt(2)),  vacuum Var = 1/2.
    Signal Hamiltonian  H = epsilon_a (a†+a) + i epsilon_p (a†-a)
                          = sqrt(2) (epsilon_a x + epsilon_p p),
    so over unit time epsilon_a displaces p by -sqrt(2) epsilon_a and
    epsilon_p displaces x by +sqrt(2) epsilon_p.
    Back-action  H_BA = g x^2 with g = ba_to_g(kappa_ba) = kappa_ba/2,
    i.e. the ponderomotive shear p -> p - kappa_ba x (b2 = a2 - kappa a1).

Loss is exact (thermal-vacuum diffusion); dephasing (pn) is non-Gaussian and
is deliberately NOT part of this track.

Key analytic facts encoded in the tests:
    - lossless QFI for epsilon_a is 8 Var(x) of the input, independent of
      kappa_ba (the shear is a known unitary commuting with the generator);
    - vacuum homodyne of p:  CFI = 4 / (1 + kappa^2)  (back-action SQL);
    - squeezing at the FD angle phi = arctan(kappa) - pi/2 restores
      CFI = 4 e^{2r} / (1 + kappa^2).
"""

import numpy as np
from scipy.linalg import expm

from .conversions import loss_to_kappa, ba_to_g

VAC_VAR = 0.5


# ---------------------------------------------------------------------------
# State constructors (V, d)
# ---------------------------------------------------------------------------

def vacuum_state():
    return VAC_VAR * np.eye(2), np.zeros(2)


def coherent_state(alpha):
    """Coherent |alpha>: vacuum covariance, mean (x, p) = sqrt(2)(Re, Im)."""
    d = np.sqrt(2) * np.array([np.real(alpha), np.imag(alpha)])
    return VAC_VAR * np.eye(2), d


def rotation(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def squeezed_state(r, phi=0.0, alpha=0.0):
    """Squeezed(-coherent) state.  phi = 0 squeezes x (Var x = e^{-2r}/2),
    matching qutip squeeze(N, r) * |0>; phi rotates the squeezed quadrature
    counter-clockwise (phi = pi/2 squeezes p)."""
    R = rotation(phi)
    V = R @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ R.T * VAC_VAR
    d = np.sqrt(2) * np.array([np.real(alpha), np.imag(alpha)])
    return V, d


# ---------------------------------------------------------------------------
# Channel building blocks
# ---------------------------------------------------------------------------

def shear_symplectic(kappa_ba):
    """Integrated ponderomotive shear: (x, p) -> (x, p - kappa_ba x)."""
    return np.array([[1.0, 0.0], [-kappa_ba, 1.0]])


def apply_symplectic(S, V, d):
    return S @ V @ S.T, S @ d


def apply_loss(V, d, eta):
    """Beam-splitter loss with transmission eta (vacuum in the open port)."""
    return eta * V + (1 - eta) * VAC_VAR * np.eye(2), np.sqrt(eta) * d


def signal_displacement(epsilon_a=0.0, epsilon_p=0.0):
    """Mean displacement from unit-time signal Hamiltonian (no back-action)."""
    return np.sqrt(2) * np.array([epsilon_p, -epsilon_a])


def _stage2_matrices(kappa_ba, eta_ch, t_final=1.0):
    """Drift A and diffusion D for the sensing stage: shear + channel loss."""
    g = ba_to_g(kappa_ba, t_final)
    A = np.array([[0.0, 0.0], [-2 * g, 0.0]])
    D = np.zeros((2, 2))
    if eta_ch < 1.0:
        kl = loss_to_kappa(1 - eta_ch)
        A = A - 0.5 * kl * np.eye(2)
        D = kl * VAC_VAR * np.eye(2)
    return A, D


def _evolve_stage2(V, d, kappa_ba, eta_ch, epsilon_a, epsilon_p, t_final=1.0):
    """Exact simultaneous evolution of (V, d) under shear + loss + signal,
    via augmented matrix exponentials (no ODE solver)."""
    A, D = _stage2_matrices(kappa_ba, eta_ch, t_final)
    s = np.sqrt(2) * np.array([epsilon_p, -epsilon_a])

    # Mean: d' = A d + s  ->  augmented 3x3 exponential.
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = s
    E = expm(M * t_final)
    d_out = E[:2, :2] @ d + E[:2, 2]

    # Covariance: V' = A V + V A^T + D  ->  vectorized 5x5 exponential.
    K = np.kron(A, np.eye(2)) + np.kron(np.eye(2), A)
    M2 = np.zeros((5, 5))
    M2[:4, :4] = K
    M2[:4, 4] = D.reshape(-1)
    E2 = expm(M2 * t_final)
    V_out = (E2[:4, :4] @ V.reshape(-1) + E2[:4, 4]).reshape(2, 2)
    return V_out, d_out


# ---------------------------------------------------------------------------
# Full channel (mirrors get_state_single_mode_rp stages)
# ---------------------------------------------------------------------------

def gaussian_rp_channel(V, d, *, epsilon_a=0.0, epsilon_p=0.0, kappa_ba=0.0,
                        eta_in=1.0, eta_ch=1.0, eta_out=1.0, t_final=1.0,
                        signal_order="simultaneous"):
    """Evolve Gaussian moments (V, d) through the RP sensing channel.

    signal_order controls where the signal displacement acts relative to the
    back-action shear (all applied inside stage 2, between eta_in and
    eta_out):
        "before"       (BA2)  displacement, then shear (+ channel loss);
        "after"        (BA1)  shear (+ channel loss), then displacement;
        "simultaneous" (BA3)  joint evolution, exact — matches the Fock code
                              get_state_single_mode_rp, which adds H_BA to
                              the signal Hamiltonian in one mesolve.
    For epsilon_a sensing the three coincide when eta_ch = 1 (the signal
    generator x commutes with H_BA).
    """
    V, d = apply_loss(V, d, eta_in)

    if signal_order == "simultaneous":
        V, d = _evolve_stage2(V, d, kappa_ba, eta_ch, epsilon_a, epsilon_p,
                              t_final)
    elif signal_order == "before":
        d = d + signal_displacement(epsilon_a, epsilon_p)
        V, d = _evolve_stage2(V, d, kappa_ba, eta_ch, 0.0, 0.0, t_final)
    elif signal_order == "after":
        V, d = _evolve_stage2(V, d, kappa_ba, eta_ch, 0.0, 0.0, t_final)
        d = d + signal_displacement(epsilon_a, epsilon_p)
    else:
        raise ValueError("signal_order must be 'before', 'after', "
                         "or 'simultaneous'")

    V, d = apply_loss(V, d, eta_out)
    return V, d


# ---------------------------------------------------------------------------
# QFI / CFI for displacement sensing
# ---------------------------------------------------------------------------

def qfi_gaussian_displacement(V, dd):
    """QFI for a parameter entering only the mean: F = dd^T V^{-1} dd.
    Exact for Gaussian states with parameter-independent covariance."""
    return float(dd @ np.linalg.solve(V, dd))

def cfi_homodyne_gaussian(V, dd, angle=np.pi / 2):
    """CFI of homodyne on quadrature q = x cos(angle) + p sin(angle)
    (default: p, the phase quadrature)."""
    u = np.array([np.cos(angle), np.sin(angle)])
    return float((u @ dd) ** 2 / (u @ V @ u))


def gaussian_qfi_rp(V, d, *, param_type="epsilon_a", kappa_ba=0.0,
                    eta_in=1.0, eta_ch=1.0, eta_out=1.0,
                    signal_order="simultaneous", homodyne_angle=None,
                    prec=1e-6):
    """QFI (or homodyne CFI, if homodyne_angle is given) for displacement
    sensing through the RP channel, starting from Gaussian moments (V, d).

    The channel is linear, so the mean derivative is computed exactly by a
    single finite difference (linearity makes it step-size independent), and
    the covariance is parameter-independent.
    """
    kwargs = dict(kappa_ba=kappa_ba, eta_in=eta_in, eta_ch=eta_ch,
                  eta_out=eta_out, signal_order=signal_order)
    V0, d0 = gaussian_rp_channel(V, d, **kwargs)
    _, d1 = gaussian_rp_channel(V, d, **{**kwargs, param_type: prec})
    dd = (d1 - d0) / prec
    if homodyne_angle is not None:
        return cfi_homodyne_gaussian(V0, dd, homodyne_angle)
    return qfi_gaussian_displacement(V0, dd)


# ---------------------------------------------------------------------------
# Reference curves (SQL and frequency-dependent squeezing)
# ---------------------------------------------------------------------------

def fd_squeeze_angle(kappa_ba):
    """Frequency-dependent squeeze angle that aligns the squeezed quadrature
    with the measured combination b2 = p - kappa x: phi = arctan(kappa) - pi/2
    (phi = -pi/2, i.e. p squeezed, at kappa = 0)."""
    return np.arctan(kappa_ba) - np.pi / 2


def sql_homodyne_cfi(kappa_ba):
    """Vacuum/coherent-input homodyne (p) CFI for epsilon_a through the RP
    channel: 4 / (1 + kappa^2) — the back-action-limited SQL curve."""
    return 4.0 / (1 + np.asarray(kappa_ba) ** 2)


def fd_squeezed_homodyne_cfi(kappa_ba, r):
    """Homodyne (p) CFI for epsilon_a with input squeezed at the FD angle:
    4 e^{2r} / (1 + kappa^2)."""
    return 4.0 * np.exp(2 * r) / (1 + np.asarray(kappa_ba) ** 2)
