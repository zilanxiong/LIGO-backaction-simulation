"""
Quantum state dynamics helpers for sensing — Python port of julia_code/DynamicsHelper.jl (CPU).

Provides MZI Heisenberg-picture evolution, two-arm evolution, single-mode evolution,
and beam-splitter transformations, all using QuTiP mesolve with identical physics
to the Julia CPU module.
"""

import numpy as np
import qutip as qt

from conversions import loss_to_kappa, phirms_to_chi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dm(state, N_basis=None):
    """Convert ket -> density matrix if needed."""
    if state.isket:
        return qt.ket2dm(state)
    return state


def _apply_noise_stage(rho, eta_vals, pn_vals, ops, n_ops):
    """Apply a loss + dephasing noise stage via mesolve (matches Julia helper)."""
    c_ops = []
    t_stage = 1.0

    for i, eta in enumerate(eta_vals):
        if eta < 1.0:
            kappa = loss_to_kappa(1 - eta, t_stage)
            c_ops.append(np.sqrt(kappa) * ops[i])

    for i, pn in enumerate(pn_vals):
        if pn > 0.0:
            chi = phirms_to_chi(pn, t_stage)
            c_ops.append(np.sqrt(chi) * n_ops[i])

    if len(c_ops) == 0:
        return rho

    H_zero = 0 * ops[0].dag() * ops[0]
    res = qt.mesolve(H_zero, rho, [0, t_stage], c_ops)
    return res.states[-1]


# ---------------------------------------------------------------------------
# MZI Heisenberg-picture evolution  (Julia: get_state_mzi_heis)
# ---------------------------------------------------------------------------

def get_state_mzi_heis(
    *,
    Delta=0.0,
    epsilon_a=0.0,
    epsilon_p=0.0,
    eta_arms=(1.0, 1.0),
    pn_arms=(0.0, 0.0),
    eta_in=(1.0, 1.0),
    pn_in=(0.0, 0.0),
    eta_out=(1.0, 1.0),
    pn_out=(0.0, 0.0),
    t_final=1.0,
    rho=None,
    N_basis=20,
    sensing_mode="single",
):
    """
    Mach-Zehnder interferometer in the Heisenberg picture.

    The density matrix stays in the input common and differential (C, D) basis.
    Beam splitters are encoded as operator transformations:
        a_A = (a_C + a_D)/sqrt(2)   (arm A)
        a_B = (a_C - a_D)/sqrt(2)   (arm B)

    Evolution stages:
        1. Input noise  (Lindblads on C, D)
        2. Arm dynamics (H + Lindblads in A, B basis)
        3. Output noise (Lindblads on C, D)
    """
    # --- IO-basis operators ---
    aC = qt.tensor(qt.destroy(N_basis), qt.qeye(N_basis))
    aD = qt.tensor(qt.qeye(N_basis), qt.destroy(N_basis))
    NC = aC.dag() * aC
    ND = aD.dag() * aD

    # --- Arm-basis operators ---
    aA = (aC + aD) / np.sqrt(2)
    aB = (aC - aD) / np.sqrt(2)
    NA = aA.dag() * aA
    NB = aB.dag() * aB

    # --- Input state ---
    if rho is None:
        psi_vac = qt.tensor(qt.fock(N_basis, 0), qt.fock(N_basis, 0))
        rho_dm = qt.ket2dm(psi_vac)
    else:
        rho_dm = _ensure_dm(rho)

    # --- Stage 1: Input noise (C, D) ---
    rho_dm = _apply_noise_stage(rho_dm, eta_in, pn_in, [aC, aD], [NC, ND])

    # --- Stage 2: Arm dynamics ---
    if sensing_mode == "single":
        H = (Delta * NA
             + epsilon_a * (aA.dag() + aA)
             + 1j * epsilon_p * (aA.dag() - aA))
    elif sensing_mode == "differential":
        H = (Delta * (NA - NB)
             + epsilon_a * ((aA.dag() + aA) - (aB.dag() + aB))
             + 1j * epsilon_p * ((aA.dag() - aA) - (aB.dag() - aB)))
    else:
        raise ValueError("sensing_mode must be 'single' or 'differential'")

    L_arms = []
    if eta_arms[0] < 1.0:
        kappaA = loss_to_kappa(1 - eta_arms[0], t_final)
        L_arms.append(np.sqrt(kappaA) * aA)
    if eta_arms[1] < 1.0:
        kappaB = loss_to_kappa(1 - eta_arms[1], t_final)
        L_arms.append(np.sqrt(kappaB) * aB)
    if pn_arms[0] > 0.0:
        chiA = phirms_to_chi(pn_arms[0], t_final)
        L_arms.append(np.sqrt(chiA) * NA)
    if pn_arms[1] > 0.0:
        chiB = phirms_to_chi(pn_arms[1], t_final)
        L_arms.append(np.sqrt(chiB) * NB)

    res_arms = qt.mesolve(H, rho_dm, np.linspace(0, t_final, 2), L_arms)
    rho_dm = res_arms.states[-1]

    # --- Stage 3: Output noise (C, D) ---
    rho_dm = _apply_noise_stage(rho_dm, eta_out, pn_out, [aC, aD], [NC, ND])

    return rho_dm


# ---------------------------------------------------------------------------
# Two-arm (direct arm basis) evolution  (Julia: get_state_arms)
# ---------------------------------------------------------------------------

def get_state_arms(
    *,
    Delta=0.0,
    eta_x=1.0,
    eta_y=1.0,
    pn_x=0.0,
    pn_y=0.0,
    t_final=1.0,
    N_basis=20,
    rho_x=None,
    rho_y=None,
    rho=None,
):
    """
    Two-arm interferometer: sensing on X arm, then loss + dephasing on both.
    """
    # Input state
    if rho is None:
        rho_x0 = rho_x if rho_x is not None else qt.coherent(N_basis, 1.0)
        rho_y0 = rho_y if rho_y is not None else qt.fock(N_basis, 0)
        rho0 = qt.tensor(rho_x0, rho_y0)
    else:
        rho0 = rho

    rho0 = _ensure_dm(rho0)

    # Operators
    ax = qt.tensor(qt.destroy(N_basis), qt.qeye(N_basis))
    ay = qt.tensor(qt.qeye(N_basis), qt.destroy(N_basis))

    # Sensing
    H = Delta * ax.dag() * ax
    t_list = [0.0, t_final]
    result = qt.mesolve(H, rho0, t_list, [])
    rho_sensed = result.states[-1]

    # Loss and dephasing
    c_ops = []
    if eta_x < 1.0:
        kappa_x = loss_to_kappa(1 - eta_x)
        c_ops.append(np.sqrt(kappa_x) * ax)
    if eta_y < 1.0:
        kappa_y = loss_to_kappa(1 - eta_y)
        c_ops.append(np.sqrt(kappa_y) * ay)
    if pn_x > 0.0:
        chi_x = phirms_to_chi(pn_x)
        c_ops.append(np.sqrt(chi_x) * ax.dag() * ax)
    if pn_y > 0.0:
        chi_y = phirms_to_chi(pn_y)
        c_ops.append(np.sqrt(chi_y) * ay.dag() * ay)

    if len(c_ops) > 0:
        H1 = 0 * ax.dag() * ax
        result1 = qt.mesolve(H1, rho_sensed, t_list, c_ops)
        return result1.states[-1]

    return rho_sensed


# ---------------------------------------------------------------------------
# Single-mode evolution  (Julia: get_state_single_mode)
# ---------------------------------------------------------------------------

def get_state_single_mode(
    *,
    Delta=0.0,
    epsilon_a=0.0,
    epsilon_p=0.0,
    eta_in=1.0,
    eta_ch=1.0,
    eta_out=1.0,
    pn_in=0.0,
    pn_ch=0.0,
    pn_out=0.0,
    sigma_a=0.0,
    sigma_p=0.0,
    t_final=1.0,
    N_basis=20,
    rho=None,
):
    """
    Single-mode channel with 3-stage noise pipeline:
        1. Input noise   (eta_in, pn_in)
        2. Sensing channel  (H + channel loss/dephasing/amplitude/phase noise)
        3. Output noise  (eta_out, pn_out)

    Mirrors Julia DynamicsHelper.get_state_single_mode exactly.
    """
    # Default state
    if rho is None:
        rho0 = qt.ket2dm(qt.coherent(N_basis, 1.0))
    else:
        rho0 = _ensure_dm(rho)

    a = qt.destroy(N_basis)
    n_op = a.dag() * a

    # --- Stage 1: Input noise ---
    L_in = []
    t_in = 1.0
    if eta_in < 1.0:
        kappa_in = loss_to_kappa(1 - eta_in)
        L_in.append(np.sqrt(kappa_in) * a)
    if pn_in > 0.0:
        chi_in = phirms_to_chi(pn_in)
        L_in.append(np.sqrt(chi_in) * n_op)

    if len(L_in) == 0:
        rho1 = rho0
    else:
        H_zero = 0 * n_op
        res_in = qt.mesolve(H_zero, rho0, [0, t_in], L_in)
        rho1 = res_in.states[-1]

    # --- Stage 2: Sensing channel ---
    H = Delta * n_op + epsilon_a * (a.dag() + a) + 1j * epsilon_p * (a.dag() - a)

    L_ch = []
    if eta_ch < 1.0:
        kappa = loss_to_kappa(1 - eta_ch)
        L_ch.append(np.sqrt(kappa) * a)
    if pn_ch > 0.0:
        chi = phirms_to_chi(pn_ch)
        L_ch.append(np.sqrt(chi) * n_op)
    if sigma_a > 0.0:
        L_ch.append(np.sqrt(sigma_a) * (a.dag() + a))
    if sigma_p > 0.0:
        L_ch.append(1j * np.sqrt(sigma_p) * (a.dag() - a))

    res_ch = qt.mesolve(H, rho1, [0.0, t_final], L_ch)
    rho_sensed = res_ch.states[-1]

    # --- Stage 3: Output noise ---
    L_out = []
    t_out = 1.0
    if eta_out < 1.0:
        kappa_out = loss_to_kappa(1 - eta_out)
        L_out.append(np.sqrt(kappa_out) * a)
    if pn_out > 0.0:
        chi_out = phirms_to_chi(pn_out)
        L_out.append(np.sqrt(chi_out) * n_op)

    if len(L_out) == 0:
        return rho_sensed

    H_zero = 0 * n_op
    res_out = qt.mesolve(H_zero, rho_sensed, [0, t_out], L_out)
    return res_out.states[-1]


# ---------------------------------------------------------------------------
# Beam splitter  (Julia: apply_bs)
# ---------------------------------------------------------------------------

def apply_bs(rho_in, N_basis, reverse=False, theta=np.pi / 4):
    """
    Apply a beam-splitter unitary U_BS = exp(theta (a_C^dag a_D - a_D^dag a_C)).

    Works on both kets and density matrices.
    """
    aC = qt.tensor(qt.destroy(N_basis), qt.qeye(N_basis))
    aD = qt.tensor(qt.qeye(N_basis), qt.destroy(N_basis))

    sign = -1 if reverse else 1
    G_BS = aC.dag() * aD - aD.dag() * aC
    U_BS = (sign * theta * G_BS).expm()

    if rho_in.isket:
        return U_BS * rho_in
    else:
        return U_BS * rho_in * U_BS.dag()
