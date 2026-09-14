"""Quantum sensing: dynamics, QFI/CFI, SLD, state reconstruction, and GKP states."""

from quantum_sensing.conversions import (
    loss_to_kappa, kappa_to_loss,
    phirms_to_chi, chi_to_phirms,
    ba_to_g, rp_power_ratio, rp_kappa,
    qfi_to_phi_min,
    dB_to_r, r_to_dB, var_to_dB, r_to_var,
    dB_to_n, n_to_dB, r_to_n, n_to_r,
    loss_lim_dB,
)

from quantum_sensing.dynamics import (
    get_state_mzi_heis, get_state_arms, get_state_single_mode,
    get_state_single_mode_rp, apply_bs,
)

from quantum_sensing.sld import (
    calculate_sld, calculate_qfi,
)

from quantum_sensing.cfi import (
    calculate_cfi, compare_cfi,
    calculate_cfi_photon_counting, calculate_cfi_homodyne,
    calculate_cfi_heterodyne, calculate_cfi_sld_optimal,
    calculate_cfi_projective,
)

from quantum_sensing.states import (
    set_data_dir,
    load_results_df, load_state_params, list_available_eta_pn,
    get_best_qfi_envelope, get_nsup_trend, get_eta_trend, get_pn_trend,
    reconstruct_fock_sup, reconstruct_sqz_vac, reconstruct_sqz_coh,
    reconstruct_state, get_wigner,
    harmonic_oscillator_wavefunction, wavefunction_to_qobj,
    STATE_COLORS, STATE_LABELS, STATE_MARKERS,
)

from quantum_sensing.gaussian import (
    vacuum_state, coherent_state, squeezed_state,
    gaussian_rp_channel, gaussian_qfi_rp,
    qfi_gaussian_displacement, cfi_homodyne_gaussian,
    fd_squeeze_angle, sql_homodyne_cfi, fd_squeezed_homodyne_cfi,
)

from quantum_sensing.channels import (
    get_state_ba1, get_state_ba2, get_state_ba3, BA_CHANNELS,
    op_signal, op_shear, op_loss_dephase, op_signal_shear_simultaneous,
    apply_chain,
)

from quantum_sensing.convergence import (
    converged_qfi, tail_population,
)

from quantum_sensing.gkp import (
    discrete_q, gkp_s_values, gkp, num_state,
    gkp_fock_coeff, gkp_fock,
)
