"""Reproduce the paper's n=5 baselines (Maliakal et al., arXiv:2606.19649).

For each (eta, sigma_phi) grid point: best Gaussian QFI and optimized Fock
superposition QFI at Ntarget = 5, compared with the paper's Fig. 2b values
and the analytic upper bounds.  Writes CSV + markdown into
quantum_sensing/results/ and the optimized states into an .npz for reuse.

Run:  python -m quantum_sensing.run_paper_baseline
"""

import json
import os
import time

import numpy as np

from .optimize import (DisplacementChannel, optimize_gaussian,
                       optimize_fock_superposition, qfi_bounds, mean_n)

N_BASIS = 40      # optimization cutoff (parameters are insensitive to the tail)
N_REFINE = 100    # cutoff for the final QFI evaluation (squeezed-vacuum tails)
NBAR = 5.0


def _qfi_big(psi, eta, sigma_phi, N_big):
    import qutip as qt
    import quantum_sensing as qs
    return qs.calculate_qfi(lambda **kw: qs.get_state_single_mode(**kw),
                            param_type="epsilon_a", rho=qt.Qobj(psi),
                            N_basis=N_big, eta_ch=eta, pn_in=sigma_phi)


def refine_qfi_fock(state, eta, sigma_phi, N_big=N_REFINE):
    """Re-evaluate a Fock-superposition ket at a large cutoff (zero-padding is
    exact: the state has no tail beyond its support)."""
    psi = np.zeros(N_big, dtype=complex)
    psi[:len(state)] = state
    return _qfi_big(psi, eta, sigma_phi, N_big)


def refine_qfi_gaussian(params, nbar, eta, sigma_phi, N_big=N_REFINE):
    """Rebuild the optimal Gaussian from its parameters at a large cutoff
    (restores the squeezed-vacuum tail the optimization cutoff clipped)."""
    from .optimize import gaussian_ket
    s, f, theta_r, theta_a = params
    s, f = np.clip(s, 0, 1), np.clip(f, 0, 1)
    r = np.arcsinh(np.sqrt(s * f * nbar))
    amp = np.sqrt(s * (1 - f) * nbar)
    psi = gaussian_ket(N_big, r, theta_r, amp * np.exp(1j * theta_a))
    return _qfi_big(psi, eta, sigma_phi, N_big)

# (eta, sigma_phi) -> paper Fig. 2b: (optimized QFI, advantage dB); None = n/a
PAPER = {
    (0.95, 0.0): (42.1, 0.0),
    (0.95, 0.1): (31.7, 1.3),
    (0.95, 0.2): (30.5, 2.2),
    (0.99, 0.0): (72.3, 0.0),
    (0.99, 0.1): (67.8, 1.0),
    (0.99, 0.2): (65.5, 1.1),
}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    states = {}
    for (eta, sig), (F_paper, adv_paper) in PAPER.items():
        t0 = time.time()
        ch = DisplacementChannel(N_BASIS, eta=eta, sigma_phi=sig)
        g = optimize_gaussian(ch, NBAR, n_starts=16)
        f = optimize_fock_superposition(ch, NBAR, n_starts=8)
        g["qfi"] = refine_qfi_gaussian(g["params"], NBAR, eta, sig)
        f["qfi"] = refine_qfi_fock(f["state"], eta, sig)
        F_best = max(g["qfi"], f["qfi"])
        adv = 10 * np.log10(F_best / g["qfi"])
        b = qfi_bounds(eta, sig, NBAR)
        rows.append({
            "eta": eta, "sigma_phi": sig,
            "qfi_gaussian": g["qfi"], "nbar_gaussian": g["nbar"],
            "qfi_fock_sup": f["qfi"], "nbar_fock_sup": f["nbar"],
            "advantage_dB": adv,
            "qfi_paper": F_paper, "advantage_paper_dB": adv_paper,
            "bound_min": b["min"],
            "runtime_s": time.time() - t0,
        })
        states[f"gauss_eta{eta}_sig{sig}"] = g["state"]
        states[f"fock_eta{eta}_sig{sig}"] = f["state"]
        r = rows[-1]
        print(f"eta={eta} sig={sig}: gauss={r['qfi_gaussian']:.2f} "
              f"fock={r['qfi_fock_sup']:.2f} adv={adv:.2f} dB "
              f"(paper {F_paper}, {adv_paper} dB) bound={b['min']:.1f} "
              f"[{r['runtime_s']:.0f}s]", flush=True)

    # CSV
    import csv
    csv_path = os.path.join(out_dir, "paper_baseline.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # states for reuse in the back-action study
    np.savez(os.path.join(out_dir, "paper_baseline_states.npz"), **states)

    # markdown table
    md = ["# Paper baseline reproduction (Ntarget = 5, N_basis = %d)\n" % N_BASIS,
          "Channel: dephasing (prep) -> displacement signal + loss (sensing), "
          "as in Maliakal et al. Fig. 1.  `advantage = 10 log10(F_best/F_gauss)`; "
          "`F_best = max(F_gauss, F_fock_sup)`.\n",
          "| eta | sigma_phi | F_gauss | F_fock_sup | adv (dB) | paper F_opt | paper adv | bound |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['eta']} | {r['sigma_phi']} | {r['qfi_gaussian']:.2f} "
                  f"| {r['qfi_fock_sup']:.2f} | {r['advantage_dB']:.2f} "
                  f"| {r['qfi_paper']} | {r['advantage_paper_dB']} "
                  f"| {r['bound_min']:.1f} |")
    with open(os.path.join(out_dir, "paper_baseline.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
