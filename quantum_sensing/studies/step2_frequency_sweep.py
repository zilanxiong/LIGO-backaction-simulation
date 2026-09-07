"""
Step 2 — how much does radiation-pressure back-action hurt the paper's
optimized states, as a function of gravitational-wave frequency?

Takes the optimized states from states.h5 (loss_ch, chosen eta/pn slice,
N_target = NT, best n_sups per family), plus a coherent-state benchmark at
the same photon number, and evolves each through the back-action channel
get_state_single_mode_rp with the KLMTV-calibrated frequency-dependent gain

    kappa_ba(Omega) = rp_kappa(Omega)     (aLIGO O4 defaults),

computing the QFI for the paper's sensing parameter epsilon_p at each
frequency, alongside the kappa_ba = 0 baseline (the paper's own channel).

The Fock cutoff is scaled with kappa_ba (the shear pumps photons into the
anti-squeezed quadrature) and a convergence check is run at the strongest
shear point.

Usage:  python quantum_sensing/studies/step2_frequency_sweep.py
Output: quantum_sensing/studies/results/step2_sweep.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qutip as qt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantum_sensing as qs
from quantum_sensing.states import load_state_params, reconstruct_state

qs.set_data_dir(ROOT / "quantum_sensing" / "data")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- Study condition (one slice of the paper's grid) -----------------------
ETA = 0.9          # channel transmission (loss_ch placement, as in the paper)
PN = 0.0           # phase noise
NT = 5.0           # photon-number budget
FREQS_HZ = np.logspace(np.log10(15.0), np.log10(2000.0), 11)


def cutoff_for(kappa_ba):
    """Fock cutoff scaled with the shear strength."""
    return int(min(150, 60 + 18 * kappa_ba))


def load_best_states():
    """Best-QFI entry per state family at (loss_ch, ETA, PN, NT)."""
    states = {}
    for st in ["fock_sup", "sqz_vac", "sqz_coh"]:
        entries = [e for e in load_state_params(st, "loss_ch", ETA, PN)
                   if abs(e["N_target"] - NT) < 0.01]
        if entries:
            best = max(entries, key=lambda e: e["qfi"])
            states[st] = best
            print(f"{st}: n_sups={best['n_sups']}  paper QFI={best['qfi']:.4f}")
    return states


def qfi_at(psi_builder, kappa_ba):
    N = cutoff_for(kappa_ba)
    psi = psi_builder(N)
    return qs.calculate_qfi(qs.get_state_single_mode_rp, param_type="epsilon_p",
                            rho=psi, N_basis=N, kappa_ba=kappa_ba, eta_ch=ETA,
                            pn_ch=PN)


def main():
    best = load_best_states()

    builders = {st: (lambda N, e=e: reconstruct_state(e, N_basis=N))
                for st, e in best.items()}
    builders["coherent"] = lambda N: qt.coherent(N, np.sqrt(NT))

    kappas = qs.rp_kappa(2 * np.pi * FREQS_HZ)
    rows = []
    for name, build in builders.items():
        q0 = qfi_at(build, 0.0)
        print(f"\n{name}: no-BA baseline QFI = {q0:.4f}")
        for f_hz, kba in zip(FREQS_HZ, kappas):
            t0 = time.time()
            q = qfi_at(build, kba)
            rows.append(dict(state=name, freq_hz=f_hz, kappa_ba=kba,
                             qfi=q, qfi_noba=q0, N_basis=cutoff_for(kba)))
            print(f"  f={f_hz:7.1f} Hz  kappa={kba:6.3f}  N={cutoff_for(kba):3d}  "
                  f"QFI={q:9.4f}  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "step2_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # Convergence check at the strongest shear
    kmax = kappas.max()
    name = "sqz_vac" if "sqz_vac" in builders else "coherent"
    N_hi = cutoff_for(kmax) + 30
    psi = builders[name](N_hi)
    q_hi = qs.calculate_qfi(qs.get_state_single_mode_rp, param_type="epsilon_p",
                            rho=psi, N_basis=N_hi, kappa_ba=kmax, eta_ch=ETA,
                            pn_ch=PN)
    q_lo = df[(df.state == name)
              & (df.kappa_ba == kmax)].qfi.iloc[0]
    print(f"convergence ({name}, kappa={kmax:.2f}): "
          f"N={cutoff_for(kmax)} -> {q_lo:.5f},  N={N_hi} -> {q_hi:.5f}, "
          f"rel diff = {abs(q_hi-q_lo)/q_hi:.2e}")


if __name__ == "__main__":
    main()
