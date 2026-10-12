"""
Step 2b — where back-action actually bites: fixed homodyne readout.

The step-2 sweep shows the QFI for epsilon_p is nearly unaffected by the
back-action shear: the shear is a parameter-independent unitary, so a
measurement optimized per frequency (the SLD basis — the QFI is attainable
by construction) evades it, which is the Fisher-information statement of
variational-readout back-action evasion.

A real interferometer reads out one fixed quadrature.  This script computes
the classical Fisher information of homodyne at the signal quadrature X
(theta_LO = 0; epsilon_p displaces x), for the same states and frequency
grid as step 2, plus homodyne optimized over LO angle per frequency
(variational readout).

Usage:  python quantum_sensing/studies/step2b_homodyne_cfi.py
Output: quantum_sensing/studies/results/step2b_homodyne.csv
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
from quantum_sensing.cfi import _get_rho_drho, calculate_cfi_homodyne
from quantum_sensing.states import load_state_params, reconstruct_state

qs.set_data_dir(ROOT / "quantum_sensing" / "data")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Same condition and grid as step 2
from step2_frequency_sweep import ETA, PN, NT, FREQS_HZ, cutoff_for, load_best_states

LO_ANGLES = np.linspace(0, np.pi, 25, endpoint=False)


def main():
    best = load_best_states()
    builders = {st: (lambda N, e=e: reconstruct_state(e, N_basis=N))
                for st, e in best.items()}
    builders["coherent"] = lambda N: qt.coherent(N, np.sqrt(NT))

    kappas = qs.rp_kappa(2 * np.pi * FREQS_HZ)
    rows = []
    for name, build in builders.items():
        for f_hz, kba in zip(FREQS_HZ, kappas):
            t0 = time.time()
            N = cutoff_for(kba)
            psi = build(N)
            # one (rho, drho) pair per point, reused across all LO angles
            rho, drho = _get_rho_drho(
                qs.get_state_single_mode_rp, param_type="epsilon_p",
                rho=psi, N_basis=N, kappa_ba=kba, eta_ch=ETA, pn_ch=PN)
            cfi_x = calculate_cfi_homodyne(rho, drho, theta_LO=0.0)["cfi"]
            cfi_by_angle = [calculate_cfi_homodyne(rho, drho, theta_LO=th)["cfi"]
                            for th in LO_ANGLES]
            i_best = int(np.argmax(cfi_by_angle))
            rows.append(dict(state=name, freq_hz=f_hz, kappa_ba=kba,
                             cfi_homodyne_x=cfi_x,
                             cfi_homodyne_opt=cfi_by_angle[i_best],
                             theta_opt=LO_ANGLES[i_best]))
            print(f"{name:9s} f={f_hz:7.1f} Hz  kappa={kba:6.3f}  "
                  f"CFI_X={cfi_x:9.4f}  CFI_opt={cfi_by_angle[i_best]:9.4f}  "
                  f"th_opt={LO_ANGLES[i_best]:.3f}  ({time.time()-t0:.1f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "step2b_homodyne.csv"
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
