"""
Loss placement vs frequency: the same total loss (eta = 0.9) placed at each
slot of the BA1 ordering (loss -> shear -> loss -> signal -> loss), plus the
BA3 concurrent channel loss:

    injection  loss before the back-action        (BA1 eta_in)
    mid        loss between back-action and signal (BA1 eta_mid)
    detection  loss after the signal               (BA1 eta_out)
    channel    loss concurrent with signal+shear   (BA3 eta_ch)

For epsilon_a the two unitaries commute, so BA1 is representative: injection
and mid differ only in whether the loss precedes the shear.  States:
sqz_vac, cat, fock at <n> = 2.

Usage:   python quantum_sensing/studies/loss_placement.py
Output:  quantum_sensing/studies/results/loss_placement.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantum_sensing as qs
from quantum_sensing import probes
from quantum_sensing.channels import get_state_ba1, get_state_ba3
from quantum_sensing.convergence import converged_qfi

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TARGET = 2.0
ETA = 0.9
FREQS_HZ = np.geomspace(1000.0, 20.0, 8)
FAMILIES = ["sqz_vac", "cat", "fock"]
N_MAX = 600

PLACEMENTS = {
    # name -> (dynamics, loss kwargs, output-tail check needed?)
    "injection": (get_state_ba1, {"eta_in": ETA}, False),
    "mid": (get_state_ba1, {"eta_mid": ETA}, False),
    "detection": (get_state_ba1, {"eta_out": ETA}, True),
    "channel": (get_state_ba3, {"eta_ch": ETA}, True),
}


def main():
    rows = []
    for family in FAMILIES:
        factory = probes.PROBE_FAMILIES[family](N_TARGET)
        print(f"\n{family}:")
        for placement, (dyn, loss_kw, check_tail) in PLACEMENTS.items():
            N_warm = 20
            for f_hz in FREQS_HZ:
                kappa = qs.rp_kappa(2 * np.pi * f_hz)
                t0 = time.time()
                res = converged_qfi(
                    factory, dynamics=dyn, param_type="epsilon_a",
                    kappa_ba=kappa, N_start=max(20, N_warm - 20),
                    N_step=20, N_max=N_MAX,
                    check_output_tail=check_tail, **loss_kw)
                N_warm = res["N_basis"]
                rows.append(dict(state=family, placement=placement,
                                 f_hz=f_hz, kappa_ba=kappa, qfi=res["qfi"],
                                 N_basis=res["N_basis"],
                                 converged=res["converged"]))
                print(f"  {placement:9s} f={f_hz:7.1f} Hz "
                      f"kappa={kappa:8.4f} QFI={res['qfi']:9.4f} "
                      f"N={res['N_basis']:3d} conv={res['converged']} "
                      f"({time.time()-t0:.1f}s)")
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / "loss_placement.csv", index=False)

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["state", "placement"], columns="f_hz",
                         values="qfi")
    print("\nQFI by placement (columns: f in Hz):")
    print(piv.round(3).to_string())
    if not df["converged"].all():
        print("\nWARNING: unconverged points — see loss_placement.csv")


if __name__ == "__main__":
    main()
