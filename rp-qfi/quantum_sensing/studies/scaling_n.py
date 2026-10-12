"""
<n>-scaling of the QFI under radiation pressure with detection loss.

For each probe family, sweep the fixed mean photon number n and fit the
scaling exponent s in QFI ~ n^s (log-log least squares over the upper half
of the n range, where the asymptotic behavior dominates):
  - lossless (analytic 8 Var(x), kappa-independent);
  - detection loss eta = 0.9 at two back-action strengths,
    kappa(106.9 Hz) ~ 0.08 (weak BA) and kappa(35 Hz) ~ 0.77 (strong BA).

Usage:   python quantum_sensing/studies/scaling_n.py
Output:  quantum_sensing/studies/results/scaling_n.csv (per-point)
         quantum_sensing/studies/results/scaling_n_exponents.csv
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
from quantum_sensing import probes
from quantum_sensing.channels import get_state_ba3
from quantum_sensing.convergence import converged_qfi

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TARGETS = [1.0, 2.0, 4.0, 8.0]
ETA_LOSS = 0.9
KAPPAS = {  # label -> (f_hz, kappa)
    "weak_ba": (106.9, qs.rp_kappa(2 * np.pi * 106.9)),
    "strong_ba": (35.0, qs.rp_kappa(2 * np.pi * 35.0)),
}
FAMILIES = ["coherent", "sqz_vac", "cat", "sqz_cat", "fock"]
N_MAX = 700


def lossless_qfi(factory, N=300):
    psi = factory(N)
    a = qt.destroy(N)
    x = (a + a.dag()) / np.sqrt(2)
    return 8 * (qt.expect(x * x, psi) - qt.expect(x, psi) ** 2).real


def fit_exponent(n, q):
    """Log-log slope over the upper half of the n range."""
    n, q = np.asarray(n, float), np.asarray(q, float)
    keep = n >= np.median(n)
    return float(np.polyfit(np.log(n[keep]), np.log(q[keep]), 1)[0])


def main():
    rows = []
    for family in FAMILIES:
        for n_t in N_TARGETS:
            factory = probes.PROBE_FAMILIES[family](n_t)
            row = dict(state=family, n_target=n_t,
                       qfi_lossless=lossless_qfi(factory))
            for label, (f_hz, kappa) in KAPPAS.items():
                t0 = time.time()
                res = converged_qfi(
                    factory, dynamics=get_state_ba3, param_type="epsilon_a",
                    kappa_ba=kappa, eta_out=ETA_LOSS,
                    N_start=30, N_step=30, N_max=N_MAX)
                row[f"qfi_{label}"] = res["qfi"]
                row[f"conv_{label}"] = res["converged"]
                print(f"{family:9s} n={n_t:4.1f} {label:9s} "
                      f"QFI={res['qfi']:9.4f} N={res['N_basis']:3d} "
                      f"conv={res['converged']} ({time.time()-t0:.1f}s)")
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "scaling_n.csv",
                                      index=False)

    df = pd.DataFrame(rows)
    exps = []
    for family, sub in df.groupby("state"):
        exps.append(dict(
            state=family,
            s_lossless=fit_exponent(sub.n_target, sub.qfi_lossless),
            s_weak_ba=fit_exponent(sub.n_target, sub.qfi_weak_ba),
            s_strong_ba=fit_exponent(sub.n_target, sub.qfi_strong_ba)))
    exps = pd.DataFrame(exps)
    exps.to_csv(RESULTS_DIR / "scaling_n_exponents.csv", index=False)
    print("\nScaling exponents s in QFI ~ n^s "
          "(upper half of n range; eta_det=0.9):")
    print(exps.to_string(index=False))
    if not (df.conv_weak_ba.all() and df.conv_strong_ba.all()):
        print("\nWARNING: unconverged points present — see scaling_n.csv")


if __name__ == "__main__":
    main()
