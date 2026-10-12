"""
Recompute band cost metrics from the state-sweep CSV under different signal
weights: flat, and inspiral |h(f)|^2 ~ f^{-7/3} (QCRB waveform-estimation
weighting, which emphasizes the low-frequency, high-kappa end of the band).

Usage:   python quantum_sensing/studies/band_metrics.py
Input:   quantum_sensing/studies/results/sweep_states_rp.csv
Output:  quantum_sensing/studies/results/band_metrics.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantum_sensing import metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WEIGHTS = {"flat": metrics.flat_weight, "inspiral": metrics.inspiral_weight}


def main():
    df = pd.read_csv(RESULTS_DIR / "sweep_states_rp.csv")
    rows = []
    for (state, config), sub in df.groupby(["state", "config"]):
        for wname, w in WEIGHTS.items():
            m = metrics.band_metrics(sub["f_hz"], sub["qfi"], w)
            rows.append(dict(state=state, config=config, weight=wname, **m))
    out = pd.DataFrame(rows).sort_values(["config", "weight", "metric_i"],
                                         ascending=[True, True, False])
    out.to_csv(RESULTS_DIR / "band_metrics.csv", index=False)
    for (config, wname), sub in out.groupby(["config", "weight"]):
        print(f"\n=== {config} loss, {wname} weight "
              f"(ranked by metric_i) ===")
        print(sub[["state", "metric_i", "metric_ii", "metric_iii"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
