"""
Noise-chain sweep, 10 Hz - 1 kHz: loss and phase-noise placements around
the signal+back-action block, all probe states at fixed <n>, evaluated in
the sheared frame (see chains.py). Parallelized over rows.

Scenarios (stage lists; strength eta for loss, chi for pn):
  lossless            []
  loss_pre / conc / post        (eta = 0.9; post also 0.7)
  pn_pre / conc / post          (chi = 0.1)
  pn_pre + loss_post            (PN -> SB -> L)
  loss_pre + pn_post            (L -> SB -> PN)

Run:  python qfi-sweep/sweep_chains.py
"""

from __future__ import annotations

import csv
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import qutip as qt

from chains import qfi_chain, basis_size_chain
from channel import kimble_K, gaussian_qfi, sqz_cov, best_sqz_angle
from probes import make_probe, var_x, mean_n

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

GAMMA = 2 * np.pi * 450.0
FREQS_HZ = np.round(np.logspace(1, 3, 9), 1)          # 10 Hz .. 1 kHz
N_TARGET_FREQ = 2.0
ANCHORS_HZ = [31.0, 100.0]
N_TARGETS_ANCHOR = [1.0, 2.0, 4.0, 8.0]
SPLITS = [0.5, 0.75]

SCENARIOS = {
    "lossless":     [],
    "loss_pre":     [("loss", 0.9, "pre")],
    "loss_conc":    [("loss", 0.9, "conc")],
    "loss_post":    [("loss", 0.9, "post")],
    "loss_post_07": [("loss", 0.7, "post")],
    "pn_pre":       [("pn", 0.1, "pre")],
    "pn_conc":      [("pn", 0.1, "conc")],
    "pn_post":      [("pn", 0.1, "post")],
    "pn_then_loss": [("pn", 0.1, "pre"), ("loss", 0.9, "post")],
    "loss_then_pn": [("loss", 0.9, "pre"), ("pn", 0.1, "post")],
}
LOSS_ONLY = {"lossless", "loss_pre", "loss_conc", "loss_post",
             "loss_post_07"}  # Gaussian-exact path usable

STATES = ["coherent", "sqz_vac", "cat", "sqz_cat", "fock"]


def _gauss_scenario(scn):
    m = {"lossless": ("none", 1.0), "loss_pre": ("inj", 0.9),
         "loss_conc": ("conc", 0.9), "loss_post": ("det", 0.9),
         "loss_post_07": ("det", 0.7)}
    return m[scn]


def eval_row(args):
    freq, n_t, kind, scn = args
    K = kimble_K(2 * np.pi * freq, GAMMA)
    stages = SCENARIOS[scn]

    if kind in ("coherent", "sqz_vac") and scn in LOSS_ONLY:
        placement, eta = _gauss_scenario(scn)
        if kind == "coherent":
            q = gaussian_qfi(np.eye(2), K, placement, eta)
            return dict(qfi=q, n_actual=n_t, method="gaussian")
        theta, q = best_sqz_angle(n_t, K, placement, eta)
        return dict(qfi=q, n_actual=n_t, theta=theta, method="gaussian")

    # Fock path (all PN chains; all non-Gaussian states)
    if kind == "sqz_vac":
        variants = [("sqz_vac", {"theta": 0.0})]
    elif kind == "sqz_cat":
        variants = [("sqz_cat", {"split": s, "theta": 0.0}) for s in SPLITS]
    else:
        variants = [(kind, {})]

    best = None
    for name, kw in variants:
        probe_small = make_probe(name, n_t, 160, **kw)
        vx = var_x(probe_small)
        N = basis_size_chain(n_t, vx, stages, K)
        psi = make_probe(name, n_t, N, **kw)
        q = qfi_chain(qt.ket2dm(psi), K, N, stages)
        row = dict(qfi=q, n_actual=mean_n(psi), method=f"fock(N={N})", **kw)
        if best is None or q > best["qfi"]:
            best = row
    return best


def main():
    RESULTS.mkdir(exist_ok=True)
    jobs = []
    for freq in FREQS_HZ:
        for kind in STATES:
            for scn in SCENARIOS:
                jobs.append((float(freq), N_TARGET_FREQ, kind, scn))
    for freq in ANCHORS_HZ:
        for n_t in N_TARGETS_ANCHOR:
            if n_t == N_TARGET_FREQ:
                continue
            for kind in STATES:
                for scn in ("lossless", "loss_post", "pn_pre", "pn_post",
                            "pn_then_loss", "loss_then_pn"):
                    jobs.append((freq, n_t, kind, scn))

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for (freq, n_t, kind, scn), out in zip(jobs,
                                               ex.map(eval_row, jobs)):
            rows.append(dict(
                freq_hz=freq, K=kimble_K(2 * np.pi * freq, GAMMA),
                n_target=n_t, state=kind, scenario=scn,
                qfi=out["qfi"], eps_min=1 / np.sqrt(out["qfi"]),
                n_actual=out["n_actual"], theta=out.get("theta", ""),
                split=out.get("split", ""), method=out["method"]))
            print(f"[{time.time()-t0:7.0f}s] f={freq:7.1f} n={n_t:4.1f} "
                  f"{kind:8s} {scn:13s} QFI={out['qfi']:.4f}", flush=True)

    with open(RESULTS / "chains.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS/'chains.csv'} ({len(rows)} rows, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
