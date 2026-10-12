"""
QFI sweep: probe states at fixed <n> through the back-action channel with
loss at {none, inj, conc, det}, over a LIGO frequency grid.

Gaussian probes (coherent, sqz_vac) use the exact covariance method
(instant); non-Gaussian probes (cat, sqz_cat, fock) use Fock-space QuTiP
with an adaptive basis. Output: results/results.csv.

Run:  python qfi-sweep/sweep.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import qutip as qt

from channel import (kimble_K, qfi_fock, gaussian_qfi, sqz_cov,
                     best_sqz_angle)
from probes import make_probe, var_x, mean_n, basis_size

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

GAMMA = 2 * np.pi * 450.0
FREQS_HZ = np.round(np.logspace(np.log10(20), np.log10(1000), 10), 1)
ANCHOR_HZ = 100.0
N_TARGETS_ANCHOR = [1.0, 2.0, 4.0, 8.0, 16.0]
N_TARGET_FREQ = 2.0
SPLITS = [0.25, 0.5, 0.75]

# (placement, eta) scenarios
SCENARIOS = [("none", 1.0), ("det", 0.9), ("det", 0.7),
             ("inj", 0.9), ("conc", 0.9)]

STATES = ["coherent", "sqz_vac", "cat", "sqz_cat", "fock"]
GAUSSIAN = {"coherent", "sqz_vac"}


def eval_state(kind, n_target, K, placement, eta):
    """Return dict(qfi, n_actual, extras...). Best-over-parameters for
    sqz_vac (angle) and sqz_cat (split; angle reused from sqz_vac)."""
    if kind == "coherent":
        q = gaussian_qfi(np.eye(2), K, placement, eta)
        return dict(qfi=q, n_actual=n_target, method="gaussian")
    if kind == "sqz_vac":
        theta, q = best_sqz_angle(n_target, K, placement, eta)
        return dict(qfi=q, n_actual=n_target, theta=theta, method="gaussian")

    # non-Gaussian: Fock space
    theta_sq, _ = best_sqz_angle(n_target, K, placement, eta)
    if kind == "cat":
        variants = [("cat", {})]
    elif kind == "sqz_cat":
        variants = [("sqz_cat", {"split": s, "theta": theta_sq})
                    for s in SPLITS]
    else:
        variants = [("fock", {})]

    best = None
    for name, kw in variants:
        psi_probe = make_probe(name, n_target, 200, **kw)
        vx = var_x(psi_probe)
        N = basis_size(n_target, K, var_x_est=vx)
        psi = make_probe(name, n_target, N, **kw)
        q = qfi_fock(qt.ket2dm(psi), K, N, placement, eta)
        row = dict(qfi=q, n_actual=mean_n(psi), method=f"fock(N={N})", **kw)
        if best is None or q > best["qfi"]:
            best = row
    return best


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()

    def record(freq_hz, n_target, kind, placement, eta, out):
        rows.append(dict(
            freq_hz=freq_hz, K=kimble_K(2 * np.pi * freq_hz, GAMMA),
            n_target=n_target, state=kind, placement=placement, eta=eta,
            qfi=out["qfi"], eps_min=1 / np.sqrt(out["qfi"]),
            n_actual=out["n_actual"], theta=out.get("theta", ""),
            split=out.get("split", ""), method=out["method"]))
        print(f"[{time.time()-t0:7.0f}s] f={freq_hz:7.1f}Hz n={n_target:4.1f} "
              f"{kind:8s} {placement:4s} eta={eta:.2f} "
              f"QFI={out['qfi']:.4f}", flush=True)

    # ---- frequency sweep at fixed <n> -----------------------------------
    for freq in FREQS_HZ:
        K = kimble_K(2 * np.pi * freq, GAMMA)
        for kind in STATES:
            for placement, eta in SCENARIOS:
                out = eval_state(kind, N_TARGET_FREQ, K, placement, eta)
                record(freq, N_TARGET_FREQ, kind, placement, eta, out)

    # ---- <n> scaling at the anchor frequency ----------------------------
    K_anchor = kimble_K(2 * np.pi * ANCHOR_HZ, GAMMA)
    for n_t in N_TARGETS_ANCHOR:
        for kind in STATES:
            for placement, eta in [("none", 1.0), ("det", 0.9), ("det", 0.7)]:
                out = eval_state(kind, n_t, K_anchor, placement, eta)
                record(ANCHOR_HZ, n_t, kind, placement, eta, out)

    with open(RESULTS / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS/'results.csv'} ({len(rows)} rows, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
