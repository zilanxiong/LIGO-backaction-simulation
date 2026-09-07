"""
Step 1 — validate the ported quantum_sensing package against the paper's data.

For a sample of optimized states stored in states.h5 (all state types and
loss placements), reconstruct the state, evolve it through
get_state_single_mode with the matching loss placement, compute the QFI with
calculate_qfi, and compare against the QFI recorded by the Julia
optimization run.

Conventions probed empirically (reported in the output):
  - sensing parameter: epsilon_p (momentum-quadrature displacement) at 0
  - loss placement:  loss_ch -> eta_ch, loss_in -> eta_in, loss_out -> eta_out
  - phase-noise placement: tried in each stage on pn > 0 cases; the stage
    that reproduces the stored QFI is reported.

Usage:  python quantum_sensing/studies/step1_validate.py
Output: quantum_sensing/studies/results/step1_validation.csv (+ stdout table)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantum_sensing as qs
from quantum_sensing.states import load_state_params, reconstruct_state

qs.set_data_dir(ROOT / "quantum_sensing" / "data")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def channel_kwargs(loss_config, eta, pn, pn_stage):
    kw = {}
    if loss_config == "loss_ch":
        kw["eta_ch"] = eta
    elif loss_config == "loss_in":
        kw["eta_in"] = eta
    elif loss_config == "loss_out":
        kw["eta_out"] = eta
    else:
        raise ValueError(loss_config)
    kw[{"in": "pn_in", "ch": "pn_ch", "out": "pn_out"}[pn_stage]] = pn
    return kw


def evaluate(entry, pn_stage, N_basis=None):
    N = N_basis if N_basis is not None else entry["N_basis"]
    psi = reconstruct_state(entry, N_basis=N)
    kw = channel_kwargs(entry["loss_config"], entry["eta"], entry["pn"], pn_stage)
    return qs.calculate_qfi(qs.get_state_single_mode, param_type="epsilon_p",
                            rho=psi, N_basis=N, **kw)


def pick_cases():
    """A spread of cases: every group, lossless / lossy / noisy, small-to-mid N."""
    wanted = [
        # (state_type, loss_config, eta, pn, N_target, n_sups)
        ("fock_sup", "loss_ch", 1.0, 0.0, 2.0, 2),
        ("fock_sup", "loss_ch", 0.8, 0.0, 2.0, 3),
        ("fock_sup", "loss_ch", 0.5, 0.0, 1.0, 4),
        ("fock_sup", "loss_ch", 0.9, 0.2, 5.0, 2),
        ("fock_sup", "loss_in", 0.8, 0.0, 2.0, 2),
        ("fock_sup", "loss_out", 0.8, 0.0, 2.0, 2),
        ("sqz_vac", "loss_ch", 0.9, 0.0, 5.0, 2),
        ("sqz_vac", "loss_ch", 0.9, 0.2, 5.0, 2),
        ("sqz_vac", "loss_in", 0.9, 0.0, 5.0, 2),
        ("sqz_vac", "loss_out", 0.9, 0.0, 5.0, 2),
        ("sqz_coh", "loss_ch", 0.9, 0.0, 5.0, 2),
        ("sqz_coh", "loss_in", 0.9, 0.0, 5.0, 2),
    ]
    cases = []
    for st, lc, eta, pn, Nt, ns in wanted:
        entries = load_state_params(st, lc, eta, pn)
        match = [e for e in entries
                 if abs(e["N_target"] - Nt) < 0.01 and e["n_sups"] == ns]
        if not match:  # fall back to any entry in the slice
            match = entries[:1]
        if match:
            cases.append(match[0])
        else:
            print(f"  (no data for {st}/{lc} eta={eta} pn={pn} Nt={Nt} ns={ns} — skipped)")
    return cases


def main():
    rows = []
    for e in pick_cases():
        ref = e["qfi"]
        row = {
            "state_type": e["state_type"], "loss_config": e["loss_config"],
            "eta": e["eta"], "pn": e["pn"], "N_target": e["N_target"],
            "n_sups": e["n_sups"], "N_basis": e["N_basis"], "qfi_ref": ref,
        }
        stages = ["ch"] if e["pn"] == 0 else ["in", "ch", "out"]
        best_stage, best_qfi, best_err = None, np.nan, np.inf
        for stg in stages:
            q = evaluate(e, stg)
            err = abs(q - ref) / ref
            if err < best_err:
                best_stage, best_qfi, best_err = stg, q, err
        row.update(qfi_ours=best_qfi, rel_err=best_err,
                   pn_stage=(best_stage if e["pn"] > 0 else "-"))
        rows.append(row)
        print(f"{e['state_type']:9s} {e['loss_config']:8s} eta={e['eta']:.2f} "
              f"pn={e['pn']:.2f} Nt={e['N_target']:4.1f} ns={e['n_sups']} | "
              f"ref={ref:9.4f} ours={best_qfi:9.4f} relerr={best_err:.2e} "
              f"pn_stage={row['pn_stage']}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "step1_validation.csv"
    df.to_csv(out, index=False)
    print(f"\nmax rel err: {df.rel_err.max():.3e}  ->  {out}")


if __name__ == "__main__":
    main()
