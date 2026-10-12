"""
Fixed-<n> state sweep: QFI vs frequency through the radiation-pressure
channel (BA3, single simultaneous Hamiltonian), O4-LIGO-calibrated
kappa_ba(Omega) = rp_kappa(Omega).

For each probe family at <n> = N_TARGET and each loss configuration
(injection: loss -> BA; detection: BA -> loss), the QFI for epsilon_a
(GW-frame phase-quadrature displacement) is computed at each frequency with
the automated Fock-cutoff convergence harness.  Lossless QFI is analytic
(8 Var(x), kappa-independent) and recorded from the input state directly.

Frequencies run high -> low so kappa_ba grows monotonically and each point
warm-starts the cutoff search from the previous converged N.

Cost metrics per (state, config) over the band (flat signal weight for now):
    metric_i   = harmonic-mean QFI  [ (mean 1/QFI)^-1, from int w/QFI ]
    metric_ii  = min QFI over the band (broadband flatness)
    metric_iii = sum QFI over the fixed set

Usage:   python quantum_sensing/studies/sweep_states_rp.py
Output:  quantum_sensing/studies/results/sweep_states_rp.csv
         quantum_sensing/studies/results/sweep_states_rp_metrics.csv
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

# Optimized-without-backaction states live in the ROOT repo package's data
# dir (not duplicated in rp-qfi; see rp-qfi/README.md).
qs.set_data_dir(ROOT.parent / "quantum_sensing" / "data")

N_TARGET = 2.0
ETA_LOSS = 0.9
FREQS_HZ = np.geomspace(1000.0, 20.0, 8)  # high -> low so kappa grows
PARAM = "epsilon_a"
N_MAX = 600

# For injection loss everything after the loss is unitary, so the QFI is
# exactly cutoff-insensitive to shear-pumped output population; skip the
# output-tail check there (see converged_qfi docstring).
CONFIGS = {
    "injection": ({"eta_in": ETA_LOSS}, {"check_output_tail": False}),
    "detection": ({"eta_out": ETA_LOSS}, {"check_output_tail": True}),
}


def build_probes():
    fams = {name: fam(N_TARGET) for name, fam in probes.PROBE_FAMILIES.items()}
    try:
        fams["opt_fock_sup"] = probes.optimized(N_TARGET, "fock_sup")
    except Exception as exc:  # data file absent or slice missing
        print(f"(optimized states skipped: {exc})")
    return fams


def lossless_qfi(factory, N=160):
    """Analytic lossless QFI for epsilon_a: 8 Var(x), kappa-independent."""
    psi = factory(N)
    a = qt.destroy(N)
    x = (a + a.dag()) / np.sqrt(2)
    var_x = (qt.expect(x * x, psi) - qt.expect(x, psi) ** 2).real
    return 8 * var_x


def main():
    fams = build_probes()
    rows = []
    for state, factory in fams.items():
        F0 = lossless_qfi(factory)
        print(f"\n{state}: <n>={probes.mean_n(factory(160)):.4f} "
              f"lossless QFI = 8 Var(x) = {F0:.4f}")
        for config, (loss_kw, conv_kw) in CONFIGS.items():
            N_STEP = 20
            N_warm = 20
            for f_hz in FREQS_HZ:
                kappa = qs.rp_kappa(2 * np.pi * f_hz)
                t0 = time.time()
                # Start one step below the previous converged cutoff so the
                # confirmation evaluations land AT it rather than above it
                # (otherwise N ratchets up by ~2 steps per frequency).
                res = converged_qfi(
                    factory, dynamics=get_state_ba3, param_type=PARAM,
                    kappa_ba=kappa, N_start=max(20, N_warm - N_STEP),
                    N_step=N_STEP, N_max=N_MAX, **conv_kw, **loss_kw)
                N_warm = res["N_basis"]
                rows.append(dict(
                    state=state, config=config, f_hz=f_hz, kappa_ba=kappa,
                    qfi=res["qfi"], qfi_lossless=F0, N_basis=res["N_basis"],
                    converged=res["converged"], n_target=N_TARGET,
                    eta=ETA_LOSS))
                print(f"  {config:9s} f={f_hz:7.1f} Hz kappa={kappa:8.4f} "
                      f"QFI={res['qfi']:9.4f} N={res['N_basis']:3d} "
                      f"conv={res['converged']} ({time.time()-t0:.1f}s)")
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / "sweep_states_rp.csv", index=False)

    df = pd.DataFrame(rows)
    met = (df.groupby(["state", "config"])["qfi"]
           .agg(metric_i=lambda q: 1.0 / np.mean(1.0 / q),
                metric_ii="min", metric_iii="sum")
           .reset_index())
    met.to_csv(RESULTS_DIR / "sweep_states_rp_metrics.csv", index=False)
    print("\nBand metrics (flat weight):")
    print(met.to_string(index=False))
    if not df["converged"].all():
        bad = df[~df["converged"]]
        print(f"\nWARNING: {len(bad)} unconverged points (N_max={N_MAX}):")
        print(bad[["state", "config", "f_hz", "N_basis"]].to_string(index=False))


if __name__ == "__main__":
    main()
