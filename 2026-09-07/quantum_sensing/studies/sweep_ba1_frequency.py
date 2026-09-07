"""
BA1 ordering study: radiation pressure FIRST, then displacement sensing.

Channel (channels.get_state_ba1, t_final = 1 conventions):

    rho -> shear(kappa_ba) -> signal(epsilon_a) -> detection loss (eta_out)

with kappa_ba(Omega) = rp_kappa(Omega), the O4-LIGO-calibrated free-mass
back-action gain.  Detection loss is what makes the ordering physical:
the shear acts on the probe BEFORE the signal is written, and the
shear-amplified anti-squeezed quadrature then couples to vacuum at the
lossy readout.  (With no loss after the shear, the epsilon_a QFI would be
exactly 8 Var(x) of the input at every frequency — the shear preserves x —
so a lossless BA1 sweep is flat by construction; detection loss is the
minimal channel where the ordering has consequences.)

Part 1 — frequency sweep at fixed <n> = N_TARGET for six probe families
(coherent, squeezed vacuum, even cat, squeezed cat, Fock, and the
optimized-without-backaction fock_sup states from the previous campaign,
bundled in quantum_sensing/data/states.h5).  Every point runs through the
automated Fock-cutoff convergence harness; frequencies go high -> low so
kappa grows monotonically and each point warm-starts from the previous
converged cutoff.

Part 2 — <n> scaling at a fixed back-action-dominated frequency
(F_SCALING_HZ, where kappa ~ 1): QFI(n) for n in N_SCALING per family,
with the fitted power-law exponent s in QFI ~ n^s reported per family.

Loss placement is selectable: "detection" (default) puts the loss AFTER
the signal (eta_out); "injection" puts it BEFORE the shear (eta_in).  For
epsilon_a the injection configuration is an analytic control: the shear
preserves x and everything after the loss is unitary, so the QFI must be
frequency-INDEPENDENT — any frequency structure seen there is numerics.

Usage:   python quantum_sensing/studies/sweep_ba1_frequency.py [injection|detection]
Output:  quantum_sensing/studies/results/ba1_frequency_sweep[_injection].csv
         quantum_sensing/studies/results/ba1_n_scaling[_injection].csv
         quantum_sensing/studies/results/ba1_qfi_vs_frequency[_injection].png
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
from quantum_sensing.channels import (get_state_ba1, get_state_ba2,
                                      get_state_ba3)
from quantum_sensing.convergence import converged_qfi

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Self-contained: optimized states ship inside this package's data dir.
qs.set_data_dir(ROOT / "quantum_sensing" / "data")

N_TARGET = 2.0
ETA_LOSS = 0.9

# Loss placement: "detection" (eta_out, after the signal) or "injection"
# (eta_in, before the shear).  With injection loss everything after the loss
# is unitary, so the output-tail convergence check is provably irrelevant
# for the QFI and is skipped (see converged_qfi docstring).
LOSS_CONFIG = (sys.argv[1] if len(sys.argv) > 1 else "detection")
DYNAMICS = get_state_ba1
if LOSS_CONFIG == "detection":
    LOSS_KW = {"eta_out": ETA_LOSS}
    CONV_KW = {"check_output_tail": True}
    SUFFIX = ""
elif LOSS_CONFIG == "injection":
    LOSS_KW = {"eta_in": ETA_LOSS}
    CONV_KW = {"check_output_tail": False}
    SUFFIX = "_injection"
elif LOSS_CONFIG == "ba2_detection":
    # BA2 (signal, then shear) with readout loss.  For epsilon_a the signal
    # generator x commutes with H_BA ~ x^2, so with no loss between the two
    # unitaries this must coincide exactly with the BA1 detection sweep —
    # a machinery consistency check as much as a physics run.
    DYNAMICS = get_state_ba2
    LOSS_KW = {"eta_out": ETA_LOSS}
    CONV_KW = {"check_output_tail": True}
    SUFFIX = "_ba2_detection"
elif LOSS_CONFIG == "ba2_injection":
    # BA2 with injection loss: everything after the loss is unitary, so
    # this must coincide with the BA1 injection sweep (frequency-flat).
    DYNAMICS = get_state_ba2
    LOSS_KW = {"eta_in": ETA_LOSS}
    CONV_KW = {"check_output_tail": False}
    SUFFIX = "_ba2_injection"
elif LOSS_CONFIG == "pn_ba1_detection":
    # Phase noise -> BA1 -> loss: input dephasing (exact elementwise mask,
    # rms PN_IN) before the shear, then signal, then readout loss.
    PN_IN = 0.1
    LOSS_KW = {"pn_in": PN_IN, "eta_out": ETA_LOSS}
    CONV_KW = {"check_output_tail": True}
    SUFFIX = "_pn_ba1_detection"
elif LOSS_CONFIG == "ba3_full":
    # The physical case: simultaneous signal + back-action (BA3) with all
    # three loss slots populated at LIGO-ish values — injection 0.95,
    # intracavity 0.99 concurrent with the interaction, readout 0.90.
    DYNAMICS = get_state_ba3
    LOSS_KW = {"eta_in": 0.95, "eta_ch": 0.99, "eta_out": 0.90}
    CONV_KW = {"check_output_tail": True}
    SUFFIX = "_ba3_full"
else:
    raise SystemExit(f"unknown loss config {LOSS_CONFIG!r}")
FREQS_HZ = np.geomspace(1000.0, 10.0, 13)   # high -> low so kappa grows
PARAM = "epsilon_a"
N_MAX = 800   # kappa(10 Hz) ~ 9.5 pumps ~kappa^2 <x^2>/2 photons; wide
              # states need cutoffs of several hundred there

F_SCALING_HZ = 30.0                          # kappa ~ 1: BA-dominated
N_SCALING = [1.0, 2.0, 5.0, 10.0]            # optimized states exist here

STATE_ORDER = ["coherent", "sqz_vac", "cat", "sqz_cat", "fock",
               "opt_fock_sup"]
STATE_LABELS = {
    "coherent": "Coherent",
    "sqz_vac": "Squeezed vac.",
    "cat": "Even cat",
    "sqz_cat": "Squeezed cat",
    "fock": "Fock",
    "opt_fock_sup": "Optimized (no-BA)",
}
STATE_COLORS = {
    "coherent": "#7f7f7f",
    "sqz_vac": "#d62728",
    "cat": "#1f77b4",
    "sqz_cat": "#9467bd",
    "fock": "#2ca02c",
    "opt_fock_sup": "#ff7f0e",
}


def build_probes(n_target):
    fams = {name: fam(n_target) for name, fam in probes.PROBE_FAMILIES.items()}
    fams["opt_fock_sup"] = probes.optimized(n_target, "fock_sup")
    return {k: fams[k] for k in STATE_ORDER}


def sweep_frequencies(state, factory, freqs_hz):
    """converged QFI at each frequency, warm-starting the cutoff search."""
    rows = []
    N_STEP = 20
    N_warm = 20
    for f_hz in freqs_hz:
        kappa = qs.rp_kappa(2 * np.pi * f_hz)
        t0 = time.time()
        res = converged_qfi(
            factory, dynamics=DYNAMICS, param_type=PARAM,
            kappa_ba=kappa, **LOSS_KW, **CONV_KW,
            N_start=max(20, N_warm - N_STEP), N_step=N_STEP, N_max=N_MAX)
        N_warm = res["N_basis"]
        rows.append(dict(
            state=state, f_hz=f_hz, kappa_ba=kappa, qfi=res["qfi"],
            N_basis=res["N_basis"], converged=res["converged"],
            n_target=N_TARGET, loss_config=LOSS_CONFIG, eta=ETA_LOSS))
        print(f"  f={f_hz:7.1f} Hz  kappa={kappa:8.4f}  "
              f"QFI={res['qfi']:9.4f}  N={res['N_basis']:3d}  "
              f"conv={res['converged']}  ({time.time()-t0:.1f}s)")
    return rows


def n_scaling(freq_hz):
    """QFI(n) at fixed frequency for every family + power-law fit."""
    kappa = qs.rp_kappa(2 * np.pi * freq_hz)
    rows = []
    for state in STATE_ORDER:
        N_STEP = 20
        N_warm = 20
        for n in N_SCALING:
            factory = build_probes(n)[state]
            res = converged_qfi(
                factory, dynamics=DYNAMICS, param_type=PARAM,
                kappa_ba=kappa, **LOSS_KW, **CONV_KW,
                N_start=max(20, N_warm - N_STEP), N_step=N_STEP, N_max=N_MAX)
            N_warm = res["N_basis"]
            rows.append(dict(state=state, n=n, f_hz=freq_hz, kappa_ba=kappa,
                             qfi=res["qfi"], N_basis=res["N_basis"],
                             converged=res["converged"],
                             loss_config=LOSS_CONFIG, eta=ETA_LOSS))
            print(f"  {state:13s} n={n:5.1f}  QFI={res['qfi']:9.4f}  "
                  f"N={res['N_basis']:3d}  conv={res['converged']}")
    df = pd.DataFrame(rows)
    # power-law exponent s: log QFI = s log n + c
    exps = {}
    for state, sub in df.groupby("state"):
        s, _ = np.polyfit(np.log(sub["n"]), np.log(sub["qfi"]), 1)
        exps[state] = s
    df["scaling_exponent"] = df["state"].map(exps)
    return df, exps


def plot(df_freq, exps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for state in STATE_ORDER:
        sub = df_freq[df_freq["state"] == state]
        label = STATE_LABELS[state]
        if state in exps:
            label += rf"  ($F \sim n^{{{exps[state]:.2f}}}$)"
        ax.loglog(sub["f_hz"], sub["qfi"], "o-", ms=4,
                  color=STATE_COLORS[state], label=label)

    ax.set_xlabel("Frequency  $\\Omega/2\\pi$  [Hz]")
    ax.set_ylabel(r"QFI for $\epsilon_a$")
    if LOSS_CONFIG == "detection":
        title = ("BA1: shear $\\to$ signal $\\to$ detection loss "
                 f"($\\langle n\\rangle$ = {N_TARGET:g}, $\\eta$ = {ETA_LOSS})")
    elif LOSS_CONFIG == "injection":
        title = ("BA1: injection loss $\\to$ shear $\\to$ signal "
                 f"($\\langle n\\rangle$ = {N_TARGET:g}, $\\eta$ = {ETA_LOSS})")
    elif LOSS_CONFIG == "ba2_detection":
        title = ("BA2: signal $\\to$ shear $\\to$ detection loss "
                 f"($\\langle n\\rangle$ = {N_TARGET:g}, $\\eta$ = {ETA_LOSS})")
    elif LOSS_CONFIG == "ba2_injection":
        title = ("BA2: injection loss $\\to$ signal $\\to$ shear "
                 f"($\\langle n\\rangle$ = {N_TARGET:g}, $\\eta$ = {ETA_LOSS})")
    elif LOSS_CONFIG == "pn_ba1_detection":
        title = ("phase noise ($\\phi_{rms}$=0.1) $\\to$ BA1 $\\to$ "
                 "detection loss "
                 f"($\\langle n\\rangle$ = {N_TARGET:g}, $\\eta$ = {ETA_LOSS})")
    else:
        title = ("BA3: $\\eta_{in}$=0.95 $\\to$ [signal+shear, "
                 "$\\eta_{ch}$=0.99] $\\to$ $\\eta_{out}$=0.90 "
                 f"($\\langle n\\rangle$ = {N_TARGET:g})")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right",
              title=f"exponent fit at {F_SCALING_HZ:g} Hz")
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda f: qs.rp_kappa(2 * np.pi * np.maximum(f, 1e-9)),
                   lambda k: np.where(k > 0, np.sqrt(
                       2 * qs.rp_power_ratio()) * (2 * np.pi * 450.0)
                       / (2 * np.pi * np.sqrt(np.maximum(k, 1e-30))), 1.0)))
    sec.set_xlabel(r"$\kappa_{ba}(\Omega)$")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"figure -> {path}")


def main():
    fams = build_probes(N_TARGET)
    freq_rows = []
    for state, factory in fams.items():
        print(f"\n{state}: <n> = {probes.mean_n(factory(160)):.4f}")
        freq_rows += sweep_frequencies(state, factory, FREQS_HZ)
        pd.DataFrame(freq_rows).to_csv(
            RESULTS_DIR / f"ba1_frequency_sweep{SUFFIX}.csv", index=False)
    df_freq = pd.DataFrame(freq_rows)

    print(f"\n<n> scaling at {F_SCALING_HZ:g} Hz "
          f"(kappa = {qs.rp_kappa(2 * np.pi * F_SCALING_HZ):.3f}):")
    df_n, exps = n_scaling(F_SCALING_HZ)
    df_n.to_csv(RESULTS_DIR / f"ba1_n_scaling{SUFFIX}.csv", index=False)

    print("\nQFI ~ n^s exponents:")
    for state in STATE_ORDER:
        print(f"  {STATE_LABELS[state]:18s} s = {exps[state]:.3f}")

    plot(df_freq, exps, RESULTS_DIR / f"ba1_qfi_vs_frequency{SUFFIX}.png")

    for df, tag in [(df_freq, "frequency sweep"), (df_n, "n scaling")]:
        if not df["converged"].all():
            bad = df[~df["converged"]]
            print(f"\nWARNING: {len(bad)} unconverged {tag} points "
                  f"(N_max={N_MAX}):")
            print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
