"""
Step 2 (figure) — QFI vs GW frequency under radiation-pressure back-action,
for three placements of the optical loss relative to the back-action stage:

    before : loss, then back-action + signal   (eta_in)
    during : loss concurrent with back-action  (eta_ch)
    after  : back-action + signal, then loss   (eta_out)

One panel per placement; series are the paper's optimized states
(states.h5, loss_ch eta=0.9, pn=0, N<=5) plus a coherent benchmark.
Dotted lines are each placement's kappa_ba = 0 baseline.

Reads  results/step2_sweep.csv (from step2_frequency_sweep.py).
Writes results/step2_qfi_vs_frequency.png.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import quantum_sensing as qs

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# CVD-validated palette (Okabe-Ito); the coherent benchmark is muted ink.
COLORS = {"fock_sup": "#0072B2", "sqz_vac": "#D55E00", "sqz_coh": "#009E73",
          "coherent": "#5f5f5f"}
MARKERS = {"fock_sup": "o", "sqz_vac": "s", "sqz_coh": "^", "coherent": ""}
LABELS = {"fock_sup": "Fock sup.", "sqz_vac": "Sqz. vac. sup.",
          "sqz_coh": "Sqz. coh. sup.", "coherent": "Coherent"}
TITLES = {"before": "A    Loss before back-action",
          "during": "B    Loss during back-action",
          "after": "C    Loss after back-action"}
SURFACE = "#fcfcfb"


def main():
    df = pd.read_csv(RESULTS_DIR / "step2_sweep.csv")

    from scipy.optimize import brentq
    f_k1 = brentq(lambda f: qs.rp_kappa(2 * np.pi * f) - 1.0, 5, 500)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, placement in zip(axes, ["before", "during", "after"]):
        sub_p = df[df.placement == placement]
        for st in ["fock_sup", "sqz_vac", "sqz_coh", "coherent"]:
            sub = sub_p[sub_p.state == st].sort_values("freq_hz")
            if sub.empty:
                continue
            ref = st == "coherent"
            ax.plot(sub.freq_hz, sub.qfi, color=COLORS[st], lw=2,
                    ls="--" if ref else "-",
                    marker=MARKERS[st], ms=0 if ref else 5,
                    markerfacecolor=SURFACE, markeredgewidth=1.5,
                    label=LABELS[st], zorder=2 if ref else 3)
            ax.axhline(sub.qfi_noba.iloc[0], color=COLORS[st], lw=1, ls=":",
                       alpha=0.55, zorder=1)
        ax.axvline(f_k1, color="#999999", lw=1, alpha=0.6, zorder=0)
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.grid(True, which="both", color="#e7e7e4", lw=0.7, zorder=0)
        ax.tick_params(colors="#444444")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.set_xlabel("GW frequency  $\\Omega/2\\pi$  [Hz]")
        ax.set_title(TITLES[placement], fontsize=10.5, loc="left",
                     color="#333333")

    axes[0].set_ylabel(r"QFI for $\epsilon_p$")
    axes[0].text(f_k1 * 1.06, axes[0].get_ylim()[0] + 0.4, r"$\kappa_{BA}=1$",
                 fontsize=8, color="#666666", rotation=90, va="bottom")
    axes[1].legend(fontsize=8.5, frameon=False, loc="center right",
                   title="dotted = no back-action", title_fontsize=8)

    fig.suptitle(r"Radiation-pressure back-action, KLMTV $\kappa_{BA}(\Omega)$"
                 r" (aLIGO O4);  optimized states from the paper: "
                 r"$\eta=0.9$, pn$=0$, $\bar N\leq 5$",
                 fontsize=10, color="#555555", y=1.0)

    out = RESULTS_DIR / "step2_qfi_vs_frequency.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
