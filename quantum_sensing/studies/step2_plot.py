"""
Step 2 (figure) — back-action vs the paper's optimized states across the
GW band.

Panel A: QFI for epsilon_p vs frequency for the paper's optimized states
         (states.h5, loss_ch eta=0.9, pn=0, N<=5) and a coherent benchmark,
         with the no-back-action baselines dotted.
Panel B: the coherent benchmark in detail — QFI vs homodyne CFI at the
         fixed signal quadrature X and homodyne optimized over LO angle
         (variational readout), showing that the low-frequency QFI gain
         from ponderomotive squeezing requires a rotated readout.

Reads  results/step2_sweep.csv and results/step2b_homodyne.csv.
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
SURFACE = "#fcfcfb"
GRID = "#e7e7e4"
INK = "#333333"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.grid(True, which="both", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors="#444444")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("GW frequency  $\\Omega/2\\pi$  [Hz]")


def main():
    qfi = pd.read_csv(RESULTS_DIR / "step2_sweep.csv")
    hom = pd.read_csv(RESULTS_DIR / "step2b_homodyne.csv")

    from scipy.optimize import brentq
    f_k1 = brentq(lambda f: qs.rp_kappa(2 * np.pi * f) - 1.0, 5, 500)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    # --- Panel A: QFI of every state ---------------------------------------
    for st in ["fock_sup", "sqz_vac", "sqz_coh", "coherent"]:
        sub = qfi[qfi.state == st].sort_values("freq_hz")
        ref = st == "coherent"
        axA.plot(sub.freq_hz, sub.qfi, color=COLORS[st], lw=2,
                 ls="--" if ref else "-",
                 marker=MARKERS[st], ms=0 if ref else 5.5,
                 markerfacecolor=SURFACE, markeredgewidth=1.6,
                 label=LABELS[st], zorder=2 if ref else 3)
        axA.axhline(sub.qfi_noba.iloc[0], color=COLORS[st], lw=1, ls=":",
                    alpha=0.55, zorder=1)
    axA.axvline(f_k1, color="#999999", lw=1, alpha=0.6, zorder=0)
    axA.text(f_k1 * 1.05, 5.4, r"$\kappa_{BA}=1$", fontsize=8.5,
             color="#666666", rotation=90, va="bottom")
    style(axA)
    axA.set_ylabel(r"QFI for $\epsilon_p$")
    axA.set_title("A    QFI: optimized states hold up under back-action",
                  fontsize=10.5, loc="left", color=INK)
    axA.legend(fontsize=8.5, frameon=False, loc="center right",
               title="dotted = no back-action", title_fontsize=8)

    # --- Panel B: coherent benchmark, measurement matters ------------------
    cq = qfi[qfi.state == "coherent"].sort_values("freq_hz")
    ch = hom[hom.state == "coherent"].sort_values("freq_hz")
    axB.plot(cq.freq_hz, cq.qfi, color="#0072B2", lw=2, marker="o", ms=5.5,
             markerfacecolor=SURFACE, markeredgewidth=1.6,
             label="QFI (optimal measurement)", zorder=3)
    axB.plot(ch.freq_hz, ch.cfi_homodyne_opt, color="#D55E00", lw=2,
             marker="s", ms=5.5, markerfacecolor=SURFACE, markeredgewidth=1.6,
             label="Homodyne, best LO angle (variational)", zorder=3)
    axB.plot(ch.freq_hz, ch.cfi_homodyne_x, color="#5f5f5f", lw=2, ls="--",
             label="Homodyne, fixed $X$ readout", zorder=2)
    axB.axvline(f_k1, color="#999999", lw=1, alpha=0.6, zorder=0)
    style(axB)
    axB.set_ylabel(r"Fisher information for $\epsilon_p$")
    axB.set_title("B    Coherent probe: the gain needs a rotated readout",
                  fontsize=10.5, loc="left", color=INK)
    axB.legend(fontsize=8.5, frameon=False, loc="upper right")

    fig.suptitle(r"Radiation-pressure back-action, KLMTV $\kappa_{BA}(\Omega)$"
                 r" (aLIGO O4);  channel: loss_ch $\eta=0.9$, pn$=0$, $\bar N\leq 5$",
                 fontsize=10, color="#555555", x=0.5, y=1.0)

    out = RESULTS_DIR / "step2_qfi_vs_frequency.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
