"""
Figures from results/results.csv — LIGO-noise-budget style:
log-log, frequency on x, minimum detectable signal eps_min = 1/sqrt(QFI)
on y (lower = better), published references drawn on the same axes.

Run after sweep.py:  python qfi-sweep/figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from channel import kimble_K

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Okabe-Ito (colorblind-safe), fixed assignment per state — never cycled
COLORS = {"coherent": "#0072B2", "sqz_vac": "#D55E00", "cat": "#009E73",
          "sqz_cat": "#CC79A7", "fock": "#E69F00"}
LABELS = {"coherent": "coherent", "sqz_vac": "squeezed vacuum",
          "cat": "cat", "sqz_cat": "squeezed cat", "fock": "Fock"}
REF_GRAY = "#666666"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.labelsize": 12, "legend.frameon": False,
})


def _fmt_freq_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency  $\\Omega/2\\pi$  [Hz]")
    ax.set_ylabel("minimum detectable signal  "
                  "$\\epsilon_{\\min} = 1/\\sqrt{F_Q}$")
    return ax


def _k_annotations(ax, y):
    for f in (20, 100, 450):
        K = kimble_K(2 * np.pi * f)
        ax.annotate(f"$\\mathcal{{K}}={K:.2g}$", (f, y), fontsize=8,
                    color="#999999", ha="center")


def fig_a(df):
    """eps_min(f) per state at <n>=2, lossless vs detection loss, with the
    BC2001 conventional-interferometer homodyne curve as the ruler."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    f_grid = np.logspace(np.log10(15), np.log10(1300), 300)
    K_grid = kimble_K(2 * np.pi * f_grid)
    conv = np.sqrt(1 + K_grid**2) / 2      # BC2001 / KLMTV, coherent homodyne

    for ax, (placement, eta, title) in zip(
            axes, [("none", 1.0, "lossless"),
                   ("det", 0.9, "detection loss  $\\eta=0.9$")]):
        sub = df[(df.placement == placement) & (df.eta == eta)
                 & (df.n_target == 2.0) & (df.freq_hz <= 1000)]
        for kind in COLORS:
            s = sub[sub.state == kind].sort_values("freq_hz")
            if s.empty:
                continue
            mk = dict(marker="o", ms=4)
            if kind == "sqz_vac":
                mk = dict(marker="o", ms=7, mfc="none")
            ax.plot(s.freq_hz, s.eps_min, "-", lw=1.8,
                    color=COLORS[kind], label=LABELS[kind], **mk)
        ax.plot(f_grid, conv, "--", color=REF_GRAY, lw=1.4,
                label="conventional IFO, fixed homodyne\n(Buonanno–Chen "
                      "2001 / KLMTV)")
        ax.axhline(0.5, color=REF_GRAY, lw=0.9, ls=":",
                   label="coherent shot-noise limit")
        _fmt_freq_axes(ax)
        ax.set_title(title, fontsize=11)
        _k_annotations(ax, sub.eps_min.max() * 1.15)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Minimum detectable signal vs frequency, "
                 "$\\langle n\\rangle = 2$ probes under radiation-pressure "
                 "back-action", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "figA_eps_min_vs_freq.png", dpi=180)
    print("wrote figA")


def fig_b(df):
    """Loss placement comparison for the squeezed-vacuum probe."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {"none": (":", "lossless"),
              "inj": ("-", "injection loss (before BA)"),
              "conc": ("--", "concurrent loss (during BA)"),
              "det": ("-.", "detection loss (after BA)")}
    shades = {"none": "#999999", "inj": "#0072B2", "conc": "#009E73",
              "det": "#D55E00"}
    sub = df[(df.state == "sqz_vac") & (df.n_target == 2.0)
             & (df.eta.isin([1.0, 0.9])) & (df.freq_hz <= 1000)]
    for placement, (ls, lab) in styles.items():
        s = sub[sub.placement == placement].sort_values("freq_hz")
        if s.empty:
            continue
        ax.plot(s.freq_hz, s.eps_min, ls, color=shades[placement], lw=1.9,
                marker="o", ms=3.5, label=lab)
    _fmt_freq_axes(ax)
    ax.set_title("Where the loss sits matters: squeezed vacuum, "
                 "$\\langle n\\rangle=2$, $\\eta=0.9$", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figB_loss_placement.png", dpi=180)
    print("wrote figB")


def fig_c(df):
    """QFI vs <n> at 100 Hz with the detection-loss ceiling 4/(1-eta)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, eta in zip(axes, (0.9, 0.7)):
        sub = df[(df.placement == "det") & (df.eta == eta)
                 & (df.freq_hz == 100.0)]
        for kind in COLORS:
            s = sub[sub.state == kind].sort_values("n_target")
            if s.empty:
                continue
            mk = dict(marker="o", ms=5)
            if kind == "sqz_vac":
                mk = dict(marker="o", ms=8, mfc="none")
            ax.plot(s.n_actual, s.qfi, "-", lw=1.8,
                    color=COLORS[kind], label=LABELS[kind], **mk)
        ax.axhline(4 / (1 - eta), color=REF_GRAY, ls="--", lw=1.4,
                   label="loss ceiling $4/(1-\\eta)$")
        # lossless squeezed-vacuum reference: F = 4 e^{2r} ~ 16 n
        nn = np.linspace(0.5, 20, 100)
        ax.plot(nn, 4 * (np.sqrt(nn) + np.sqrt(nn + 1))**2, ":",
                color=REF_GRAY, lw=1.1,
                label="lossless sqz. vac. $4e^{2r}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mean photon number  $\\langle n\\rangle$")
        ax.set_title(f"detection loss  $\\eta={eta}$", fontsize=11)
    axes[0].set_ylabel("quantum Fisher information  $F_Q$")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Photon-number scaling at 100 Hz "
                 "($\\mathcal{K}\\approx 0.1$): loss caps every state",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "figC_n_scaling.png", dpi=180)
    print("wrote figC")


def main():
    df = pd.read_csv(RESULTS / "results.csv")
    fig_a(df)
    fig_b(df)
    fig_c(df)


if __name__ == "__main__":
    main()
