"""
Strain-referred figures: y = h_min/h_SQL = eps_min * sqrt(2/K(Omega)).

The GW enters the phase quadrature with amplitude sqrt(2 K) h/h_SQL
(KLMTV input-output relation), so channel-referred eps_min converts to
strain via sqrt(2/K). Sanity anchors on the same axes:
  - SQL line at 1
  - conventional IFO fixed homodyne: sqrt((1/K + K)/2)   (BC2001 Fig. 2)
  - coherent QCRB: sqrt(1/(2K))                          (Miao et al. 2017)

Reads results/results.csv (first sweep) and, if present,
results/chains.csv (noise-chain sweep). Writes results/figS*.png.
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


def strain(eps_min, K):
    return eps_min * np.sqrt(2.0 / K)


def _refs(ax, f_grid):
    Kg = kimble_K(2 * np.pi * f_grid)
    ax.plot(f_grid, np.sqrt((1 / Kg + Kg) / 2), "--", color=REF_GRAY,
            lw=1.4, label="conventional IFO, fixed homodyne\n"
                          "(Buonanno–Chen 2001 / KLMTV)")
    ax.plot(f_grid, np.sqrt(1 / (2 * Kg)), ":", color=REF_GRAY, lw=1.2,
            label="coherent QCRB (Miao et al. 2017)")
    ax.axhline(1.0, color="#999999", lw=1.0, label="SQL")


def _axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency  $\\Omega/2\\pi$  [Hz]")
    ax.set_ylabel("$\\sqrt{S_h}\\,/\\,\\sqrt{S_h^{\\rm SQL}}$")


def fig_states(df, scen_col, lossless_key, lossy_key, lossy_title, out,
               left_title="lossless"):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    f_grid = np.logspace(1, 3, 300)
    for ax, key, title in zip(axes, (lossless_key, lossy_key),
                              (left_title, lossy_title)):
        sub = df[(df[scen_col] == key) & (df.n_target == 2.0)]
        sub = sub.drop_duplicates(subset=["state", "freq_hz"])
        for kind in COLORS:
            s = sub[sub.state == kind].sort_values("freq_hz")
            if s.empty:
                continue
            mk = dict(marker="o", ms=7, mfc="none") if kind == "sqz_vac" \
                else dict(marker="o", ms=4)
            ax.plot(s.freq_hz, strain(s.eps_min, s.K), "-", lw=1.8,
                    color=COLORS[kind], label=LABELS[kind], **mk)
        _refs(ax, f_grid)
        _axes(ax)
        ax.set_title(title, fontsize=11)
    axes[0].legend(fontsize=8, loc="upper center")
    fig.suptitle("Strain-referred quantum limits, "
                 "$\\langle n\\rangle = 2$ probes", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=180)
    print("wrote", out)


def fig_band(df, scen_col, keys, out, state="sqz_vac",
             title="Loss placement band", band_label="sequential bounds"):
    """Injection/detection band with the concurrent curve inside."""
    pre_k, conc_k, post_k, lossless_k = keys
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    f_grid = np.logspace(1, 3, 300)
    sub = df[(df.state == state) & (df.n_target == 2.0)]

    common = None
    for k in keys:
        fs = set(sub[sub[scen_col] == k].freq_hz)
        common = fs if common is None else (common & fs)

    def curve(key):
        s = (sub[(sub[scen_col] == key) & (sub.freq_hz.isin(common))]
             .drop_duplicates(subset="freq_hz").sort_values("freq_hz"))
        return s.freq_hz.to_numpy(), strain(s.eps_min, s.K).to_numpy()

    f_pre, h_pre = curve(pre_k)
    f_post, h_post = curve(post_k)
    ax.fill_between(f_pre, h_pre, h_post, color="#0072B2", alpha=0.15,
                    label=band_label)
    ax.plot(f_pre, h_pre, "-", color="#0072B2", lw=1.5)
    ax.plot(f_post, h_post, "-.", color="#D55E00", lw=1.5)
    f_c, h_c = curve(conc_k)
    ax.plot(f_c, h_c, "--", color="#009E73", lw=2.0, marker="o", ms=4,
            label="concurrent (physical arm loss)")
    f_l, h_l = curve(lossless_k)
    ax.plot(f_l, h_l, ":", color="#999999", lw=1.5, label="lossless")
    _refs(ax, f_grid)
    _axes(ax)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=180)
    print("wrote", out)


def main():
    df1 = pd.read_csv(RESULTS / "results.csv")
    # first sweep: scenario = placement+eta
    df1["scen"] = df1.placement + "_" + df1.eta.astype(str)
    fig_states(df1, "scen", "none_1.0", "det_0.9",
               "detection loss  $\\eta = 0.9$", "figS1_states_strain.png")
    fig_band(df1, "scen",
             ("inj_0.9", "conc_0.9", "det_0.9", "none_1.0"),
             "figS2_loss_band_strain.png",
             band_label="sequential bounds (loss before ... after BA)",
             title="Squeezed vacuum, $\\langle n\\rangle=2$: where 10% "
                   "loss sits")

    chains = RESULTS / "chains.csv"
    if chains.exists():
        df2 = pd.read_csv(chains)
        fig_states(df2, "scenario", "lossless", "pn_post",
                   "phase noise after BA  $\\chi = 0.1$",
                   "figS3_states_pn_strain.png")
        fig_band(df2, "scenario",
                 ("pn_pre", "pn_conc", "pn_post", "lossless"),
                 "figS4_pn_band_strain.png",
                 band_label="sequential bounds (PN before ... after BA)",
                 title="Squeezed vacuum, $\\langle n\\rangle=2$: where "
                       "phase noise sits")
        fig_states(df2, "scenario", "pn_then_loss", "loss_then_pn",
                   "loss $\\to$ BA $\\to$ PN",
                   "figS5_two_noise_strain.png",
                   left_title="PN $\\to$ BA $\\to$ loss")


if __name__ == "__main__":
    main()
