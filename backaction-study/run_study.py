"""
BA1/BA2/BA3 backaction-ordering QFI study.

Self-contained runner for the first-milestone experiment grid:

  E1  orderings          BA1 (ba->sig), BA2 (sig->ba), BA3 (simultaneous),
                         lossless, GW-like configuration (theta_sig = 0)
  E1b off-axis contrast  signal generator p (theta_sig = pi/2), shear BA1
  E2  loss placement     injection (loss->BA->sig) vs detection (BA->sig->loss)
  E3  phase-noise chain  pn -> BA1/BA2 -> loss
  E4  <n> scaling        QFI vs input mean photon number per configuration

for five state families (coherent, squeezed vacuum, Fock, cat, squeezed cat)
at fixed INPUT mean photon number, and both backaction forms (linearized
ponderomotive shear exp(-i chi/2 X^2) and Kerr exp(-i chi n^2)).

The squeezed families use the sensing-optimal orientation (anti-squeezed
along the signal generator X_0, squeeze_angle = pi), so their lossless
baseline is QFI = 2 e^{+2r}.

Every QFI value is computed with an automated Fock-cutoff convergence check
(N_basis grown geometrically until successive values agree to CONV_RTOL);
the starting cutoff is shear-aware, since the ponderomotive shear is a
squeezer that grows photon number by ~cosh(2r), r = arcsinh(chi/2).

Physics lives in the quantum_sensing package (../quantum-sensing-py); this
script only needs numpy/scipy/qutip/pandas/matplotlib (requirements.txt).

Run from anywhere:  python run_study.py
Outputs: results/qfi_results.csv, results/fig_*.png, results/study_log.txt
(when run via the shell redirect), and summary tables on stdout.
"""

import sys
import time
from pathlib import Path

try:
    import quantum_sensing  # noqa: F401  (installed)
except ImportError:  # fresh clone: use the sibling package directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "quantum-sensing-py"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_sensing import calculate_qfi_converged

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

FAMILIES = ["coherent", "squeezed", "fock", "cat", "sqz_cat"]
LABELS = {
    "coherent": "Coherent",
    "squeezed": "Squeezed vac.",
    "fock": "Fock",
    "cat": "Cat",
    "sqz_cat": "Squeezed cat",
}
# Anti-squeeze the signal-generator quadrature (sensing-optimal orientation).
SQUEEZE_ANGLE = {"squeezed": np.pi, "sqz_cat": np.pi}
# Fixed family -> categorical slot assignment (validated palette order).
COLORS = {
    "coherent": "#2a78d6",
    "squeezed": "#eb6834",
    "fock": "#1baf7a",
    "cat": "#eda100",
    "sqz_cat": "#e87ba4",
}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

CHAINS = {"BA1": ("ba", "sig"), "BA2": ("sig", "ba"), "BA3": ("basig",)}
CHI_GRID = {"shear": [0.0, 0.5, 1.0, 2.0], "kerr": [0.0, 0.05, 0.1, 0.2]}
NBAR = 4.0
CONV_RTOL = 5e-4


def n_start_for(nbar, ba_type, chi):
    """Starting Fock cutoff; shear-aware (the shear grows <n> by ~cosh 2r)."""
    n_eff = nbar
    if ba_type == "shear" and chi > 0.0:
        r = np.arcsinh(chi / 2.0)
        n_eff = nbar * np.cosh(2 * r) + np.sinh(r) ** 2
    return int(min(max(70, round(12 * n_eff)), 600))


def qfi(family, nbar, ba_type, chi, **kw):
    q, N_used, _ = calculate_qfi_converged(
        param_value=0.0, param_type="s",
        N_start=n_start_for(nbar, ba_type, chi), N_max=1000, growth=1.4,
        rtol=CONV_RTOL, state=family, nbar=nbar, ba_type=ba_type, chi_ba=chi,
        squeeze_angle=SQUEEZE_ANGLE.get(family, 0.0), **kw)
    return q, N_used


def run_experiments():
    rows = []
    t0 = time.time()

    def record(exp, family, ba_type, chain_label, chain, chi, nbar,
               theta_sig=0.0, eta_in=1.0, eta_out=1.0, pn=0.0):
        q, N_used = qfi(family, nbar, ba_type, chi, chain=chain,
                        theta_sig=theta_sig, eta_in=eta_in, eta_out=eta_out,
                        pn=pn)
        rows.append(dict(exp=exp, family=family, ba_type=ba_type,
                         chain=chain_label, chi=chi, nbar=nbar,
                         theta_sig=theta_sig, eta_in=eta_in, eta_out=eta_out,
                         pn=pn, qfi=q, N_used=N_used))

    # E1: orderings, lossless, GW-like axis
    for family in FAMILIES:
        for ba_type, chis in CHI_GRID.items():
            for chain_label, chain in CHAINS.items():
                for chi in chis:
                    record("orderings", family, ba_type, chain_label, chain,
                           chi, NBAR)
        print(f"[{time.time()-t0:6.1f}s] E1 done: {family}", flush=True)

    # E1b: off-axis signal (generator p), shear BA1
    for family in FAMILIES:
        for chi in CHI_GRID["shear"]:
            record("offaxis", family, "shear", "BA1", CHAINS["BA1"], chi, NBAR,
                   theta_sig=np.pi / 2)
    print(f"[{time.time()-t0:6.1f}s] E1b done", flush=True)

    # E2: loss placement at eta = 0.8
    placements = {"injection": (("loss_in", "ba", "sig"), 0.8, 1.0),
                  "detection": (("ba", "sig", "loss_out"), 1.0, 0.8)}
    for family in FAMILIES:
        for ba_type, chis in CHI_GRID.items():
            for place, (chain, eta_in, eta_out) in placements.items():
                for chi in chis:
                    record("loss_placement", family, ba_type, place, chain,
                           chi, NBAR, eta_in=eta_in, eta_out=eta_out)
        print(f"[{time.time()-t0:6.1f}s] E2 done: {family}", flush=True)

    # E3: pn -> BA1/BA2 -> loss (eta_out = 0.9, chi fixed per ba_type)
    chi_fix = {"shear": 2.0, "kerr": 0.1}
    pn_chains = {"BA1": ("pn", "ba", "sig", "loss_out"),
                 "BA2": ("pn", "sig", "ba", "loss_out")}
    for family in FAMILIES:
        for ba_type in ("shear", "kerr"):
            for chain_label, chain in pn_chains.items():
                for pn in (0.0, 0.1, 0.3):
                    record("pn_chain", family, ba_type, chain_label, chain,
                           chi_fix[ba_type], NBAR, eta_out=0.9, pn=pn)
        print(f"[{time.time()-t0:6.1f}s] E3 done: {family}", flush=True)

    # E4: <n> scaling for three configurations
    scaling_cfgs = {
        "lossless": dict(chain=("sig",), ba_type="shear", chi=0.0,
                         eta_out=1.0),
        "kerr_ba1": dict(chain=CHAINS["BA1"], ba_type="kerr", chi=0.05,
                         eta_out=1.0),
        "shear_detloss": dict(chain=("ba", "sig", "loss_out"), ba_type="shear",
                              chi=2.0, eta_out=0.8),
    }
    for family in FAMILIES:
        for cfg_name, cfg in scaling_cfgs.items():
            for nbar in (1.0, 2.0, 4.0, 8.0, 16.0):
                record("scaling", family, cfg["ba_type"], cfg_name,
                       cfg["chain"], cfg["chi"], nbar, eta_out=cfg["eta_out"])
        print(f"[{time.time()-t0:6.1f}s] E4 done: {family}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "qfi_results.csv", index=False)
    print(f"[{time.time()-t0:6.1f}s] wrote {RESULTS/'qfi_results.csv'} "
          f"({len(df)} rows); max N_basis used: {df.N_used.max()}",
          flush=True)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


LINESTYLES = {"BA1": "-", "BA2": "--", "BA3": ":",
              "injection": "--", "detection": "-"}
MARKERS = {"BA1": "o", "BA2": "s", "BA3": "^",
           "injection": "s", "detection": "o"}


def _panel_grid(df, exp, x, row_key, row_vals, line_key, fname, xlabel,
                ylabel, suptitle, norm_to_x0=False):
    ncols = len(FAMILIES)
    nrows = len(row_vals)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, rv in enumerate(row_vals):
        for j, family in enumerate(FAMILIES):
            ax = axes[i][j]
            _style_ax(ax)
            sub = df[(df.exp == exp) & (df[row_key] == rv)
                     & (df.family == family)]
            for lv in sub[line_key].unique():
                ss = sub[sub[line_key] == lv].sort_values(x)
                y = ss.qfi.to_numpy()
                if norm_to_x0:
                    y = y / y[0]
                ax.plot(ss[x], y, LINESTYLES.get(lv, "-"),
                        marker=MARKERS.get(lv, "o"), markersize=5,
                        linewidth=2, color=COLORS[family], label=str(lv),
                        markerfacecolor=SURFACE, markeredgewidth=1.6,
                        markeredgecolor=COLORS[family])
            if i == 0:
                ax.set_title(LABELS[family], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{rv}\n{ylabel}", fontsize=9, color=INK2)
            if i == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=9, color=INK2)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle(suptitle, fontsize=12, color=INK, y=1.00)
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=160, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/fname}", flush=True)


def make_figures(df):
    _panel_grid(df, "orderings", "chi", "ba_type", ["shear", "kerr"], "chain",
                "fig_orderings.png", "backaction strength χ", "QFI",
                "Lossless BA1/BA2/BA3 orderings, phase-quadrature signal, ⟨n⟩ = 4\n"
                "(shear row: flat = the [X, X²] = 0 invariance; Kerr row: BA1/BA3 split from BA2)")

    # Off-axis: one panel, all families
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)
    for family in FAMILIES:
        sub = df[(df.exp == "offaxis") & (df.family == family)].sort_values("chi")
        ax.plot(sub.chi, sub.qfi, "-", marker="o", markersize=5, linewidth=2,
                color=COLORS[family], label=LABELS[family],
                markerfacecolor=SURFACE, markeredgewidth=1.6,
                markeredgecolor=COLORS[family])
    ax.set_xlabel("shear strength χ", fontsize=9, color=INK2)
    ax.set_ylabel("QFI", fontsize=9, color=INK2)
    ax.set_title("Off-axis signal (generator p): shear BA1 changes QFI\n"
                 "vacuum reference: QFI = 2(1+χ²)", fontsize=10, color=INK)
    chis = np.array(CHI_GRID["shear"])
    ax.plot(chis, 2 * (1 + chis ** 2), lw=1, ls="--", color=MUTED, zorder=0)
    ax.annotate("2(1+χ²)", xy=(chis[-1], 2 * (1 + chis[-1] ** 2)),
                fontsize=8, color=MUTED, xytext=(-30, 4),
                textcoords="offset points")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_offaxis.png", dpi=160, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_offaxis.png'}", flush=True)

    _panel_grid(df, "loss_placement", "chi", "ba_type", ["shear", "kerr"],
                "chain", "fig_loss_placement.png", "backaction strength χ",
                "QFI(χ) / QFI(0)",
                "Injection (loss→BA→sig) vs detection (BA→sig→loss) at η = 0.8, ⟨n⟩ = 4\n"
                "normalized to χ = 0 to isolate the backaction penalty",
                norm_to_x0=True)

    _panel_grid(df, "pn_chain", "pn", "ba_type", ["shear", "kerr"], "chain",
                "fig_pn_chain.png", "phase-noise rms (rad)", "QFI",
                "Phase noise → BA1/BA2 → loss (η_out = 0.9; χ = 2.0 shear / 0.1 Kerr), ⟨n⟩ = 4")

    # Scaling: one row of three config panels, lines = families, log-log
    cfg_titles = {"lossless": "Lossless (signal only)",
                  "kerr_ba1": "Kerr BA1 (χ = 0.05)",
                  "shear_detloss": "Shear + detection loss\n(χ = 2, η = 0.8)"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (cfg, title) in zip(axes, cfg_titles.items()):
        _style_ax(ax)
        for family in FAMILIES:
            sub = df[(df.exp == "scaling") & (df.chain == cfg)
                     & (df.family == family)].sort_values("nbar")
            ax.plot(sub.nbar, sub.qfi, "-", marker="o", markersize=5,
                    linewidth=2, color=COLORS[family], label=LABELS[family],
                    markerfacecolor=SURFACE, markeredgewidth=1.6,
                    markeredgecolor=COLORS[family])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("input ⟨n⟩", fontsize=9, color=INK2)
    axes[0].set_ylabel("QFI", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle("QFI vs input mean photon number", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_scaling.png", dpi=160, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_scaling.png'}", flush=True)


def print_summary(df):
    pd.set_option("display.width", 120)
    print("\n=== E1 orderings (lossless, theta_sig=0, nbar=4): QFI ===")
    for ba_type in ("shear", "kerr"):
        sub = df[(df.exp == "orderings") & (df.ba_type == ba_type)]
        piv = sub.pivot_table(index=["family", "chain"], columns="chi",
                              values="qfi").round(4)
        print(f"\n-- {ba_type} --\n{piv}")
    print("\n=== E2 loss placement (eta=0.8): QFI(chi)/QFI(0) ===")
    for ba_type in ("shear", "kerr"):
        sub = df[(df.exp == "loss_placement") & (df.ba_type == ba_type)].copy()
        base = sub[sub.chi == 0].set_index(["family", "chain"]).qfi
        sub["rel"] = sub.apply(
            lambda r: r.qfi / base.loc[(r.family, r.chain)], axis=1)
        piv = sub.pivot_table(index=["family", "chain"], columns="chi",
                              values="rel").round(4)
        print(f"\n-- {ba_type} --\n{piv}")
    print("\n=== E3 pn chain: QFI ===")
    piv = df[df.exp == "pn_chain"].pivot_table(
        index=["ba_type", "family", "chain"], columns="pn",
        values="qfi").round(3)
    print(piv)
    print("\n=== E4 scaling: QFI ===")
    piv = df[df.exp == "scaling"].pivot_table(
        index=["chain", "family"], columns="nbar", values="qfi").round(3)
    print(piv)


if __name__ == "__main__":
    df = run_experiments()
    make_figures(df)
    print_summary(df)
    print("\nDone.")
