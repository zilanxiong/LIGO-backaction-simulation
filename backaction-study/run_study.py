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

The folder is self-contained: the quantum_sensing package it uses is the
local copy sitting next to this script (a snapshot of ../quantum-sensing-py),
and the only external dependencies are in requirements.txt.

Run from anywhere:  python run_study.py
Outputs: results/qfi_results.csv, results/fig_*.png, results/study_log.txt
(when run via the shell redirect), and summary tables on stdout.
"""

import sys
import time
from pathlib import Path

# Use the local quantum_sensing copy even if another one is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_sensing import (calculate_qfi_converged, get_state_backaction,
                             make_state, mean_photon_number,
                             quadrature_covariance)

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
CONV_RTOL = 1e-3  # invisible at plot scale; 5e-4 needs N_basis > 1000 for the
                  # sheared squeezed-cat at nbar = 16


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
            # A parameter-independent QFI autoscales to numerical jitter and
            # renders as fake structure; variation below the convergence
            # tolerance is not resolved, so pin such panels to a real scale.
            lo, hi = ax.get_ylim()
            mid = 0.5 * (lo + hi)
            if hi - lo < 2 * CONV_RTOL * max(abs(mid), 1.0):
                pad = max(0.1 * abs(mid), 0.5)
                ax.set_ylim(mid - pad, mid + pad)
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


# ---------------------------------------------------------------------------
# E5: linearization bridge (Kerr chi_K vs shear chi_eff = 4 chi_K <n>)
# ---------------------------------------------------------------------------

BRIDGE_CHI_EFF = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
BRIDGE_NBARS = [2.0, 8.0, 32.0]
SEQ_BLUE = ["#86b6ef", "#3987e5", "#0d366b"]  # sequential ramp for the nbar sweep


def _bridge_error(family, nbar, N_b, chi_eff):
    """
    Relative error of the linearized (shear) prediction for the largest
    quadrature-covariance eigenvalue (the ponderomotive anti-squeezing
    magnitude, rotation-invariant) against the exact Kerr output, with the
    bridge chi_eff = 4 chi_K <n>.
    """
    psi = make_state(family, nbar, N_b,
                     squeeze_angle=SQUEEZE_ANGLE.get(family, 0.0))
    n_act = mean_photon_number(psi)
    C_in = quadrature_covariance(psi)
    S = np.array([[1.0, 0.0], [-chi_eff, 1.0]])
    lam_lin = np.linalg.eigvalsh(S @ C_in @ S.T)[-1]
    out = get_state_backaction(chi_ba=chi_eff / (4.0 * n_act), ba_type="kerr",
                               chain=("ba",), N_basis=N_b, rho=psi)
    lam_kerr = np.linalg.eigvalsh(quadrature_covariance(out))[-1]
    return lam_lin, lam_kerr, abs(lam_kerr - lam_lin) / lam_lin


def run_bridge():
    rows = []
    for family in FAMILIES:
        N_b = 130
        for chi_eff in BRIDGE_CHI_EFF:
            lam_lin, lam_kerr, err = _bridge_error(family, NBAR, N_b, chi_eff)
            rows.append(dict(panel="families", family=family, nbar=NBAR,
                             chi_eff=chi_eff, lam_lin=lam_lin,
                             lam_kerr=lam_kerr, rel_err=err))
    for nbar in BRIDGE_NBARS:
        N_b = int(max(90, 6 * nbar + 8 * np.sqrt(nbar) + 40))
        for chi_eff in BRIDGE_CHI_EFF:
            lam_lin, lam_kerr, err = _bridge_error("coherent", nbar, N_b,
                                                   chi_eff)
            rows.append(dict(panel="coherent_nbar", family="coherent",
                             nbar=nbar, chi_eff=chi_eff, lam_lin=lam_lin,
                             lam_kerr=lam_kerr, rel_err=err))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "bridge_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    _style_ax(ax)
    for family in FAMILIES:
        sub = df[(df.panel == "families") & (df.family == family)]
        ax.plot(sub.chi_eff, sub.rel_err, "-", marker="o", markersize=5,
                linewidth=2, color=COLORS[family], label=LABELS[family],
                markerfacecolor=SURFACE, markeredgewidth=1.6,
                markeredgecolor=COLORS[family])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("State families, ⟨n⟩ = 4", fontsize=10, color=INK)
    ax.set_xlabel("effective shear strength χ_eff = 4χ_K⟨n⟩",
                  fontsize=9, color=INK2)
    ax.set_ylabel("linearization error\n|λ_Kerr − λ_shear| / λ_shear",
                  fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    ax = axes[1]
    _style_ax(ax)
    for nbar, color in zip(BRIDGE_NBARS, SEQ_BLUE):
        sub = df[(df.panel == "coherent_nbar") & (df.nbar == nbar)]
        ax.plot(sub.chi_eff, sub.rel_err, "-", marker="o", markersize=5,
                linewidth=2, color=color, label=f"⟨n⟩ = {nbar:g}",
                markerfacecolor=SURFACE, markeredgewidth=1.6,
                markeredgecolor=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Coherent carrier vs ⟨n⟩", fontsize=10, color=INK)
    ax.set_xlabel("effective shear strength χ_eff = 4χ_K⟨n⟩",
                  fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    fig.suptitle("Where the linearized (per-frequency shear) description "
                 "breaks: Kerr vs shear at matched strength",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bridge.png", dpi=160, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_bridge.png'}", flush=True)

    print("\n=== E5 linearization bridge: |lam_Kerr - lam_shear|/lam_shear ===")
    piv = df[df.panel == "families"].pivot_table(
        index="family", columns="chi_eff", values="rel_err").round(4)
    print(piv)
    piv = df[df.panel == "coherent_nbar"].pivot_table(
        index="nbar", columns="chi_eff", values="rel_err").round(5)
    print(piv)
    return df


def run_bridge_visuals(chi_wigner=0.8):
    """
    Intuitive companions to fig_bridge: (1) what each model actually predicts
    (anti-squeezing vs coupling, exact Kerr vs linearized shear overlaid);
    (2) Wigner functions from both models at one coupling.
    """
    import qutip as qt

    N_b = 130
    chis = np.linspace(0.0, 1.6, 9)

    def outputs(family, chi_eff):
        psi = make_state(family, NBAR, N_b,
                         squeeze_angle=SQUEEZE_ANGLE.get(family, 0.0))
        n_act = mean_photon_number(psi)
        kerr = get_state_backaction(chi_ba=chi_eff / (4.0 * n_act),
                                    ba_type="kerr", chain=("ba",),
                                    N_basis=N_b, rho=psi)
        shear = get_state_backaction(chi_ba=chi_eff, ba_type="shear",
                                     chain=("ba",), N_basis=N_b, rho=psi)
        return psi, kerr, shear

    # --- (1) prediction-vs-truth curves ---
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(3.0 * len(FAMILIES), 3.0),
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, family in zip(axes, FAMILIES):
        _style_ax(ax)
        lam_in = None
        db_k, db_s = [], []
        for chi_eff in chis:
            psi, kerr, shear = outputs(family, chi_eff)
            if lam_in is None:
                lam_in = np.linalg.eigvalsh(quadrature_covariance(psi))[-1]
            db_k.append(10 * np.log10(
                np.linalg.eigvalsh(quadrature_covariance(kerr))[-1] / lam_in))
            db_s.append(10 * np.log10(
                np.linalg.eigvalsh(quadrature_covariance(shear))[-1] / lam_in))
        ax.plot(chis, db_s, "--", linewidth=2, color=MUTED,
                label="linearized (shear)")
        ax.plot(chis, db_k, "-", marker="o", markersize=4.5, linewidth=2,
                color=COLORS[family], label="exact (Kerr)",
                markerfacecolor=SURFACE, markeredgewidth=1.5,
                markeredgecolor=COLORS[family])
        ax.set_title(LABELS[family], fontsize=10, color=INK)
        ax.set_xlabel("χ_eff", fontsize=9, color=INK2)
    axes[0].set_ylabel("backaction-induced\nanti-squeezing (dB)",
                       fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle("What each model predicts: ponderomotive anti-squeezing above "
                 "the input state, exact Kerr vs linearized shear (⟨n⟩ = 4)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bridge_prediction.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_bridge_prediction.png'}", flush=True)

    # --- (2) Wigner gallery at one coupling ---
    xv = np.linspace(-5.5, 5.5, 181)
    fig, axes = plt.subplots(2, len(FAMILIES),
                             figsize=(2.6 * len(FAMILIES), 5.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, family in enumerate(FAMILIES):
        _, kerr, shear = outputs(family, chi_wigner)
        W_s = qt.wigner(shear, xv, xv)
        W_k = qt.wigner(kerr, xv, xv)
        vmax = max(np.abs(W_s).max(), np.abs(W_k).max())
        for i, W in enumerate([W_s, W_k]):
            ax = axes[i][j]
            ax.pcolormesh(xv, xv, W, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GRID)
        axes[0][j].set_title(LABELS[family], fontsize=10, color=INK)
    axes[0][0].set_ylabel("linearized\n(shear)", fontsize=10, color=INK2)
    axes[1][0].set_ylabel("exact\n(Kerr)", fontsize=10, color=INK2)
    fig.suptitle(f"Phase space (Wigner) after backaction at χ_eff = "
                 f"{chi_wigner}, ⟨n⟩ = 4 — same coupling, two models\n"
                 "(blue/red = positive/negative quasi-probability; "
                 "x horizontal, p vertical)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bridge_wigner.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_bridge_wigner.png'}", flush=True)


def run_bridge_frequency():
    """
    Frequency-domain versions of the bridge figures: the coupling axis is
    mapped through the Kimble factor, chi_eff(Omega) = K(Omega) =
    2 (I0/I_SQL) gamma^4 / (Omega^2 (gamma^2 + Omega^2)), so the
    linearization breakdown becomes a function of sideband frequency
    (strong backaction at low Omega, none at high Omega). Interpretation:
    a frequency-multiplexed preparation placing an identical copy of the
    state in each sideband mode.
    """
    import qutip as qt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    from quantum_sensing import kimble_K

    N_b = 130
    omegas = np.logspace(np.log10(0.5), np.log10(5.0), 13)  # Omega / gamma

    def _omega_axis(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0, 5.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Ω / γ", fontsize=9, color=INK2)

    def one(family, chi_eff):
        psi = make_state(family, NBAR, N_b,
                         squeeze_angle=SQUEEZE_ANGLE.get(family, 0.0))
        n_act = mean_photon_number(psi)
        C_in = quadrature_covariance(psi)
        S = np.array([[1.0, 0.0], [-chi_eff, 1.0]])
        lam_in = np.linalg.eigvalsh(C_in)[-1]
        lam_lin = np.linalg.eigvalsh(S @ C_in @ S.T)[-1]
        kerr = get_state_backaction(chi_ba=chi_eff / (4.0 * n_act),
                                    ba_type="kerr", chain=("ba",),
                                    N_basis=N_b, rho=psi)
        lam_kerr = np.linalg.eigvalsh(quadrature_covariance(kerr))[-1]
        return psi, kerr, lam_in, lam_lin, lam_kerr

    # --- (1) prediction-vs-truth curves over frequency ---
    fig, axes = plt.subplots(1, len(FAMILIES),
                             figsize=(3.0 * len(FAMILIES), 3.1), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, family in zip(axes, FAMILIES):
        _style_ax(ax)
        db_k, db_s = [], []
        for om in omegas:
            chi_eff = float(kimble_K(om))
            _, _, lam_in, lam_lin, lam_kerr = one(family, chi_eff)
            db_s.append(10 * np.log10(lam_lin / lam_in))
            db_k.append(10 * np.log10(lam_kerr / lam_in))
        ax.plot(omegas, db_s, "--", linewidth=2, color=MUTED,
                label="linearized (shear)")
        ax.plot(omegas, db_k, "-", marker="o", markersize=4.5, linewidth=2,
                color=COLORS[family], label="exact (Kerr)",
                markerfacecolor=SURFACE, markeredgewidth=1.5,
                markeredgecolor=COLORS[family])
        _omega_axis(ax)
        ax.axvline(1.0, color=GRID, linewidth=1)
        ax.set_title(LABELS[family], fontsize=10, color=INK)
    axes[0].set_ylabel("backaction-induced\nanti-squeezing (dB)",
                       fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle("What each model predicts vs sideband frequency: "
                 "χ(Ω) = K(Ω), I₀ = I_SQL, ⟨n⟩ = 4 "
                 "(vertical line: Ω = γ, where K = 1)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bridge_freq_prediction.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_bridge_freq_prediction.png'}", flush=True)

    # --- (2) Wigner ladder vs frequency (squeezed vacuum) ---
    xv = np.linspace(-5.5, 5.5, 361)
    om_gallery = [0.6, 1.0, 2.0, 4.0]
    fig, axes = plt.subplots(2, len(om_gallery),
                             figsize=(2.6 * len(om_gallery), 5.4),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    XX, PP = np.meshgrid(xv, xv)
    for j, om in enumerate(om_gallery):
        chi_eff = float(kimble_K(om))
        psi, kerr, *_ = one("squeezed", chi_eff)
        # The linearized model's output is Gaussian by definition: render its
        # Wigner analytically from Sigma = S C_in S^T (a Fock-basis rendering
        # of the strongly sheared state would need a far larger cutoff).
        S = np.array([[1.0, 0.0], [-chi_eff, 1.0]])
        Sig = S @ quadrature_covariance(psi) @ S.T
        Sinv = np.linalg.inv(Sig)
        W_s = (np.exp(-0.5 * (Sinv[0, 0] * XX**2 + 2 * Sinv[0, 1] * XX * PP
                              + Sinv[1, 1] * PP**2))
               / (2 * np.pi * np.sqrt(np.linalg.det(Sig))))
        W_k = qt.wigner(kerr, xv, xv)
        vmax = max(np.abs(W_s).max(), np.abs(W_k).max())
        for i, W in enumerate([W_s, W_k]):
            ax = axes[i][j]
            ax.pcolormesh(xv, xv, W, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GRID)
        axes[0][j].set_title(f"Ω = {om:g} γ   (K = {chi_eff:.2g})",
                             fontsize=10, color=INK)
    axes[0][0].set_ylabel("linearized\n(shear)", fontsize=10, color=INK2)
    axes[1][0].set_ylabel("exact\n(Kerr)", fontsize=10, color=INK2)
    fig.suptitle("Squeezed vacuum after backaction across the band "
                 "(I₀ = I_SQL, ⟨n⟩ = 4): the two models agree at high "
                 "frequency and diverge below Ω ≈ γ", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bridge_freq_wigner.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_bridge_freq_wigner.png'}", flush=True)

    # --- (3) linearization error vs frequency; power dependence ---
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    _style_ax(ax)
    for family in FAMILIES:
        errs = []
        for om in omegas:
            chi_eff = float(kimble_K(om))
            _, _, _, lam_lin, lam_kerr = one(family, chi_eff)
            errs.append(abs(lam_kerr - lam_lin) / lam_lin)
        ax.plot(omegas, errs, "-", marker="o", markersize=4.5, linewidth=2,
                color=COLORS[family], label=LABELS[family],
                markerfacecolor=SURFACE, markeredgewidth=1.5,
                markeredgecolor=COLORS[family])
    _omega_axis(ax)
    ax.set_yscale("log")
    ax.axhline(0.1, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("10%", xy=(omegas[-1], 0.1), fontsize=8, color=MUTED,
                xytext=(2, 3), textcoords="offset points")
    ax.set_title("State families, I₀ = I_SQL", fontsize=10, color=INK)
    ax.set_ylabel("linearization error", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    ax = axes[1]
    _style_ax(ax)
    for i_ratio, color in zip([0.3, 1.0, 3.0], SEQ_BLUE):
        errs = []
        for om in omegas:
            chi_eff = float(kimble_K(om, I_ratio=i_ratio))
            _, _, _, lam_lin, lam_kerr = one("squeezed", chi_eff)
            errs.append(abs(lam_kerr - lam_lin) / lam_lin)
        ax.plot(omegas, errs, "-", marker="o", markersize=4.5, linewidth=2,
                color=color, label=f"I₀/I_SQL = {i_ratio:g}",
                markerfacecolor=SURFACE, markeredgewidth=1.5,
                markeredgecolor=color)
    _omega_axis(ax)
    ax.set_yscale("log")
    ax.axhline(0.1, color=MUTED, linewidth=1, linestyle="--")
    ax.set_title("Squeezed vacuum vs circulating power", fontsize=10,
                 color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    fig.suptitle("Where the linearized description breaks, as a function of "
                 "sideband frequency (χ(Ω) = K(Ω), ⟨n⟩ = 4)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bridge_freq_error.png", dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'fig_bridge_freq_error.png'}", flush=True)


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
    run_bridge()
    run_bridge_visuals()
    run_bridge_frequency()
    print("\nDone.")
