#!/usr/bin/env python3
"""Run the full radiation-pressure QFI study and write tables and figures.

Usage::

    python scripts/run_study.py [--quick] [--outdir results]

Every section writes a CSV next to its figure so the numbers are inspectable
without re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotstyle import THEMES, finish, label_lines, legend_below, theme  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from ligo_backaction import (  # noqa: E402
    ALIGO,
    ChannelSpec,
    IFOParams,
    converge,
    fd_squeeze_angle,
    homodyne_noise_spectrum,
    make_state,
    optimal_no_backaction_state,
    optimize_state_for_channel,
    optimal_squeeze_angle,
    qfi_h_from_qfi_lambda,
    qfi_lambda,
    suggested_cutoff,
    total_qfi,
    weighted_inverse_cost,
    worst_case_qfi,
)
from ligo_backaction.metrics import power_law_weight  # noqa: E402
from ligo_backaction.operators import squeeze_op  # noqa: E402
from ligo_backaction.twomode import (  # noqa: E402
    TwoModeSpec,
    additivity_defect,
    entangled_cat,
    product_state,
    two_mode_squeezed_vacuum,
)

FAMILIES = ["coherent", "squeezed", "fock", "cat_even", "cat_odd", "squeezed_cat"]
LABELS = {
    "coherent": "coherent",
    "squeezed": "squeezed",
    "fock": "Fock",
    "cat_even": "even cat",
    "cat_odd": "odd cat",
    "squeezed_cat": "squeezed cat",
    "opt_no_ba": "opt. (no BA)",
    "opt_with_ba": "opt. (with BA)",
    "vacuum": "vacuum",
}
MAX_CUTOFF = 600


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def converged_family_qfi(family: str, nbar: float, spec: ChannelSpec, rtol: float = 3e-3):
    """QFI with an automated Fock-cutoff convergence check.

    Returns ``(value, cutoff, converged)``.  The ladder starts from the
    rule-of-thumb cutoff for this ``(nbar, kappa)`` and is capped so a single
    scan point cannot run away.
    """
    n0 = min(max(suggested_cutoff(nbar, spec.kappa), 40), MAX_CUTOFF)
    ladder, n = [], n0
    while n < MAX_CUTOFF:
        ladder.append(n)
        n = int(np.ceil(n * 1.35))
    ladder.append(MAX_CUTOFF)

    def value(N):
        return qfi_lambda(make_state(family, N, nbar), spec)

    res = converge(value, cutoffs=ladder, rtol=rtol)
    return res.value, res.cutoff, res.converged


# --------------------------------------------------------------------------
# 1. Gaussian benchmark curves: SQL, variational readout, FD squeezing
# --------------------------------------------------------------------------
def study_benchmark_curves(outdir: Path):
    print("[1] Gaussian benchmark curves (SQL, variational readout, FD squeezing)")
    f = np.logspace(np.log10(10.0), np.log10(3000.0), 400)
    kappa = ALIGO.kappa(f)
    r = 10 * np.log(10) / 20  # 10 dB of squeezing
    curves = {
        "no squeezing, phase readout": homodyne_noise_spectrum(kappa),
        "10 dB frozen-angle squeezing": homodyne_noise_spectrum(kappa, r=r, squeeze_angle=0.0),
        "10 dB FD squeezing": homodyne_noise_spectrum(kappa, r=r, squeeze_angle=fd_squeeze_angle(kappa)),
        "QCRB, vacuum (= variational)": homodyne_noise_spectrum(kappa, zeta=np.arctan(kappa)),
    }
    write_csv(
        outdir / "benchmark_curves.csv",
        ["f_Hz", "kappa"] + list(curves),
        [[f[i], kappa[i]] + [c[i] for c in curves.values()] for i in range(len(f))],
    )
    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            drawn = []
            for i, (label, sp) in enumerate(curves.items()):
                ax.loglog(f, np.sqrt(sp), color=t["series"][i], label=label)
                drawn.append((label, f, np.sqrt(sp)))
            label_lines(ax, drawn, t, fracs=[0.06, 0.93, 0.30, 0.62])
            ax.axhline(1.0, color=t["muted"], lw=1.0, ls=(0, (4, 3)))
            ax.annotate("SQL", (11, 1.06), color=t["muted"], fontsize=8)
            ax.set_xlabel("sideband frequency  f  [Hz]")
            ax.set_ylabel(r"strain noise  $\sqrt{S_h}\,/\,h_{\rm SQL}$")
            ax.set_title("Gaussian benchmarks reproduced by the covariance track")
            ax.set_xlim(10, 3000)
            legend_below(fig, ax, ncol=2)
            finish(fig, str(outdir / "fig1_benchmark_curves"), name)


# --------------------------------------------------------------------------
# 2. Probe states versus kappa, at fixed photon number, for several losses
# --------------------------------------------------------------------------
def study_states_vs_kappa(outdir: Path, nbar: float, kappas, etas):
    print("[2] QFI of each probe family versus kappa, at fixed nbar")
    rows, data = [], {}
    for eta in etas:
        for family in FAMILIES:
            vals, flags = [], []
            for k in kappas:
                spec = ChannelSpec(kappa=k, ordering="BA1", eta_out=eta)
                v, N, ok = converged_family_qfi(family, nbar, spec)
                vals.append(v)
                flags.append(ok)
                rows.append([eta, family, k, v, N, ok])
            data[(eta, family)] = np.array(vals)
            if not all(flags):
                print(f"    ! {family} eta={eta}: unconverged at kappa="
                      f"{[k for k, ok in zip(kappas, flags) if not ok]}")
    write_csv(outdir / "states_vs_kappa.csv", ["eta_out", "state", "kappa", "qfi_lambda", "cutoff", "converged"], rows)

    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, len(etas), figsize=(3.6 * len(etas), 3.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, eta in zip(axes, etas):
                for i, family in enumerate(FAMILIES):
                    ax.plot(kappas, data[(eta, family)], color=t["series"][i], label=LABELS[family])
                ax.set_xlabel(r"opto-mechanical coupling  $\kappa$")
                ax.set_title(rf"detection efficiency $\eta = {eta:g}$")
                ax.set_yscale("log")
            axes[0].set_ylabel(r"$\mathcal{F}_\lambda$")
            fig.suptitle(rf"Probe states at fixed $\bar n = {nbar:g}$ under radiation pressure", y=1.0)
            fig.subplots_adjust(wspace=0.12)
            legend_below(fig, axes[0], ncol=6)
            finish(fig, str(outdir / "fig2_states_vs_kappa"), name)
    return data


# --------------------------------------------------------------------------
# 3. Where the loss sits: injection versus detection
# --------------------------------------------------------------------------
def study_loss_placement(outdir: Path, nbar: float, kappas, eta: float):
    print("[3] Injection loss versus detection loss")
    rows, data = [], {}
    for family in ["squeezed", "cat_even"]:
        for placement, kw in (("injection", dict(eta_in=eta)), ("detection", dict(eta_out=eta))):
            vals = []
            for k in kappas:
                spec = ChannelSpec(kappa=k, ordering="BA1", **kw)
                v, N, ok = converged_family_qfi(family, nbar, spec)
                vals.append(v)
                rows.append([family, placement, k, v, N, ok])
            data[(family, placement)] = np.array(vals)
    write_csv(outdir / "loss_placement.csv", ["state", "loss_placement", "kappa", "qfi_lambda", "cutoff", "converged"], rows)

    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            drawn = []
            for i, key in enumerate(data):
                label = f"{LABELS[key[0]]}, {key[1]}"
                ax.plot(kappas, data[key], color=t["series"][i], label=label,
                        ls="-" if key[1] == "injection" else (0, (5, 2)))
                drawn.append((label, kappas, data[key]))
            label_lines(ax, drawn, t)
            ax.set_xlabel(r"opto-mechanical coupling  $\kappa$")
            ax.set_ylabel(r"$\mathcal{F}_\lambda$")
            ax.set_yscale("log")
            ax.set_title(rf"Injection loss vs detection loss, $\eta = {eta:g}$, $\bar n = {nbar:g}$")
            legend_below(fig, ax, ncol=2)
            finish(fig, str(outdir / "fig3_loss_placement"), name)
    return data


# --------------------------------------------------------------------------
# 4. Optimal phase-space extent: how much anti-squeezing does BA + loss want?
# --------------------------------------------------------------------------
def study_optimal_extent(outdir: Path, nbar: float, kappas, etas):
    print("[4] Optimal input squeeze angle at fixed photon number (Gaussian track)")
    rows, data = [], {}
    for eta in etas:
        thetas, extents, qfis = [], [], []
        for k in kappas:
            spec = ChannelSpec(kappa=float(k), ordering="BA1", eta_out=eta)
            th, q, var_x = optimal_squeeze_angle(nbar, spec)
            thetas.append(np.degrees(th))
            extents.append(10 * np.log10(2 * var_x))
            qfis.append(q)
            rows.append([eta, float(k), np.degrees(th), 10 * np.log10(2 * var_x), q])
        data[eta] = (np.array(thetas), np.array(extents), np.array(qfis))
    write_csv(outdir / "optimal_extent.csv",
              ["eta_out", "kappa", "theta_opt_deg", "x_extent_dB", "qfi_lambda"], rows)

    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
            for i, eta in enumerate(etas):
                axes[0].plot(kappas, data[eta][0], color=t["series"][i], label=rf"$\eta = {eta:g}$")
                axes[1].plot(kappas, data[eta][1], color=t["series"][i], label=rf"$\eta = {eta:g}$")
            axes[0].set_ylabel("optimal input squeeze angle  [deg]")
            axes[1].set_ylabel(r"phase-space extent along $x$  [dB re vacuum]")
            axes[1].axhline(0.0, color=t["muted"], lw=1.0, ls=(0, (4, 3)))
            for ax in axes:
                ax.set_xlabel(r"opto-mechanical coupling  $\kappa$")
            fig.suptitle(
                rf"At fixed $\bar n = {nbar:g}$, radiation pressure plus loss caps the useful extent along $x$",
                y=1.0,
            )
            fig.subplots_adjust(wspace=0.28)
            legend_below(fig, axes[0], ncol=5)
            finish(fig, str(outdir / "fig4_optimal_extent"), name)
    return data


# --------------------------------------------------------------------------
# 5. Broadband strain sensitivity and the three cost metrics
# --------------------------------------------------------------------------
def study_broadband(outdir: Path, params: IFOParams, freqs, nbar: float):
    print("[5] Broadband strain QFI and cost metrics")
    kappa = params.kappa(freqs)
    hs = params.h_sql(freqs)
    eta = 0.9
    families = FAMILIES + ["opt_no_ba"]
    curves, rows, metrics = {}, [], []
    for family in families:
        fh = []
        for f_, k, h in zip(freqs, kappa, hs):
            spec = ChannelSpec(kappa=float(k), ordering="BA1", eta_out=eta)
            if family == "opt_no_ba":
                Nused = min(max(suggested_cutoff(nbar, float(k)), 40), MAX_CUTOFF)
                v, conv = qfi_lambda(optimal_no_backaction_state(Nused, nbar), spec), True
            else:
                v, Nused, conv = converged_family_qfi(family, nbar, spec)
            fh.append(float(qfi_h_from_qfi_lambda(v, k, h)))
            rows.append([family, float(f_), float(k), v, fh[-1],
                         float(1.0 / np.sqrt(fh[-1]) / h), Nused, conv])
            if not conv:
                print(f"    ! {family}: unconverged at f = {f_:.0f} Hz (kappa = {k:.2f})")
        curves[family] = np.array(fh)
    write_csv(outdir / "broadband_qfi.csv",
              ["state", "f_Hz", "kappa", "qfi_lambda", "qfi_h", "sigma_h_over_hSQL", "cutoff", "converged"], rows)

    w = power_law_weight(freqs)
    for family in families:
        metrics.append([
            family,
            weighted_inverse_cost(freqs, curves[family], w),
            worst_case_qfi(curves[family]),
            total_qfi(curves[family]),
        ])
    write_csv(outdir / "broadband_metrics.csv",
              ["state", "cost_i_weighted_inverse", "cost_ii_worst_case_qfi", "cost_iii_total_qfi"], metrics)

    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            for i, family in enumerate(families):
                sens = 1.0 / np.sqrt(curves[family]) / hs
                # The no-back-action optimum is drawn dashed on top because it
                # lands exactly on the squeezed-vacuum curve -- that coincidence
                # is the result, so both have to stay visible.
                ax.loglog(freqs, sens, color=t["series"][i % len(t["series"])], label=LABELS[family],
                          ls=(0, (5, 3)) if family == "opt_no_ba" else "-",
                          lw=2.4 if family == "opt_no_ba" else 2.0)
            ax.axhline(1.0, color=t["muted"], lw=1.0, ls=(0, (4, 3)))
            ax.annotate("$h_{\\rm SQL}$", (freqs[0] * 1.02, 1.06), color=t["muted"], fontsize=8)
            ax.set_xlabel("sideband frequency  f  [Hz]")
            ax.set_ylabel(r"QCRB strain  $\mathcal{F}_h^{-1/2}\,/\,h_{\rm SQL}$")
            ax.set_title(rf"Broadband QCRB, $\bar n = {nbar:g}$, detection $\eta = {eta:g}$")
            legend_below(fig, ax, ncol=4)
            finish(fig, str(outdir / "fig5_broadband"), name)
    return curves, metrics


# --------------------------------------------------------------------------
# 6. Channel orderings and phase noise
# --------------------------------------------------------------------------
def study_orderings_and_noise(outdir: Path, nbar: float):
    print("[6] Channel orderings (BA1/BA2/BA3) and phase noise")
    N = 200
    rows = []
    for family in ["squeezed", "cat_even"]:
        psi = make_state(family, N, nbar)
        for phi, tag in ((0.0, "p-quadrature (physical)"), (np.pi / 2, "x-quadrature")):
            vals = {}
            for ordering in ("none", "BA1", "BA2", "BA3"):
                spec = ChannelSpec(kappa=1.5, lam=0.3, phi_sig=phi, ordering=ordering, eta_out=0.9)
                vals[ordering] = qfi_lambda(psi, spec)
            rows.append([family, tag, vals["none"], vals["BA1"], vals["BA2"], vals["BA3"],
                         max(abs(vals["BA3"] - vals["BA1"]), abs(vals["BA3"] - vals["BA2"]))])
    write_csv(outdir / "orderings.csv",
              ["state", "signal_quadrature", "no_BA", "BA1", "BA2", "BA3", "max_BA3_minus_BA12"], rows)

    sigmas = np.linspace(0.0, 0.5, 11)
    rows, data = [], {}
    for family in ["squeezed", "cat_even", "coherent"]:
        vals = []
        for s in sigmas:
            spec = ChannelSpec(kappa=1.0, ordering="BA1", sigma_in=float(s), eta_out=0.9)
            v, Nused, ok = converged_family_qfi(family, nbar, spec)
            vals.append(v)
            rows.append([family, float(s), v, Nused, ok])
        data[family] = np.array(vals)
    write_csv(outdir / "phase_noise.csv", ["state", "sigma_phase_rad", "qfi_lambda", "cutoff", "converged"], rows)

    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            drawn = []
            for i, family in enumerate(data):
                y = data[family] / data[family][0]
                ax.plot(sigmas, y, color=t["series"][i], label=LABELS[family])
                drawn.append((LABELS[family], sigmas, y))
            label_lines(ax, drawn, t)
            ax.set_xlabel(r"input phase noise  $\sigma_\phi$  [rad]")
            ax.set_ylabel(r"$\mathcal{F}_\lambda(\sigma_\phi)\,/\,\mathcal{F}_\lambda(0)$")
            ax.set_title(rf"Phase noise $\to$ radiation pressure $\to$ loss, $\bar n = {nbar:g}$")
            legend_below(fig, ax, ncol=3)
            finish(fig, str(outdir / "fig6_phase_noise"), name)
    return rows


# --------------------------------------------------------------------------
# 7. State optimisation with back-action
# --------------------------------------------------------------------------
def study_optimization(outdir: Path, nbar: float, kappas, eta: float, N: int = 120, maxiter: int = 60):
    print("[7] Optimising the probe state with back-action present")
    rows = []
    no_ba = optimal_no_backaction_state(N, nbar)
    for k in kappas:
        spec = ChannelSpec(kappa=float(k), ordering="BA1", eta_out=eta)
        # Seed with the Gaussian optimum for *this* channel at the same photon
        # number, so the local search starts from the best known Gaussian probe.
        theta, _, _ = optimal_squeeze_angle(nbar, spec)
        r = np.arcsinh(np.sqrt(nbar))
        vac = np.zeros(N, dtype=complex)
        vac[0] = 1.0
        gauss_seed = squeeze_op(N, -r * np.exp(2j * theta)) @ vac
        gauss_seed = gauss_seed / np.linalg.norm(gauss_seed)

        base = qfi_lambda(no_ba, spec)
        gauss = qfi_lambda(gauss_seed, spec)
        res = optimize_state_for_channel(
            N, nbar, spec, seeds=("cat_even",), extra_seeds=(gauss_seed,),
            n_params=16, maxiter=maxiter, real_only=False,
        )
        rows.append([float(k), base, gauss, res.qfi, res.qfi / base, res.qfi / gauss,
                     res.nbar, float(abs(np.vdot(res.state, gauss_seed))), res.n_iter])
        print(f"    kappa={k:4.2f}  no-BA opt {base:8.4f}  Gaussian opt {gauss:8.4f}  "
              f"free opt {res.qfi:8.4f}  (x{res.qfi / base:.3f} / x{res.qfi / gauss:.3f})")
    write_csv(outdir / "optimization.csv",
              ["kappa", "qfi_no_BA_optimal_state", "qfi_gaussian_optimum", "qfi_free_optimum",
               "gain_over_no_BA_state", "gain_over_gaussian", "nbar_realised",
               "overlap_with_gaussian_optimum", "n_iter"], rows)
    return rows


# --------------------------------------------------------------------------
# 8. Two-mode additivity
# --------------------------------------------------------------------------
def study_twomode(outdir: Path):
    print("[8] Two-mode additivity of the per-frequency description")
    N = 14
    cases = [
        ("product squeezed", product_state(make_state("squeezed", N, 1.0), make_state("squeezed", N, 1.0)), 0.0),
        ("product cat", product_state(make_state("cat_even", N, 1.0), make_state("cat_even", N, 1.0)), 0.0),
        ("product squeezed, cross BA", product_state(make_state("squeezed", N, 1.0), make_state("squeezed", N, 1.0)), 0.8),
        ("product cat, cross BA", product_state(make_state("cat_even", N, 1.0), make_state("cat_even", N, 1.0)), 0.8),
        ("two-mode squeezed", two_mode_squeezed_vacuum(N, 0.6), 0.0),
        ("entangled cat", entangled_cat(N, 1.0), 0.0),
    ]
    rows = []
    for label, psi, kc in cases:
        spec = TwoModeSpec(kappa1=1.0, kappa2=1.0, kappa_cross=kc, eta=0.8)
        d = additivity_defect(psi, spec, N)
        rows.append([label, kc, d["trace_joint"], d["trace_per_frequency"], d["rel_defect"], d["max_offdiag"]])
    write_csv(outdir / "twomode_additivity.csv",
              ["case", "kappa_cross", "joint_sum_qfi", "per_frequency_sum_qfi", "rel_defect", "max_offdiag"], rows)

    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.6, 3.8))
            labels = [r[0] for r in rows]
            vals = [100 * r[4] for r in rows]
            colors = [t["series"][0] if abs(v) < 1e-4 else t["series"][1] for v in vals]
            bars = ax.barh(labels, vals, color=colors, height=0.55)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:+.2f}%", (b.get_width(), b.get_y() + b.get_height() / 2),
                            xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
                            va="center", ha="left" if v >= 0 else "right",
                            color=t["text_secondary"], fontsize=8)
            ax.axvline(0.0, color=t["muted"], lw=1.0)
            ax.set_xlabel("joint QFI relative to the per-frequency sum  [%]")
            ax.set_title("Where the independent-frequency description breaks")
            ax.grid(axis="y", visible=False)
            span = max(max(vals), 1.0)
            ax.set_xlim(-0.12 * span, span * 1.28)
            ax.invert_yaxis()
            finish(fig, str(outdir / "fig7_twomode_additivity"), name)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="coarser grids for a fast smoke run")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nbar = 2.0
    kappas = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]) if not args.quick else np.array([0.0, 1.0, 3.0])
    etas = [1.0, 0.9, 0.7]
    # Circulating power reduced so that kappa stays inside the range the Fock
    # track can resolve across the whole band (kappa ~ 1/Omega^2 diverges at low
    # frequency; see README).
    params = IFOParams(power_ratio=0.25)
    freqs = np.logspace(np.log10(200.0), np.log10(2000.0), 12 if not args.quick else 5)

    t0 = time.time()
    study_benchmark_curves(outdir)
    study_states_vs_kappa(outdir, nbar, kappas, etas)
    study_loss_placement(outdir, nbar, kappas, 0.8)
    study_optimal_extent(outdir, nbar, np.linspace(0.0, 5.0, 51), [1.0, 0.95, 0.9, 0.8, 0.5])
    study_broadband(outdir, params, freqs, nbar)
    study_orderings_and_noise(outdir, nbar)
    study_optimization(outdir, nbar, [0.0, 0.5, 1.5, 3.0] if not args.quick else [1.5], 0.9)
    study_twomode(outdir)
    print(f"\nDone in {time.time() - t0:.1f} s -> {outdir}/")


if __name__ == "__main__":
    main()
