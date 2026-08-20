#!/usr/bin/env python3
"""
Figures for the radiation-pressure QFI study.

Reads the CSVs written by ``run_study.py`` and renders one figure per section of
the report.  Every figure is produced in a light and a dark variant, and every
figure has the CSV it came from sitting beside it.

Usage::

    python backaction-qfi/make_figures.py [--dir backaction-qfi/results]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import THEMES, finish, label_lines, legend_below, theme  # noqa: E402

LABELS = {
    "coherent": "coherent",
    "squeezed": "squeezed",
    "fock": "Fock",
    "cat_even": "even cat",
    "cat_odd": "odd cat",
    "squeezed_cat": "squeezed cat",
    "opt_no_ba": "opt. (no BA)",
}
ORDER = ["coherent", "squeezed", "fock", "cat_even", "cat_odd", "squeezed_cat", "opt_no_ba"]


def read(path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def series(rows, key_fields, x_field, y_field, where=None):
    """Group rows into {key: (xs, ys)} sorted by x."""
    out = defaultdict(list)
    for r in rows:
        if where and not where(r):
            continue
        key = tuple(r[f] for f in key_fields)
        out[key].append((float(r[x_field]), float(r[y_field])))
    return {k: tuple(np.array(z) for z in zip(*sorted(v))) for k, v in out.items()}


# ---------------------------------------------------------------------------
def fig_orderings(d, out):
    """BA1 / BA2 / BA3: identical for the physical quadrature, separated for the other."""
    rows = read(d / "orderings.csv")
    phys = [r for r in rows if r["signal_quadrature"].startswith("epsilon_a")]
    nonc = [r for r in rows if r["signal_quadrature"].startswith("epsilon_p")]
    orderings = ["none", "BA1", "BA2", "BA3"]

    phys_states = sorted({r["state"] for r in phys}, key=lambda s: ORDER.index(s))
    kappa = sorted({float(r["kappa"]) for r in phys})[-1]
    left = {s: [next(float(r["qfi_epsilon_a"]) for r in phys
                     if r["state"] == s and r["ordering"] == o and float(r["kappa"]) == kappa)
                for o in orderings] for s in phys_states}
    nonc_states = sorted({r["state"] for r in nonc}, key=lambda s: ORDER.index(s))
    right = {s: [next(float(r["qfi_epsilon_a"]) for r in nonc
                      if r["state"] == s and r["ordering"] == o) for o in orderings]
             for s in nonc_states}

    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
            w = 0.8 / max(len(left), 1)
            for i, s in enumerate(phys_states):
                axes[0].bar(np.arange(4) + i * w - 0.4 + w / 2, left[s], width=w * 0.9,
                            color=t["series"][ORDER.index(s)], label=LABELS[s])
            for i, s in enumerate(nonc_states):
                axes[1].bar(np.arange(4) + i * w * len(left) / len(right) - 0.4
                            + w * len(left) / len(right) / 2, right[s],
                            width=w * len(left) / len(right) * 0.9,
                            color=t["series"][ORDER.index(s)], label=LABELS[s])
            axes[0].set_title(rf"signal in $p$ (physical), $\kappa={kappa:g}$" "\nBA1 = BA2 = BA3")
            axes[1].set_title("signal in $x$ (non-commuting)\nBA3 bracketed by BA1, BA2")
            for ax in axes:
                ax.set_xticks(np.arange(4))
                ax.set_xticklabels(["no BA", "BA1", "BA2", "BA3"])
                ax.grid(axis="x", visible=False)
                ax.set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            legend_below(fig, axes[0], ncol=3)
            fig.subplots_adjust(wspace=0.25)
            finish(fig, str(out / "fig1_orderings"), name)


def fig_loss_placement(d, out):
    rows = read(d / "loss_placement.csv")
    data = series(rows, ("state", "loss_placement"), "kappa", "qfi_epsilon_a")
    states = sorted({k[0] for k in data}, key=lambda s: ORDER.index(s))
    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            for s in states:
                c = t["series"][ORDER.index(s)]
                for placement, ls in (("injection", "-"), ("detection", (0, (5, 2)))):
                    xs, ys = data[(s, placement)]
                    ax.plot(xs, ys, color=c, ls=ls,
                            label=f"{LABELS[s]}, {placement}" if placement == "injection" else None)
            ax.set_yscale("log")
            ax.set_xlabel(r"opto-mechanical coupling  $\kappa$")
            ax.set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            ax.set_title("Injection loss is flat in $\\kappa$; detection loss is not\n"
                         "(solid: loss $\\to$ BA;  dashed: BA $\\to$ loss;  $\\eta=0.8$)")
            legend_below(fig, ax, ncol=4)
            finish(fig, str(out / "fig2_loss_placement"), name)


def fig_phase_noise(d, out):
    rows = read(d / "phase_noise.csv")
    data = series(rows, ("state",), "sigma_phi_rad", "qfi_epsilon_a")
    states = sorted({k[0] for k in data}, key=lambda s: ORDER.index(s))
    for name in THEMES:
        with theme(name) as t:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            drawn = []
            for s in states:
                xs, ys = data[(s,)]
                ax.plot(xs, ys, color=t["series"][ORDER.index(s)], label=LABELS[s])
                drawn.append((LABELS[s], xs, ys))
            label_lines(ax, drawn, t)
            ax.set_xlabel(r"input phase noise  $\sigma_\phi$  [rad]")
            ax.set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            ax.set_title("Phase noise $\\to$ back-action $\\to$ detection loss\n"
                         "the Fock state is exactly flat; the cat overtakes squeezing")
            legend_below(fig, ax, ncol=4)
            finish(fig, str(out / "fig3_phase_noise"), name)


def fig_states_vs_kappa(d, out):
    rows = read(d / "states_vs_kappa.csv")
    etas = sorted({float(r["eta_out"]) for r in rows}, reverse=True)
    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, len(etas), figsize=(3.3 * len(etas), 3.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, eta in zip(axes, etas):
                data = series(rows, ("state",), "kappa", "qfi_epsilon_a",
                              where=lambda r, e=eta: float(r["eta_out"]) == e)
                for s in sorted(data, key=lambda k: ORDER.index(k[0])):
                    xs, ys = data[s]
                    ax.plot(xs, ys, color=t["series"][ORDER.index(s[0])], label=LABELS[s[0]])
                ax.set_yscale("log")
                ax.set_xlabel(r"$\kappa$")
                ax.set_title(rf"$\eta_{{\rm out}} = {eta:g}$")
            axes[0].set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            fig.suptitle(r"Probe states at fixed $\bar n = 2$: flat without loss, "
                         r"ranking inverts with it", y=1.0)
            fig.subplots_adjust(wspace=0.1)
            legend_below(fig, axes[0], ncol=7)
            finish(fig, str(out / "fig4_states_vs_kappa"), name)


def fig_nbar_scaling(d, out):
    rows = read(d / "nbar_scaling.csv")
    configs = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])
    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, len(configs), figsize=(3.2 * len(configs), 3.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, cfg in zip(axes, configs):
                data = series(rows, ("state",), "nbar", "qfi_epsilon_a",
                              where=lambda r, c=cfg: r["config"] == c)
                for s in sorted(data, key=lambda k: ORDER.index(k[0])):
                    xs, ys = data[s]
                    ax.loglog(xs, ys, color=t["series"][ORDER.index(s[0])], label=LABELS[s[0]],
                              marker="o", markersize=3)
                ax.set_xlabel(r"$\bar n$")
                ax.set_title(cfg.replace(", ", ",\n"), fontsize=9)
                # A log axis auto-labels decades and minor ticks, which collide
                # badly on a narrow panel; label the actual grid points instead.
                ticks = sorted({float(r["nbar"]) for r in rows})
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8)
                ax.set_xticks([], minor=True)
            axes[0].set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            fig.suptitle(r"Photon-number scaling: linear without loss, saturating or "
                         r"$\it{falling}$ with back-action plus loss", y=1.02)
            fig.subplots_adjust(wspace=0.1)
            legend_below(fig, axes[0], ncol=7)
            finish(fig, str(out / "fig5_nbar_scaling"), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "results"))
    args = ap.parse_args()
    d = Path(args.dir)
    for fn in (fig_orderings, fig_loss_placement, fig_phase_noise,
               fig_states_vs_kappa, fig_nbar_scaling):
        fn(d, d)
        print(f"  {fn.__name__} ok")
    print(f"figures -> {d}/")


if __name__ == "__main__":
    main()
