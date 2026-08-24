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
}
ORDER = ["coherent", "squeezed", "fock", "cat_even", "cat_odd", "squeezed_cat"]


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
            axes[0].set_title(rf"signal in $p$ (physical), $\kappa={kappa:g}$, detection loss"
                              "\nBA1 = BA2 = BA3")
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
                for placement, ls in (("injection", "-"), ("concurrent", (0, (1, 1.6))),
                                      ("detection", (0, (5, 2)))):
                    if (s, placement) not in data:
                        continue
                    xs, ys = data[(s, placement)]
                    ax.plot(xs, ys, color=c, ls=ls,
                            label=LABELS[s] if placement == "injection" else None)
            ax.set_yscale("log")
            ax.set_xlabel(r"opto-mechanical coupling  $\kappa$")
            ax.set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            ax.set_title("Where the loss sits decides how much back-action costs\n"
                         "(solid: loss $\\to$ BA;  dotted: loss $\\it{during}$ BA;  "
                         "dashed: BA $\\to$ loss;  $\\eta=0.8$)")
            legend_below(fig, ax, ncol=4)
            finish(fig, str(out / "fig2_loss_placement"), name)


def fig_phase_noise(d, out):
    """Phase noise -> BA -> detection loss, plus evidence that BA1/BA2/BA3 coincide here."""
    rows = read(d / "phase_noise.csv")
    has_ordering = bool(rows) and "ordering" in rows[0]
    main = [r for r in rows if r["ordering"] == "BA3"] if has_ordering else rows
    data = series(main, ("state",), "sigma_phi_rad", "qfi_epsilon_a")
    states = sorted({k[0] for k in data}, key=lambda s: ORDER.index(s))

    spread = {}
    if has_ordering:
        for st in states:
            per = {o: series([r for r in rows if r["state"] == st and r["ordering"] == o],
                             ("state",), "sigma_phi_rad", "qfi_epsilon_a")[(st,)]
                   for o in ("BA1", "BA2", "BA3")}
            xs = per["BA3"][0]
            stack = np.vstack([per[o][1] for o in ("BA1", "BA2", "BA3")])
            spread[st] = (xs, stack.max(axis=0) - stack.min(axis=0))

    for name in THEMES:
        with theme(name) as t:
            ncols = 2 if spread else 1
            fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.2))
            axes = np.atleast_1d(axes)
            ax = axes[0]
            drawn = []
            for st in states:
                xs, ys = data[(st,)]
                ax.plot(xs, ys, color=t["series"][ORDER.index(st)], label=LABELS[st])
                drawn.append((LABELS[st], xs, ys))
            label_lines(ax, drawn, t)
            ax.set_xlabel(r"input phase noise  $\sigma_\phi$  [rad]")
            ax.set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            ax.set_title("Phase noise $\\to$ back-action $\\to$ detection loss\n"
                         "the Fock state is exactly flat; the cat overtakes squeezing")
            if spread:
                ax2 = axes[1]
                for st in states:
                    xs, dy = spread[st]
                    ax2.semilogy(xs, np.maximum(dy, 1e-16), color=t["series"][ORDER.index(st)],
                                 marker="o", markersize=4, label=LABELS[st])
                ax2.axhline(1e-6, color=t["muted"], lw=1.0, ls=(0, (4, 3)))
                ax2.annotate("1e-6", (0.01, 1.3e-6), color=t["muted"], fontsize=8)
                ax2.set_xlabel(r"input phase noise  $\sigma_\phi$  [rad]")
                ax2.set_ylabel(r"$\max_{ij}|\mathcal{F}_{\rm BA_i} - \mathcal{F}_{\rm BA_j}|$")
                ax2.set_ylim(1e-14, 1e-4)
                ax2.set_title("all three orderings were run\n"
                              "they agree to ~1e-11, far below any effect here")
            legend_below(fig, ax, ncol=4)
            fig.subplots_adjust(wspace=0.32)
            finish(fig, str(out / "fig4_phase_noise"), name)


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
            fig.suptitle(r"Probe states at fixed $\bar n = 2$, detection loss: flat at "
                         r"$\eta = 1$, ranking inverts once loss is on", y=1.0)
            fig.subplots_adjust(wspace=0.1)
            legend_below(fig, axes[0], ncol=6)
            finish(fig, str(out / "fig5_states_vs_kappa"), name)


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
            legend_below(fig, axes[0], ncol=6)
            finish(fig, str(out / "fig6_nbar_scaling"), name)


def fig_concurrent_orderings(d, out):
    """With loss acting during the interaction, BA1/BA2/BA3 separate and BA3 is bracketed."""
    path = d / "concurrent_orderings.csv"
    if not path.exists():
        print("  fig_concurrent_orderings skipped (no concurrent_orderings.csv)")
        return
    rows = read(path)
    states = sorted({r["state"] for r in rows}, key=lambda s: ORDER.index(s))
    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, len(states), figsize=(3.4 * len(states), 3.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, st in zip(axes, states):
                data = series(rows, ("ordering",), "kappa", "qfi_epsilon_a",
                              where=lambda r, s0=st: r["state"] == s0)
                xs = data[("BA1",)][0]
                lo = np.minimum(data[("BA1",)][1], data[("BA2",)][1])
                hi = np.maximum(data[("BA1",)][1], data[("BA2",)][1])
                # The band is the claim: BA1 and BA2 bound BA3.
                ax.fill_between(xs, lo, hi, color=t["series"][0], alpha=0.15, lw=0,
                                label="between BA1 and BA2")
                for i, o in enumerate(("BA1", "BA2", "BA3")):
                    x, y = data[(o,)]
                    ax.plot(x, y, color=t["series"][i], marker="o", markersize=4, label=o)
                ax.set_xlabel(r"$\kappa$")
                ax.set_title(LABELS[st])
            axes[0].set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            fig.suptitle(r"Loss acting $\it{during}$ the interaction separates the orderings "
                         r"($\eta_{\rm ch}=0.8$); BA3 stays bracketed", y=1.0)
            fig.subplots_adjust(wspace=0.12)
            legend_below(fig, axes[0], ncol=4)
            finish(fig, str(out / "fig3_concurrent_orderings"), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "results"))
    args = ap.parse_args()
    d = Path(args.dir)
    for fn in (fig_orderings, fig_loss_placement, fig_concurrent_orderings,
               fig_phase_noise, fig_states_vs_kappa, fig_nbar_scaling):
        fn(d, d)
        print(f"  {fn.__name__} ok")
    print(f"figures -> {d}/")


if __name__ == "__main__":
    main()
