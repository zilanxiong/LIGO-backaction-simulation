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
from matplotlib.lines import Line2D  # noqa: E402

from plotstyle import THEMES, finish, label_lines, legend_below, theme  # noqa: E402
from ifo import ALIGO, f_at_kappa  # noqa: E402

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
            axes[1].set_title("signal in $x$ (non-commuting)\n"
                              "orderings separate; Fock breaks the bracketing")
            for ax in axes:
                ax.set_xticks(np.arange(4))
                ax.set_xticklabels(["no BA", "BA1", "BA2", "BA3"])
                ax.grid(axis="x", visible=False)
                ax.set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            legend_below(fig, axes[0], ncol=3)
            fig.subplots_adjust(wspace=0.25)
            finish(fig, str(out / "fig1_orderings"), name)


def fig_loss_placement(d, out):
    """Loss placement x ordering.

    One panel per ordering.  Injection and detection are identical across the
    three panels -- with stage-separated loss the ordering cannot matter -- so
    the panels differ only in the dotted concurrent curves.
    """
    rows = read(d / "loss_placement.csv")
    data = series(rows, ("state", "ordering", "loss_placement"), "kappa", "qfi_epsilon_a")
    states = sorted({k[0] for k in data}, key=lambda s: ORDER.index(s))
    orderings = [o for o in ("BA1", "BA2", "BA3") if any(k[1] == o for k in data)]
    titles = {"BA1": "BA1:  RP $\\to$ sensing",
              "BA2": "BA2:  sensing $\\to$ RP",
              "BA3": "BA3:  simultaneous (physical)"}
    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(1, len(orderings), figsize=(4.0 * len(orderings), 4.3),
                                     sharey=True)
            axes = np.atleast_1d(axes)
            for ax, o in zip(axes, orderings):
                for s in states:
                    c = t["series"][ORDER.index(s)]
                    for placement, ls in (("injection", "-"), ("concurrent", (0, (1, 1.6))),
                                          ("detection", (0, (5, 2)))):
                        if (s, o, placement) not in data:
                            continue
                        xs, ys = data[(s, o, placement)]
                        ax.plot(xs, ys, color=c, ls=ls,
                                label=LABELS[s] if placement == "injection" else None)
                ax.set_yscale("log")
                ax.set_xlabel(r"opto-mechanical coupling  $\kappa$")
                ax.set_title(titles.get(o, o), fontsize=10)
            axes[0].set_ylabel(r"$\mathcal{F}_{\epsilon_a}$")
            fig.suptitle(r"Where the loss sits decides how much back-action costs"
                         "\n" r"($\eta = 0.8$;  injection and detection are identical "
                         r"across the three panels)", fontsize=11, y=1.0)
            legend_below(fig, axes[0], ncol=4)
            # Second legend row for the linestyle key, so it does not have to ride
            # along in the title where it collides with the panel headings.
            keys = [Line2D([], [], color=t["text_secondary"], ls=ls, label=lab)
                    for ls, lab in ((("-"), "loss $\\to$ BA (injection)"),
                                    ((0, (1, 1.6)), "loss $\\it{during}$ BA (concurrent)"),
                                    ((0, (5, 2)), "BA $\\to$ loss (detection)"))]
            fig.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                       ncol=3, fontsize=8, frameon=False)
            fig.subplots_adjust(top=0.82, wspace=0.12)
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



def fig_frequency(d, out):
    """Section 2 with the x-axis calibrated to signal frequency.

    Top row: the QFI, the same quantity fig2 plots against kappa.
    Bottom row: the Cramer-Rao bound on the strain in units of h_SQL, which is
    what the frequency calibration is actually for -- it folds in the sqrt(2 kappa)
    signal transfer, so the two rows do not have the same shape.
    """
    path = d / "frequency_sweep.csv"
    if not path.exists():
        print("  fig_frequency skipped (run run_study.py --only F)")
        return
    rows = read(path)
    qfi_d = series(rows, ("state", "ordering", "loss_placement"), "f_hz", "qfi_epsilon_a")
    sig_d = series(rows, ("state", "ordering", "loss_placement"), "f_hz", "sigma_h_over_hsql")
    states = sorted({k[0] for k in qfi_d}, key=lambda s: ORDER.index(s))
    orderings = [o for o in ("BA1", "BA2", "BA3") if any(k[1] == o for k in qfi_d)]
    titles = {"BA1": "BA1:  RP $\\to$ sensing",
              "BA2": "BA2:  sensing $\\to$ RP",
              "BA3": "BA3:  simultaneous (physical)"}
    styles = (("injection", "-"), ("concurrent", (0, (1, 1.6))), ("detection", (0, (5, 2))))
    f_sql = float(f_at_kappa(1.0))

    for name in THEMES:
        with theme(name) as t:
            fig, axes = plt.subplots(2, len(orderings),
                                     figsize=(4.0 * len(orderings), 7.4),
                                     sharex=True, sharey="row")
            axes = np.atleast_2d(axes)
            for col, o in enumerate(orderings):
                for row, data in enumerate((qfi_d, sig_d)):
                    ax = axes[row][col]
                    for st in states:
                        c = t["series"][ORDER.index(st)]
                        for placement, ls in styles:
                            key = (st, o, placement)
                            if key not in data:
                                continue
                            xs, ys = data[key]
                            ax.plot(xs, ys, color=c, ls=ls,
                                    label=LABELS[st] if placement == "injection" else None)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.axvline(f_sql, color=t["muted"], lw=0.8, ls=":", zorder=0)
                    if row == 1:
                        ax.axhline(1.0, color=t["muted"], lw=0.8, zorder=0)
                        ax.set_xlabel("signal frequency  $f$  [Hz]")
                        # The default log locator crowds this narrow decade.
                        ax.set_xticks([400, 600, 1000, 2000, 3000])
                        ax.set_xticklabels(["400", "600", "1k", "2k", "3k"])
                        ax.set_xticks([], minor=True)
                axes[0][col].set_title(titles.get(o, o), fontsize=10)
                # kappa is a function of f, so label it on a second axis rather
                # than asking the reader to carry the conversion in their head.
                sec = axes[0][col].secondary_xaxis(
                    "top", functions=(lambda f: ALIGO.kappa(f), lambda k: f_at_kappa(k)))
                sec.set_xlabel(r"opto-mechanical coupling  $\kappa$", fontsize=9)
                sec.set_xticks([3, 1, 0.3, 0.1, 0.01, 0.001])
                sec.set_xticklabels(["3", "1", "0.3", "0.1", "0.01", "0.001"], fontsize=8)
            axes[0][0].set_ylabel(r"$\mathcal{F}_{\epsilon_a}$" "\n"
                                  "(same quantity as fig2)", fontsize=9)
            axes[1][0].set_ylabel(r"$\sigma_h\,/\,h_{\rm SQL}$   (lower is better)" "\n"
                                  r"strain bound, $\sqrt{2\kappa}$ transfer folded in",
                                  fontsize=9)
            axes[0][-1].annotate(r"$\kappa=1$", xy=(f_sql, 1.0),
                                 xycoords=("data", "axes fraction"),
                                 xytext=(3, -12), textcoords="offset points",
                                 fontsize=8, color=t["muted"], ha="left", va="top")
            fig.suptitle(f"Section 2 in frequency  —  {ALIGO.name}, "
                         r"$I_0 = I_{\rm SQL}$, $\gamma/2\pi = 500$ Hz, "
                         r"$\eta = 0.8$", fontsize=11, y=1.0)
            legend_below(fig, axes[0][0], ncol=4)
            keys = [Line2D([], [], color=t["text_secondary"], ls=ls, label=lab)
                    for ls, lab in ((("-"), "loss $\\to$ BA (injection)"),
                                    ((0, (1, 1.6)), "loss $\\it{during}$ BA (concurrent)"),
                                    ((0, (5, 2)), "BA $\\to$ loss (detection)"))]
            fig.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                       ncol=3, fontsize=8, frameon=False)
            fig.subplots_adjust(top=0.86, hspace=0.20, wspace=0.10)
            finish(fig, str(out / "fig7_frequency"), name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "results"))
    args = ap.parse_args()
    d = Path(args.dir)
    for fn in (fig_orderings, fig_loss_placement, fig_concurrent_orderings,
               fig_phase_noise, fig_states_vs_kappa, fig_nbar_scaling,
               fig_frequency):
        fn(d, d)
        print(f"  {fn.__name__} ok")
    print(f"figures -> {d}/")


if __name__ == "__main__":
    main()
