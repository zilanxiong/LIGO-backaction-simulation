"""Shared matplotlib styling for the study figures.

Uses the validated categorical palette (adjacent-pair CVD gates pass in both
light and dark; see the palette validator).  Every figure is rendered twice, in
light and dark, and every figure has a CSV table beside it -- which is also the
relief required for the light-mode contrast warning on the aqua/yellow/magenta
slots.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#8a8880",
        "grid": "#e4e3df",
        "series": SERIES_LIGHT,
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#8a8880",
        "grid": "#333331",
        "series": SERIES_DARK,
    },
}


@contextmanager
def theme(name: str):
    """Apply the light or dark rc parameters for the duration of the block."""
    t = THEMES[name]
    rc = {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["text_primary"],
        "axes.labelcolor": t["text_secondary"],
        "axes.edgecolor": t["grid"],
        "axes.titlecolor": t["text_primary"],
        "xtick.color": t["text_secondary"],
        "ytick.color": t["text_secondary"],
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
    }
    with plt.rc_context(rc):
        yield t


MAX_DIRECT_LABELS = 4


def label_lines(ax, series, t, log_x=False, fracs=None):
    """Direct-label up to four lines, staggered along x so the text cannot collide.

    ``series`` is a list of ``(label, xs, ys)``.  Labels are drawn in text ink --
    identity comes from the line the label sits on, never from the text colour.
    Above four series the legend carries identity on its own.
    """
    if len(series) > MAX_DIRECT_LABELS:
        return
    if fracs is None:
        fracs = np.linspace(0.55, 0.97, len(series)) if len(series) > 1 else [0.9]
    for (label, xs, ys), frac in zip(series, fracs):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        i = int(np.clip(round(frac * (xs.size - 1)), 0, xs.size - 1))
        ax.annotate(
            label,
            xy=(xs[i], ys[i]),
            xytext=(0, 9),
            textcoords="offset points",
            color=t["text_secondary"],
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.18", fc=t["surface"], ec="none", alpha=0.85),
        )


def legend_below(fig, ax, ncol=3):
    """Legend under the axes, so it never sits on top of the data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, fontsize=8, frameon=False)


def finish(fig, path_stem: str, name: str):
    """Save the figure under ``results/<stem>.png`` / ``results/<stem>.dark.png``."""
    suffix = "" if name == "light" else ".dark"
    fig.savefig(f"{path_stem}{suffix}.png", bbox_inches="tight")
    plt.close(fig)
