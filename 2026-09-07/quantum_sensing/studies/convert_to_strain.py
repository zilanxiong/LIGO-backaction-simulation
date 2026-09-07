"""
Convert a QFI frequency sweep into the literature presentation:
SQL-normalized strain noise sqrt(S_h / S_SQL) vs frequency.

Mapping (KLMTV two-photon conventions, single mode per frequency bin):
the signal enters the output phase quadrature as sqrt(2 kappa) h / h_SQL,
while our epsilon_a displaces p by -sqrt(2) epsilon_a over unit time, so
epsilon_a = sqrt(kappa) h / h_SQL and

    F_h = (kappa / h_SQL^2) F_eps  =>  S_h / S_SQL = 2 / (kappa F_eps),

with the constant fixed so that the lossless vacuum-homodyne CFI
4/(1+kappa^2) reproduces the textbook SQL curve
S_h/S_SQL = (kappa + 1/kappa)/2 (touching 1 at kappa = 1).  The script
asserts that anchor against cfi_homodyne_gaussian before plotting.

The quantum-CRB curves plotted from the QFI therefore lie BELOW the
homodyne SQL "V" wherever the optimal measurement beats fixed homodyne.
This is CRB-per-measurement mapped into strain units: shapes and ratios
are comparable to literature noise budgets; absolute strain would need
the continuous-measurement bandwidth normalization.

Usage:  python quantum_sensing/studies/convert_to_strain.py [suffix]
        (suffix as in ba1_frequency_sweep<suffix>.csv; default "_ba3_full")
Output: quantum_sensing/studies/results/strain_sql<suffix>.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantum_sensing as qs
from quantum_sensing import gaussian as g

RESULTS_DIR = Path(__file__).resolve().parent / "results"

STATE_ORDER = ["coherent", "sqz_vac", "sqz_vac_p", "cat", "sqz_cat", "fock",
               "opt_fock_sup"]
STATE_LABELS = {
    "coherent": "Coherent",
    "sqz_vac": "Sqz. vac. (x, wrong angle)",
    "sqz_vac_p": "Squeezed vac. (p)",
    "cat": "Even cat",
    "sqz_cat": "Squeezed cat", "fock": "Fock",
    "opt_fock_sup": "Optimized (no-BA)",
}
STATE_COLORS = {
    "coherent": "#7f7f7f", "sqz_vac": "#f2a0a5", "sqz_vac_p": "#d62728",
    "cat": "#1f77b4",
    "sqz_cat": "#9467bd", "fock": "#2ca02c", "opt_fock_sup": "#ff7f0e",
}

TITLES = {
    "_ba3_full": ("BA3 physical channel: $\\eta_{in}$=0.95, concurrent "
                  "$\\eta_{ch}$=0.99, $\\eta_{out}$=0.90"),
    "": "BA1: shear $\\to$ signal $\\to$ detection loss ($\\eta$=0.9)",
    "_injection": "BA1: injection loss ($\\eta$=0.9) $\\to$ shear $\\to$ signal",
    "_ba2_detection": "BA2: signal $\\to$ shear $\\to$ detection loss ($\\eta$=0.9)",
    "_ba2_injection": "BA2: injection loss ($\\eta$=0.9) $\\to$ signal $\\to$ shear",
    "_pn_ba1_detection": ("phase noise ($\\phi_{rms}$=0.1) $\\to$ BA1 $\\to$ "
                          "detection loss ($\\eta$=0.9)"),
    "_paper_pn_ba1": ("phase noise (200 mrad) $\\to$ BA1 $\\to$ 5% loss"),
}


def sql_ratio(kappa, qfi):
    """S_h / S_SQL = 2 / (kappa * F_eps)."""
    return 2.0 / (kappa * qfi)


def _check_anchor():
    """Lossless vacuum homodyne must land on the textbook SQL V-curve."""
    for kappa in (0.3, 1.0, 3.0):
        V, d = g.vacuum_state()
        cfi = g.gaussian_qfi_rp(V, d, param_type="epsilon_a", kappa_ba=kappa,
                                homodyne_angle=np.pi / 2)
        assert np.isclose(sql_ratio(kappa, cfi), (kappa + 1 / kappa) / 2,
                          rtol=1e-9), "SQL normalization anchor failed"


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_ba3_full"
    if suffix == "none":
        suffix = ""
    _check_anchor()

    df = pd.read_csv(RESULTS_DIR / f"ba1_frequency_sweep{suffix}.csv")
    n_target = df["n_target"].iloc[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    f_ref = np.geomspace(10, 1000, 200)
    k_ref = qs.rp_kappa(2 * np.pi * f_ref)

    # References: fixed-p-homodyne vacuum SQL "V" and the lossless
    # coherent-state quantum CRB (F_eps = 4).
    ax.loglog(f_ref, np.sqrt((k_ref + 1 / k_ref) / 2), "k--", lw=1.2,
              label="SQL (vacuum + $p$-homodyne, lossless)")
    ax.loglog(f_ref, np.sqrt(1 / (2 * k_ref)), "k:", lw=1.2,
              label="Coherent quantum CRB (lossless)")

    for state in STATE_ORDER:
        sub = df[df["state"] == state].sort_values("f_hz")
        if sub.empty:
            continue
        ax.loglog(sub["f_hz"], np.sqrt(sql_ratio(sub["kappa_ba"], sub["qfi"])),
                  "o-", ms=4, color=STATE_COLORS[state],
                  label=STATE_LABELS[state])

    ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Frequency  $\\Omega/2\\pi$  [Hz]")
    ax.set_ylabel(r"$\sqrt{S_h\,/\,S_h^{SQL}}$")
    ax.set_title(f"{TITLES.get(suffix, suffix)}  "
                 f"($\\langle n\\rangle$ = {n_target:g})", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    out = RESULTS_DIR / f"strain_sql{suffix}.png"
    fig.savefig(out, dpi=160)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
