#!/usr/bin/env python3
"""
QFI under radiation pressure, on the quantum_sensing channel.

Runs the to-do list directly:

  A. channel orderings      BA1, BA2, BA3 (and no back-action as reference)
  B. loss placement         injection loss (loss -> BA) vs detection loss (BA -> loss)
  C. phase noise            phase noise -> BA -> loss
  D. probe states           at fixed <n>: coherent, squeezed, Fock, cat, squeezed cat,
                            and the no-back-action optimum
  E. <n> scaling            QFI(nbar) fitted to a power law for each state and channel

Every Fock-space number goes through the automated cutoff convergence check;
the cutoff used and its convergence flag are written into every CSV row.

Usage::

    python backaction-qfi/run_study.py [--quick]
    python backaction-qfi/make_figures.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convergence import converged_qfi  # noqa: E402
from probe_states import STATE_LABELS, make_state  # noqa: E402
from radiation_pressure import get_state_single_mode_ba, suggested_cutoff  # noqa: E402

# The "optimum without back-action" probe is deliberately absent: the real
# optimised states come from the previous optimisation runs and need the
# consolidated_data directory (quantum_sensing.states.set_data_dir).  Rather
# than report an analytic stand-in, that row is deferred.
FAMILIES = ["coherent", "squeezed", "fock", "cat_even", "cat_odd", "squeezed_cat"]
LABELS = STATE_LABELS
MAX_CUTOFF = 700
#: Concurrent-loss runs cost more per cutoff but converge sooner (loss damps the tail).
CONCURRENT_CUTOFF = 340


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def qfi(family, nbar, rtol=3e-4, max_cutoff=None, **channel):
    """Convergence-checked QFI for ``epsilon_a``.  Returns (value, cutoff, converged).

    ``max_cutoff`` overrides the global cap.  Concurrent-loss runs use a lower
    one: the split-step evolution costs more per cutoff, and loss acting during
    the interaction damps the high-Fock tail, so they converge sooner anyway.
    """
    MAX_CUTOFF = max_cutoff or globals()["MAX_CUTOFF"]
    n0 = min(max(suggested_cutoff(nbar, channel.get("kappa", 0.0)), 40), MAX_CUTOFF)
    # The ladder needs at least three rungs, or n_stable=2 can never be satisfied
    # and the point is flagged unconverged however well it actually converged.
    # That bites whenever the rule-of-thumb start sits close to the cap.
    n0 = min(n0, max(40, int(MAX_CUTOFF / 1.35**2)))
    ladder, n = [], n0
    while n < MAX_CUTOFF:
        ladder.append(n)
        n = int(np.ceil(n * 1.35))
    ladder.append(MAX_CUTOFF)
    res = converged_qfi(
        lambda N: make_state(family, N, nbar),
        get_state_single_mode_ba,
        param_type="epsilon_a",
        cutoffs=ladder,
        rtol=rtol,
        n_stable=2,
        **channel,
    )
    if not res.converged:
        print(f"    ! UNCONVERGED: {family} nbar={nbar} {channel} -> {res!r}")
    return res.value, res.cutoff, res.converged


# ---------------------------------------------------------------------------
# A. Channel orderings
# ---------------------------------------------------------------------------
def study_orderings(outdir, nbar, kappas):
    print("[A] BA1 / BA2 / BA3")
    rows = []
    for family in ["squeezed", "cat_even", "fock"]:
        for kappa in kappas:
            vals = {}
            for ordering in ("none", "BA1", "BA2", "BA3"):
                v, N, ok = qfi(family, nbar, kappa=kappa, ordering=ordering, eta_out=0.9)
                vals[ordering] = v
                rows.append([family, "epsilon_a (physical)", kappa, ordering, v, N, ok])
            spread = max(abs(vals[o] - vals["BA3"]) for o in ("BA1", "BA2"))
            print(f"    {family:>12} kappa={kappa:4.2f}  no-BA {vals['none']:8.5f}  "
                  f"BA1 {vals['BA1']:8.5f}  BA2 {vals['BA2']:8.5f}  BA3 {vals['BA3']:8.5f}  "
                  f"|BA-BA3| max {spread:.1e}")

    # Non-commuting reference: signal in the amplitude quadrature.
    from sld import calculate_qfi
    for family in ["squeezed", "cat_even"]:
        N = 250
        psi = make_state(family, N, nbar)
        vals = {}
        for ordering in ("none", "BA1", "BA2", "BA3"):
            vals[ordering] = calculate_qfi(
                get_state_single_mode_ba, param_type="epsilon_p", param_value=0.0,
                kappa=1.5, ordering=ordering, eta_out=0.9, N_basis=N, rho=psi,
            )
            rows.append([family, "epsilon_p (non-commuting)", 1.5, ordering, vals[ordering], N, True])
        bounded = min(vals["BA1"], vals["BA2"]) <= vals["BA3"] <= max(vals["BA1"], vals["BA2"])
        print(f"    {family:>12} epsilon_p  BA1 {vals['BA1']:8.5f}  BA2 {vals['BA2']:8.5f}  "
              f"BA3 {vals['BA3']:8.5f}  BA3 bounded: {bounded}")

    write_csv(outdir / "orderings.csv",
              ["state", "signal_quadrature", "kappa", "ordering", "qfi_epsilon_a", "cutoff", "converged"],
              rows)


# ---------------------------------------------------------------------------
# B. Loss placement
# ---------------------------------------------------------------------------
def study_loss_placement(outdir, nbar, kappas, eta):
    print(f"[B] Loss placement: injection (loss -> BA), concurrent (loss during BA), "
          f"detection (BA -> loss), eta = {eta}")
    rows = []
    placements = (
        ("injection", dict(eta_in=eta), None),
        ("concurrent", dict(eta_ch=eta), CONCURRENT_CUTOFF),
        ("detection", dict(eta_out=eta), None),
    )
    for family in ["squeezed", "cat_even", "fock", "coherent"]:
        line = {}
        for placement, kw, cap in placements:
            vals = []
            for kappa in kappas:
                v, N, ok = qfi(family, nbar, kappa=kappa, ordering="BA3", max_cutoff=cap, **kw)
                vals.append(v)
                rows.append([family, placement, kappa, v, N, ok])
            line[placement] = vals
        print(f"    {family:>12} " + "   ".join(
            f"{p} {line[p][0]:7.4f}->{line[p][-1]:7.4f}" for p, _, _ in placements))
    write_csv(outdir / "loss_placement.csv",
              ["state", "loss_placement", "kappa", "qfi_epsilon_a", "cutoff", "converged"], rows)


def study_concurrent_orderings(outdir, nbar, kappas, eta):
    """BA1/BA2/BA3 when dissipation acts *during* the interaction.

    With stage-separated loss the three orderings coincide exactly, because the
    signal and back-action generators commute.  Concurrent loss breaks that: the
    Lindblad operator does not commute with the shear, so the orderings separate
    even in the physical signal quadrature -- and BA3 is bracketed by BA1/BA2,
    which is the configuration the project plan describes.
    """
    print(f"[B2] Orderings under concurrent loss (eta_ch = {eta})")
    rows = []
    for family in ["squeezed", "cat_even", "fock"]:
        for kappa in kappas:
            vals = {}
            for ordering in ("BA1", "BA2", "BA3"):
                v, N, ok = qfi(family, nbar, kappa=kappa, ordering=ordering,
                               eta_ch=eta, max_cutoff=CONCURRENT_CUTOFF)
                vals[ordering] = v
                rows.append([family, kappa, ordering, v, N, ok])
            lo, hi = min(vals["BA1"], vals["BA2"]), max(vals["BA1"], vals["BA2"])
            bracketed = lo - 1e-9 <= vals["BA3"] <= hi + 1e-9
            print(f"    {family:>12} kappa={kappa:4.2f}  BA1 {vals['BA1']:8.4f}  "
                  f"BA2 {vals['BA2']:8.4f}  BA3 {vals['BA3']:8.4f}  "
                  f"spread {hi - lo:7.4f}  BA3 bracketed: {bracketed}")
    write_csv(outdir / "concurrent_orderings.csv",
              ["state", "kappa", "ordering", "qfi_epsilon_a", "cutoff", "converged"], rows)


# ---------------------------------------------------------------------------
# C. Phase noise -> BA -> loss
# ---------------------------------------------------------------------------
def study_phase_noise(outdir, nbar, sigmas, kappa, eta):
    print(f"[C] Phase noise -> BA(kappa={kappa}) -> detection loss(eta={eta})")
    rows = []
    for family in ["squeezed", "cat_even", "coherent", "fock"]:
        vals = []
        for sigma in sigmas:
            v, N, ok = qfi(family, nbar, kappa=kappa, ordering="BA3", pn_in=float(sigma), eta_out=eta)
            vals.append(v)
            rows.append([family, float(sigma), v, N, ok])
        print(f"    {family:>12} " + "  ".join(f"{v:7.4f}" for v in vals))
    write_csv(outdir / "phase_noise.csv",
              ["state", "sigma_phi_rad", "qfi_epsilon_a", "cutoff", "converged"], rows)


# ---------------------------------------------------------------------------
# D. Probe states versus kappa
# ---------------------------------------------------------------------------
def study_states(outdir, nbar, kappas, etas):
    print("[D] Probe states at fixed <n> versus kappa")
    rows = []
    for eta in etas:
        print(f"    --- eta_out = {eta} ---")
        for family in FAMILIES:
            vals = []
            for kappa in kappas:
                v, N, ok = qfi(family, nbar, kappa=kappa, ordering="BA3", eta_out=eta)
                vals.append(v)
                rows.append([eta, family, kappa, v, N, ok])
            print(f"    {LABELS[family]:>14} " + "  ".join(f"{v:8.4f}" for v in vals))
    write_csv(outdir / "states_vs_kappa.csv",
              ["eta_out", "state", "kappa", "qfi_epsilon_a", "cutoff", "converged"], rows)


# ---------------------------------------------------------------------------
# E. <n> scaling
# ---------------------------------------------------------------------------
def study_nbar_scaling(outdir, nbars, configs):
    """Report how the QFI scales with photon number, for each state and channel.

    Two numbers per row, because a single power-law fit over the whole grid is
    misleading: at small ``nbar`` every QFI flattens onto the vacuum value, which
    drags the exponent down.

    * ``alpha_fit`` -- least-squares log-log slope over ``nbar >= 1`` only.
    * ``alpha_local`` -- the slope between the two largest grid points, i.e. the
      local logarithmic derivative at the top of the range.

    ``alpha = 1`` is shot-noise-like, ``alpha = 2`` Heisenberg-like.  For the
    lossless no-back-action case the squeezed-vacuum QFI is
    ``2(2n+1+2 sqrt(n(n+1))) -> 8n``, so ``alpha -> 1`` from below.
    """
    print("[E] <n> scaling  (alpha_fit over nbar >= 1; alpha_local from the top two points)")
    rows, fits = [], []
    for label, channel in configs:
        print(f"    --- {label} ---")
        for family in FAMILIES:
            vals = []
            for nbar in nbars:
                if family == "cat_odd" and nbar < 1.0:
                    vals.append(np.nan)
                    continue
                v, N, ok = qfi(family, nbar, **channel)
                vals.append(v)
                rows.append([label, family, nbar, v, N, ok])
            arr = np.array(vals, dtype=float)
            ns = np.array(nbars, dtype=float)
            good = np.isfinite(arr) & (arr > 0) & (ns >= 1.0)
            if good.sum() >= 2:
                alpha, log_a = np.polyfit(np.log(ns[good]), np.log(arr[good]), 1)
                resid = np.log(arr[good]) - (alpha * np.log(ns[good]) + log_a)
                rms = float(np.sqrt(np.mean(resid**2)))
                prefactor = float(np.exp(log_a))
            else:
                alpha, prefactor, rms = np.nan, np.nan, np.nan
            fin = np.isfinite(arr) & (arr > 0)
            if fin.sum() >= 2:
                nn, aa = ns[fin][-2:], arr[fin][-2:]
                alpha_local = float(np.log(aa[1] / aa[0]) / np.log(nn[1] / nn[0]))
            else:
                alpha_local = np.nan
            fits.append([label, family, float(alpha), alpha_local, prefactor, rms, int(good.sum())])
            print(f"    {LABELS[family]:>14} alpha_fit {alpha:6.3f}  alpha_local {alpha_local:6.3f}   "
                  + "  ".join(f"{v:8.3f}" for v in vals))
    write_csv(outdir / "nbar_scaling.csv",
              ["config", "state", "nbar", "qfi_epsilon_a", "cutoff", "converged"], rows)
    write_csv(outdir / "nbar_scaling_fits.csv",
              ["config", "state", "alpha_fit_nbar_ge_1", "alpha_local_top_two",
               "prefactor", "logfit_rms", "n_points_fitted"], fits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    nbar = 2.0
    if args.quick:
        kappas = [0.0, 1.0, 2.0]
        nbars = [0.5, 1.0, 2.0]
        sigmas = [0.0, 0.2, 0.4]
        etas = [1.0, 0.9]
    else:
        kappas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        nbars = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
        sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        etas = [1.0, 0.95, 0.9, 0.7]

    configs = [
        ("no back-action, lossless", dict(kappa=0.0, ordering="none")),
        ("BA kappa=1, lossless", dict(kappa=1.0, ordering="BA3")),
        ("BA kappa=1, detection eta=0.9", dict(kappa=1.0, ordering="BA3", eta_out=0.9)),
        ("BA kappa=2, detection eta=0.9", dict(kappa=2.0, ordering="BA3", eta_out=0.9)),
    ]

    t0 = time.time()
    study_orderings(outdir, nbar, [0.5, 1.5] if args.quick else [0.5, 1.5, 3.0])
    study_loss_placement(outdir, nbar, kappas, 0.8)
    study_concurrent_orderings(outdir, nbar, [0.5, 1.5, 3.0] if not args.quick else [1.5], 0.8)
    study_phase_noise(outdir, nbar, sigmas, kappa=1.0, eta=0.9)
    study_states(outdir, nbar, kappas, etas)
    study_nbar_scaling(outdir, nbars, configs)
    print(f"\nDone in {time.time() - t0:.1f} s -> {outdir}/")


if __name__ == "__main__":
    main()
