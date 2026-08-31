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
from ifo import ALIGO, f_at_kappa, strain_uncertainty  # noqa: E402
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


def qfi(family, nbar, rtol=3e-4, max_cutoff=None, param_type="epsilon_a", **channel):
    """Convergence-checked QFI.  Returns (value, cutoff, converged).

    ``param_type`` selects the estimated parameter: ``"epsilon_a"`` (the physical
    signal quadrature) or ``"epsilon_p"`` (the non-commuting reference).

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
        param_type=param_type,
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

    # Non-commuting reference: signal in the amplitude quadrature.  Convergence
    # checked like every other point, rather than trusting a single fixed cutoff.
    for family in ["squeezed", "cat_even", "fock"]:
        vals, cutoffs = {}, {}
        for ordering in ("none", "BA1", "BA2", "BA3"):
            v, N, ok = qfi(family, nbar, param_type="epsilon_p",
                           kappa=1.5, ordering=ordering, eta_out=0.9)
            vals[ordering], cutoffs[ordering] = v, N
            rows.append([family, "epsilon_p (non-commuting)", 1.5, ordering, v, N, ok])
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
    """Injection / concurrent / detection loss, for every ordering of the shear.

    Run for all three orderings, not just the physical BA3.  With stage-separated
    loss the ordering cannot matter -- the shear and the signal are both functions
    of x -- so injection and detection reproduce across BA1/BA2/BA3 and the check
    is a null result on a second axis.  Concurrent loss is where they part: BA2
    puts the shear after the lossy evolution, where it is a QFI-preserving
    unitary, so BA2 stays flat in kappa while BA1 pays the full cost.
    """
    print(f"[B] Loss placement: injection (loss -> BA), concurrent (loss during BA), "
          f"detection (BA -> loss), eta = {eta}, all orderings")
    rows = []
    placements = (
        ("injection", dict(eta_in=eta), None),
        ("concurrent", dict(eta_ch=eta), CONCURRENT_CUTOFF),
        ("detection", dict(eta_out=eta), None),
    )
    for family in ["squeezed", "cat_even", "fock", "coherent"]:
        for ordering in ("BA1", "BA2", "BA3"):
            line = {}
            for placement, kw, cap in placements:
                vals = []
                for kappa in kappas:
                    v, N, ok = qfi(family, nbar, kappa=kappa, ordering=ordering,
                                   max_cutoff=cap, **kw)
                    vals.append(v)
                    rows.append([family, ordering, placement, kappa, v, N, ok])
                line[placement] = vals
            print(f"    {family:>12} {ordering}  " + "   ".join(
                f"{p} {line[p][0]:7.4f}->{line[p][-1]:7.4f}" for p, _, _ in placements))
    write_csv(outdir / "loss_placement.csv",
              ["state", "ordering", "loss_placement", "kappa", "qfi_epsilon_a",
               "cutoff", "converged"], rows)


# ---------------------------------------------------------------------------
# F. Loss placement vs signal frequency
# ---------------------------------------------------------------------------
def study_frequency(outdir, nbar, eta, n_points=12, kappa_max=3.0, decades=1.0,
                    params=ALIGO):
    """Section B again, with the x-axis calibrated to signal frequency.

    ``kappa`` is not a free knob in a real interferometer -- it is fixed by the
    sideband frequency through

        kappa(Omega) = 2 (I0/I_SQL) gamma^4 / [Omega^2 (gamma^2 + Omega^2)] ,

    so every point on a kappa sweep corresponds to one frequency.  The grid runs
    from the frequency where ``kappa = kappa_max`` upward, because kappa grows
    without bound towards low frequency and the Fock cutoff cannot follow it
    (see the README on the low-frequency wall).

    Alongside the QFI this records the Cramer-Rao bound on the strain,

        sigma_h = h_SQL * sqrt(2 / (kappa * F_epsilon_a)) ,

    which is the quantity a noise budget actually wants: it folds in the fact
    that the signal transfer itself carries a factor sqrt(2 kappa).
    """
    f_lo = float(f_at_kappa(kappa_max, params))
    freqs = np.logspace(np.log10(f_lo), np.log10(f_lo * 10 ** decades), n_points)
    print(f"[F] Loss placement vs frequency: {freqs[0]:.0f}-{freqs[-1]:.0f} Hz, "
          f"kappa {float(params.kappa(freqs[0])):.2f}-{float(params.kappa(freqs[-1])):.4f}, "
          f"eta = {eta}, {params.name}")
    placements = (
        ("injection", dict(eta_in=eta), None),
        ("concurrent", dict(eta_ch=eta), CONCURRENT_CUTOFF),
        ("detection", dict(eta_out=eta), None),
    )
    rows = []
    for family in ["squeezed", "cat_even", "fock", "coherent"]:
        for ordering in ("BA1", "BA2", "BA3"):
            best = {}
            for placement, kw, cap in placements:
                sig = []
                for f_hz in freqs:
                    kappa = float(params.kappa(f_hz))
                    hs = float(params.h_sql(f_hz))
                    v, N, ok = qfi(family, nbar, kappa=kappa, ordering=ordering,
                                   max_cutoff=cap, **kw)
                    sigma_h, sigma_rel = strain_uncertainty(v, kappa, hs)
                    sig.append(float(sigma_rel))
                    rows.append([family, ordering, placement, f_hz, kappa, v,
                                 float(sigma_h), float(sigma_rel), N, ok])
                i = int(np.argmin(sig))
                best[placement] = (freqs[i], sig[i])
            print(f"    {family:>12} {ordering}  " + "   ".join(
                f"{p} best {best[p][1]:.3f} h_SQL @ {best[p][0]:.0f} Hz"
                for p, _, _ in placements))
    write_csv(outdir / "frequency_sweep.csv",
              ["state", "ordering", "loss_placement", "f_hz", "kappa",
               "qfi_epsilon_a", "sigma_h", "sigma_h_over_hsql", "cutoff", "converged"],
              rows)


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
    """Phase noise -> back-action -> detection loss, for all three orderings.

    The project plan asks for "phase noise -> BA1/2 -> loss", so all three
    orderings are recorded rather than just the physical one.  Under this
    stage-separated loss model they coincide exactly, and the printed spread is
    the evidence for that rather than an appeal to the separate unit test.
    """
    print(f"[C] Phase noise -> BA(kappa={kappa}) -> detection loss(eta={eta}), all orderings")
    rows = []
    for family in ["squeezed", "cat_even", "coherent", "fock"]:
        per_ordering = {}
        for ordering in ("BA1", "BA2", "BA3"):
            vals = []
            for sigma in sigmas:
                v, N, ok = qfi(family, nbar, kappa=kappa, ordering=ordering,
                               pn_in=float(sigma), eta_out=eta)
                vals.append(v)
                rows.append([family, ordering, float(sigma), v, N, ok])
            per_ordering[ordering] = vals
        spread = max(abs(a - b) for o1 in per_ordering for o2 in per_ordering
                     for a, b in zip(per_ordering[o1], per_ordering[o2]))
        print(f"    {family:>12} " + "  ".join(f"{v:7.4f}" for v in per_ordering["BA3"])
              + f"   max|BAi - BAj| {spread:.1e}")
    write_csv(outdir / "phase_noise.csv",
              ["state", "ordering", "sigma_phi_rad", "qfi_epsilon_a", "cutoff", "converged"], rows)


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
    ap.add_argument("--only", default="",
                    help="comma-separated section letters to run (A,B,B2,C,D,E,F); "
                         "default runs all.  Each section writes its own CSV, so "
                         "re-running one leaves the others in place.")
    args = ap.parse_args()
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    run = (lambda s: True) if not only else (lambda s: s in only)
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
    if run("A"):
        study_orderings(outdir, nbar, [0.5, 1.5] if args.quick else [0.5, 1.5, 3.0])
    if run("B"):
        study_loss_placement(outdir, nbar, kappas, 0.8)
    if run("B2"):
        study_concurrent_orderings(outdir, nbar, [0.5, 1.5, 3.0] if not args.quick else [1.5], 0.8)
    if run("C"):
        study_phase_noise(outdir, nbar, sigmas, kappa=1.0, eta=0.9)
    if run("D"):
        study_states(outdir, nbar, kappas, etas)
    if run("E"):
        study_nbar_scaling(outdir, nbars, configs)
    if run("F"):
        study_frequency(outdir, nbar, 0.8, n_points=6 if args.quick else 12)
    print(f"\nDone in {time.time() - t0:.1f} s -> {outdir}/")


if __name__ == "__main__":
    main()
