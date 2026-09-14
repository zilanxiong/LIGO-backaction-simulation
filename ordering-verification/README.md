# Ordering verification: signal vs. back-action vs. noise

Numerical verification that the ordering of signal/displacement with back-action doesn't matter even with loss incorporated.

## The claim

The GW signal is the unitary
`S = exp(-i eps x)` with generator `x = a + a†` (it creates
phase-quadrature fluctuations), and radiation-pressure back-action is
`B = exp(-i g x²)`. Since `[x, x²] = 0`:

1. **S and B commute** — swapping them wherever they are *adjacent* leaves
   the output state, and hence the QFI for `eps`, unchanged.
2. That does **not** make the signal's position relative to **loss**
   irrelevant. Loss `D[a]` commutes with neither `S` nor `B`, so moving
   either block across the loss changes the channel: loss after the signal
   damps the displacement; loss before it does not.

Prediction: the 6 orderings of `{S, B, L}` collapse into exactly 4
equivalence classes

```
{SBL, BSL}   {LSB, LBS}   {SLB}   {BLS}
```

with within-class agreement at numerical precision.

## What `verify_ordering.py` does

For probes (coherent, squeezed vacuum, even cat) at fixed `<n> = 2`, it
computes the QFI for `eps` under all 6 orderings, for
`kappa_ba ∈ {0.5, 1.0}`, loss `eta ∈ {0.9, 0.7}` and dephasing
`chi = 0.1`, plus controls:

- **No decoherence** (`eta = 1`): all orderings agree and equal `4 Var(x)`
  of the probe, independent of `kappa_ba` — back-action alone is invisible
  to the optimal measurement; only decoherence makes it bite.
- **`kappa_ba = 0`**: collapse to 2 classes (signal before/after loss).
- **Dephasing table**: same 6 permutations with `D[n]` in place of loss.
- **SLD cross-check**: direct QFI sum vs `Tr[rho L²]`.
- **Exact Gaussian reference**: for coherent and squeezed probes every
  block is Gaussian, so the QFI is computed exactly from
  `F = u^T V^{-1} u` (covariance propagation) and used as the truth
  anchor for the Fock-basis numbers.

## Numerical caveats found (and why they matter for the sweep design)

- **Units/calibration**: with `x = a + a†` (`[x, p] = 2i`, vacuum
  `Var = 1`), the unitary `exp(-i (kappa/2) x²)` shears
  `p -> p - 2 kappa x` — a factor 2 vs. the naive reading of `ba_to_g`.
  Ordering conclusions are unaffected, but the `kappa_ba` calibration to
  LIGO's `rp_kappa(Omega)` must pin this convention down.
- **Truncation**: the shear inflates `Var(p)` by `(2 kappa)² Var(x)`, so
  the Fock basis needed grows fast with `kappa`. At `<n> = 2`,
  `kappa = 1` converges at `N ≈ 200`; `kappa = 2` needs `N ≈ 700` for
  1e-3 accuracy. The exact Gaussian reference makes this cheap to detect;
  any future sweep should use it as a per-point convergence guard.

## Files

- `verify_ordering.py` — the whole verification, self-contained
  (`python ordering-verification/verify_ordering.py`, needs `qutip`).
- `results/SUMMARY.md` — human-readable pass/fail report.
- `results/ordering_qfi.csv` — all QFI values (with exact Gaussian
  reference where applicable).

## Consequence for the study design

Signal↔back-action order is never a variable; only the position of each
decoherence element relative to the `S,B` block matters. The sweep's
scenario space is therefore: {loss, dephasing} placement ∈
{before, concurrent, after} × strength × frequency-dependent
`kappa_ba(Omega)`, with `S` and `B` kept adjacent by convention — and one
redundant ordering pair retained in the test suite as a regression check.
