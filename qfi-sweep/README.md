# QFI sweep: probe states under radiation-pressure back-action

Which fixed-photon-number probe states best survive the combination of
ponderomotive back-action and decoherence, measured by the quantum Fisher
information for a GW-like displacement signal. Output is deliberately in
the format LIGO papers use: minimum detectable signal vs frequency,
log-log, published curves drawn on the same axes.

## Conventions (unit-tested)

- Quadratures `x = a + a†`, `p = -i(a - a†)`, `[x, p] = 2i`, vacuum
  variance 1.
- Signal `S = exp(-i eps x)` — displaces the phase quadrature; `eps` is the
  estimated parameter, `eps_min = 1/sqrt(F_Q)` the quantum Cramér-Rao
  bound on it.
- Back-action `B = exp(-i g x²)` with **g = K/4**, so vacuum input
  reproduces KLMTV's input-output relation `p_out = p_in - K x_in` with the
  same numerical Kimble factor
  `K(Ω) = 2 X γ⁴ / (Ω²(γ²+Ω²))`, `X = 4 P ω₀ /(m L γ³ c)`
  (O4-like defaults: 360 kW, γ = 2π·450 Hz, 40 kg, 3995 m → K = 1 near
  31 Hz). `test_kimble_input_output_relation` enforces this; note the
  factor-2 relative to reading `ba_to_g` naively in these units.
- Loss `D[a]` at transmission `eta`, placed `inj` (before the signal+BA
  block), `det` (after), or `conc` (Lindblad running *during* the same
  evolution — the physical arm-loss case that no sequential ordering
  reproduces).

## What runs

`sweep.py` evaluates probes {coherent, squeezed vacuum, cat, squeezed cat,
Fock} at fixed ⟨n⟩:

- frequency grid 20–1000 Hz at ⟨n⟩ = 2, scenarios {lossless, det 0.9,
  det 0.7, inj 0.9, conc 0.9};
- ⟨n⟩ ∈ {1, 2, 4, 8, 16} at the 100 Hz anchor, {lossless, det 0.9, det 0.7}.

Gaussian probes use the exact covariance method (`gaussian_qfi`, including
a Lyapunov ODE for the concurrent case); non-Gaussian probes use QuTiP
Fock-space evolution with an adaptive cutoff (`basis_size`, inflated by
`K² Var(x)` — the shear pumps the state up). Squeeze angle is optimized
per point via the exact Gaussian QFI; the squeezed cat optimizes its
photon split over {0.25, 0.5, 0.75} and reuses the squeezed-vacuum angle.

## Cross-checks (pytest, 9 tests)

1. KLMTV input-output relation reproduced exactly by the Fock channel.
2. Kimble factor calibration: K = 1 at ~31 Hz.
3. Lossless QFI = 4 Var(x), independent of K (back-action invisible to the
   optimal measurement — the Miao et al. PRL 119, 050801 (2017) statement).
4. Gaussian-exact vs Fock-numerics agreement for all three loss placements.
5. Independent closed form for coherent + shear + detection loss.
6. QFI curve sits at/below the Buonanno-Chen 2001 / KLMTV conventional-
   interferometer fixed-homodyne curve sqrt(1+K²)/2, merging as K → 0.
7. Detection-loss ceiling F_Q ≤ 4/(1-η) approached monotonically.
8. Injection ≠ detection loss; concurrent lies between them.
9. Probe photon budgets exact.

## Outputs (`results/`)

- `results.csv` — one row per (state, ⟨n⟩, frequency, placement, η):
  QFI, eps_min, actual ⟨n⟩, optimal angle/split, method + basis size.
  Any row is independently recomputable (e.g. with the group's Julia code).
- `figA_eps_min_vs_freq.png` — eps_min(Ω) per state, lossless and lossy,
  with the BC2001 conventional-homodyne curve and shot-noise line as rulers.
- `figB_loss_placement.png` — injection vs concurrent vs detection loss for
  the squeezed probe.
- `figC_n_scaling.png` — QFI vs ⟨n⟩ at 100 Hz against the 4/(1-η) ceiling.

Run: `python sweep.py && python figures.py`; verify: `pytest`.

## Deliberate scope cuts (next steps)

- QFI only — homodyne CFI (attainability with a fixed readout) drops into
  `eval_state` later without restructuring.
- Dephasing scenarios excluded from this run (machinery exists in
  `ordering-verification/`).
- Squeezed-cat angle tied to the squeezed-vacuum optimum; full joint
  (split, angle) optimization is a later refinement.
- Detuned/signal-recycled dynamics (complex squeezing) not modeled; that
  extension unlocks recreating Buonanno-Chen figures directly.
