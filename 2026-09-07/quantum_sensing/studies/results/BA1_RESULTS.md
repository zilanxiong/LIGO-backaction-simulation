# BA1 ordering study: radiation pressure first, then displacement sensing

Channel: `shear(kappa_ba(Omega)) -> signal(epsilon_a) -> detection loss (eta_out = 0.9)`
with the O4-LIGO-calibrated `kappa_ba(Omega) = rp_kappa(Omega)`; fixed `<n> = 2`
for every probe. Produced by `sweep_ba1_frequency.py` (N_max = 800).

## QFI vs frequency (10 Hz - 1 kHz)

- Above ~100 Hz (kappa << 1) every state sits at its loss-degraded plateau
  (the shear is negligible): cat 15.37 > Fock 12.18 > squeezed cat 11.00 >
  optimized(no-BA) 7.94 > coherent 3.60 (= eta * 4/(1+(1-eta)Var...) SQL-like)
  >> squeezed vacuum 0.40.
- Squeezed vacuum is squeezed in x — the epsilon_a generator — so its QFI
  (8 Var(x), degraded by loss) is *small* at every frequency; all its photons
  are spent anti-squeezing the quadrature that doesn't help.
- Below ~50 Hz the growing shear (kappa ~ 1 at 30 Hz, ~9.5 at 10 Hz) mixes
  the anti-squeezed/amplified quadrature into the readout loss and all states
  collapse toward a common back-action-limited floor (~0.3-0.6 at 10 Hz).
- The state ranking is preserved down to ~20 Hz; at 10 Hz the differences
  are largely erased.

## Validation

Coherent and squeezed-vacuum rows agree with the exact Gaussian
covariance-matrix ground truth (`gaussian_qfi_rp`, `signal_order="after"`)
to <= 0.2% everywhere, including the four 10 Hz points the convergence
harness flags as unconverged (the strict output-tail check fails at
kappa = 9.5 with N_max = 800; the QFI values themselves are accurate:
Fock 0.3916 vs exact 0.392427 for the coherent probe).

## <n> scaling at 30 Hz (kappa = 1.05), QFI ~ n^s

| state             | s      | note                                        |
|-------------------|--------|---------------------------------------------|
| coherent          | 0.000  | QFI independent of n (displacement adds no Var(x)) |
| squeezed vacuum   | -0.847 | more squeezing shrinks Var(x) further       |
| even cat          | -0.432 | larger alpha -> more BA-amplified x-spread lost |
| squeezed cat      | +0.166 | mild growth, saturating by n = 10           |
| Fock              | +0.375 | best absolute QFI at n = 10 (16.4)          |
| optimized (no-BA) | +0.442 | steepest fit, but non-monotonic (opt. states were not optimized for this channel) |

No state approaches Heisenberg (s = 2) — the 10% detection loss after the
shear bounds the useful n-scaling, and states optimized without back-action
do not transfer their advantage to the BA1 channel.

## Injection-loss variant (eta_in = 0.9, BEFORE the shear)

Run with `python sweep_ba1_frequency.py injection`. Confirms the analytic
control exactly: with the loss before the shear, everything downstream is
unitary and the shear preserves x, so the QFI is frequency-INDEPENDENT for
every state, 10 Hz - 1 kHz, all points converged:

| state             | QFI (all freqs) | detection-loss high-f plateau |
|-------------------|-----------------|-------------------------------|
| cat               | 17.07           | 15.37                         |
| fock              | 13.54           | 12.18                         |
| squeezed cat      | 12.22           | 11.00                         |
| optimized (no-BA) | 8.83            | 7.94                          |
| coherent          | 4.00            | 3.60                          |
| squeezed vacuum   | 0.44            | 0.40                          |

Notably the coherent state's QFI is exactly 4 (the lossless value):
injection loss shrinks the displacement but not the vacuum covariance, and
the epsilon_a QFI depends only on Var(x). For the non-Gaussian states the
loss does bite (17.07 < lossless 8 Var(x)), but frequency-uniformly.

<n>-scaling exponents at 30 Hz are close to the detection-loss ones
(coherent 0.00, sqz_vac -0.85, cat -0.32, sqz_cat +0.29, fock +0.34,
optimized +0.45): the scaling behaviour is set by how each family spends
photons on Var(x) versus loss-vulnerable structure, not by loss placement.

Placement comparison at fixed total efficiency (0.9): detection loss is
strictly worse at low frequency (e.g. cat at 10 Hz: 0.48 vs 17.07) —
the back-action penalty comes entirely from loss AFTER the shear.

## BA2 orderings (signal first, then shear)

`ba2_detection` and `ba2_injection` configs.  For epsilon_a the signal
generator x commutes with the shear Hamiltonian x^2, so with no loss
between the two unitaries BA2 must equal BA1 — confirmed numerically:
max relative difference over all 78 (state, frequency) points is 7e-12
(detection) and 1.4e-12 (injection).  Ordering matters only with loss
interleaved (see below).

## BA3 physical channel, all three loss slots

`ba3_full` config: eta_in = 0.95 -> [signal + shear + CONCURRENT
eta_ch = 0.99, Strang-split] -> eta_out = 0.90.  The physical LIGO
ordering: signal and back-action act simultaneously, with intracavity
loss during the interaction.

- High-frequency plateaus sit ~4-12% below the BA1 detection-only run
  (cat 10.80 vs 15.37, Fock 9.93 vs 12.18, sqz cat 9.24 vs 11.00,
  optimized 7.36 vs 7.94, coherent 3.58 vs 3.60, sqz vac 0.42 vs 0.40)
  — the extra injection + intracavity loss costs the non-Gaussian states
  far more than the Gaussian ones.
- The low-frequency collapse is unchanged in shape: all states converge
  to ~0.26-0.41 at 10 Hz.
- Coherent / squeezed-vacuum rows agree with the exact Gaussian ground
  truth (signal_order="simultaneous") to <= 0.2% at every frequency,
  including the four 10 Hz points flagged unconverged by the tail check
  (coherent 0.3819 vs exact 0.3826).
- <n>-scaling at 30 Hz: coherent 0.00, sqz_vac -0.84, cat -0.39,
  sqz_cat -0.38, Fock +0.20, optimized +0.40.  Note squeezed cat flips
  sign vs the single-loss runs (+0.17/+0.29 -> -0.38): with loss on both
  sides of the interaction its photon budget stops paying off.

Files: `ba1_frequency_sweep[_injection|_ba2_detection|_ba2_injection|_ba3_full].csv`,
matching `ba1_n_scaling*.csv`, `ba1_qfi_vs_frequency*.png`, `ba1_run*.log`.
