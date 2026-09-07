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

Files: `ba1_frequency_sweep[_injection].csv`, `ba1_n_scaling[_injection].csv`,
`ba1_qfi_vs_frequency[_injection].png`, `ba1_run[_injection].log`.
