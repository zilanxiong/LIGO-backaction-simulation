# Findings — Gaussian verification track

> The back-action study run on the group's own `quantum_sensing` channel is in
> [`results/backaction/RESULTS.md`](results/backaction/RESULTS.md).  This
> document covers the independent `ligo_backaction` track, which additionally
> carries the interferometer calibration, the broadband cost metrics and the
> two-frequency additivity sandbox.  The two agree on every result they share.

Answers this code gives to the questions in the project plan. Every number
below comes from `results/*.csv`, reproduced by `python scripts/run_study.py`.
Unless stated otherwise: mean photon number `nbar = 2`, ordering `BA1`,
detection efficiency as labelled, Fock cutoffs chosen by the automated
convergence checker (all 126 points of the state scan converged).

---

## 1. Radiation pressure alone costs no information at all

At `eta = 1` the QFI is **exactly independent of `kappa`** for every probe
state tested — coherent, squeezed, Fock, even/odd cat, squeezed cat — to the
last digit the solver produces (`results/states_vs_kappa.csv`):

| state | `kappa = 0` | `kappa = 1` | `kappa = 3` |
|---|---|---|---|
| coherent | 2.000 | 2.000 | 2.000 |
| Fock | 10.000 | 10.000 | 10.000 |
| even cat | 18.261 | 18.261 | 18.261 |
| squeezed | 19.797 | 19.798 | 19.798 |

This is not a numerical coincidence, and it is worth being explicit about
because it sets up everything else. The back-action generator is `x²` and the
signal generator is `x`; they commute. So

* `U_BA` applied *after* the signal is a parameter-independent unitary and
  cannot change the QFI;
* `U_BA` applied *before* the signal is absorbed into the generator, because the
  QFI of the family `e^{i lambda x} (U rho U†) e^{-i lambda x}` equals the QFI of
  `rho` with generator `U† x U = x`. (For a pure state this is just the statement
  that the shear leaves `4 Var(x)` alone; the argument holds for mixed states
  too, which is why the injection-loss column of section 2 is exactly flat.)

**Consequence for the BA1/BA2/BA3 question:** in the physical configuration
(signal in the phase quadrature) all three orderings are *exactly* equal, with
or without loss and phase noise — `results/orderings.csv` gives agreement to
8e-15 for squeezed vacuum and 4e-15 for the even cat. BA1 and BA2 do not
merely bound BA3; they coincide with it.

The bounding picture the plan describes is the *non-commuting* case. Rotating
the signal into the amplitude quadrature (`phi_sig = pi/2`, not physical for a
free mass but a clean test of the algebra) makes the orderings differ by more
than an order of magnitude, and BA3 does land between them:

| state | no BA | BA1 | BA2 | BA3 |
|---|---|---|---|---|
| squeezed | 0.200 | 9.780 | 0.201 | 2.541 |
| even cat | 1.584 | 7.360 | 1.611 | 2.813 |

**So: radiation pressure only destroys information in combination with
something else.** Everything below is about what that something else is.

---

## 2. It is *detection* loss that makes back-action bite — injection loss does not

`results/loss_placement.csv`, `eta = 0.8`:

| | `kappa = 0` | `kappa = 1` | `kappa = 3` |
|---|---|---|---|
| squeezed, injection loss | 7.122 | 7.122 | 7.122 |
| squeezed, detection loss | 5.698 | 3.364 | 0.786 |
| even cat, injection loss | 4.659 | 4.659 | 4.659 |
| even cat, detection loss | 3.727 | 2.183 | 0.680 |

Injection loss gives a QFI that is *exactly flat* in `kappa`: loss before the
interaction degrades the input, and the shear that follows is still a
QFI-preserving unitary. Detection loss falls by a factor of 7 across the same
range. For vacuum input the Gaussian track gives the closed form (a unit test):

```
F(BA -> signal -> detection loss) = 2 eta / (1 + eta (1 - eta) kappa²)
```

The `eta(1-eta)` factor is the signature: the damage is maximal at 50 % loss
and vanishes at both `eta = 1` and `eta = 0`. Physically, the shear writes the
probe's `x` fluctuations into `p`; loss then mixes in vacuum that is *not*
correlated with them, and the correlation that would have made the extra `p`
noise harmless is destroyed.

**Practical reading:** for back-action, squeezing losses on the readout side
are qualitatively worse than losses on the injection side, and the usual
lumping of "total loss" into one number hides the effect.

---

## 3. Are phase-space-compact states better under radiation pressure? Yes, once there is loss

This is the sharpest result in the study, and it needs the question posed at
*fixed photon number* to be well posed. At fixed `nbar` a Gaussian probe has
exactly one free knob — the squeeze *angle* (a coherent displacement does not
change the covariance and so does not change the QFI, so every photon should go
into squeezing). Scanning that angle (`results/optimal_extent.csv`, `nbar = 2`,
i.e. 9.96 dB available):

| `eta` | `kappa = 0` | `kappa = 1` | `kappa = 3` | `kappa = 5` |
|---|---|---|---|---|
| 1.00 | 0.0°, 9.96 dB | 0.0°, 9.96 dB | 0.0°, 9.96 dB | 0.0°, 9.96 dB |
| 0.95 | 0.0°, 9.96 dB | 19.0°, 9.47 dB | 45.8°, 6.88 dB | 59.8°, 4.13 dB |
| 0.90 | 0.0°, 9.96 dB | 27.8°, 8.91 dB | 57.5°, 4.67 dB | 69.0°, 1.33 dB |
| 0.80 | 0.0°, 9.96 dB | 35.5°, 8.19 dB | 65.0°, 2.67 dB | 74.3°, **−0.85 dB** |
| 0.50 | 0.0°, 9.96 dB | 42.2°, 7.38 dB | 69.8°, 1.05 dB | 77.5°, **−2.52 dB** |

(angle, then the resulting phase-space extent along `x` in dB relative to
vacuum.)

Without loss the answer is "as extended as possible" — anti-squeeze `x` to the
hilt. With loss the optimum rotates away monotonically with `kappa`, and beyond
`kappa ~ 4` at 20 % loss it goes **negative**: the best probe is *squeezed*
along the quadrature that drives the back-action, i.e. phase-space compact,
even though that throws away most of the naive `4 Var(x)` advantage.

The state scan says the same thing without any Gaussian assumption. At
`eta = 0.7` (`results/states_vs_kappa.csv`) the ranking inverts as `kappa` grows:

| state | `4 Var(x)` | `kappa = 0` | `kappa = 1` | `kappa = 3` |
|---|---|---|---|---|
| squeezed | 19.80 | **3.776** | **2.129** | 0.473 |
| even cat | 18.26 | 2.184 | 1.425 | 0.478 |
| Fock | 10.00 | 2.372 | 2.050 | **0.914** |
| coherent | 2.00 | 1.400 | 1.157 | 0.484 |

The Fock state — by far the most phase-space-compact of the high-QFI probes —
starts 1.6× *behind* squeezed vacuum and ends 1.9× *ahead* of it. The crossing
is near `kappa ~ 1.5`. Nothing about the Fock state is special here except its
small `Var(x)`: what is being rewarded is compactness along the back-action
quadrature.

**Answer: at fixed photon number there is an optimal, finite phase-space extent
along the back-action quadrature, and it shrinks as `kappa` and loss grow.
"More squeezing is better" is only true in the lossless limit.**

---

## 4. How do states optimised without back-action perform with it?

The exact no-back-action optimum at fixed `nbar` (maximise `4 Var(x)` subject
to `<n> = nbar`) is squeezed vacuum with `sinh²r = nbar`, giving
`F = 2(2nbar + 1 + 2 sqrt(nbar(nbar+1)))` — the code recovers it to 1e-5 from a
Lagrange-multiplier eigenproblem, and it is a unit test.

Pushed through a real channel it is no longer optimal, but re-optimising is
worth at most tens of percent, and **every bit of that gain is a rotation**
(`results/optimization.csv`, `nbar = 2`, `eta = 0.9`):

| `kappa` | no-BA optimum | best Gaussian (rotated) | free 16-mode search | gain vs no-BA | gain vs Gaussian |
|---|---|---|---|---|---|
| 0.0 | 9.428 | 9.428 | 9.428 | 1.000 | 1.000 |
| 0.5 | 8.347 | 8.875 | 8.875 | 1.063 | 1.000 |
| 1.5 | 4.354 | 6.039 | 6.039 | **1.387** | 1.000 |
| 3.0 | 2.357 | 2.906 | 2.935 | 1.245 | 1.010 |

The "free" column is a local L-BFGS search over 16 complex Fock amplitudes,
seeded both from the rotated Gaussian optimum and from an even cat. At
`kappa <= 1.5` it never improves on its Gaussian seed at all (the reported
optimum *is* the seed); at `kappa = 3` it finds 1 % more, with 0.979 overlap on
the Gaussian state — i.e. still essentially Gaussian. This is a local search, so
it cannot rule out a distant non-Gaussian optimum, but it does say that none
sits in the basin around the physically motivated seeds.

**Answer: previously optimised (no-back-action) states remain good probes; the
correction that matters is a frequency-dependent rotation of the input, not a
different, more exotic state.** That is the input-side analogue of the
filter-cavity trick, and it is a much cheaper fix than redesigning the state.

---

## 5. A fixed broadband state versus frequency-dependent squeezing

`results/broadband_metrics.csv`, `nbar = 2`, detection `eta = 0.9`, 12
frequencies from 200 Hz to 2 kHz, weight `w(f) ~ f^(-4/3)`:

| state | (i) `∫ w/F df` (lower better) | (ii) worst-case `F_h` | (iii) `Σ F_h` |
|---|---|---|---|
| squeezed | **4.90e−49** | **5.19e+47** | 3.11e+49 |
| opt. (no BA) | 4.89e−49 | 5.19e+47 | **3.12e+49** |
| squeezed cat | 4.91e−49 | 5.18e+47 | 3.11e+49 |
| even cat | 6.27e−49 | 4.23e+47 | 2.42e+49 |
| odd cat | 6.37e−49 | 4.15e+47 | 2.38e+49 |
| Fock | 7.07e−49 | 3.35e+47 | 2.21e+49 |
| coherent | 2.25e−48 | 0.99e+47 | 0.75e+49 |

All three metrics rank the states identically here — the curves in
`fig5_broadband` are close to parallel, so there is no flatness-versus-total
tension to exploit at this operating point. Metric (ii) only starts to
discriminate when a probe is optimised *per frequency*, which is exactly what
section 3 shows the input rotation doing.

Two honest caveats. First, these are *bounds*: the QCRB assumes the optimal
measurement, so comparing them with a real interferometer's noise curve compares
a bound with an achievable readout. In this band the bound crosses `h_SQL`
around 400-600 Hz (fig 5) — below that every probe beats the SQL, above it none
does, because `kappa` has fallen and the signal transduction with it. Second, `kappa ~ 1/Omega²` diverges at low
frequency, so the Fock track cannot reach the 10–100 Hz band at all (see below);
that band is exactly where back-action dominates in a real detector, and it is
reachable only on the Gaussian track.

---

## 6. Phase noise reverses the ranking

`results/phase_noise.csv`, input phase noise `sigma_phi` then `kappa = 1` then
`eta = 0.9`:

| `sigma_phi` (rad) | 0.00 | 0.10 | 0.15 | 0.20 | 0.35 | 0.50 |
|---|---|---|---|---|---|---|
| squeezed | **6.211** | **5.058** | **4.365** | 3.825 | 2.919 | 2.528 |
| even cat | 4.566 | 4.407 | 4.257 | **4.098** | **3.718** | **3.464** |
| coherent | 1.651 | 1.550 | 1.442 | 1.317 | 0.971 | 0.767 |

The even cat overtakes squeezed vacuum at about **0.17 rad (10°)** of input
phase noise, and by 0.5 rad it is 37 % ahead. Squeezed vacuum loses 59 % of its
QFI over that range; the cat loses 24 %. Phase noise rotates the anti-squeezed
axis into the squeezed one, which costs a squeezed state its entire advantage,
whereas the cat's information sits in a discrete superposition that a small
random rotation smears much more slowly.

**This is the one place in the study where a non-Gaussian probe wins outright**,
and it argues for reporting cat performance against a phase-noise budget rather
than against loss alone.

---

## 7. Where does the independent single-frequency description break?

`results/twomode_additivity.csv` compares the exact joint two-mode QFI matrix
with the per-frequency prediction (each sideband traced out, propagated through
its own single-frequency channel, QFIs summed), at `eta = 0.8`:

| case | joint / per-frequency − 1 | max off-diagonal `F_ij` |
|---|---|---|
| product squeezed, no cross term | ~1e−9 | ~1e−16 |
| product cat, no cross term | ~1e−9 | ~1e−16 |
| product squeezed, cross-frequency BA | +1.5 % | 1.18 |
| product cat, cross-frequency BA | +3.6 % | 0.89 |
| two-mode squeezed input | +78 % | 1.02 |
| entangled cat input | +26 % | 0.43 |

The result is sharper than "Gaussian additive, non-Gaussian not". **Being
non-Gaussian is not what breaks additivity** — a product of two *cats* through
two independent channels is additive to nine digits, exactly like a product of
squeezed states. Two things break it:

1. **cross-frequency back-action.** The radiation-pressure force is quadratic in
   the total field, so two sidebands beat against each other and produce an
   `x1 x2` term at `Omega1 - Omega2` that the per-frequency linearisation drops.
   Restoring it moves the answer by 1.5 % for Gaussian probes and **2.4× more
   (3.6 %) for cats** — so the non-Gaussianity amplifies the error, it does not
   cause it.
2. **cross-frequency correlations in the probe.** A frequency-multiplexed
   preparation — anything entangled across sidebands — makes the per-frequency
   sum wrong by tens of percent, and puts large off-diagonal terms in the QFI
   matrix that the independent description says are zero by construction.

**Answer: the per-frequency sum is safe for a probe prepared independently at
each sideband, Gaussian or not. It fails as soon as the preparation is
broadband-correlated, and it acquires a non-Gaussianity-amplified error from the
beat term whenever the sidebands are strongly coupled.** Since the interesting
non-Gaussian broadband sources are precisely the correlated ones, the joint
treatment is not optional for them.

The two-mode module is a deliberately minimal toy — it locates the breakdown,
it does not model a real broadband response.

---

## Numerical caveats worth carrying forward

* **The Fock cutoff is the limiting resource.** The shear inflates the output
  photon number by roughly `(1 + kappa²)(nbar + 1)`, and the truncated `x²`
  exponential reflects amplitude off the cutoff. At `kappa = 3`, `nbar = 2`,
  reaching 1e-5 accuracy needs `N ~ 560`. Every scan point here went through the
  convergence checker and all 126 converged. The coupling range of each Fock-track
  scan was chosen so that this holds: at `kappa ~ 5` even `N = 700` is still ~1 %
  from the Gaussian reference, so that regime is not reported on the Fock track
  at all.
* **The low-frequency wall is real.** `kappa ~ 1/Omega²`, so at the aLIGO-like
  calibration `kappa(100 Hz) ~ 48` at full power and no reachable cutoff
  converges. The broadband study runs at quarter power over 200 Hz – 2 kHz
  (`kappa <= 2.7`). Extending the non-Gaussian analysis into the back-action
  dominated band below 100 Hz needs a different representation — a Gaussian +
  perturbative-correction scheme, or a phase-space method — not a bigger cutoff.
* The QFI is an upper bound over all measurements. Nothing here says a
  homodyne, or any realisable readout, achieves it.

## Suggested next steps

1. **Frequency-dependent input rotation.** Section 3 says the whole re-optimisation
   gain is an angle. Compute the optimal `theta(Omega)` across a real band and
   compare it with the filter-cavity rotation used for FD squeezing on the
   readout side — they optimise different things and should not coincide.
2. **Phase-noise budget for cats.** Section 6's crossover at 0.17 rad is the
   number that decides whether a cat is worth preparing. Map it against `nbar`,
   `kappa` and loss.
3. **Below the Fock wall.** The 10–100 Hz band is where back-action dominates and
   where the Fock track cannot go. This is the main methodological gap.
4. **Three-mode check.** The two-mode toy shows the beat term matters at the
   percent level; whether that accumulates or cancels across a dense frequency
   grid is not answerable with two modes.
