# Backaction-ordering QFI study (BA1/BA2/BA3)

First milestone of the radiation-pressure backaction program: for a defined
single-mode channel, test how the quantum Fisher information (QFI) for a
displacement signal depends on the **ordering** of backaction, signal, loss,
and phase noise, for five input-state families at fixed mean photon number.

This folder is **fully self-contained**: it carries its own copy of the
`quantum_sensing` package, so it can be zipped and run anywhere with just the
dependencies in `requirements.txt` — no install step, no reference to the rest
of the repository. (The copy is a snapshot of `../quantum-sensing-py`, the
package's canonical home for future milestones; the script and tests prefer
the local copy even if another version is pip-installed.)

```
backaction-study/
├── run_study.py             # the full experiment grid + figures (python run_study.py)
├── quantum_sensing/         # local package copy: backaction.py, sld.py,
│                            #   conversions.py, dynamics.py, states.py, gkp.py
├── tests/test_backaction.py # regression tests pinning every analytic anchor (pytest tests/)
├── requirements.txt         # numpy, scipy, qutip, pandas, h5py, matplotlib, pytest
└── results/                 # qfi_results.csv (tidy), fig_*.png, study_log.txt
```

## The channel

Per-frequency, light-only description: the mirror is eliminated and radiation
pressure acts on the optical mode as a unitary, composable with noise stages.
Implemented in `quantum_sensing/backaction.py`.

**Conventions.** Quadratures `X_θ = (a e^{-iθ} + a† e^{iθ})/√2`, so `X_0 = x`,
`X_{π/2} = p`, `[x, p] = i`, vacuum variance 1/2.

| Stage | Map | Implementation |
|---|---|---|
| signal `sig` | `U = exp(-i s X_θsig)` (translates `X_{θsig+π/2}` by −s) | dense `expm` |
| backaction `ba` (shear) | `U = exp(-i (χ/2) X_θba²)`: `x → x`, `p → p − χx` | dense `expm` |
| backaction `ba` (kerr) | `U = exp(-i χ n²)` (self-phase modulation) | dense `expm` |
| simultaneous `basig` | `U = exp(-i (s X_θsig + G_ba))` — BA3, one generator | dense `expm` |
| `loss_in` / `loss_out` | attenuation channel, transmission η | exact Kraus sum, O(N²) per term |
| `pn` | number dephasing, rms `pn` rad | exact `ρ_nm → ρ_nm e^{-pn²(n-m)²/2}` |

The shear is the **linearized ponderomotive** unitary (rotation·squeeze·rotation
with `r = arcsinh(χ/2)`); the Kerr is the **nonlinear** form. With the default
axes (`θ_sig = θ_ba = 0`) the signal displaces the phase quadrature and
backaction is driven by the amplitude quadrature — the GW-like configuration.

The channel is a pure function `get_state_backaction(**params) → ρ`, so the
package's black-box `calculate_qfi(dynamics, param_type="s", ...)` (central
finite difference + SLD eigendecomposition) applies unchanged.

**States** (`make_state`, all at fixed input ⟨n⟩; root-solve where no closed
form exists): coherent `α = √n̄`; squeezed vacuum `sinh²r = n̄`; Fock `|n̄⟩`;
even cat `∝ |α⟩+|−α⟩` with `n̄ = α² tanh α²`; squeezed cat `S(r)(|α⟩+|−α⟩)`
with half of n̄ in squeezing. The squeezed families are anti-squeezed along the
signal generator (`squeeze_angle = π`), the sensing-optimal orientation, so
their lossless baseline is `QFI = 2e^{+2r}`.

**Convergence.** Every QFI value is computed via `calculate_qfi_converged`:
the Fock cutoff grows geometrically until successive values agree to 5×10⁻⁴.
The starting cutoff is shear-aware — the shear is a squeezer and grows ⟨n⟩ by
~`cosh 2r` — and each CSV row records the cutoff actually used (`N_used`).

## The experiment grid (`run_study.py`)

| Exp | Chains | Sweep |
|---|---|---|
| E1 orderings | BA1 `(ba,sig)`, BA2 `(sig,ba)`, BA3 `(basig)` | χ, lossless, ⟨n⟩=4 |
| E1b off-axis | BA1 with `θ_sig = π/2` (generator p) | χ (shear) |
| E2 loss placement | injection `(loss,ba,sig)` vs detection `(ba,sig,loss)` | χ at η=0.8 |
| E3 pn chain | `(pn,ba,sig,loss)` vs `(pn,sig,ba,loss)` | pn at η=0.9, fixed χ |
| E4 scaling | lossless / Kerr-BA1 / shear+detection-loss | ⟨n⟩ ∈ {1,2,4,8,16} |
| E5 linearization bridge | Kerr `(ba,)` vs shear congruence at matched strength | χ_eff = 4χ_K⟨n⟩ |

## What the results mean (analytic anchors, all pinned in `tests/`)

1. **BA2 is always invisible to QFI.** Any parameter-independent unitary
   applied *after* the signal cannot change QFI (unitary invariance). BA2
   curves are exactly flat for both backaction forms — a structural theorem,
   not a finding about backaction being harmless.
2. **The shear is invisible even in BA1/BA3 for the GW-like axis.** The signal
   generator `X_0` commutes with the shear generator `X_0²`, so lossless QFI
   is χ-independent in every ordering. This is the single-mode statement that
   backaction at one frequency is fully evadable (variational readout): the
   *optimal* measurement loses nothing. It would NOT hold for fixed-quadrature
   homodyne CFI — that is where the SQL lives (next milestone).
3. **Backaction becomes real when something breaks the commutation:**
   - *Off-axis signal* (`θ_sig = π/2`, generator `p`, `[p, x²] ≠ 0`): BA1 QFI
     changes with χ; vacuum reference `QFI = 2(1+χ²)` (backaction *helps* here —
     ponderomotive anti-squeezing of the measured quadrature).
   - *Kerr backaction* (`[X_0, n²] ≠ 0`): BA1 and BA3 split from BA2 even
     lossless; BA3 is **not** bounded by BA1/BA2 in this channel.
   - *Detection loss* (loss after BA+signal): Gaussian closed form for vacuum,
     `QFI = 2η / (1 + η(1−η)χ²)` — the χ-dependent factor is the backaction
     penalty, maximal at η = 1/2, absent at η ∈ {0, 1}.
   - *Injection loss* (loss before BA): **no** χ-dependent penalty for
     coherent/vacuum inputs — loss placement is physics, not bookkeeping.
4. **The linearization bridge (E5) quantifies where the per-frequency shear
   description breaks.** Expanding the Kerr `n²` around a coherent carrier
   gives the shear with `χ_eff = 4χ_K⟨n⟩` (plus a rotation and displacement
   that drop out of covariance eigenvalues), so we compare the largest
   quadrature-covariance eigenvalue — the ponderomotive anti-squeezing
   magnitude, rotation-invariant — of the exact Kerr output against the
   linearized prediction `eig_max(S C_in Sᵀ)`, `S = [[1,0],[−χ_eff,1]]`.
   For a coherent carrier the error is tiny and falls ∝1/⟨n⟩ at fixed χ_eff
   (0.1% at χ_eff = 0.2, ⟨n⟩ = 2; 10⁻⁴ by ⟨n⟩ = 32) — the classical-carrier
   limit where the frequency-domain description becomes exact. Non-Gaussian
   states break it early and at *first order* in χ_eff: Fock (no carrier at
   all) is already 5% off at χ_eff = 0.05, and squeezed/cat families reach
   O(10%) by χ_eff ≈ 0.2–0.4 and O(1) beyond — the single-mode answer to
   "where does the independent-frequency linearized channel description
   break for non-Gaussian states". See `results/fig_bridge.png` and
   `results/bridge_results.csv`.
5. **Displacement sensing is powered by generator variance, not amplitude.**
   Lossless `QFI = 4 Var(X_θsig)`: coherent stays at 2 for all ⟨n⟩; Fock gives
   `2(2n+1)`; anti-squeezed vacuum `2e^{2r}` ≈ 8⟨n⟩ (Heisenberg-like); cat and
   squeezed-cat sit between. The ⟨n⟩-scaling under backaction+loss (E4) is the
   state-comparison question from the project notes.

## Reproducing

```bash
pip install -r requirements.txt
python run_study.py               # ~20–40 min; writes results/
pytest tests/                     # analytic-anchor regression suite
```
