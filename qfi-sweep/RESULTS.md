# Sweep results — first full run

Parameters: O4-like calibration (K = 1 at ~31 Hz), probes at fixed <n>,
grid and conventions per README. All 9 anchor tests pass; every Gaussian
row is computed by the exact covariance method and non-Gaussian rows by
Fock-space QuTiP with adaptive cutoff.

## Headline findings

1. **Lossless: back-action costs nothing, at any frequency.** Every
   state's eps_min(f) is flat and equals 1/sqrt(4 Var(x)) — the Miao et
   al. (2017) QCRB statement. Ranking is pure Var(x) at fixed <n>:
   squeezed vacuum ~ squeezed cat < cat < Fock < coherent
   (0.159, 0.165, 0.224, 0.5 at <n> = 2).

2. **With loss, the low-frequency penalty appears and is state-dependent**
   (fig A, right): 10% detection loss bends every curve up below ~50 Hz
   where K > 1; the non-classical states keep a ~2x advantage over
   coherent across the band but lose their large lossless edge.

3. **Loss placement is a big effect at low frequency** (fig B): for
   squeezed vacuum at <n> = 2, eta = 0.9, the same 10% loss gives
   eps_min = 0.22 (injection, frequency-flat — loss before the unitary
   block cannot know about K), 0.29 (concurrent) and 0.34 (detection) at
   20 Hz; all placements converge above ~100 Hz. Detection-chain loss is
   the expensive kind exactly where radiation pressure is strong.

4. **The squeezed cat collapses onto squeezed vacuum.** The split
   optimizer saturates at its maximum squeezing fraction and matches
   pure squeezed vacuum to <0.01% everywhere tested: for this
   displacement-sensing channel the cat component buys nothing. (Fig C
   shows the two curves exactly coincident — open orange circles under
   the pink line.)

5. **Cat states are fragile at large <n>** (fig C): under detection loss
   the cat's QFI *decreases* with photon number beyond <n> ~ 2 (eta=0.9)
   — the alpha^2-growing coherence is exactly what loss kills — while
   squeezed vacuum climbs monotonically toward the 4/(1-eta) ceiling.
   Fock states plateau early. At eta = 0.7 the large-<n> cat is no better
   than a coherent state.

6. **Nothing beats squeezed vacuum anywhere on this grid.** At fixed <n>,
   maximal x-anti-squeezing wins at every frequency and every loss
   configuration tested — consistent with the loss ceiling being
   approached from below by the Gaussian family. The interesting
   optimal-state question is therefore pushed to (a) fixed *homodyne*
   readout (CFI), where non-Gaussian states may pay off, and (b) the
   detuned/signal-recycled channel.

## Cross-check status

pytest: 9/9 pass (KLMTV input-output relation, K=1 at ~31 Hz, lossless
QFI = 4 Var(x) and K-independent, Gaussian-vs-Fock for all placements,
closed-form coherent+det-loss, below the BC2001 conventional curve,
4/(1-eta) ceiling monotone, inj != det with conc bracketed, photon
budgets). Any CSV row is independently recomputable from the stated
conventions.

## Noise-chain sweep (10 Hz - 1 kHz, chains.csv, figS3-S5)

7. **Back-action-amplified phase noise destroys carrier-carrying probes.**
   With chi = 0.1 dephasing after (or concurrent with) the BA block, the
   coherent state's QFI collapses at low frequency (4 -> 0.05 at 10 Hz,
   K = 9.4): its large mean field <x> != 0 couples to the K^2-amplified
   sheared dephasing operator. Zero-mean states are nearly immune
   (squeezed vacuum 39.6 -> 35.8, cat 36.5 -> 33.0, Fock 20 -> 18.1).
   This is the first genuine ranking inversion mechanism found: phase
   noise + back-action punishes displacement-based probes specifically.

8. **PN placement matters little for zero-mean states** (figS4): for
   squeezed vacuum the pre/conc/post band is only a few percent wide even
   at K = 9.4 - unlike loss, whose placement band is wide (figS2).

9. **Cross-validation:** the pn_pre chain rows reproduce the
   ordering-verification exact-map tables digit-for-digit (coherent
   2.336, sqz 36.044, cat 33.528 at the shared parameters).
