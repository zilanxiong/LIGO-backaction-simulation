# Study outputs

Regenerate everything with `python scripts/run_study.py` (about 4 minutes).
Each figure has a CSV beside it holding the numbers behind it. Figures come in
a light and a `.dark` variant.

| output | contents |
|---|---|
| `benchmark_curves.csv`, `fig1_benchmark_curves` | Gaussian benchmarks: SQL curve, frozen-angle and frequency-dependent squeezing, and the vacuum QCRB (= variational readout) |
| `states_vs_kappa.csv`, `fig2_states_vs_kappa` | QFI of each probe family versus `kappa` at fixed `nbar`, for three detection efficiencies; includes the cutoff used and its convergence flag |
| `loss_placement.csv`, `fig3_loss_placement` | injection loss versus detection loss |
| `optimal_extent.csv`, `fig4_optimal_extent` | optimal input squeeze angle and the resulting phase-space extent along `x`, at fixed `nbar` |
| `broadband_qfi.csv`, `broadband_metrics.csv`, `fig5_broadband` | strain QCRB across the band and the three cost metrics |
| `orderings.csv` | BA1 / BA2 / BA3 for the physical and the rotated signal quadrature |
| `phase_noise.csv`, `fig6_phase_noise` | input phase noise into radiation pressure into detection loss |
| `optimization.csv` | no-back-action optimum versus the best rotated Gaussian versus a free Fock-amplitude search |
| `twomode_additivity.csv`, `fig7_twomode_additivity` | joint two-mode QFI against the per-frequency sum |

Every CSV that involves the Fock track carries a `converged` column from the
automated cutoff check. Do not use a row where it is `False`.
