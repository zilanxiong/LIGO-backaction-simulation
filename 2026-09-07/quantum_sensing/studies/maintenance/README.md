# One-shot data repair scripts (historical record)

These scripts were each run ONCE, from the repository root, to repair
specific rows of the committed sweep CSVs in `../results/`. They are kept
for provenance and reproducibility — their effects are already in the
CSVs, and each invocation is recorded in the corresponding
`../results/*.log`. Re-running them is harmless but unnecessary
(they are idempotent against already-repaired data, except where a
convergence re-run would simply reproduce the same values).

- `patch_sqzp.py` — replaced truncation-corrupted `sqz_vac_p` rows
  (anti-squeezed state under kappa ~ 10 needs N >~ 1500; the N_max = 800
  Fock values were wrong) with EXACT Gaussian covariance-matrix values in
  the loss-only configs (detection, ba2_detection, ba3_full). Patched rows
  are marked `N_basis = 0` (= analytic). A follow-up inline pass (same
  criterion, `converged -> True` where |Fock - Gaussian|/Gaussian <= 1e-3)
  marked the accurate-but-flagged 14.7 Hz rows as Gaussian-validated.

- `pn_highN.py` — re-ran the two `sqz_vac_p` low-frequency points of the
  `pn_ba1_detection` config (non-Gaussian channel, no analytic shortcut)
  at N_max = 1700 and wrote the results into the CSV.

- `paper_highN.py` — re-ran the ten flagged low-frequency points of the
  `paper_pn_ba1` config (n = 5) at N_max = 1800 after the N_MAX-override
  ordering bug (fixed in sweep_ba1_frequency.py) had capped the first
  pass at N = 800.

Note: these scripts predate the optimized-state orientation fix in
`probes.py`; the `opt_fock_sup` / `opt_matched` rows they touched were
subsequently re-scored wholesale via
`sweep_ba1_frequency.py <config> only=<state>`.
