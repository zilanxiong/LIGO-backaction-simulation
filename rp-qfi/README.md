# rp-qfi — QFI with radiation pressure

Working folder for the radiation-pressure QFI study. Contains a snapshot copy
of the mentor's `quantum_sensing` package (snapshot of the repo-root package
as of commit 84c9422, 2026-09-07); all new work for this study happens here.

This copy DIVERGES INDEPENDENTLY from the root `quantum_sensing/` by design:
the root package continues to evolve in parallel and is not re-synced here.
Additions unique to this folder so far: `gaussian.py` (covariance-matrix
ground truth for the RP channel) and `convergence.py` (automated Fock-cutoff
convergence), with tests in `tests/test_gaussian_rp.py` and
`tests/test_convergence.py`.

Run from this directory (or add it to PYTHONPATH) so `import quantum_sensing`
resolves to the local copy.

Data: the 30MB `states.h5` is NOT duplicated here — use
`quantum_sensing.set_data_dir("../quantum_sensing/data")` (relative to this
folder) to load the optimized-state results from the root package copy.
