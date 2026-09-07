# rp-qfi — QFI with radiation pressure

Working folder for the radiation-pressure QFI study. Contains a snapshot copy
of the mentor's `quantum_sensing` package (copied from repo root at commit
d04cbfa); all new work for this study happens here, leaving the root package
untouched.

Run from this directory (or add it to PYTHONPATH) so `import quantum_sensing`
resolves to the local copy.

Data: the 30MB `states.h5` is NOT duplicated here — use
`quantum_sensing.set_data_dir("../quantum_sensing/data")` (relative to this
folder) to load the optimized-state results from the root package copy.
