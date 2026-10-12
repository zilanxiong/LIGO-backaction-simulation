# reference/

Frozen material kept for comparison only. Nothing in the live code imports it.

| path | what it is |
|---|---|
| `quantum_sensing/` | the earlier in-tree copy of the group's package, with local additions: tight `mesolve` tolerances, CFI `rho` forwarding, `optimize.py`, studies, tests and their data/results. Formerly at the repository root |
| `local-fixes.patch` | the two local fixes (`dynamics.py`, `cfi.py`) as a unified diff against the upstream package, ready to hand to the package maintainer |

The live, importable package is the upstream snapshot in `../quantum-sensing-py/`,
installed by `pip install -r requirements.txt`.

To run the frozen tests against the frozen copy: `pytest reference/quantum_sensing/tests`
(the `conftest.py` here puts this directory first on `sys.path`).
