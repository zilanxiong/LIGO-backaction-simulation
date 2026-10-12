"""Make `import quantum_sensing` resolve to reference/quantum_sensing (the frozen
local copy with its extra modules) when running the reference tests, instead of
the installed upstream package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _m in [m for m in sys.modules if m == "quantum_sensing" or m.startswith("quantum_sensing.")]:
    del sys.modules[_m]
