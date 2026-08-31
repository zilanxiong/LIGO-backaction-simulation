"""Make quantum_sensing importable from a fresh clone (no pip install needed)."""

import sys
from pathlib import Path

try:
    import quantum_sensing  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "quantum-sensing-py"))
