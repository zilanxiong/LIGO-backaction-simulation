"""Put the group's ``quantum_sensing`` package on the import path for the tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "quantum-sensing-py"))
