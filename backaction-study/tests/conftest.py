"""Use the study folder's local quantum_sensing copy (self-contained)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
