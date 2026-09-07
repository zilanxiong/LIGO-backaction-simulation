"""Make this folder's flat modules importable, so it runs standalone."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
