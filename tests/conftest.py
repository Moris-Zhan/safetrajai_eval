import sys
from pathlib import Path

# Pre-packaging only: the legacy layout is importable solely from the repo root.
# Removed once the project is installed as a package (Phase 2).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
