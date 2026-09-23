from pathlib import Path

# src/safetrajai/paths.py -> repo root. Editable install keeps __file__ in the source tree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
