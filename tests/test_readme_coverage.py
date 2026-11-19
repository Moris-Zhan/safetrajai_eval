"""README and the command index must mention every script and scenario config (spec readme-command-coverage).

Only reads README.md, Markdown/command-index.md and the file names under scripts/ and config/;
never touches dataset/, models/, runs/.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED_FILES = ("README.md", "Markdown/command-index.md")
README_TEXT = "\n".join((REPO_ROOT / name).read_text(encoding="utf-8") for name in DOCUMENTED_FILES)

# file name -> reason it is intentionally not documented in README or the command index.
EXEMPT: Mapping[str, str] = {
    "main.py": "Nuclio function entry point for CVAT auto-annotation, not a CLI",
    "model_handler.py": "Nuclio function helper for CVAT auto-annotation, not a CLI",
}


def entry_point_names() -> list[str]:
    scripts = sorted(path.name for path in (REPO_ROOT / "scripts").rglob("*.py") if path.name != "__init__.py")
    config_dirs = ("", "tracking", "grid_search", "reid", "collision")
    configs = sorted(path.name for d in config_dirs for path in (REPO_ROOT / "config" / d).glob("*.yaml"))
    return scripts + configs


def find_undocumented(readmeText: str, names: list[str], exempt: Mapping[str, str]) -> list[str]:
    blankReasons = sorted(name for name, reason in exempt.items() if not reason.strip())
    if blankReasons:
        raise ValueError(f"EXEMPT entries need a reason: {blankReasons}")
    return [name for name in names if name not in exempt and name not in readmeText]


def test_readme_mentions_every_script_and_config() -> None:
    missing = find_undocumented(README_TEXT, entry_point_names(), EXEMPT)
    assert not missing, f"README.md does not mention {len(missing)} entry points: {missing}"


def test_exempt_entry_without_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="need a reason"):
        find_undocumented("", ["a.py"], {"a.py": "  "})


def test_exempt_entry_is_not_reported_and_documented_entry_is_not_flagged() -> None:
    missing = find_undocumented("see b.py", ["a.py", "b.py", "c.py"], {"a.py": "internal helper"})
    assert missing == ["c.py"]
