"""Smoke tests for the legacy (pre-restructure) layout."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = sorted((REPO_ROOT / "config").glob("*.yaml"))


def project_modules() -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(
        path[: -len(".py")].replace("/", ".").removesuffix(".__init__")
        for path in tracked
        if not path.startswith(("third_party/", "src/", "tests/"))
    )


@pytest.mark.parametrize("module_name", project_modules())
def test_import_project_module(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.name)
def test_load_config(config_path: Path) -> None:
    from evaluation.config_loader import load_config

    load_config(config_path)


@pytest.mark.parametrize(
    "config_path",
    [p for p in CONFIG_PATHS if "class_tracking" in (yaml.safe_load(p.read_text(encoding="utf-8")) or {})],
    ids=lambda p: p.name,
)
def test_load_track_config(config_path: Path) -> None:
    from module.tracker.config import CLASS_NAMES, load_track_config

    load_track_config(config_path)
    assert CLASS_NAMES
