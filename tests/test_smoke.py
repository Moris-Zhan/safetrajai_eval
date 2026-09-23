"""Smoke tests for the packaged layout (see openspec restructure-project-layout, design D10)."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import safetrajai

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = sorted((REPO_ROOT / "config").glob("*.yaml"))
SCRIPTS = sorted((REPO_ROOT / "scripts").rglob("*.py"))
BASE_COMMIT = (REPO_ROOT / "tests" / "golden" / "BASE_COMMIT").read_text().strip()

# Legacy literals from BASE_COMMIT (evaluation/mot_format.py, evaluation/trackeval_runner.py,
# dataset/reid/*.py) that the single-source class constants must reproduce exactly.
LEGACY_NAMES = {0: "person", 1: "car", 2: "scooter", 3: "small_truck", 4: "bus", 5: "truck"}
LEGACY_MOT_CLASS_ID = {"person": 1, "car": 2, "scooter": 3, "small_truck": 4, "bus": 5, "truck": 6}
LEGACY_REID_CLASSES = ["bus", "car", "person", "scooter", "small_truck", "truck"]

FROZEN_TRACKER_METHODS = {
    ("module/tracker/class_aware_botsort.py", "src/safetrajai/tracking/class_aware_botsort.py", "ClassAwareBOTSORT"): (
        "__init__", "update", "remove_duplicate_stracks", "init_track",
    ),
    ("module/tracker/class_aware_bytetrack.py", "src/safetrajai/tracking/class_aware_bytetrack.py", "ClassAwareBYTETracker"): (
        "__init__", "update", "remove_duplicate_stracks",
    ),
}


def package_modules() -> list[str]:
    return sorted(
        info.name for info in pkgutil.walk_packages(safetrajai.__path__, prefix="safetrajai.")
    ) + ["utils.model"]


def scripts_with_cli() -> list[Path]:
    return [path for path in SCRIPTS if "argparse" in path.read_text(encoding="utf-8")]


def method_bodies(source: str, class_name: str) -> dict[str, str]:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: ast.dump(ast.Module(body=node.body, type_ignores=[]))
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.parametrize("module_name", package_modules())
def test_import_package_module(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_import_script_without_running(script: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"smoke_script_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.parametrize("script", scripts_with_cli(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_script_help_exits_zero(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.name)
def test_load_config(config_path: Path) -> None:
    from safetrajai.evaluation.config_loader import load_config

    load_config(config_path)


@pytest.mark.parametrize(
    "config_path",
    [p for p in CONFIG_PATHS if "class_tracking" in (yaml.safe_load(p.read_text(encoding="utf-8")) or {})],
    ids=lambda p: p.name,
)
def test_track_config_class_names_match_contract(config_path: Path) -> None:
    from safetrajai.classes import CLASS_NAMES
    from safetrajai.tracking import config as tracking_config

    tracking_config.load_track_config(config_path)
    assert dict(tracking_config.CLASS_NAMES) == CLASS_NAMES


def test_class_constants_match_legacy_values() -> None:
    from safetrajai.classes import CLASS_NAMES, MOT_CLASS_ID
    from safetrajai.evaluation.mot_format import NAMES
    from safetrajai.evaluation.trackeval_runner import class_name_to_class_id

    assert CLASS_NAMES == LEGACY_NAMES and list(CLASS_NAMES) == list(LEGACY_NAMES)
    assert MOT_CLASS_ID == {name: class_id + 1 for class_id, name in CLASS_NAMES.items()}
    assert MOT_CLASS_ID == LEGACY_MOT_CLASS_ID and list(MOT_CLASS_ID) == list(LEGACY_MOT_CLASS_ID)
    assert NAMES == LEGACY_NAMES
    assert class_name_to_class_id == LEGACY_MOT_CLASS_ID


@pytest.mark.parametrize(
    "script", sorted((REPO_ROOT / "scripts" / "reid_data").glob("*.py")), ids=lambda p: p.name
)
def test_reid_data_class_order_unchanged(script: Path) -> None:
    if "CLASSES" not in script.read_text(encoding="utf-8"):
        pytest.skip("script does not iterate classes")
    spec = importlib.util.spec_from_file_location(f"smoke_reid_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CLASSES == LEGACY_REID_CLASSES


@pytest.mark.parametrize(
    ("old_path", "new_path", "class_name"), list(FROZEN_TRACKER_METHODS), ids=lambda v: str(v).split("/")[-1]
)
def test_tracker_override_bodies_unchanged(old_path: str, new_path: str, class_name: str) -> None:
    old_source = subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:{old_path}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    old_bodies = method_bodies(old_source, class_name)
    new_bodies = method_bodies((REPO_ROOT / new_path).read_text(encoding="utf-8"), class_name)
    for method in FROZEN_TRACKER_METHODS[(old_path, new_path, class_name)]:
        assert method in old_bodies, f"{class_name}.{method} missing at BASE_COMMIT"
        assert new_bodies.get(method) == old_bodies[method], f"{class_name}.{method} body changed"


def test_no_sys_path_mutation() -> None:
    pattern = re.compile(r"\bsys\.path\b")
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "utils"]
    sources = [p for root in roots for p in root.rglob("*.py")] + [REPO_ROOT / "track_eval.py"]
    offenders = [str(p.relative_to(REPO_ROOT)) for p in sources if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_utils_model_is_reexport_shim() -> None:
    import utils.model
    from safetrajai.reid import model

    for name in ("ReIDNet", "CameraEmbedding", "OcclusionRepair"):
        assert getattr(utils.model, name) is getattr(model, name)


QUARANTINED_REID_WEIGHT = REPO_ROOT / "models" / "class_reid_weights" / "temp" / "person" / "best_full_model.pt"


@pytest.mark.skipif(not QUARANTINED_REID_WEIGHT.exists(), reason="quarantined Re-ID weight not present")
def test_quarantined_full_model_weight_still_loads_via_shim() -> None:
    import torch

    reid_model = torch.load(QUARANTINED_REID_WEIGHT, map_location="cpu", weights_only=False)
    assert type(reid_model).__name__ == "ReIDNet"


def camera_role_conflicts(registry: dict[str, dict]) -> set[str]:
    eval_cameras = {cam for entry in registry.values() if entry.get("role") == "eval" for cam in entry.get("cameras", [])}
    source_cameras = {cam for entry in registry.values() if entry.get("role") == "source" for cam in entry.get("cameras", [])}
    return eval_cameras & source_cameras


def test_dataset_registry_roles_are_valid_and_cameras_disjoint() -> None:
    from safetrajai.evaluation.dataset_layout import load_dataset_registry

    registry = load_dataset_registry()
    assert registry, "config/datasets.yaml has no datasets"
    for dataset_id, entry in registry.items():
        assert entry.get("role") in {"eval", "source"}, f"{dataset_id}: role must be eval or source"
        assert entry.get("cameras"), f"{dataset_id}: cameras must be listed"
    assert camera_role_conflicts(registry) == set()


def test_camera_role_conflict_is_detected() -> None:
    registry = {
        "eval_set": {"role": "eval", "cameras": ["C000025"]},
        "source_set": {"role": "source", "cameras": ["C000054", "C000025"]},
    }
    assert camera_role_conflicts(registry) == {"C000025"}


def yaml_string_values(node) -> list[str]:
    if isinstance(node, dict):
        return [s for value in node.values() for s in yaml_string_values(value)]
    if isinstance(node, list):
        return [s for value in node for s in yaml_string_values(value)]
    return [node] if isinstance(node, str) else []


@pytest.mark.parametrize(
    "config_path",
    CONFIG_PATHS + sorted((REPO_ROOT / "tests" / "golden" / "configs").glob("*.yaml")),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_config_does_not_reference_quarantined_assets(config_path: Path) -> None:
    values = yaml_string_values(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    assert [v for v in values if "/temp/" in v or v.startswith("temp/")] == []
