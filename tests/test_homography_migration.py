"""Unit tests for migrating tag-keyed homography data to camera-keyed configs."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from safetrajai.collision.homography.camera_config import load_camera_config
from safetrajai.collision.homography.migration import MigrationConflict, apply_migration, plan_migration

TAG_25 = "C000025 中華三路 民生二路(東北角)-2-20-1280-1280"
TAG_297 = "C000297 一心 凱旋(西北角)-2-20-1280-1280"
YAML_25 = "# 信心度說明：目視估計\ncctv_points:\n  - [518, 348]\nsat_points:\n  - [1, 2]\n"


@pytest.fixture()
def layout(tmp_path: Path) -> dict:
    matrices = tmp_path / "models" / "matrices"
    npy_dir = tmp_path / "models" / "checkpoints"
    checkpoints = tmp_path / "config" / "homography_checkpoints"
    cameras = tmp_path / "config" / "homography"
    for d in (matrices, npy_dir, checkpoints):
        d.mkdir(parents=True)
    for tag in (TAG_25, TAG_297):
        np.save(matrices / f"H_{tag}.npy", np.eye(3))
    (checkpoints / f"{TAG_25}.yaml").write_text(YAML_25, encoding="utf-8")
    (checkpoints / "README.md").write_text("doc", encoding="utf-8")
    cctv = np.array([[[1.25, 2.5]], [[3.0, 4.0]], [[5.125, 6.0]], [[7.0, 8.0]]], dtype=np.float64)
    sat = cctv * 2.0 + 0.1
    np.save(npy_dir / f"check_pts_cctv_{TAG_297}.npy", cctv)
    np.save(npy_dir / f"check_pts_sat_{TAG_297}.npy", sat)
    return {
        "matrices_dir": matrices, "npy_checkpoints_dir": npy_dir,
        "checkpoints_dir": checkpoints, "camera_dir": cameras,
        "cctv": cctv, "sat": sat,
    }


def run(layout: dict) -> list:
    keys = ("matrices_dir", "npy_checkpoints_dir", "checkpoints_dir", "camera_dir")
    return plan_migration(**{k: layout[k] for k in keys})


def snapshot(root: Path) -> dict:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_camera_configs_from_matrices(layout: dict) -> None:
    apply_migration(run(layout))
    cfg = load_camera_config(layout["camera_dir"] / "C000025.yaml")
    assert cfg.matrix_tag == TAG_25
    assert cfg.name == "C000025 中華三路 民生二路(東北角)"
    assert (cfg.satellite.zoom, cfg.satellite.scale, cfg.satellite.size) == (20, 2, (640, 640))
    assert cfg.scale_check == "unchecked"
    assert cfg.latlon is None
    assert (layout["camera_dir"] / "C000297.yaml").exists()


def test_tag_yaml_is_copied_verbatim(layout: dict) -> None:
    apply_migration(run(layout))
    assert (layout["checkpoints_dir"] / "C000025.yaml").read_text(encoding="utf-8") == YAML_25


def test_npy_checkpoints_roundtrip_exactly(layout: dict) -> None:
    apply_migration(run(layout))
    raw = yaml.safe_load((layout["checkpoints_dir"] / "C000297.yaml").read_text(encoding="utf-8"))
    assert np.array_equal(np.asarray(raw["cctv_points"]), layout["cctv"].reshape(-1, 2))
    assert np.array_equal(np.asarray(raw["sat_points"]), layout["sat"].reshape(-1, 2))


def test_sources_are_untouched(layout: dict) -> None:
    sources = {**snapshot(layout["matrices_dir"]), **snapshot(layout["npy_checkpoints_dir"])}
    old_yaml = (layout["checkpoints_dir"] / f"{TAG_25}.yaml").read_bytes()
    apply_migration(run(layout))
    assert {**snapshot(layout["matrices_dir"]), **snapshot(layout["npy_checkpoints_dir"])} == sources
    assert (layout["checkpoints_dir"] / f"{TAG_25}.yaml").read_bytes() == old_yaml


def test_conflicting_target_writes_nothing(layout: dict) -> None:
    layout["camera_dir"].mkdir(parents=True)
    (layout["checkpoints_dir"] / "C000297.yaml").write_text("cctv_points: []\n", encoding="utf-8")
    before = {**snapshot(layout["camera_dir"]), **snapshot(layout["checkpoints_dir"])}
    with pytest.raises(MigrationConflict) as excinfo:
        apply_migration(run(layout))
    assert "C000297.yaml" in str(excinfo.value)
    assert {**snapshot(layout["camera_dir"]), **snapshot(layout["checkpoints_dir"])} == before


def test_rerun_is_noop(layout: dict) -> None:
    apply_migration(run(layout))
    assert all(step.action == "unchanged" for step in run(layout))
