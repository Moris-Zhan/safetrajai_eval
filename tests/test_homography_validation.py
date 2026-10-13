"""Unit tests for the camera-keyed homography validation report."""

from pathlib import Path

import numpy as np
import pytest

from safetrajai.collision.homography.camera_config import load_camera_config
from safetrajai.collision.homography.validation import validate_camera

TAG = "C000025 中華三路 民生二路(東北角)-2-20-1280-1280"
TRAIN_CCTV = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


@pytest.fixture()
def dirs(tmp_path: Path) -> dict:
    matrices = tmp_path / "matrices"
    matrices.mkdir()
    np.save(matrices / f"H_{TAG}.npy", np.diag([2.0, 2.0, 1.0]))
    np.save(matrices / f"pts_cctv_{TAG}.npy", TRAIN_CCTV)
    np.save(matrices / f"pts_sate_{TAG}.npy", TRAIN_CCTV * 2 + 0.5)
    cameras = tmp_path / "cameras"
    cameras.mkdir()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    return {"matrices": matrices, "cameras": cameras, "checkpoints": checkpoints, "sat": tmp_path / "sat"}


def camera(dirs: dict, scale_check: str = "passed", latlon: bool = True) -> object:
    body = (
        f"camera: C000025\nmatrix_tag: {TAG}\n"
        "satellite: {provider: google_static, zoom: 20, scale: 2, size: [640, 640]}\n"
        f"scale_check: {scale_check}\n" + ("latlon: [22.6, 120.3]\n" if latlon else "")
    )
    path = dirs["cameras"] / "C000025.yaml"
    path.write_text(body, encoding="utf-8")
    return load_camera_config(path)


def write_checkpoints(dirs: dict, cctv, sat) -> None:
    lines = ["cctv_points:"] + [f"  - [{x}, {y}]" for x, y in cctv] + ["sat_points:"] + [f"  - [{x}, {y}]" for x, y in sat]
    (dirs["checkpoints"] / "C000025.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(dirs: dict, cfg) -> dict:
    return validate_camera(cfg, dirs["matrices"], dirs["sat"], dirs["checkpoints"])


def test_report_has_check_train_and_meter_columns(dirs: dict) -> None:
    cctv = [[10 * i, 5 * i] for i in range(1, 9)]
    sat = [[2 * x + 3.0, 2 * y + 4.0] for x, y in cctv]
    write_checkpoints(dirs, cctv, sat)
    row = run(dirs, camera(dirs))
    assert row["validated"] is True and row["camera"] == "C000025" and row["scale_check"] == "passed"
    assert row["check_N"] == 8 and row["check_excluded"] == 0
    assert row["check_dist_rmse_px"] == pytest.approx(5.0)
    assert row["check_dist_rmse_m"] == pytest.approx(5.0 * row["m_per_pixel"])
    assert row["train_residual_N"] == 4
    assert row["train_residual_rmse_px"] == pytest.approx(np.hypot(0.5, 0.5))


def test_training_points_are_excluded(dirs: dict) -> None:
    cctv = [[0, 0]] + [[10 * i, 7 * i] for i in range(1, 6)]
    sat = [[2 * x, 2 * y] for x, y in cctv]
    write_checkpoints(dirs, cctv, sat)
    row = run(dirs, camera(dirs))
    assert row["check_excluded"] == 1 and row["check_N"] == 5


def test_too_few_checkpoints(dirs: dict) -> None:
    write_checkpoints(dirs, [[10, 10], [20, 30]], [[21, 21], [41, 61]])
    row = run(dirs, camera(dirs))
    assert row["validated"] is False and row["reason"] == "too_few_checkpoints"
    assert row["check_N"] == 2 and row["check_dist_mean_px"] == pytest.approx(np.sqrt(2))


def test_meter_columns_blank_without_scale(dirs: dict) -> None:
    write_checkpoints(dirs, [[10 * i, 3 * i] for i in range(1, 6)], [[20 * i, 6 * i] for i in range(1, 6)])
    row = run(dirs, camera(dirs, scale_check="unchecked"))
    assert row["m_per_pixel"] is None and row["check_dist_rmse_m"] is None
    assert row["check_dist_rmse_px"] == pytest.approx(0.0)


def test_no_checkpoints(dirs: dict) -> None:
    row = run(dirs, camera(dirs))
    assert row["validated"] is False and row["reason"] == "no_checkpoints"
    assert row["train_residual_N"] == 4
