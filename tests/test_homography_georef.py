"""Unit tests for camera-keyed homography config, resolution and georeferencing."""

from pathlib import Path

import numpy as np
import pytest

from safetrajai.collision.homography.camera_config import (
    load_camera_config,
    load_camera_config_for_sequence,
)
from safetrajai.collision.homography.points import (
    assert_no_legacy_point_files,
    resolve_camera_homography,
)

SEQ_C000025 = "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1"
TAG_C000025 = "C000025 中華三路 民生二路(東北角)-2-20-1280-1280"
SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]
SQUARE_X2 = [[0, 0], [200, 0], [200, 200], [0, 200]]


def write_camera(camera_dir: Path, camera: str, body: str) -> Path:
    camera_dir.mkdir(parents=True, exist_ok=True)
    path = camera_dir / f"{camera}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


FULL_CONFIG = f"""\
camera: C000025
name: C000025 中華三路 民生二路(東北角)
matrix_tag: {TAG_C000025}
satellite: {{provider: google_static, zoom: 20, scale: 2, size: [640, 640]}}
latlon: [22.625869, 120.297201]
latlon_source: osm_overpass
latlon_date: '2026-10-13'
scale_check: passed
"""


def test_load_full_config(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000025", FULL_CONFIG))
    assert cfg.camera == "C000025"
    assert cfg.matrix_tag == TAG_C000025
    assert cfg.satellite.zoom == 20 and cfg.satellite.scale == 2
    assert cfg.satellite.size == (640, 640)
    assert cfg.latlon == (22.625869, 120.297201)
    assert cfg.scale_check == "passed"


def test_scale_check_defaults_to_unchecked(tmp_path: Path) -> None:
    cfg = load_camera_config(write_camera(tmp_path, "C000025", "camera: C000025\n"))
    assert cfg.scale_check == "unchecked"
    assert cfg.matrix_tag is None


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ("name: x\n", "camera"),
        ("camera: C000043\n", "camera"),
        (f"camera: C000025\ncctv_points: {SQUARE}\nmap_points: {SQUARE[:3]}\n", "數量不符"),
        (f"camera: C000025\ncctv_points: {SQUARE[:3]}\nmap_points: {SQUARE[:3]}\n", "cctv_points"),
        (f"camera: C000025\ncctv_points: {SQUARE}\n", "map_points"),
        ("camera: C000025\nscale_check: ok\n", "scale_check"),
        ("camera: C000025\nlatlon: [95.0, 120.0]\n", "latlon"),
        ("camera: C000025\nsatellite: {provider: esri, zoom: 19, scale: 1, size: [256, 256]}\n", "satellite"),
        ("camera: C000025\nunknown_field: 1\n", "unknown_field"),
    ],
)
def test_invalid_config_names_file_and_field(tmp_path: Path, body: str, field: str) -> None:
    path = write_camera(tmp_path, "C000025", body)
    with pytest.raises(ValueError) as excinfo:
        load_camera_config(path)
    assert str(path) in str(excinfo.value)
    assert field in str(excinfo.value)


def test_config_for_sequence_uses_camera_id(tmp_path: Path) -> None:
    write_camera(tmp_path, "C000025", FULL_CONFIG)
    assert load_camera_config_for_sequence(tmp_path, SEQ_C000025).camera == "C000025"
    assert load_camera_config_for_sequence(tmp_path, "C000043 民族一路 九如一路_1") is None


def test_resolve_prefers_existing_matrix(tmp_path: Path) -> None:
    matrices = tmp_path / "matrices"
    matrices.mkdir()
    H = np.diag([3.0, 3.0, 1.0])
    np.save(matrices / f"H_{TAG_C000025}.npy", H)
    write_camera(
        tmp_path / "cameras", "C000025",
        f"camera: C000025\nmatrix_tag: {TAG_C000025}\ncctv_points: {SQUARE}\nmap_points: {SQUARE_X2}\n",
    )
    resolved, tag = resolve_camera_homography(SEQ_C000025, tmp_path / "cameras", matrices)
    assert tag == TAG_C000025
    np.testing.assert_array_equal(resolved, H)


def test_resolve_from_manual_points(tmp_path: Path) -> None:
    write_camera(
        tmp_path / "cameras", "C000025",
        f"camera: C000025\ncctv_points: {SQUARE}\nmap_points: {SQUARE_X2}\n",
    )
    resolved, tag = resolve_camera_homography(SEQ_C000025, tmp_path / "cameras", tmp_path / "matrices")
    assert tag is None
    np.testing.assert_allclose(resolved / resolved[2, 2], np.diag([2.0, 2.0, 1.0]), atol=1e-9)


def test_resolve_without_config_returns_none(tmp_path: Path) -> None:
    assert resolve_camera_homography(SEQ_C000025, tmp_path / "cameras", tmp_path / "matrices") == (None, None)


def test_resolve_missing_matrix_file_is_error(tmp_path: Path) -> None:
    write_camera(tmp_path / "cameras", "C000025", f"camera: C000025\nmatrix_tag: {TAG_C000025}\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_camera_homography(SEQ_C000025, tmp_path / "cameras", tmp_path / "matrices")
    assert f"H_{TAG_C000025}.npy" in str(excinfo.value)


def test_legacy_point_files_are_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / "homography_points"
    legacy.mkdir()
    (legacy / "README.md").write_text("doc", encoding="utf-8")
    assert_no_legacy_point_files(legacy)

    (legacy / f"{SEQ_C000025}.yaml").write_text("cctv_points: []\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        assert_no_legacy_point_files(legacy)
    assert "config/homography/<camera>.yaml" in str(excinfo.value)


def test_meters_per_pixel_kaohsiung(tmp_path: Path) -> None:
    from safetrajai.collision.homography.georef import meters_per_pixel

    body = FULL_CONFIG.replace("latlon: [22.625869, 120.297201]", "latlon: [22.6, 120.3]")
    mpp = meters_per_pixel(load_camera_config(write_camera(tmp_path, "C000025", body)))
    expected = 156543.03392 * np.cos(np.radians(22.6)) / 2 ** 20 / 2
    assert abs(mpp - expected) < 1e-12
    assert mpp == pytest.approx(0.0689, abs=1e-4)


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.replace("latlon: [22.625869, 120.297201]\n", ""),
        lambda t: t.replace("satellite: {provider: google_static, zoom: 20, scale: 2, size: [640, 640]}\n", ""),
        lambda t: t.replace("scale_check: passed", "scale_check: failed"),
        lambda t: t.replace("scale_check: passed", "scale_check: unchecked"),
        lambda t: t.replace(f"matrix_tag: {TAG_C000025}\n", f"cctv_points: {SQUARE}\nmap_points: {SQUARE_X2}\n"),
    ],
    ids=["no_latlon", "no_satellite", "scale_failed", "scale_unchecked", "manual_points"],
)
def test_meters_per_pixel_unavailable(tmp_path: Path, edit) -> None:
    from safetrajai.collision.homography.georef import meters_per_pixel

    assert meters_per_pixel(load_camera_config(write_camera(tmp_path, "C000025", edit(FULL_CONFIG)))) is None
