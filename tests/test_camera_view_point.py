"""Optional camera_view_point field of config/homography/<camera>.yaml (satellite-image pixel of the camera)."""

from pathlib import Path

import pytest

from safetrajai.collision.homography.camera_config import load_camera_config, update_camera_config_fields


def write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "C000054.yaml"
    path.write_text(f"camera: C000054\nscale_check: unchecked\n{extra}", encoding="utf-8")
    return path


def test_view_point_is_read_as_floats(tmp_path: Path) -> None:
    cfg = load_camera_config(write_config(tmp_path, "camera_view_point: [320.5, 240]\n"))

    assert cfg.camera_view_point == (320.5, 240.0)


def test_view_point_defaults_to_none(tmp_path: Path) -> None:
    assert load_camera_config(write_config(tmp_path)).camera_view_point is None


@pytest.mark.parametrize("value", ["[1, 2, 3]", "[1]", "['a', 'b']", "[true, 2]", "5", "{x: 1, y: 2}"])
def test_malformed_view_point_names_the_field(tmp_path: Path, value: str) -> None:
    path = write_config(tmp_path, f"camera_view_point: {value}\n")

    with pytest.raises(ValueError, match=r"camera_view_point.*\[x, y\]"):
        load_camera_config(path)


def test_view_point_can_be_written_in_place_and_read_back(tmp_path: Path) -> None:
    path = write_config(tmp_path, "# keep this comment\n")

    update_camera_config_fields(path, {"camera_view_point": [10.5, 20]})

    assert "camera_view_point: [10.5, 20]" in path.read_text(encoding="utf-8")
    assert "# keep this comment" in path.read_text(encoding="utf-8")
    assert load_camera_config(path).camera_view_point == (10.5, 20.0)
