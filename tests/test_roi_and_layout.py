"""Unit tests for camera-keyed ROI resolution and dataset layout discovery."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from safetrajai.evaluation.config_loader import load_config
from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name, discover_source_sequences
from safetrajai.paths import CONFIG_DIR
from safetrajai.roi import resolve_roi

SEQ_C000025 = "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1"


@pytest.mark.parametrize(
    ("seq_name", "camera_id"),
    [
        (SEQ_C000025, "C000025"),
        ("BEV0001 高雄 大順路富民路_20250101000000", "BEV0001"),
        ("037- [翠華路、海功東路 通道-01]_20250101000000", "037-"),
    ],
)
def test_camera_id_is_text_before_first_space(seq_name: str, camera_id: str) -> None:
    assert camera_id_from_seq_name(seq_name) == camera_id


def test_camera_id_without_space_is_rejected() -> None:
    with pytest.raises(ValueError):
        camera_id_from_seq_name("C000025_20251209145630_15s_1")


def test_raw_layout_only_discovers_listed_cameras(tmp_path: Path) -> None:
    for camera, name in [("C000054", "C000054 a_1.mp4"), ("C000054", "Demo.mp4"), ("C000025", "C000025 b_1.mp4")]:
        (tmp_path / camera).mkdir(exist_ok=True)
        (tmp_path / camera / name).touch()

    found = discover_source_sequences(tmp_path, "unused", layout="raw", cameras=["C000054"])

    assert [p.name for p in found] == ["C000054 a_1.mp4"]


def test_raw_layout_requires_cameras(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        discover_source_sequences(tmp_path, "unused", layout="raw", cameras=[])


def test_annotated_layout_is_unchanged(tmp_path: Path) -> None:
    seq_dir = tmp_path / "annotations_all8" / f"annotation_{SEQ_C000025}"
    seq_dir.mkdir(parents=True)
    (seq_dir / f"{SEQ_C000025}.mp4").touch()

    found = discover_source_sequences(tmp_path, "annotations_all8")

    assert [p.stem for p in found] == [SEQ_C000025]


def test_c000025_roi_matches_legacy_det_corner() -> None:
    polygon = resolve_roi(SEQ_C000025, (640, 480))
    np.testing.assert_array_equal(polygon, np.array([[210, 90], [580, 100], [640, 320], [0, 280]], dtype=np.float32))


def test_missing_roi_falls_back_to_full_frame(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    polygon = resolve_roi("C999999 無 ROI 鏡頭_20250101000000", (1280, 720), roi_dir=tmp_path)

    np.testing.assert_array_equal(polygon, np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32))
    assert "C999999" in capsys.readouterr().out


def test_roi_image_size_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "C000054.yaml").write_text(
        yaml.safe_dump({"camera": "C000054", "image_size": [704, 480], "polygon": [[0, 0], [10, 0], [10, 10]]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        resolve_roi("C000054 民族一路 天祥二路_20250623070611", (1280, 720), roi_dir=tmp_path)


def test_config_with_det_corner_is_rejected(tmp_path: Path) -> None:
    legacy = yaml.safe_load((CONFIG_DIR / "bytetrack.yaml").read_text(encoding="utf-8"))
    legacy["det_corner"] = [[0, 0], [10, 0], [10, 10]]
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(yaml.safe_dump(legacy, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="config/roi"):
        load_config(str(legacy_path))
