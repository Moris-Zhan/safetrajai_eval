"""draw_collision_heatmap writes the TTC, PET and conflict event heat maps."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from safetrajai import collision
from safetrajai.collision.output import draw_collision_heatmap, draw_satellite_heatmap


def background() -> np.ndarray:
    return np.full((120, 160, 3), 90, dtype=np.uint8)


def test_heatmap_is_written_with_background_size(tmp_path: Path) -> None:
    output_path = tmp_path / "heatmap.png"
    assert draw_collision_heatmap([(40, 50), (42, 52), (120, 80)], background(), output_path) is True
    assert cv2.imread(str(output_path)).shape[:2] == (120, 160)


def test_no_points_writes_nothing_and_removes_stale_file(tmp_path: Path) -> None:
    output_path = tmp_path / "heatmap.png"
    assert draw_collision_heatmap([(40, 50)], background(), output_path) is True
    assert draw_collision_heatmap([], background(), output_path) is False
    assert not output_path.exists()


def hot_spot(heatmap: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Centre of mass (x, y) of where `heatmap` is redder than `reference`."""
    redder = np.clip(heatmap[..., 2].astype(float) - reference[..., 2].astype(float), 0, None)
    rows, cols = np.indices(redder.shape)
    return float((cols * redder).sum() / redder.sum()), float((rows * redder).sum() / redder.sum())


def test_hot_spot_follows_the_event_point(tmp_path: Path) -> None:
    left, right = tmp_path / "left.png", tmp_path / "right.png"
    draw_collision_heatmap([(30, 60)] * 5, background(), left)
    draw_collision_heatmap([(110, 60)] * 5, background(), right)
    left_image, right_image = cv2.imread(str(left)), cv2.imread(str(right))
    left_x, left_y = hot_spot(left_image, right_image)
    right_x, right_y = hot_spot(right_image, left_image)
    assert abs(left_x - 30) <= 8 and abs(left_y - 60) <= 8
    assert abs(right_x - 110) <= 8 and abs(right_y - 60) <= 8


def stage3_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_name="CCTV", split_name="val", tracker_name="YOLO11x",
        collision_lookback_sec=2.0, collision_max_ttc_frames=50, collision_iou_threshold=0.1,
        collision_overlap_threshold=0.1, collision_output_dir=tmp_path / "collision_output",
        homography_camera_dir=tmp_path / "no_homography", homography_matrices_dir=tmp_path / "no_matrices",
        homography_satellite_dir=tmp_path / "no_satellite",
    )


@pytest.mark.parametrize("with_background", [True, False])
def test_stage3_draws_heatmaps_only_with_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_background: bool,
) -> None:
    seq_dir = tmp_path / "stage1" / "C000999 sample"
    seq_dir.mkdir(parents=True)
    if with_background:
        cv2.imwrite(str(seq_dir / "first_frame_with_box.jpg"), background())
    monkeypatch.setattr(
        collision, "detection_core", SimpleNamespace(
            compute_ttc_for_all_pairs=lambda *args, **kwargs: [
                {"collision_point": (40, 50), "future_box1": None, "future_box2": None}],
            compute_pet_for_all_pairs=lambda *args, **kwargs: [{"cross_point": (100, 70)}],
        ),
    )
    monkeypatch.setattr(collision, "_reproject_ttc_events", lambda events, H_inv: events)
    monkeypatch.setattr(collision, "_reproject_pet_events", lambda events, H_inv: events)
    monkeypatch.setattr(collision.collision_output, "write_events_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(collision.collision_output, "write_events_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(collision.collision_output, "write_ttc_legacy_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(collision.collision_output, "write_pet_legacy_log", lambda *args, **kwargs: None)
    cfg = stage3_config(tmp_path)

    collision.run_collision_detection_for_sequence(seq_dir, cfg, fps=10.0, bounds=(120, 160))

    out_dir = cfg.collision_output_dir / seq_dir.name
    assert (out_dir / "ttc_collision_heatmap.png").exists() is with_background
    assert (out_dir / "pet_collision_heatmap.png").exists() is with_background


def test_satellite_heatmap_keeps_only_points_inside_the_image(tmp_path: Path) -> None:
    satellite = np.full((400, 400, 3), 90, dtype=np.uint8)
    output_path = tmp_path / "satellite.png"
    assert draw_satellite_heatmap([(200, 200), (500, 100), (100, 450)], satellite, output_path, title="t") is True
    assert cv2.imread(str(output_path)).shape[:2] == (400, 400)
    assert draw_satellite_heatmap([(500, 100)], satellite, output_path, title="t") is False
    assert not output_path.exists()


def test_stage3_draws_satellite_heatmaps_when_the_image_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq_dir = tmp_path / "stage1" / "C000999 sample"
    seq_dir.mkdir(parents=True)
    cfg = stage3_config(tmp_path)
    satellite_dir = tmp_path / "satellite"
    satellite_dir.mkdir()
    cv2.imwrite(str(satellite_dir / "tag.png"), np.full((200, 200, 3), 90, dtype=np.uint8))
    cfg.homography_satellite_dir = satellite_dir
    monkeypatch.setattr(collision.homography_points, "resolve_camera_homography", lambda *args: (np.eye(3), "tag"))
    monkeypatch.setattr(collision, "add_homography_fields", lambda *args: None)
    monkeypatch.setattr(
        collision, "detection_core", SimpleNamespace(
            compute_ttc_for_all_pairs=lambda *args, **kwargs: [{"collision_point": (40, 50)}],
            compute_pet_for_all_pairs=lambda *args, **kwargs: [{"cross_point": (100, 70)}],
        ),
    )
    monkeypatch.setattr(collision, "_reproject_ttc_events", lambda events, H_inv: events)
    monkeypatch.setattr(collision, "_reproject_pet_events", lambda events, H_inv: events)
    for writer in ("write_events_json", "write_events_csv", "write_ttc_legacy_log", "write_pet_legacy_log"):
        monkeypatch.setattr(collision.collision_output, writer, lambda *args, **kwargs: None)

    collision.run_collision_detection_for_sequence(seq_dir, cfg, fps=10.0, bounds=(120, 160))
    out_dir = cfg.collision_output_dir / seq_dir.name
    assert (out_dir / "ttc_collision_heatmap_satellite.png").exists()
    assert (out_dir / "pet_collision_heatmap_satellite.png").exists()

    (satellite_dir / "tag.png").unlink()
    (out_dir / "ttc_collision_heatmap_satellite.png").unlink()
    (out_dir / "pet_collision_heatmap_satellite.png").unlink()
    collision.run_collision_detection_for_sequence(seq_dir, cfg, fps=10.0, bounds=(120, 160))
    assert not list(out_dir.glob("*_satellite.png"))
