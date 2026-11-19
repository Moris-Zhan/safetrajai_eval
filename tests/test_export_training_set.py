"""Pure helpers of the fine-tuning export: exclusion ledger, split rule, mask drop/paint rules, YOLO lines."""

import importlib.util
from pathlib import Path

import numpy as np

from safetrajai.paths import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "export_training_set", PROJECT_ROOT / "scripts" / "detection_data" / "export_training_set.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)


def test_group_deleted_frames_are_excluded(tmp_path: Path) -> None:
    ledger = tmp_path / "box_issues.csv"
    ledger.write_text(
        "track_id,kind,old_label,manifest_frames,note\n"
        "1,group_deleted,bus,132;134,x\n"
        "2,loose_box,car,500,y\n", encoding="utf-8")
    assert export.parse_excluded_frames(ledger) == {132, 134}


def test_validation_split_matches_camera_and_sequence_text() -> None:
    val = ["C000012", "C000054:20250629"]
    assert export.is_validation("C000012", "C000012 road_20250512065724", val)
    assert export.is_validation("C000054", "C000054 road_20250629082435", val)
    assert not export.is_validation("C000054", "C000054 road_20250623065608", val)
    assert not export.is_validation("C000011", "C000011 road_20250512065725", val)


def test_masked_box_is_dropped_but_kept_box_is_not_painted_over() -> None:
    canvas = export.mask_canvas([[[0, 0], [60, 0], [60, 100], [0, 100]]], 100, 100)
    mostly_inside = (1, 0.0, 0.0, 50.0, 50.0)
    mostly_outside = (1, 40.0, 0.0, 100.0, 50.0)
    assert export.masked_fraction(canvas, mostly_inside) >= 0.5
    assert export.masked_fraction(canvas, mostly_outside) < 0.5

    image = np.full((100, 100, 3), 200, np.uint8)
    painted = export.apply_mask(image, canvas, [mostly_outside])
    assert painted[80, 10, 0] == export.GRAY  # masked, no box
    assert painted[10, 45, 0] == 200  # inside the kept box, not painted
    assert painted[10, 80, 0] == 200  # outside the mask


def test_yolo_line_is_normalised_and_clamped() -> None:
    assert export.yolo_line((2, -10.0, 20.0, 50.0, 80.0), 100, 200) == "2 0.250000 0.250000 0.500000 0.300000"


def test_boxes_by_frame_skips_outside_shapes() -> None:
    annotations = {"tracks": [{"label": "scooter", "shapes": [
        {"frame": 3, "points": [1, 2, 3, 4], "outside": False},
        {"frame": 4, "points": [1, 2, 3, 4], "outside": True}]}]}
    assert export.boxes_by_frame(annotations) == {3: [(2, 1, 2, 3, 4)]}


def test_frame_stem_uses_camera_timestamp_and_frame() -> None:
    assert export.frame_stem("C000012", "C000012 中正路_20250512065724", 21) == "C000012_20250512065724_000021"
