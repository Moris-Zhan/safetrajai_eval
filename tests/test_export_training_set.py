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


def test_group_boxes_are_taken_only_on_the_listed_frames_and_skip_outside_shapes(tmp_path: Path) -> None:
    ledger = tmp_path / "box_issues.csv"
    ledger.write_text(
        "track_id,kind,old_label,manifest_frames,note\n"
        "7,group_deleted,bus,10;12,x\n"
        "8,loose_box,car,10,y\n", encoding="utf-8")
    snapshot = {"tracks": [
        {"track_id": 7, "shapes": [
            {"frame": 10, "points": [0, 0, 5, 5], "outside": False},
            {"frame": 11, "points": [1, 1, 6, 6], "outside": False},
            {"frame": 12, "points": [2, 2, 7, 7], "outside": True}]},
        {"track_id": 8, "shapes": [{"frame": 10, "points": [9, 9, 10, 10], "outside": False}]}]}
    assert export.group_boxes_by_frame(snapshot, ledger) == {10: [(0, 0.0, 0.0, 5.0, 5.0)]}


def test_restored_group_frame_paints_group_region_gray_but_keeps_labeled_boxes() -> None:
    image = np.full((20, 20, 3), 200, np.uint8)
    canvas = export.mask_canvas([], 20, 20)
    group = [(0, 0.0, 0.0, 10.0, 10.0)]
    kept = [(1, 2.0, 2.0, 4.0, 4.0)]
    painted = export.apply_mask(image, export.add_boxes_to_canvas(canvas, group), kept)
    assert painted[8, 8, 0] == export.GRAY       # inside the group box, outside any kept box
    assert painted[3, 3, 0] == 200               # kept box is never painted over
    assert painted[15, 15, 0] == 200             # elsewhere untouched


PERSONS = [[0, 0, 10, 20, 0.9], [20, 0, 30, 20, 0.8], [40, 0, 50, 20, 0.7], [60, 0, 70, 20, 0.6], [80, 0, 90, 20, 0.5]]


def test_review_answers_become_labels_gray_regions_and_frame_exclusion() -> None:
    answer = {"boxes": {"0": "person", "1": "none", "2": "unsure", "3": "partial:scooter", "4": "bus"},
              "missed": False, "corrections": {"3": [58.0, 0.0, 72.0, 22.0]}}
    kept, gray, reason = export.resolve_review(PERSONS, answer)
    assert reason == ""
    assert kept == [(0, 0.0, 0.0, 10.0, 20.0), (2, 58.0, 0.0, 72.0, 22.0), (4, 80.0, 0.0, 90.0, 20.0)]
    assert gray == [(0, 40.0, 0.0, 50.0, 20.0)]


def test_uncorrected_partial_box_or_missed_person_excludes_the_frame() -> None:
    partial = {"boxes": {"0": "partial:car"}, "missed": False, "corrections": {}}
    assert export.resolve_review(PERSONS, partial)[2] == "partial_uncorrected"
    assert export.resolve_review(PERSONS, {"boxes": {"0": "partial"}, "missed": False})[2] == "partial_uncorrected"
    assert export.resolve_review(PERSONS, {"boxes": {"0": "person"}, "missed": True})[2] == "missed_person"


def test_base_other_class_boxes_are_dropped_when_they_overlap_a_reviewed_box() -> None:
    reviewed = [(2, 0.0, 0.0, 10.0, 20.0)]
    detections = [(2, 1.0, 1.0, 10.0, 20.0), (1, 100.0, 0.0, 120.0, 20.0)]
    assert export.fill_other_boxes(detections, reviewed) == [(1, 100.0, 0.0, 120.0, 20.0)]


def test_tiny_person_pseudo_boxes_keep_only_small_unoccupied_persons() -> None:
    detections = [(0, 0.0, 0.0, 10.0, 20.0), (0, 100.0, 0.0, 140.0, 80.0), (1, 200.0, 0.0, 210.0, 20.0),
                  (0, 300.0, 0.0, 310.0, 20.0)]
    occupied = [(1, 299.0, 0.0, 310.0, 20.0)]
    assert export.tiny_person_boxes(detections, occupied, 1280, 720, 32.0) == [(0, 0.0, 0.0, 10.0, 20.0)]


def test_person_repeat_list_repeats_only_person_frames() -> None:
    assert export.repeated_train_list(["a", "b", "c"], [True, False, True], 3) == ["a", "a", "a", "b", "c", "c", "c"]
