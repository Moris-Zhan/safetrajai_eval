"""Scenarios from specs/evaluation-roi-filtering: ignore-region matching before TrackEval."""

from pathlib import Path

import numpy as np

from safetrajai.evaluation.ignore_region import apply_ignore_region, ignore_masks

ROI = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


def rows(*boxes: tuple[float, float, float, float], frame: int = 1) -> np.ndarray:
    return np.array([[frame, i + 1, *box] for i, box in enumerate(boxes)], dtype=np.float64).reshape(-1, 6)


def test_detection_outside_roi_matched_to_ignored_gt_is_dropped() -> None:
    gt = rows((200, 200, 20, 20))
    pred = rows((201, 201, 20, 20))

    gt_keep, pred_keep = ignore_masks(gt, pred, ROI)

    assert not gt_keep.any()
    assert not pred_keep.any()


def test_prediction_crossing_roi_boundary_is_kept_as_true_positive() -> None:
    gt = rows((85, 40, 20, 20))      # center (95, 50): inside ROI
    pred = rows((91, 40, 20, 20))    # center (101, 50): outside ROI, IoU 0.54 with GT

    gt_keep, pred_keep = ignore_masks(gt, pred, ROI)

    assert gt_keep.all()
    assert pred_keep.all()


def test_unmatched_prediction_outside_roi_is_not_scored() -> None:
    gt = rows((200, 200, 20, 20))
    pred = rows((300, 300, 20, 20))

    _, pred_keep = ignore_masks(gt, pred, ROI)

    assert not pred_keep.any()


def test_unmatched_prediction_inside_roi_is_kept_as_false_positive() -> None:
    gt = rows((200, 200, 20, 20))
    pred = rows((40, 40, 20, 20))

    _, pred_keep = ignore_masks(gt, pred, ROI)

    assert pred_keep.all()


def test_prediction_outside_roi_below_iou_threshold_is_dropped() -> None:
    gt = rows((85, 40, 20, 20))      # center (95, 50): inside ROI
    pred = rows((95, 40, 20, 20))    # center (105, 50): outside ROI, IoU 0.33 with GT

    gt_keep, pred_keep = ignore_masks(gt, pred, ROI)

    assert gt_keep.all()             # the in-ROI GT stays and becomes a false negative
    assert not pred_keep.any()


def test_prediction_prefers_valid_gt_over_ignored_gt() -> None:
    gt = rows((80, 40, 20, 20), (100, 40, 20, 20))   # centers (90,50) inside, (110,50) outside
    pred = rows((82, 40, 20, 20))                     # IoU 0.82 with valid GT, 0.05 with ignored GT

    gt_keep, pred_keep = ignore_masks(gt, pred, ROI)

    assert gt_keep.tolist() == [True, False]
    assert pred_keep.all()


def test_apply_ignore_region_rewrites_only_output_copies_and_is_idempotent(tmp_path: Path) -> None:
    seq, split = "C000025 test_1", "CCTV-val"
    gt_txt = tmp_path / "gt" / "mot_challenge" / split / seq / "gt" / "gt.txt"
    gt_txt.parent.mkdir(parents=True)
    gt_txt.write_text("1,1,10,10,20,20,1,2,1\n1,2,200,200,20,20,1,2,1\n", encoding="utf-8")
    trk_txt = tmp_path / "trackers" / "mot_challenge" / split / "YOLO11x__car" / "data" / f"{seq}.txt"
    trk_txt.parent.mkdir(parents=True)
    trk_txt.write_text("1,7,11,11,20,20,0.9\n1,8,201,201,20,20,0.8\n", encoding="utf-8")

    first = apply_ignore_region(tmp_path, split, seq, ["YOLO11x__car"], ROI)
    gt_txt.write_text("1,1,10,10,20,20,1,2,1\n1,2,200,200,20,20,1,2,1\n", encoding="utf-8")
    second = apply_ignore_region(tmp_path, split, seq, ["YOLO11x__car"], ROI)

    assert first == second == {"car": (1, 1)}
    assert trk_txt.read_text(encoding="utf-8") == "1,7,11,11,20,20,0.9\n"
    assert gt_txt.read_text(encoding="utf-8") == "1,1,10,10,20,20,1,2,1\n"
    raw = tmp_path / "trackers_raw" / "mot_challenge" / split / "YOLO11x__car" / "data" / f"{seq}.txt"
    assert raw.read_text(encoding="utf-8").count("\n") == 2
