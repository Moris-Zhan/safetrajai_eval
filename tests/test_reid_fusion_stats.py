"""Scenarios from specs/reid-fusion: threshold calibration uses only tuning sequences."""

import numpy as np
import pytest

from safetrajai.reid.fusion_stats import (
    MIN_HARD_NEGATIVES, candidates, check_tuning_sequences, class_stats, hard_negative_pairs, positive_pairs,
    read_mot_gt,
)

TUNE_SEQ = "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1"
HOLDOUT_SEQ = "C000025 中華三路 民生二路(東北角)_20251209152639_15s_1"


def write_gt(path, rows) -> None:
    path.write_text("".join(f"{f},{i},{x - 10},{y - 10},20,20,1,{c},1.0\n" for f, i, x, y, c in rows))


def test_holdout_sequences_are_rejected() -> None:
    check_tuning_sequences([TUNE_SEQ])
    with pytest.raises(ValueError, match="152639"):
        check_tuning_sequences([TUNE_SEQ, HOLDOUT_SEQ])


def test_read_mot_gt_uses_one_based_class_and_centers(tmp_path) -> None:
    write_gt(tmp_path / "gt.txt", [(1, 7, 100, 50, 2)])
    (box,) = read_mot_gt(tmp_path / "gt.txt", TUNE_SEQ)
    assert (box["class"], box["gt_id"], box["cx"], box["cy"]) == ("car", 7, 100.0, 50.0)


def test_positive_pairs_need_one_second_gap(tmp_path) -> None:
    write_gt(tmp_path / "gt.txt", [(f, 1, 100, 100, 2) for f in (1, 3, 6, 11)])
    boxes = read_mot_gt(tmp_path / "gt.txt", TUNE_SEQ)
    pairs = positive_pairs(boxes, {TUNE_SEQ: 5.0})
    assert sorted((boxes[a]["frame"], boxes[b]["frame"]) for a, b in pairs) == [(1, 6), (1, 11), (3, 11), (6, 11)]


def test_hard_negatives_same_frame_same_class_within_radius(tmp_path) -> None:
    write_gt(tmp_path / "gt.txt", [(1, 1, 100, 100, 2), (1, 2, 200, 100, 2), (1, 3, 400, 100, 2),
                                   (1, 4, 110, 100, 3), (2, 2, 105, 100, 2)])
    boxes = read_mot_gt(tmp_path / "gt.txt", TUNE_SEQ)
    pairs = hard_negative_pairs(boxes, {"car": 160.0, "scooter": 400.0})
    assert [(boxes[a]["gt_id"], boxes[b]["gt_id"]) for a, b in pairs] == [(1, 2)]


def test_too_few_hard_negatives_fall_back_and_are_flagged(tmp_path) -> None:
    rows = [(f, i, 100 + 1000 * i, 100, 6) for f in range(1, 11) for i in (1, 2)]
    write_gt(tmp_path / "gt.txt", rows)
    boxes = read_mot_gt(tmp_path / "gt.txt", TUNE_SEQ)
    features = np.eye(2, dtype=np.float32)[[b["gt_id"] - 1 for b in boxes]]
    stats = class_stats(features, boxes, {TUNE_SEQ: 5.0}, {"truck": 200.0}, seed=0)
    assert stats["truck"]["negative_kind"] == "any_fallback"
    assert stats["truck"]["negative"]["pairs"] > MIN_HARD_NEGATIVES
    assert stats["truck"]["positive"]["p50"] == 0.0 and stats["truck"]["negative"]["p50"] == 0.5


def test_candidates_are_in_similarity_direction() -> None:
    stats = {"car": {"positive": {"pairs": 10, "p95": 0.2, "p99": 0.3, "p99.9": 0.35},
                     "negative": {"pairs": 10, "p5": 0.15}, "negative_kind": "hard"}}
    c = candidates(stats)["car"]
    assert c["appearance_thresh_wide"] == 0.8
    assert c["appearance_thresh_strict"] == 0.85
    assert (c["appearance_veto_sim_p99"], c["appearance_veto_sim_p99_9"]) == (0.7, 0.65)
    stats["car"]["negative"]["p5"] = 0.4
    assert candidates(stats)["car"]["appearance_thresh_strict"] == 0.8
