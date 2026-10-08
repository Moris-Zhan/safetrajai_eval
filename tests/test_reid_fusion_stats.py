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
    stats = class_stats(features, boxes, {TUNE_SEQ: 5.0}, {"truck": 200.0}, seed=0, retention_by_class={"truck": 15})
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


def test_auc_perfect_and_identical() -> None:
    from safetrajai.reid.fusion_stats import auc

    assert auc(np.array([0.1, 0.2]), np.array([0.5, 0.6, 0.7])) == 1.0
    assert auc(np.array([0.3, 0.3]), np.array([0.3, 0.3])) == 0.5
    assert auc(np.array([0.8]), np.array([0.1])) == 0.0


def test_tpr_at_fpr5() -> None:
    from safetrajai.reid.fusion_stats import tpr_at_fpr5

    negatives = np.linspace(0.2, 1.0, 101)
    p5 = np.percentile(negatives, 5)
    positives = np.array([0.1, p5 - 0.01, p5 + 0.01, 0.9])
    assert tpr_at_fpr5(positives, negatives) == 0.5


def test_positive_window_is_capped_by_retention(tmp_path) -> None:
    from safetrajai.reid.fusion_stats import positive_pairs_in_window

    write_gt(tmp_path / "gt.txt", [(f, 1, 100, 100, 3) for f in (1, 9, 34)])
    boxes = read_mot_gt(tmp_path / "gt.txt", TUNE_SEQ)
    one_second = positive_pairs(boxes, {TUNE_SEQ: 5.0})
    gaps = sorted(abs(boxes[a]["frame"] - boxes[b]["frame"]) for a, b in one_second)
    assert gaps == [8, 25, 33]
    window = positive_pairs_in_window(boxes, min_gap_frames=5, retention_by_class={"scooter": 10})
    assert sorted(abs(boxes[a]["frame"] - boxes[b]["frame"]) for a, b in window) == [8]


def test_retention_is_track_buffer_without_conversion() -> None:
    from safetrajai.reid.fusion_stats import retention_frames

    config = {"tracking_params": {"tracker": "bytetrack", "frame_rate": 30},
              "class_tracking": {"class_tracking_params": {"car": {"track_buffer": 15, "norm_factor": 80.0}}}}
    assert retention_frames(config) == {"car": 15}
    config["class_tracking"]["class_tracking_params"]["car"]["track_buffer"] = 1.5
    with pytest.raises(ValueError, match="track_buffer"):
        retention_frames(config)


def test_size_bin_uses_smaller_box() -> None:
    from safetrajai.reid.fusion_stats import size_bin

    boxes = [{"w": 12.0, "h": 40.0}, {"w": 30.0, "h": 50.0}, {"w": 20.0, "h": 18.0}, {"w": 40.0, "h": 33.0},
             {"w": 50.0, "h": 32.0}]
    assert size_bin(boxes, (0, 1)) == "<16"
    assert size_bin(boxes, (1, 2)) == "16-32"
    assert size_bin(boxes, (3, 4)) == ">=32"


def test_class_stats_reports_auc_tpr_and_size_bins(tmp_path) -> None:
    rows = [(f, i, 100 + 20 * i, 100, 2) for f in range(1, 21) for i in range(1, 5)]
    write_gt(tmp_path / "gt.txt", rows)
    boxes = read_mot_gt(tmp_path / "gt.txt", TUNE_SEQ)
    features = np.eye(4, dtype=np.float32)[[b["gt_id"] - 1 for b in boxes]]
    stats = class_stats(features, boxes, {TUNE_SEQ: 5.0}, {"car": 160.0}, seed=0, retention_by_class={"car": 15})
    car = stats["car"]
    assert car["negative_kind"] == "hard"
    assert car["auc"] == 1.0 and car["tpr_at_fpr5"] == 1.0
    assert car["auc_1s"] == 1.0
    assert 0 < car["positive"]["pairs"] < car["positive_1s"]["pairs"]
    assert set(car["by_size"]) == {"16-32"}
    assert car["by_size"]["16-32"]["auc"] == 1.0


def metrics(car_auc, car_tpr, scooter_auc, scooter_tpr, others=None) -> dict:
    out = {"car": {"auc": car_auc, "tpr_at_fpr5": car_tpr, "negative_kind": "hard"},
           "scooter": {"auc": scooter_auc, "tpr_at_fpr5": scooter_tpr, "negative_kind": "hard"}}
    out.update(others or {})
    return out


REFERENCE = metrics(0.80, 0.30, 0.70, 0.20, {"person": {"auc": 0.75, "tpr_at_fpr5": 0.2, "negative_kind": "hard"},
                                              "truck": {"auc": 0.9, "tpr_at_fpr5": 0.5, "negative_kind": "any_fallback"}})
NO_NOISE = {cls: {"auc": 0.0, "tpr_at_fpr5": 0.0} for cls in ("car", "scooter")}


def test_gate_rejects_lower_scooter_auc() -> None:
    from safetrajai.reid.fusion_stats import separability_gate

    result = separability_gate(metrics(0.85, 0.40, 0.69, 0.30), REFERENCE, NO_NOISE)
    assert not result["passed"]
    assert any("scooter" in reason for reason in result["reasons"])


def test_gate_rejects_too_small_improvement() -> None:
    from safetrajai.reid.fusion_stats import separability_gate

    assert not separability_gate(metrics(0.804, 0.40, 0.72, 0.30), REFERENCE, NO_NOISE)["passed"]
    assert separability_gate(metrics(0.81, 0.32, 0.72, 0.22), REFERENCE, NO_NOISE)["passed"]


def test_gate_uses_noise_floor_when_larger() -> None:
    from safetrajai.reid.fusion_stats import noise_floor, separability_gate

    noise = noise_floor(metrics(0.808, 0.30, 0.70, 0.20), metrics(0.80, 0.30, 0.70, 0.20))
    assert noise["car"]["auc"] == pytest.approx(0.008)
    result = separability_gate(metrics(0.812, 0.40, 0.75, 0.30), REFERENCE, noise)
    assert result["min_improvement"]["car"]["auc"] == pytest.approx(0.016)
    assert not result["passed"]


def test_gate_rejects_when_only_tail_does_not_improve() -> None:
    from safetrajai.reid.fusion_stats import separability_gate

    assert not separability_gate(metrics(0.82, 0.305, 0.72, 0.30), REFERENCE, NO_NOISE)["passed"]


def test_gate_limits_drop_of_other_classes_with_enough_distractors() -> None:
    from safetrajai.reid.fusion_stats import separability_gate

    good = metrics(0.85, 0.40, 0.75, 0.30)
    dropped = good | {"person": {"auc": 0.72, "tpr_at_fpr5": 0.2, "negative_kind": "hard"},
                      "truck": {"auc": 0.5, "tpr_at_fpr5": 0.1, "negative_kind": "any_fallback"}}
    assert not separability_gate(dropped, REFERENCE, NO_NOISE)["passed"]
    fine = dropped | {"person": {"auc": 0.74, "tpr_at_fpr5": 0.2, "negative_kind": "hard"}}
    assert separability_gate(fine, REFERENCE, NO_NOISE)["passed"]


def test_select_best_keeps_highest_mean_auc_or_none() -> None:
    from safetrajai.reid.fusion_stats import select_best, separability_gate

    results = {
        "a": separability_gate(metrics(0.82, 0.40, 0.78, 0.30), REFERENCE, NO_NOISE),
        "b": separability_gate(metrics(0.86, 0.40, 0.80, 0.30), REFERENCE, NO_NOISE),
        "c": separability_gate(metrics(0.70, 0.40, 0.90, 0.30), REFERENCE, NO_NOISE),
    }
    assert results["a"]["mean_auc"] == pytest.approx(0.80) and results["b"]["mean_auc"] == pytest.approx(0.83)
    assert select_best(results) == "b"
    assert select_best({"c": results["c"]}) is None
