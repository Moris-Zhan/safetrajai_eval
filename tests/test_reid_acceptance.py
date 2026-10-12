"""Scenarios from specs/reid-model: per-class tracking acceptance and its aggregation."""

import pytest

from safetrajai.reid.acceptance import (
    aggregate, apply_overrides, judge_class, parse_summary, parse_track_time,
)

HEADER = "HOTA DetA AssA IDSW CLR_TP IDTP IDFP GT_Dets"


def metrics(idsw: int, tp: int, idp: float, assa: float, gt: int = 100) -> dict:
    return {"IDSW": idsw, "CLR_TP": tp, "IDP": idp, "AssA": assa, "HOTA": assa, "GT_Dets": gt}


BASE = metrics(idsw=4, tp=500, idp=0.95, assa=80.0)


def test_parse_summary_reads_header_and_values(tmp_path) -> None:
    path = tmp_path / "car_summary.txt"
    path.write_text(f"{HEADER}\n70.5 60 80.25 3 400 380 20 420\n")
    assert parse_summary(path) == {"HOTA": 70.5, "DetA": 60.0, "AssA": 80.25, "IDSW": 3.0, "CLR_TP": 400.0,
                                   "IDTP": 380.0, "IDFP": 20.0, "GT_Dets": 420.0}


def test_aggregate_sums_counts_and_weights_by_gt() -> None:
    rows = [
        {"HOTA": 60.0, "AssA": 80.0, "IDSW": 2, "CLR_TP": 100, "IDTP": 90, "IDFP": 10, "GT_Dets": 100},
        {"HOTA": 90.0, "AssA": 50.0, "IDSW": 1, "CLR_TP": 300, "IDTP": 270, "IDFP": 0, "GT_Dets": 300},
    ]
    result = aggregate(rows)
    assert result["IDSW"] == 3 and result["CLR_TP"] == 400 and result["GT_Dets"] == 300 + 100
    assert result["IDP"] == pytest.approx(360 / 370)
    assert result["AssA"] == pytest.approx((80 * 100 + 50 * 300) / 400)
    assert result["HOTA"] == pytest.approx((60 * 100 + 90 * 300) / 400)


def test_aggregate_without_gt_has_no_rates() -> None:
    result = aggregate([{"HOTA": 0, "AssA": 0, "IDSW": 0, "CLR_TP": 0, "IDTP": 0, "IDFP": 0, "GT_Dets": 0}])
    assert result["GT_Dets"] == 0 and result["IDP"] is None and result["AssA"] is None


def test_more_id_switches_fail() -> None:
    assert not judge_class(metrics(5, 510, 0.96, 81.0), BASE)["passed"]


def test_only_assa_gain_fails() -> None:
    result = judge_class(metrics(4, 500, 0.95, 80.6), BASE)
    assert not result["passed"] and "身分" in result["reason"]


def test_fewer_id_switches_without_regression_pass() -> None:
    assert judge_class(metrics(3, 500, 0.95, 80.0), BASE)["passed"]
    assert judge_class(metrics(4, 500, 0.96, 80.0), BASE)["passed"]


def test_lower_tp_or_assa_fails_even_with_fewer_switches() -> None:
    assert not judge_class(metrics(3, 499, 0.96, 80.0), BASE)["passed"]
    assert not judge_class(metrics(3, 500, 0.96, 79.9), BASE)["passed"]


def test_class_without_gt_fails() -> None:
    no_gt = {"IDSW": 0, "CLR_TP": 0, "IDP": None, "AssA": None, "HOTA": None, "GT_Dets": 0}
    result = judge_class(no_gt, no_gt)
    assert not result["passed"] and "GT" in result["reason"]


def test_parse_track_time_weights_by_frames() -> None:
    log = ("\x1b[33mTracking complete — 10 frames\x1b[0m\n  \x1b[35mtrack avg\x1b[0m : \x1b[32m0.0100s\x1b[0m\n"
           "\x1b[33mTracking complete — 30 frames\x1b[0m\n  \x1b[35mtrack avg\x1b[0m : \x1b[32m0.0300s\x1b[0m\n")
    assert parse_track_time(log) == (40, pytest.approx(0.025))
    assert parse_track_time("no timing here") == (0, None)


def test_apply_overrides_parses_values_and_keeps_input() -> None:
    config = {"class_tracking": {"class_tracking_params": {"car": {"use_reid": False, "appearance_thresh": 0.1}}}}
    result = apply_overrides(config, ["car.use_reid=true", "car.appearance_thresh=0.79",
                                      "car.model=models/class_reid_weights/x/model.pt"])
    car = result["class_tracking"]["class_tracking_params"]["car"]
    assert car == {"use_reid": True, "appearance_thresh": 0.79, "model": "models/class_reid_weights/x/model.pt"}
    assert config["class_tracking"]["class_tracking_params"]["car"]["use_reid"] is False


def test_apply_overrides_rejects_unknown_class_and_bad_format() -> None:
    config = {"class_tracking": {"class_tracking_params": {"car": {}}}}
    with pytest.raises(ValueError, match="bike"):
        apply_overrides(config, ["bike.use_reid=true"])
    with pytest.raises(ValueError, match="CLASS.KEY=VALUE"):
        apply_overrides(config, ["car.use_reid"])


def run_cli(*args: str):
    import subprocess
    import sys

    from safetrajai.paths import PROJECT_ROOT

    script = PROJECT_ROOT / "scripts" / "reid_train" / "acceptance_grid.py"
    return subprocess.run([sys.executable, str(script), *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          timeout=120)


def test_configs_apply_overrides_per_subset(tmp_path) -> None:
    import yaml

    result = run_cli("configs", "--name", "a_car", "--out-root", str(tmp_path),
                     "--set", "car.use_reid=true", "--set", "car.reid_center_gate_px=80")
    assert result.returncode == 0, result.stderr[-2000:]
    for subset in ("tune", "holdout"):
        config = yaml.safe_load((tmp_path / "configs" / f"a_car_{subset}.yaml").read_text(encoding="utf-8"))
        assert config["paths"]["SOURCE_VIDEO_DIR"] == f"./dataset/eval/cctv_all8_subsets/{subset}/"
        assert config["paths"]["OUTPUT_ROOT"] == str(tmp_path / "a_car" / subset)
        car = config["class_tracking"]["class_tracking_params"]["car"]
        assert car["use_reid"] is True and car["reid_center_gate_px"] == 80


def test_configs_reject_invalid_parameter(tmp_path) -> None:
    result = run_cli("configs", "--name", "bad", "--out-root", str(tmp_path), "--subsets", "tune",
                     "--set", "car.use_reid=true", "--set", "car.appearance_veto_sim=1.5")
    assert result.returncode != 0
    assert "appearance_veto_sim" in result.stderr


def write_run(root, name: str, idsw: int) -> None:
    from safetrajai.classes import CLASS_NAMES

    run_cli("configs", "--name", name, "--out-root", str(root), "--subsets", "tune")
    seq_root = root / name / "tune" / "mot_annotations_all8"
    for seq in ("seqA", "seqB"):
        for cls in CLASS_NAMES.values():
            folder = seq_root / seq / "trackers" / "mot_challenge" / "CCTV-val" / f"YOLO11x__{cls}"
            folder.mkdir(parents=True)
            gt = 100 if cls == "car" else 0
            values = f"80 70 {80 if gt else 0} {idsw if gt else 0} {400 if gt else 0} {380 if gt else 0} 20 {gt}"
            (folder / f"{cls}_summary.txt").write_text(f"{HEADER}\n{values}\n")
    (root / name / "tune" / "run.log").write_text(
        "Tracking complete — 10 frames\n  track avg : 0.0100s\nTracking complete — 30 frames\n  track avg : 0.0300s\n")


def test_summarize_judges_against_baseline(tmp_path) -> None:
    import csv

    write_run(tmp_path, "baseline", idsw=3)
    write_run(tmp_path, "better", idsw=2)
    result = run_cli("summarize", "--baseline", "baseline", "--runs", "better", "--out-root", str(tmp_path),
                     "--out", str(tmp_path / "summary"))
    assert result.returncode == 0, result.stderr[-2000:]
    rows = {(r["run"], r["class"]): r for r in csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8"))}
    assert rows[("baseline", "car")]["IDSW"] == "6.0" and rows[("better", "car")]["IDSW"] == "4.0"
    assert rows[("better", "car")]["passed"] == "True"
    assert rows[("better", "bus")]["passed"] == "False" and "GT" in rows[("better", "bus")]["reason"]
    assert float(rows[("better", "car")]["track_s_per_frame"]) == pytest.approx(0.025)
    assert (tmp_path / "summary.md").exists()
