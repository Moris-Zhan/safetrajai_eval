"""Unit tests for grid_search/core.py (spec grid-search-trials).

Heavy collaborators (YOLO, cv2, TrackEval) are replaced on the core module; nothing here reads dataset/, models/
or runs/.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from safetrajai.grid_search import core
from safetrajai.tracking import config

# Importing core pulls in the vendored TrackEval, which rebinds np.bool to the builtin bool at import time. numpy.testing
# calls np.bool(array) internally, so every later assert_allclose in the same session would raise. TrackEval is read-only
# (ADR 0002), so undo the rebinding here; np.int/np.float/np.object/np.str do not exist natively and are left alone.
if np.bool is not np.bool_:
    np.bool = np.bool_


def test_importing_core_does_not_leave_numpy_bool_rebound() -> None:
    assert np.bool is np.bool_
    np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_parameter_grid_expands_in_sorted_key_order() -> None:
    grid = core.ParameterGrid({"b": [1, 2], "a": [3, 4, 5]})
    combinations = list(grid)
    assert len(grid) == len(combinations) == 6
    assert combinations[:3] == [{"a": 3, "b": 1}, {"a": 3, "b": 2}, {"a": 4, "b": 1}]
    assert combinations[-1] == {"a": 5, "b": 2}


def test_parameter_grid_with_an_empty_value_list_has_no_combinations() -> None:
    grid = core.ParameterGrid({"a": [], "b": [1, 2]})
    assert len(grid) == 0
    assert list(grid) == []


def test_combo_id_ignores_key_order_and_distinguishes_values() -> None:
    grid = core.ParameterGrid({"a": [1, 2], "b": [3, 4]})
    assert grid.combo_id({"a": 1, "b": 3}) == grid.combo_id({"b": 3, "a": 1})
    assert grid.combo_id({"a": 1, "b": 3}) != grid.combo_id({"a": 2, "b": 3})
    assert json.loads(grid.combo_id({"a": 1, "b": 3})) == {"a": 1, "b": 3}


def trackeval_results(assa: Any, *, tracker: str = "YOLO11x__scooter", sequences: dict[str, Any] | None = None) -> dict:
    classEntry = {"scooter": {"HOTA": {"AssA": assa}}}
    return {"MotChallenge2DBox": {tracker: sequences if sequences is not None else {"COMBINED_SEQ": classEntry}}}


def test_extract_assa_mean_averages_the_alpha_array() -> None:
    results = trackeval_results(np.array([0.2, 0.4, 0.6]))
    assert core.extract_assa_mean(results, "scooter", "YOLO11x__scooter") == pytest.approx(0.4)


def test_extract_assa_mean_accepts_a_scalar_and_a_result_message_tuple() -> None:
    results = trackeval_results(0.5)
    assert core.extract_assa_mean((results, {"messages": "ignored"}), "scooter", "YOLO11x__scooter") == 0.5


def test_extract_assa_mean_prefers_the_combined_sequence() -> None:
    perSequence = {"seq_a": {"scooter": {"HOTA": {"AssA": 0.1}}}, "COMBINED_SEQ": {"scooter": {"HOTA": {"AssA": 0.9}}}}
    results = trackeval_results(None, sequences=perSequence)
    assert core.extract_assa_mean(results, "scooter", "YOLO11x__scooter") == 0.9


def test_extract_assa_mean_falls_back_to_the_first_tracker_when_variant_is_unknown() -> None:
    results = trackeval_results(0.7, tracker="some_other_tracker")
    assert core.extract_assa_mean(results, "scooter", "YOLO11x__scooter") == 0.7


@pytest.mark.parametrize("results", [
    None,
    [],
    ([], {}),
    {"MotChallenge2DBox": {"YOLO11x__scooter": {"COMBINED_SEQ": {"scooter": {}}}}},
    {"MotChallenge2DBox": {"YOLO11x__scooter": {"COMBINED_SEQ": {"scooter": {"HOTA": {}}}}}},
    {"MotChallenge2DBox": {}},
])
def test_extract_assa_mean_returns_none_when_a_level_is_missing(results: Any) -> None:
    assert core.extract_assa_mean(results, "scooter", "YOLO11x__scooter") is None


# Known gap G4 (openspec change add-config-and-grid-search-unit-tests, design.md): desired behavior, expected to fail
# until extract_assa_mean stops falling back to an arbitrary entry.
@pytest.mark.xfail(strict=True, reason="G4: a missing target class falls back to the first entry, e.g. a combined average")
def test_extract_assa_mean_does_not_borrow_another_entry_when_the_class_is_missing() -> None:
    sequences = {"COMBINED_SEQ": {"cls_comb_cls_av": {"HOTA": {"AssA": np.array([0.7])}}}}
    results = trackeval_results(None, sequences=sequences)
    assert core.extract_assa_mean(results, "scooter", "YOLO11x__scooter") is None


def write_gt_source(root: Path, *, root_seqinfo: str | None = None, gt_seqinfo: str | None = None) -> None:
    (root / "gt").mkdir(parents=True)
    (root / "gt" / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")
    if root_seqinfo is not None:
        (root / "seqinfo.ini").write_text(root_seqinfo, encoding="utf-8")
    if gt_seqinfo is not None:
        (root / "gt" / "seqinfo.ini").write_text(gt_seqinfo, encoding="utf-8")


def test_setup_trial_gt_overwrites_stale_gt_without_touching_the_source(tmp_path: Path) -> None:
    source, trial = tmp_path / "source", tmp_path / "trial"
    write_gt_source(source)
    (trial / "gt").mkdir(parents=True)
    (trial / "gt" / "gt.txt").write_text("stale and modified", encoding="utf-8")
    sourceBefore = (source / "gt" / "gt.txt").read_bytes()
    assert core._setup_trial_gt(str(source), trial) is True
    assert (trial / "gt" / "gt.txt").read_bytes() == sourceBefore
    assert (source / "gt" / "gt.txt").read_bytes() == sourceBefore


@pytest.mark.parametrize(("rootSeqinfo", "gtSeqinfo", "expected"), [
    ("root", None, "root"),
    (None, "gt", "gt"),
    ("root", "gt", "root"),
])
def test_setup_trial_gt_copies_seqinfo_from_root_or_gt_folder(
    tmp_path: Path, rootSeqinfo: str | None, gtSeqinfo: str | None, expected: str,
) -> None:
    source, trial = tmp_path / "source", tmp_path / "trial"
    write_gt_source(source, root_seqinfo=rootSeqinfo, gt_seqinfo=gtSeqinfo)
    trial.mkdir()
    assert core._setup_trial_gt(str(source), trial) is True
    assert (trial / "seqinfo.ini").read_text(encoding="utf-8") == expected


def test_setup_trial_gt_returns_false_when_the_source_has_no_gt(tmp_path: Path) -> None:
    source, trial = tmp_path / "source", tmp_path / "trial"
    source.mkdir()
    trial.mkdir()
    assert core._setup_trial_gt(str(source), trial) is False
    assert not (trial / "gt").exists()


def gt_cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(OUTPUT_ROOT=str(tmp_path), run_track_eval_dir="mot_out")


def test_resolve_gt_source_root_prefers_the_exact_sequence_directory(tmp_path: Path) -> None:
    prefix = "C000025 sequence_dup"  # exactly 20 characters, the length of the fallback glob prefix
    for name in (f"{prefix}_1", f"{prefix}_2"):
        (tmp_path / "mot_out" / name).mkdir(parents=True)
    assert core._resolve_gt_source_root(gt_cfg(tmp_path), f"{prefix}_2") == str(tmp_path / "mot_out" / f"{prefix}_2")


def test_resolve_gt_source_root_returns_none_when_nothing_matches(tmp_path: Path) -> None:
    (tmp_path / "mot_out").mkdir()
    assert core._resolve_gt_source_root(gt_cfg(tmp_path), "no_such_sequence") is None


# Known gap G3 (openspec change add-config-and-grid-search-unit-tests, design.md): desired behavior, expected to fail
# until the prefix fallback refuses ambiguous matches.
@pytest.mark.xfail(strict=True, reason="G3: the 20-character prefix fallback can return another sequence's GT directory")
def test_resolve_gt_source_root_does_not_return_a_sibling_sequence(tmp_path: Path) -> None:
    prefix = "C000025 sequence_dup"
    for name in (f"{prefix}_1", f"{prefix}_2"):
        (tmp_path / "mot_out" / name).mkdir(parents=True)
    assert core._resolve_gt_source_root(gt_cfg(tmp_path), f"{prefix}_3") is None


def det_cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(OUTPUT_ROOT=str(tmp_path), run_det_eval_dir="det")


def test_discover_videos_lists_only_sequences_with_a_video_in_sorted_order(tmp_path: Path) -> None:
    for sequence, hasVideo in (("seq_c", True), ("seq_a", True), ("seq_b", False)):
        directory = tmp_path / "det" / sequence
        directory.mkdir(parents=True)
        if hasVideo:
            (directory / f"{sequence}.mp4").write_bytes(b"")
    found = core.discover_grid_search_videos(det_cfg(tmp_path), None, max_videos=10)
    assert found == [str(tmp_path / "det" / "seq_a" / "seq_a.mp4"), str(tmp_path / "det" / "seq_c" / "seq_c.mp4")]
    assert core.discover_grid_search_videos(det_cfg(tmp_path), None, max_videos=1) == found[:1]


def test_discover_videos_returns_empty_when_stage1_output_is_missing(tmp_path: Path) -> None:
    assert core.discover_grid_search_videos(det_cfg(tmp_path), None, max_videos=5) == []


def test_discover_videos_glob_override_replaces_the_default_scan(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("b.mp4", "a.mp4"):
        (videos / name).write_bytes(b"")
    found = core.discover_grid_search_videos(det_cfg(tmp_path), str(videos / "*.mp4"), max_videos=1)
    assert found == [str(videos / "a.mp4")]


class FakeCapture:
    def __init__(self, path: str) -> None:
        self.path = path

    def get(self, prop: int) -> float:
        return 640.0

    def release(self) -> None:
        return None


@pytest.fixture
def trial_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Replace every heavy collaborator of score_params and record what the tracking step sees."""
    monkeypatch.setitem(config.CLASS_TRACKING_PARAMS, "scooter", {"track_buffer": 30, "use_reid": False, "match_thresh": 0.8})
    monkeypatch.setitem(config.CLASS_TRACKING_PARAMS, "car", {"track_buffer": 45, "use_reid": False})
    gtRoot = tmp_path / "gt_root"
    gtRoot.mkdir()
    env = SimpleNamespace(seen=[], scores=[], missing_gt=set(), tracking_error=None, gt_root=gtRoot)

    def fake_tracking(**kwargs: Any) -> None:
        env.seen.append(copy.deepcopy(dict(config.CLASS_TRACKING_PARAMS)))
        if env.tracking_error is not None:
            raise env.tracking_error

    def fake_resolve_gt(cfg: Any, seq_stem: str) -> str | None:
        return None if seq_stem in env.missing_gt else str(env.gt_root)

    monkeypatch.setattr(core, "Detector", lambda model_path: object())
    monkeypatch.setattr(core, "_run_tracking_for_video", fake_tracking)
    monkeypatch.setattr(core.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(core, "save_mot_format", lambda **kwargs: None)
    monkeypatch.setattr(core, "_resolve_gt_source_root", fake_resolve_gt)
    monkeypatch.setattr(core, "_setup_trial_gt", lambda **kwargs: True)
    monkeypatch.setattr(core, "run_trackeval_mot", lambda *args, **kwargs: {})
    monkeypatch.setattr(core, "extract_assa_mean", lambda results, target_class, tracker_variant: env.scores.pop(0))
    return env


def run_score(tmp_path: Path, params: dict[str, Any], videos: tuple[str, ...] = ("a.mp4",)) -> float:
    cfg = SimpleNamespace(tracker_name="YOLO11x", benchmark_name="MOT", split_name="val")
    return core.score_params(
        params=params, target_class="scooter", video_paths=[str(tmp_path / name) for name in videos],
        model_path="unused.pt", cfg=cfg, output_root=tmp_path / "trial",
    )


def test_trial_sees_merged_parameters_for_the_target_class_only(trial_env: SimpleNamespace, tmp_path: Path) -> None:
    trial_env.scores = [0.5]
    run_score(tmp_path, {"match_thresh": 0.5, "new_track_thresh": 0.3})
    [duringTrial] = trial_env.seen
    # Unspecified fields such as use_reid survive the merge; other classes are untouched.
    assert duringTrial["scooter"] == {"track_buffer": 30, "use_reid": False, "match_thresh": 0.5, "new_track_thresh": 0.3}
    assert duringTrial["car"] == {"track_buffer": 45, "use_reid": False}


def test_parameters_are_restored_after_a_normal_trial(trial_env: SimpleNamespace, tmp_path: Path) -> None:
    before = copy.deepcopy(dict(config.CLASS_TRACKING_PARAMS))
    trial_env.scores = [0.5]
    assert run_score(tmp_path, {"match_thresh": 0.5}) == pytest.approx(0.5)
    assert dict(config.CLASS_TRACKING_PARAMS) == before


def test_parameters_are_restored_and_minus_infinity_returned_after_an_exception(trial_env: SimpleNamespace, tmp_path: Path) -> None:
    before = copy.deepcopy(dict(config.CLASS_TRACKING_PARAMS))
    trial_env.tracking_error = RuntimeError("boom")
    assert run_score(tmp_path, {"match_thresh": 0.5}) == float("-inf")
    assert dict(config.CLASS_TRACKING_PARAMS) == before


def test_parameters_are_restored_when_every_sequence_is_skipped_for_missing_gt(trial_env: SimpleNamespace, tmp_path: Path) -> None:
    before = copy.deepcopy(dict(config.CLASS_TRACKING_PARAMS))
    trial_env.missing_gt = {"a", "b"}
    assert run_score(tmp_path, {"match_thresh": 0.5}, videos=("a.mp4", "b.mp4")) == float("-inf")
    assert dict(config.CLASS_TRACKING_PARAMS) == before


def test_zero_scores_are_dropped_before_taking_the_25th_percentile(trial_env: SimpleNamespace, tmp_path: Path) -> None:
    trial_env.scores = [0.0, 0.2, 0.4, 0.6, 0.8]
    result = run_score(tmp_path, {"match_thresh": 0.5}, videos=("a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"))
    assert result == pytest.approx(float(np.percentile([0.2, 0.4, 0.6, 0.8], 25)))


@pytest.mark.parametrize(("scores", "videos"), [([None, 0.0], ("a.mp4", "b.mp4")), ([], ())])
def test_without_a_usable_score_the_result_is_minus_infinity(
    trial_env: SimpleNamespace, tmp_path: Path, scores: list[float | None], videos: tuple[str, ...],
) -> None:
    trial_env.scores = scores
    assert run_score(tmp_path, {"match_thresh": 0.5}, videos=videos) == float("-inf")


# Known gap G2 (openspec change add-config-and-grid-search-unit-tests, design.md): desired behavior, expected to fail
# until score_params removes the key it added for a class that had no entry before the trial.
@pytest.mark.xfail(strict=True, reason="G2: a class without an entry is restored as {} instead of being removed")
def test_class_without_an_entry_is_removed_again_after_the_trial(
    trial_env: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(config.CLASS_TRACKING_PARAMS, "scooter")
    trial_env.scores = [0.5]
    run_score(tmp_path, {"match_thresh": 0.5})
    assert "scooter" not in config.CLASS_TRACKING_PARAMS


FULL_GRID = {"a": [1, 2], "b": [10, 20]}
SCORE_TABLE = {(1, 10): 0.1, (1, 20): 0.4, (2, 10): 0.3, (2, 20): 0.2}


def install_scores(monkeypatch: pytest.MonkeyPatch, table: dict[tuple[int, int], float]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_score_params(**kwargs: Any) -> float:
        params = kwargs["params"]
        calls.append(dict(params))
        return table[(params["a"], params["b"])]

    monkeypatch.setattr(core, "score_params", fake_score_params)
    return calls


def make_runner(tmp_path: Path, *, grid: dict[str, list] | None = None, resume: bool = False) -> core.GridSearchRunner:
    return core.GridSearchRunner(
        param_grid=grid if grid is not None else FULL_GRID, target_class="scooter", video_paths=[], model_path="unused.pt",
        cfg=None, output_root=tmp_path / "gs", result_csv=tmp_path / "results.csv", resume=resume,
    )


def read_rows(tmp_path: Path) -> list[dict[str, str]]:
    with open(tmp_path / "results.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_run_records_every_trial_and_tracks_the_best(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_scores(monkeypatch, SCORE_TABLE)
    outcome = make_runner(tmp_path).run()
    assert outcome["best_params"] == {"a": 1, "b": 20}
    assert outcome["best_score"] == 0.4
    assert [entry["params"] for entry in outcome["all_results"]] == [
        {"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20},
    ]
    rows = read_rows(tmp_path)
    assert len(rows) == 4
    assert json.loads(rows[1]["params"]) == {"a": 1, "b": 20}
    assert rows[1]["score"] == "0.400000"
    assert (tmp_path / "results.csv").read_text(encoding="utf-8").count("combo_id,score") == 1


def test_ties_keep_the_first_best(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_scores(monkeypatch, {key: 0.5 for key in SCORE_TABLE})
    assert make_runner(tmp_path).run()["best_params"] == {"a": 1, "b": 10}


def test_resume_scores_only_unrecorded_combinations_and_keeps_the_overall_best(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    install_scores(monkeypatch, SCORE_TABLE)
    make_runner(tmp_path, grid={"a": [1], "b": [10, 20]}).run()
    calls = install_scores(monkeypatch, SCORE_TABLE)
    outcome = make_runner(tmp_path, resume=True).run()
    assert calls == [{"a": 2, "b": 10}, {"a": 2, "b": 20}]
    assert len(outcome["all_results"]) == 4
    assert outcome["best_params"] == {"a": 1, "b": 20}
    assert len(read_rows(tmp_path)) == 4


def test_without_resume_every_combination_is_scored_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_scores(monkeypatch, SCORE_TABLE)
    make_runner(tmp_path).run()
    calls = install_scores(monkeypatch, SCORE_TABLE)
    make_runner(tmp_path).run()
    assert len(calls) == 4


# Known gap G5 (openspec change add-config-and-grid-search-unit-tests, design.md): desired behavior, expected to fail
# until a failed trial (-inf) is no longer counted as completed. The expectation itself is still to be confirmed.
@pytest.mark.xfail(strict=True, reason="G5: a failed (-inf) trial is skipped by --resume as if it had completed")
def test_resume_retries_a_failed_trial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_scores(monkeypatch, {(1, 10): float("-inf")})
    make_runner(tmp_path, grid={"a": [1], "b": [10]}).run()
    calls = install_scores(monkeypatch, SCORE_TABLE)
    make_runner(tmp_path, grid={"a": [1], "b": [10]}, resume=True).run()
    assert {"a": 1, "b": 10} in calls
