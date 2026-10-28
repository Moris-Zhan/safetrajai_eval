"""Sequence selection and soft-skip behaviour of TrackingEvalPipeline.run_stage3."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import safetrajai.pipeline.tracking_pipeline as tracking_pipeline
from safetrajai.evaluation import dataset_layout


def make_cfg(tmp_path: Path, conflict_enabled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        OUTPUT_ROOT=tmp_path, run_det_eval_dir="det", collision_output_dir=tmp_path / "collision",
        homography_camera_dir=tmp_path / "homography", frame_rate=10,
        conflict_analysis=SimpleNamespace(enabled=conflict_enabled),
    )


def make_sequence(det_dir: Path, name: str, with_video: bool) -> Path:
    seq_dir = det_dir / name
    seq_dir.mkdir(parents=True)
    if with_video:
        avi = seq_dir / f"{name}.avi"
        writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
        for _ in range(2):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        avi.rename(seq_dir / f"{name}.mp4")
    return seq_dir


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    recorded = {"collision": [], "conflict": []}
    monkeypatch.setattr(tracking_pipeline, "run_collision_detection_for_sequence",
                        lambda **kwargs: recorded["collision"].append(kwargs))
    monkeypatch.setattr(tracking_pipeline, "run_conflict_analysis_for_sequence",
                        lambda **kwargs: recorded["conflict"].append(kwargs))
    return recorded


@pytest.mark.parametrize("create_empty_dir", [False, True])
def test_missing_stage1_output_is_a_soft_skip(tmp_path: Path, calls: dict, capsys, create_empty_dir: bool) -> None:
    if create_empty_dir:
        (tmp_path / "det").mkdir()

    assert tracking_pipeline.TrackingEvalPipeline(make_cfg(tmp_path)).run_stage3() is None

    assert f"No stage1 output found under: {tmp_path / 'det'}" in capsys.readouterr().out
    assert calls == {"collision": [], "conflict": []}


def test_only_sequences_with_a_video_are_processed(tmp_path: Path, calls: dict, capsys) -> None:
    det_dir = tmp_path / "det"
    seq_a = make_sequence(det_dir, "seqA", with_video=True)
    make_sequence(det_dir, "seqB", with_video=False)
    (det_dir / "notes.txt").write_text("not a sequence")

    tracking_pipeline.TrackingEvalPipeline(make_cfg(tmp_path)).run_stage3()

    assert len(calls["collision"]) == 1
    call = calls["collision"][0]
    assert call["seq_output_dir"] == seq_a
    assert call["fps"] == 5.0
    assert call["bounds"] == (48, 64)
    assert calls["conflict"] == []
    assert "seqB" in capsys.readouterr().out


def test_conflict_analysis_runs_after_collision_when_enabled(tmp_path: Path, calls: dict) -> None:
    make_sequence(tmp_path / "det", "seqA", with_video=True)

    tracking_pipeline.TrackingEvalPipeline(make_cfg(tmp_path, conflict_enabled=True)).run_stage3()

    assert [call["seq_output_dir"] for call in calls["conflict"]] == [call["seq_output_dir"] for call in calls["collision"]]
    assert calls["conflict"][0]["bounds"] == (48, 64)


def test_stage1_output_is_listed_once(tmp_path: Path, calls: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    make_sequence(tmp_path / "det", "seqA", with_video=True)
    listings = []
    original = dataset_layout.list_stage1_sequences
    monkeypatch.setattr(dataset_layout, "list_stage1_sequences", lambda path: listings.append(path) or original(path))

    tracking_pipeline.TrackingEvalPipeline(make_cfg(tmp_path)).run_stage3()

    assert len(listings) == 1
