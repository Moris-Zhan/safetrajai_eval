"""Stage3 counting-line statistics of one sequence (synthetic stage1 output)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from safetrajai.collision.counting.config import CountingConfig
from safetrajai.collision.counting.runner import run_counting_for_sequence

SEQ = "C000025 seq_a"
LINES_YAML = """camera: C000025
image_size: [64, 48]
lines:
  - name: gate
    points: [[0, 10], [10, 10]]
    forward_label: down
    backward_label: up
"""
HEADER = "frame,time_s,line,direction,label,class,track_id,cumulative"


def write_tracks(seq_dir: Path, rows_by_class: dict) -> None:
    """rows: (frame, track_id, bottom_centre_x, bottom_centre_y) drawn as 10x10 boxes."""
    for class_name in ("person", "car", "scooter", "small_truck", "bus", "truck"):
        data_dir = seq_dir / "trackers" / "mot_challenge" / "CCTV-val" / f"YOLO11x__{class_name}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = "".join(f"{f},{i},{x - 5},{y - 10},10,10,0.9\n" for f, i, x, y in rows_by_class.get(class_name, []))
        (data_dir / f"{seq_dir.name}.txt").write_text(text)


@pytest.fixture
def setup(tmp_path: Path):
    seq_dir = tmp_path / "det" / SEQ
    seq_dir.mkdir(parents=True)
    lines_dir = tmp_path / "counting_lines"
    lines_dir.mkdir()
    (lines_dir / "C000025.yaml").write_text(LINES_YAML, encoding="utf-8")
    cfg = SimpleNamespace(counting=CountingConfig(enabled=True), collision_output_dir=tmp_path / "out",
                          benchmark_name="CCTV", split_name="val", tracker_name="YOLO11x")
    return seq_dir, cfg, lines_dir, tmp_path / "out" / SEQ / "counting"


def run(seq_dir: Path, cfg, lines_dir: Path, fps: float = 5.0):
    return run_counting_for_sequence(seq_dir, cfg, fps=fps, bounds=(48, 64), lines_dir=lines_dir)


def test_disabled_writes_nothing_and_returns_none(setup) -> None:
    seq_dir, cfg, lines_dir, out_dir = setup
    cfg.counting = CountingConfig(enabled=False)
    write_tracks(seq_dir, {"car": [(1, 1, 5, 5), (2, 1, 5, 15)]})

    assert run(seq_dir, cfg, lines_dir) is None
    assert not out_dir.exists()


def test_camera_without_lines_only_writes_a_skipped_summary(setup) -> None:
    seq_dir, cfg, lines_dir, out_dir = setup
    (lines_dir / "C000025.yaml").unlink()

    summary = run(seq_dir, cfg, lines_dir)

    assert summary["skipped_reason"] == "no_counting_lines"
    assert sorted(p.name for p in out_dir.iterdir()) == ["summary.json"]
    assert json.loads((out_dir / "summary.json").read_text())["skipped_reason"] == "no_counting_lines"


def test_crossings_are_counted_per_line_direction_and_class(setup) -> None:
    seq_dir, cfg, lines_dir, out_dir = setup
    write_tracks(seq_dir, {
        "car": [(1, 1, 5, 5), (2, 1, 5, 15), (3, 2, 5, 5), (4, 2, 5, 15), (5, 3, 5, 15), (6, 3, 5, 5)],
        "scooter": [(1, 1, 5, 5), (2, 1, 5, 15)],
    })

    summary = run(seq_dir, cfg, lines_dir, fps=5.0)

    rows = (out_dir / "counts.csv").read_text().splitlines()
    assert rows == [
        HEADER,
        "2,0.4,gate,forward,down,car,1,1",
        "2,0.4,gate,forward,down,scooter,1,1",
        "4,0.8,gate,forward,down,car,2,2",
        "6,1.2,gate,backward,up,car,3,1",
    ]
    totals = summary["totals"]["gate"]
    assert totals["forward"]["car"] == 2 and totals["forward"]["scooter"] == 1 and totals["forward"]["person"] == 0
    assert totals["forward"]["all"] == 3 and totals["backward"]["all"] == 1 and totals["backward"]["car"] == 1
    assert summary["skipped_reason"] is None and summary["num_tracks"] == 4
    assert json.loads((out_dir / "summary.json").read_text())["totals"] == summary["totals"]


def test_summary_states_its_limitations_and_the_lines(setup) -> None:
    seq_dir, cfg, lines_dir, out_dir = setup
    write_tracks(seq_dir, {})

    summary = run(seq_dir, cfg, lines_dir)

    assert {item["code"] for item in summary["limitations"]} == {"id_switches", "not_roi_filtered"}
    assert summary["lines"] == [{"name": "gate", "points": [[0.0, 10.0], [10.0, 10.0]],
                                 "forward_label": "down", "backward_label": "up"}]
    assert (out_dir / "counts.csv").read_text() == HEADER + "\n"


def test_image_size_mismatch_is_an_error(setup) -> None:
    seq_dir, cfg, lines_dir, _ = setup

    with pytest.raises(ValueError, match="image_size"):
        run_counting_for_sequence(seq_dir, cfg, fps=5.0, bounds=(720, 1280), lines_dir=lines_dir)


def test_tracks_are_not_filtered_by_a_roi(setup, tmp_path: Path) -> None:
    seq_dir, cfg, lines_dir, out_dir = setup
    (lines_dir / "C000025.yaml").write_text(LINES_YAML.replace("[[0, 10], [10, 10]]", "[[0, 10], [500, 10]]"))
    write_tracks(seq_dir, {"car": [(1, 1, 300, 5), (2, 1, 300, 15)]})   # centre far outside the 64x48 frame

    assert run(seq_dir, cfg, lines_dir)["totals"]["gate"]["forward"]["car"] == 1
