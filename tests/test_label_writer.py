"""Per-frame label file written by stage1 and by the grid search tracking loop."""

from pathlib import Path

import numpy as np
import pytest

import safetrajai.grid_search.core as grid_search_core
import safetrajai.pipeline.sequence_tracker as sequence_tracker
from safetrajai.evaluation.mot_format import write_frame_labels


def test_writes_one_normalised_line_per_track(tmp_path: Path) -> None:
    label_file = tmp_path / "frame.txt"
    tracks = np.array([
        [10.0, 20.0, 50.0, 80.0, 3.0, 0.8765, 2.0],
        [0.0, 0.0, 100.0, 200.0, 12.0, 0.5, 0.0],
    ])

    write_frame_labels(label_file, tracks, width=100, height=200)

    assert label_file.read_text() == (
        "2 0.300000 0.250000 0.400000 0.300000 3 0.8765\n"
        "0 0.500000 0.500000 1.000000 1.000000 12 0.5000\n"
    )


def test_ignores_rows_with_fewer_than_seven_columns_and_extra_columns(tmp_path: Path) -> None:
    label_file = tmp_path / "frame.txt"
    tracks = [
        np.array([1.0, 1.0, 5.0, 5.0, 1.0, 0.9]),
        np.array([10.0, 20.0, 50.0, 80.0, 3.0, 0.8765, 2.0, 99.0]),
    ]

    write_frame_labels(label_file, tracks, width=100, height=200)

    assert label_file.read_text() == "2 0.300000 0.250000 0.400000 0.300000 3 0.8765\n"


def test_empty_tracks_still_create_an_empty_file(tmp_path: Path) -> None:
    label_file = tmp_path / "frame.txt"

    write_frame_labels(label_file, np.array([]), width=100, height=200)

    assert label_file.exists() and label_file.read_text() == ""


def test_overwrites_an_existing_file(tmp_path: Path) -> None:
    label_file = tmp_path / "frame.txt"
    label_file.write_text("stale\n")

    write_frame_labels(label_file, np.array([]), width=100, height=200)

    assert label_file.read_text() == ""


@pytest.mark.parametrize("module", [sequence_tracker, grid_search_core])
def test_callers_use_the_shared_writer(module) -> None:
    assert "cx = (x1 + x2) / 2 / width" not in Path(module.__file__).read_text(encoding="utf-8")
