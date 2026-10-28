"""Video header probing shared by stage1, stage3 and grid search."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import safetrajai.pipeline.tracking_pipeline as tracking_pipeline
from safetrajai.evaluation.utils import probe_video


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_reads_size_and_fps_from_the_header(tiny_video: Path) -> None:
    assert probe_video(tiny_video) == (64, 48, 5.0)


def test_accepts_str_paths(tiny_video: Path) -> None:
    assert probe_video(str(tiny_video)) == (64, 48, 5.0)


def test_unreadable_video_gives_zeros_instead_of_raising(tmp_path: Path) -> None:
    assert probe_video(tmp_path / "missing.mp4") == (0, 0, 0.0)


def test_tracking_pipeline_uses_the_shared_probe() -> None:
    assert "CAP_PROP_FRAME_WIDTH" not in Path(tracking_pipeline.__file__).read_text(encoding="utf-8")
