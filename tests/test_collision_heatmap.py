"""draw_collision_heatmap writes the TTC, PET and conflict event heat maps."""

from pathlib import Path

import cv2
import numpy as np

from safetrajai.collision.output import draw_collision_heatmap


def background() -> np.ndarray:
    return np.full((120, 160, 3), 90, dtype=np.uint8)


def test_heatmap_is_written_with_background_size(tmp_path: Path) -> None:
    output_path = tmp_path / "heatmap.png"
    assert draw_collision_heatmap([(40, 50), (42, 52), (120, 80)], background(), output_path) is True
    assert cv2.imread(str(output_path)).shape[:2] == (120, 160)


def test_no_points_writes_nothing_and_removes_stale_file(tmp_path: Path) -> None:
    output_path = tmp_path / "heatmap.png"
    assert draw_collision_heatmap([(40, 50)], background(), output_path) is True
    assert draw_collision_heatmap([], background(), output_path) is False
    assert not output_path.exists()
