"""
Stage1 MOT tracks for the conflict analysis (spec「位置取點與軌跡識別」).

Each box is reduced to its bottom-centre (ground contact) point, projected to the top-down
plane with the camera homography and scaled to metres. ROI membership is still decided by
the box centre, as in the existing stage3 (stage1 tracks are not ROI-filtered; ROI is applied downstream). Tracks are keyed by (class, track_id)
because track ids are only unique within a class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from safetrajai.collision.trajectory_adapter import _read_class_trajectory_file

from .config import CLASSES


@dataclass(frozen=True)
class Track:
    class_name: str
    track_id: int
    frames: np.ndarray    # (N,) int, ascending
    cctv_xy: np.ndarray   # (N, 2) bottom-centre in CCTV pixels
    map_xy: np.ndarray    # (N, 2) top-down plane pixels
    xy_m: np.ndarray      # (N, 2) metres

    @property
    def key(self) -> Tuple[str, int]:
        return (self.class_name, self.track_id)


def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))[:, 0, :]


def load_tracks(
    seq_output_dir: Path,
    benchmark_name: str,
    split_name: str,
    tracker_name: str,
    roi: np.ndarray,
    H: np.ndarray,
    m_per_pixel: float,
) -> Tuple[Track, ...]:
    """All tracks of one sequence, ordered by class contract order then track id."""
    seq_output_dir = Path(seq_output_dir)
    traj_root = seq_output_dir / "trackers" / "mot_challenge" / f"{benchmark_name}-{split_name}"
    tracks = []
    for class_name in CLASSES:
        path = traj_root / f"{tracker_name}__{class_name}" / "data" / f"{seq_output_dir.name}.txt"
        for track_id, rows in sorted(_read_class_trajectory_file(path, roi).items()):
            rows = sorted(rows, key=lambda r: r[0])
            frames = np.array([r[0] for r in rows], dtype=np.int64)
            cctv = np.array([(left + w / 2.0, top + h) for _, left, top, w, h, _ in rows], dtype=np.float64)
            map_xy = project_points(cctv, H)
            tracks.append(Track(class_name, int(track_id), frames, cctv, map_xy, map_xy * m_per_pixel))
    return tuple(tracks)
