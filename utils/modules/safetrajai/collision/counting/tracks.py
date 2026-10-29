"""Stage1 tracks for the counting lines: bottom-centre of every box in CCTV pixels, not ROI-filtered."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.collision.counting.crossing import CountingTrack
from safetrajai.collision.trajectory_adapter import _read_class_trajectory_file

CLASSES: Tuple[str, ...] = tuple(CLASS_NAMES[i] for i in sorted(CLASS_NAMES))
# Stage1 tracks are not ROI-filtered; the reader wants a polygon, so give it one that holds every box centre.
EVERYWHERE_POLYGON = np.array([[-1e9, -1e9], [1e9, -1e9], [1e9, 1e9], [-1e9, 1e9]], dtype=np.float32)


def load_counting_tracks(seq_output_dir: Path, cfg) -> Tuple[CountingTrack, ...]:
    traj_root = seq_output_dir / "trackers" / "mot_challenge" / f"{cfg.benchmark_name}-{cfg.split_name}"
    tracks = []
    for class_name in CLASSES:
        path = traj_root / f"{cfg.tracker_name}__{class_name}" / "data" / f"{seq_output_dir.name}.txt"
        for track_id, rows in sorted(_read_class_trajectory_file(path, EVERYWHERE_POLYGON).items()):
            rows = sorted(rows, key=lambda r: r[0])
            frames = np.array([r[0] for r in rows], dtype=np.int64)
            points = np.array([(left + w / 2.0, top + h) for _, left, top, w, h, _ in rows], dtype=np.float64)
            tracks.append(CountingTrack(class_name, int(track_id), frames, points))
    return tuple(tracks)
