"""Per-frame drawing of the bird's-eye video (spec F3): satellite base map, class-coloured trails and ids, speed.

Pure functions on numpy images; positions are satellite-image pixels and are scaled once by `scale`
(see `bev_scale`). The caller owns the video writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from safetrajai.classes import CLASS_NAMES

CLASS_ORDER: Tuple[str, ...] = tuple(CLASS_NAMES[i] for i in sorted(CLASS_NAMES))
# BGR, one fixed colour per class of the class contract.
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 140, 255), "car": (0, 200, 0), "scooter": (255, 0, 255),
    "small_truck": (255, 200, 0), "bus": (0, 0, 255), "truck": (255, 0, 0),
}
_WHITE = (255, 255, 255)


@dataclass(frozen=True)
class BevTrack:
    class_name: str
    track_id: int
    frames: np.ndarray      # (N,) int, ascending
    map_xy: np.ndarray      # (N, 2) satellite-image pixels, unscaled
    speed_kmh: np.ndarray   # (N,) NaN where unknown


def bev_scale(size_wh: Tuple[int, int], max_side_px: int) -> float:
    """Factor that fits the longest side into max_side_px; never enlarges."""
    return min(1.0, max_side_px / max(size_wh))


def _font_scale(image: np.ndarray) -> float:
    return max(0.3, min(image.shape[:2]) / 1800.0)


def _draw_legend(image: np.ndarray) -> None:
    font = _font_scale(image)
    row = int(30 * font) + 3
    for index, class_name in enumerate(CLASS_ORDER):
        top = 4 + index * row
        cv2.rectangle(image, (4, top), (4 + row - 3, top + row - 3), CLASS_COLORS_BGR[class_name], -1)
        cv2.putText(image, class_name, (8 + row, top + row - 4), cv2.FONT_HERSHEY_SIMPLEX, font, _WHITE, 1, cv2.LINE_AA)


def draw_bev_frame(
    image: np.ndarray, scale: float, frame: int, tracks: Sequence[BevTrack], tail_frames: int,
) -> np.ndarray:
    """A copy of `image` with the tracks present at `frame` (trail of the last `tail_frames` frames, id, speed)."""
    out = image.copy()
    font = _font_scale(out)
    radius = max(2, int(round(10 * font)))
    for track in tracks:
        current = np.flatnonzero(track.frames == frame)
        if not len(current):
            continue
        color = CLASS_COLORS_BGR[track.class_name]
        in_tail = (track.frames >= frame - tail_frames) & (track.frames <= frame)
        trail = np.rint(track.map_xy[in_tail] * scale).astype(np.int32)
        cv2.polylines(out, [trail.reshape(-1, 1, 2)], False, color, 1)
        x, y = (int(v) for v in np.rint(track.map_xy[current[0]] * scale))
        cv2.circle(out, (x, y), radius, color, -1)
        speed = track.speed_kmh[current[0]]
        label = f"{track.track_id}" if np.isnan(speed) else f"{track.track_id} {speed:.0f}km/h"
        cv2.putText(out, label, (x + radius + 3, y - radius), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)
    _draw_legend(out)
    return out
