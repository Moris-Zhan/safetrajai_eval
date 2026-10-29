"""forward/backward labels of a counting line, derived from its geometry instead of typed in.

For a line p1 -> p2 with d = p2 - p1 (image coordinates, y down) the forward travel direction is
(-dy, dx): from the left of the drawing direction to its right. Without a homography it is named
by screen direction (up, up_right, ...). With a homography the direction is moved to the satellite
plane, whose top is north, and named by compass (northbound, northeastbound, ...). Labels typed in
the file win; forward and backward are decided on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from safetrajai.collision.counting.lines import CountingLine

SCREEN_NAMES = ("up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left")
COMPASS_NAMES = tuple(
    f"{name}bound" for name in ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"))
_PROBE_FRACTION = 0.05   # how far (relative to the line length) the homography probes the local direction


@dataclass(frozen=True)
class ResolvedLabels:
    forward: str
    backward: str
    forward_source: str    # "manual" | "compass" | "screen"
    backward_source: str


def _sector(vector: Tuple[float, float], names: Tuple[str, ...]) -> str:
    """Name of the 45-degree sector around `vector` (image coordinates, y down, sector 0 = up)."""
    bearing = math.degrees(math.atan2(vector[0], -vector[1])) % 360.0
    return names[int((bearing + 22.5) // 45) % 8]


def _map_direction(line: CountingLine, vector: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    middle = np.array([(line.p1[0] + line.p2[0]) / 2.0, (line.p1[1] + line.p2[1]) / 2.0])
    length = math.hypot(line.p2[0] - line.p1[0], line.p2[1] - line.p1[1])
    unit = np.array(vector) / math.hypot(*vector)
    probe = np.array([middle, middle + unit * max(1.0, _PROBE_FRACTION * length)], dtype=np.float64)
    moved = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), np.asarray(homography, dtype=np.float64))[:, 0, :]
    return float(moved[1][0] - moved[0][0]), float(moved[1][1] - moved[0][1])


def resolve_labels(line: CountingLine, homography: Optional[np.ndarray] = None) -> ResolvedLabels:
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    forward_vector = (-dy, dx)
    backward_vector = (dy, -dx)
    if homography is None:
        names, source = SCREEN_NAMES, "screen"
        forward_auto = _sector(forward_vector, names)
        backward_auto = _sector(backward_vector, names)
    else:
        names, source = COMPASS_NAMES, "compass"
        forward_auto = _sector(_map_direction(line, forward_vector, homography), names)
        backward_auto = _sector(_map_direction(line, backward_vector, homography), names)
    forward, forward_source = (line.forward_label, "manual") if line.forward_label is not None else (forward_auto, source)
    backward, backward_source = (
        (line.backward_label, "manual") if line.backward_label is not None else (backward_auto, source))
    return ResolvedLabels(forward, backward, forward_source, backward_source)
