"""Counting-line crossings of tracks (pure functions, CCTV pixel coordinates, image y axis down).

For a line p1 -> p2 with d = p2 - p1, side(p) = dx * (py - y1) - dy * (px - x1). A point with
side < 0 is on the left of the drawing direction (as seen on screen), side >= 0 on its right.
Crossing from left to right is "forward", right to left "backward". A track crosses when two
consecutive samples change side and the path between them meets the segment p1-p2 (not only its
extension). A track counts once per line and direction, so jitter on the line adds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.collision.counting.lines import CountingLine, Point

FORWARD = "forward"
BACKWARD = "backward"
_CLASS_RANK = {name: class_id for class_id, name in CLASS_NAMES.items()}


@dataclass(frozen=True)
class CountingTrack:
    class_name: str
    track_id: int
    frames: np.ndarray    # (N,) int, ascending
    points: np.ndarray    # (N, 2) bottom-centre in CCTV pixels


@dataclass(frozen=True)
class Crossing:
    line: str
    direction: str
    class_name: str
    track_id: int
    frame: int


def side_of(line: CountingLine, point: Point) -> float:
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    return dx * (point[1] - line.p1[1]) - dy * (point[0] - line.p1[0])


def _meets_segment(line: CountingLine, start: Point, end: Point, side_start: float, side_end: float) -> bool:
    t = side_start / (side_start - side_end)
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    u = ((x - line.p1[0]) * dx + (y - line.p1[1]) * dy) / (dx * dx + dy * dy)
    return 0.0 <= u <= 1.0


def _track_crossings(track: CountingTrack, line: CountingLine) -> List[Crossing]:
    found: Dict[str, Crossing] = {}
    sides = [side_of(line, (float(x), float(y))) for x, y in track.points]
    for index in range(1, len(sides)):
        was_left, is_left = sides[index - 1] < 0, sides[index] < 0
        if was_left == is_left:
            continue
        start = (float(track.points[index - 1][0]), float(track.points[index - 1][1]))
        end = (float(track.points[index][0]), float(track.points[index][1]))
        if not _meets_segment(line, start, end, sides[index - 1], sides[index]):
            continue
        direction = FORWARD if was_left else BACKWARD
        found.setdefault(direction, Crossing(line.name, direction, track.class_name, track.track_id,
                                             int(track.frames[index])))
    return list(found.values())


def find_crossings(tracks: Iterable[CountingTrack], lines: Sequence[CountingLine]) -> Tuple[Crossing, ...]:
    """Crossings ordered by frame, line (as given), class (contract order) and track id."""
    order = {line.name: index for index, line in enumerate(lines)}
    crossings = [c for track in tracks for line in lines for c in _track_crossings(track, line)]
    crossings.sort(key=lambda c: (c.frame, order[c.line], _CLASS_RANK.get(c.class_name, len(_CLASS_RANK)), c.track_id))
    return tuple(crossings)
