"""Per-frame drawing of the bird's-eye video (spec F3): satellite base map, class-coloured trails and ids, speed.

Pure functions on numpy images; positions are satellite-image pixels and are scaled once by `scale`
(see `bev_scale`). The caller owns the video writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.collision.counting.crossing import Crossing
from safetrajai.pipeline.video_io import VideoWriterWithFfmpeg

from .config import BevVideoConfig
from .kinematics import Kinematics

CLASS_ORDER: Tuple[str, ...] = tuple(CLASS_NAMES[i] for i in sorted(CLASS_NAMES))
# BGR, one fixed colour per class of the class contract.
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 140, 255), "car": (0, 200, 0), "scooter": (255, 0, 255),
    "small_truck": (255, 200, 0), "bus": (0, 0, 255), "truck": (255, 0, 0),
}
_WHITE = (255, 255, 255)
_EVENT_LINK_BGR = (0, 0, 255)
_COUNTING_LINE_BGR = (0, 255, 255)
_PANEL_BGR = (30, 30, 30)


@dataclass(frozen=True)
class BevTrack:
    class_name: str
    track_id: int
    frames: np.ndarray      # (N,) int, ascending
    map_xy: np.ndarray      # (N, 2) satellite-image pixels, unscaled
    speed_kmh: np.ndarray   # (N,) NaN where unknown


@dataclass(frozen=True)
class BevEvent:
    """A conflict event: tracks i and j (indices into the drawn tracks) are linked from start to end frame."""
    start_frame: int
    end_frame: int
    i: int
    j: int


@dataclass(frozen=True)
class BevCountingLine:
    name: str
    p1: Tuple[float, float]   # satellite-image pixels, unscaled
    p2: Tuple[float, float]
    forward_label: str
    backward_label: str


@dataclass(frozen=True)
class BevCounting:
    lines: Tuple[BevCountingLine, ...]
    crossings: Tuple[Crossing, ...]


def cumulative_counts(counting: BevCounting, frame: int) -> Dict[Tuple[str, str], int]:
    """Crossings up to and including `frame`, all classes, for every line and both directions."""
    counts = {(line.name, direction): 0 for line in counting.lines for direction in ("forward", "backward")}
    for crossing in counting.crossings:
        if crossing.frame <= frame:
            counts[(crossing.line, crossing.direction)] += 1
    return counts


def bev_scale(size_wh: Tuple[int, int], max_side_px: int) -> float:
    """Factor that fits the longest side into max_side_px; never enlarges."""
    return min(1.0, max_side_px / max(size_wh))


def _font_scale(image: np.ndarray) -> float:
    return max(0.3, min(image.shape[:2]) / 1800.0)


def _draw_legend(image: np.ndarray) -> None:
    font = _font_scale(image)
    row = int(30 * font) + 3
    text_width = max(cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0] for name in CLASS_ORDER)
    cv2.rectangle(image, (0, 0), (12 + row + text_width, 8 + row * len(CLASS_ORDER)), _PANEL_BGR, -1)
    for index, class_name in enumerate(CLASS_ORDER):
        top = 4 + index * row
        cv2.rectangle(image, (4, top), (4 + row - 3, top + row - 3), CLASS_COLORS_BGR[class_name], -1)
        cv2.putText(image, class_name, (8 + row, top + row - 4), cv2.FONT_HERSHEY_SIMPLEX, font, _WHITE, 1, cv2.LINE_AA)


def _draw_counting(out: np.ndarray, scale: float, frame: int, counting: BevCounting) -> None:
    font = _font_scale(out)
    for line in counting.lines:
        p1 = tuple(int(v) for v in np.rint(np.array(line.p1) * scale))
        p2 = tuple(int(v) for v in np.rint(np.array(line.p2) * scale))
        cv2.line(out, p1, p2, _COUNTING_LINE_BGR, 2)
    counts = cumulative_counts(counting, frame)
    rows = [f"{l.name}  {l.forward_label} {counts[(l.name, 'forward')]} / {l.backward_label} {counts[(l.name, 'backward')]}"
            for l in counting.lines]
    sizes = [cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0] for r in rows]
    row_h = int(30 * font) + 6
    width = max(w for w, _ in sizes) + 8
    left = out.shape[1] - width - 4
    cv2.rectangle(out, (left, 4), (left + width, 4 + row_h * len(rows) + 2), _PANEL_BGR, -1)
    for index, row in enumerate(rows):
        cv2.putText(out, row, (left + 4, 4 + row_h * (index + 1) - 3), cv2.FONT_HERSHEY_SIMPLEX, font, _WHITE, 1, cv2.LINE_AA)


def draw_bev_frame(
    image: np.ndarray, scale: float, frame: int, tracks: Sequence[BevTrack], tail_frames: int,
    *, events: Sequence[BevEvent] = (), counting: "BevCounting | None" = None,
) -> np.ndarray:
    """A copy of `image` with the tracks present at `frame` (trail of the last `tail_frames` frames, id, speed)."""
    out = image.copy()
    font = _font_scale(out)
    radius = max(2, int(round(10 * font)))
    if counting is not None:
        _draw_counting(out, scale, frame, counting)
    for event in events:
        if not event.start_frame <= frame <= event.end_frame:
            continue
        ends = []
        for track in (tracks[event.i], tracks[event.j]):
            at = np.flatnonzero(track.frames == frame)
            if len(at):
                ends.append(tuple(int(v) for v in np.rint(track.map_xy[at[0]] * scale)))
        if len(ends) == 2:
            cv2.line(out, ends[0], ends[1], _EVENT_LINK_BGR, 1)
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


def bev_tracks(kins: Sequence[Kinematics]) -> Tuple[BevTrack, ...]:
    """Tracks with the smoothed speed of the conflict analysis in km/h (NaN where it is undefined)."""
    return tuple(
        BevTrack(k.track.class_name, k.track.track_id, k.track.frames, k.track.map_xy, k.speed * 3.6) for k in kins)


def write_bev_video(
    out_path: Path, background: np.ndarray, tracks: Sequence[BevTrack], cfg: BevVideoConfig, fps: float,
    events: Sequence[BevEvent] = (), counting: "BevCounting | None" = None,
) -> Dict[str, Any]:
    """Write `out_path` from the first to the last frame that has a track; returns what was written."""
    if not tracks:
        return {"written": False, "skipped_reason": "no_tracks"}
    height, width = background.shape[:2]
    scale = bev_scale((width, height), cfg.max_side_px)
    out_w, out_h = int(round(width * scale)) // 2 * 2, int(round(height * scale)) // 2 * 2   # even sizes for H.264
    base = cv2.resize(background, (out_w, out_h)) if (out_w, out_h) != (width, height) else background
    tail_frames = max(1, int(round(cfg.tail_s * fps)))
    first = int(min(t.frames[0] for t in tracks))
    last = int(max(t.frames[-1] for t in tracks))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriterWithFfmpeg(out_path, 5, out_w, out_h)
    for frame in range(first, last + 1):
        writer.write(draw_bev_frame(base, scale, frame, tracks, tail_frames, events=events, counting=counting))
    writer.close()
    info = {"written": True, "frames": last - first + 1, "size": [out_w, out_h], "scale": scale,
            "tail_frames": tail_frames, "event_links": len(events)}
    if counting is not None:
        final = cumulative_counts(counting, last)
        info["counting"] = {"lines": [l.name for l in counting.lines],
                            "final_counts": {l.name: {d: final[(l.name, d)] for d in ("forward", "backward")}
                                             for l in counting.lines}}
    return info
