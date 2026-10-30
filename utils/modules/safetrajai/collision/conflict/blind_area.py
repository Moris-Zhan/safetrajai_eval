"""Blind area behind large vehicles, seen from a camera viewpoint, on the satellite pixel grid.

Every occluder is a body rectangle. Its shadow is the region behind it between the two silhouette rays
from the viewpoint. Shadows are rasterised with cv2 (no polygon-clipping dependency) and counted inside
the ROI; the area is accurate to about one pixel along every edge. Objects are treated as flat, so the
shadows overestimate the blind area (nothing is hidden behind the roof line of a real car).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .kinematics import Kinematics, segment_bounds

Point = Tuple[float, float]


def body_rectangle(center: Point, heading_rad: float, length: float, width: float) -> np.ndarray:
    """Corners (4, 2) of a rectangle whose long side points along `heading_rad` (image coordinates)."""
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    side = np.array([-forward[1], forward[0]])
    middle = np.asarray(center, dtype=np.float64)
    half_l, half_w = length / 2.0, width / 2.0
    return np.array([
        middle + forward * half_l + side * half_w,
        middle + forward * half_l - side * half_w,
        middle - forward * half_l - side * half_w,
        middle - forward * half_l + side * half_w,
    ])


def _viewpoint_in_body(viewpoint: np.ndarray, corners: np.ndarray) -> bool:
    """True when the viewpoint is inside the convex rectangle or on its boundary."""
    edges = np.roll(corners, -1, axis=0) - corners
    offsets = viewpoint - corners
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(cross >= -1e-9) or np.all(cross <= 1e-9))


def shadow_polygon(viewpoint: Point, corners: np.ndarray, reach: float) -> Optional[np.ndarray]:
    """Quad behind the body between the silhouette rays, extended `reach` away from the viewpoint.

    None when the viewpoint is inside or on the body (nothing sensible to project). The quad still
    covers the rear half of the body itself; callers remove the body from the rasterised shadow.
    """
    eye = np.asarray(viewpoint, dtype=np.float64)
    if _viewpoint_in_body(eye, corners):
        return None
    toward = corners - eye
    angles = np.arctan2(toward[:, 1], toward[:, 0])
    centre_vector = corners.mean(axis=0) - eye
    centre_angle = math.atan2(centre_vector[1], centre_vector[0])
    relative = (angles - centre_angle + math.pi) % (2.0 * math.pi) - math.pi     # (-pi, pi]; spread < pi outside a convex body
    left, right = corners[int(np.argmin(relative))], corners[int(np.argmax(relative))]
    ray_left = (left - eye) / np.linalg.norm(left - eye)
    ray_right = (right - eye) / np.linalg.norm(right - eye)
    return np.array([left, right, right + ray_right * reach, left + ray_left * reach])


def _fill(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
    return canvas.astype(bool)


def polygon_mask(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask (height, width) of a polygon given in pixel coordinates."""
    return _fill(np.asarray(points, dtype=np.float64), shape)


def blind_area_px(viewpoint: Point, bodies: Sequence[np.ndarray], roi: np.ndarray) -> Tuple[float, int]:
    """(blind pixels inside the ROI, number of bodies skipped because the viewpoint is inside them).

    The union of the shadows, each without its own body: a vehicle standing in another's shadow stays blind.
    """
    shape = roi.shape
    reach = 2.0 * math.hypot(*shape)
    blind = np.zeros(shape, dtype=bool)
    skipped = 0
    for corners in bodies:
        shadow = shadow_polygon(viewpoint, corners, reach)
        if shadow is None:
            skipped += 1
            continue
        blind |= _fill(shadow, shape) & ~_fill(corners, shape)
    return float(np.count_nonzero(blind & roi)), skipped


@dataclass(frozen=True)
class BlindAreaResult:
    columns: Dict[str, List[Any]]
    viewpoint_inside_skips: int       # body-frames without a shadow because the viewpoint was inside the body
    occluders_without_heading: int    # tracks that never moved fast enough to have a heading
    frames_without_heading: int       # body-frames skipped because no heading was known yet
    shadows: Mapping[int, Tuple[np.ndarray, ...]]   # frame -> shadow quads, only filled with keep_shadows


def _last_valid(values: np.ndarray, frames: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Forward fill of NaN inside each segment of the track; leading NaN of a segment stay NaN.

    A frame gap longer than `max_gap_frames` starts a new segment (as in the kinematics), so the heading
    from before the gap is not carried over to a vehicle that may now move in another direction.
    """
    filled = np.full_like(values, np.nan)
    for start, end in segment_bounds(frames, max_gap_frames):
        segment = values[start:end]
        valid = np.where(~np.isnan(segment), np.arange(len(segment)), -1)
        index = np.maximum.accumulate(valid)
        filled[start:end] = np.where(index >= 0, segment[np.maximum(index, 0)], np.nan)
    return filled


def compute_blind_area(
    occluders: Sequence[Kinematics],
    viewpoint: Point,
    roi: np.ndarray,
    meters_per_pixel: float,
    fps: float,
    frames: Sequence[int],
    lengths_m: Mapping[str, float],
    widths_m: Mapping[str, float],
    max_gap_frames: int,
    keep_shadows: bool = False,
) -> BlindAreaResult:
    """Blind-area ratio inside `roi` for every frame in `frames`.

    Bodies sit on each track's bottom-centre point in map pixels and point along the smoothed velocity
    direction; a vehicle that is (nearly) still keeps its last valid heading, and one that has not had a
    heading yet is skipped. `num_occluders` counts the bodies that cast a shadow. `keep_shadows` also returns
    the shadow quads per frame (for the bird's-eye video).
    """
    bodies_by_frame: Dict[int, List[np.ndarray]] = defaultdict(list)
    never_moved = frames_skipped = 0
    for kin in occluders:
        heading = _last_valid(kin.heading_deg, kin.track.frames, max_gap_frames)
        if np.all(np.isnan(heading)):
            never_moved += 1
            continue
        track = kin.track
        length_px = lengths_m[track.class_name] / meters_per_pixel
        width_px = widths_m[track.class_name] / meters_per_pixel
        for frame, centre, angle in zip(track.frames, track.map_xy, heading):
            if np.isnan(angle):
                frames_skipped += 1
                continue
            bodies_by_frame[int(frame)].append(body_rectangle(tuple(centre), math.radians(angle), length_px, width_px))

    roi_px = float(np.count_nonzero(roi))
    pixel_area = meters_per_pixel ** 2
    columns: Dict[str, List[Any]] = {name: [] for name in (
        "frame", "time_s", "num_occluders", "blind_area_m2", "roi_area_m2", "blind_ratio")}
    inside_skips = 0
    shadows: Dict[int, Tuple[np.ndarray, ...]] = {}
    reach = 2.0 * math.hypot(*roi.shape)
    for frame in frames:
        bodies = bodies_by_frame.get(int(frame), [])
        if keep_shadows:
            quads = tuple(q for q in (shadow_polygon(viewpoint, body, reach) for body in bodies) if q is not None)
            if quads:
                shadows[int(frame)] = quads
        blind_px, skipped = blind_area_px(viewpoint, bodies, roi)
        inside_skips += skipped
        columns["frame"].append(int(frame))
        columns["time_s"].append(int(frame) / fps)
        columns["num_occluders"].append(len(bodies) - skipped)
        columns["blind_area_m2"].append(blind_px * pixel_area)
        columns["roi_area_m2"].append(roi_px * pixel_area)
        columns["blind_ratio"].append(blind_px / roi_px if roi_px > 0 else float("nan"))
    return BlindAreaResult(columns, inside_skips, never_moved, frames_skipped, shadows)
