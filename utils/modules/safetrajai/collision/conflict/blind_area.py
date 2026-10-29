"""Blind area behind large vehicles, seen from a camera viewpoint, on the satellite pixel grid.

Every occluder is a body rectangle. Its shadow is the region behind it between the two silhouette rays
from the viewpoint. Shadows are rasterised with cv2 (no polygon-clipping dependency) and counted inside
the ROI; the area is accurate to about one pixel along every edge. Objects are treated as flat, so the
shadows overestimate the blind area (nothing is hidden behind the roof line of a real car).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

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
