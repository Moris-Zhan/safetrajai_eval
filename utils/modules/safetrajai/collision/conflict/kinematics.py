"""
Velocity, acceleration and heading of a track in metres (spec「速度與加速度」, design §3).

Positions are smoothed with a centred moving average whose length comes from seconds
(`smooth_window_s`); near segment ends the window shrinks symmetrically so straight
constant-velocity motion is preserved exactly. Velocity and acceleration are central
differences over real time; undefined values are NaN, never 0 (the reference implementation
filled them with 0). Low-fps sequences get no acceleration at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import ConflictConfig
from .tracks import Track


@dataclass(frozen=True)
class Kinematics:
    track: Track
    t: np.ndarray            # (N,) seconds
    pos: np.ndarray          # (N, 2) smoothed metres
    vel: np.ndarray          # (N, 2) m/s, NaN where undefined
    acc: np.ndarray          # (N, 2) m/s², NaN where undefined
    speed: np.ndarray        # (N,) m/s
    heading_deg: np.ndarray  # (N,) [0, 360), 0 = +x (east), 90 = +y (south); NaN when slow
    acceleration_available: bool


def smoothing_window_frames(window_s: float, fps: float) -> int:
    return 2 * int(math.floor(window_s * fps / 2.0)) + 1


def segment_bounds(frames: np.ndarray, max_gap_frames: int) -> List[Tuple[int, int]]:
    """[start, end) index ranges split wherever the frame gap exceeds `max_gap_frames`."""
    if len(frames) == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) > max_gap_frames) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(frames)]
    return list(zip(starts.tolist(), ends.tolist()))


def _centred_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    if window <= 1 or n < window:
        return x.copy()
    half = window // 2
    cumsum = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    out = np.empty_like(x)
    for i in range(n):
        h = min(half, i, n - 1 - i)
        out[i] = (cumsum[i + h + 1] - cumsum[i - h]) / (2 * h + 1)
    return out


def _central_difference(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= 3:
        out[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])[:, None]
    return out


def compute_kinematics(track: Track, fps: float, cfg: ConflictConfig) -> Kinematics:
    t = track.frames.astype(np.float64) / fps
    window = smoothing_window_frames(cfg.smooth_window_s, fps)
    acceleration_available = fps >= cfg.min_fps_for_acceleration

    pos = np.empty_like(track.xy_m, dtype=np.float64)
    vel = np.full_like(pos, np.nan)
    acc = np.full_like(pos, np.nan)
    for start, end in segment_bounds(track.frames, cfg.max_gap_frames):
        seg_t = t[start:end]
        pos[start:end] = _centred_moving_average(track.xy_m[start:end].astype(np.float64), window)
        vel[start:end] = _central_difference(pos[start:end], seg_t)
        if acceleration_available:
            acc[start:end] = _central_difference(vel[start:end], seg_t)

    speed = np.hypot(vel[:, 0], vel[:, 1])
    heading = np.degrees(np.arctan2(vel[:, 1], vel[:, 0])) % 360.0
    heading = np.where(speed >= cfg.min_speed_mps, heading, np.nan)
    return Kinematics(track, t, pos, vel, acc, speed, heading, acceleration_available)
