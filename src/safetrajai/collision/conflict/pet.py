"""
Post-Encroachment Time per track pair (spec「PET」, design §8, ADR 0011).

As in the reference implementation, only track pairs that appear together in at least one
frame are considered by default; `pet_max_gap_s > 0` additionally admits pairs whose time
ranges are at most that far apart. The conflict point is the midpoint of the closest pair of
path points (raw positions, metres); each track's entry and exit are its first and last times
within `pet_zone_m` of that point. One conflict point per pair is a known limitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import ConflictConfig
from .tracks import Track


@dataclass(frozen=True)
class PetRow:
    i: int                     # index of the first track of the pair (track order)
    j: int
    conflict_xy_m: np.ndarray  # (2,)
    min_distance_m: float
    t1_in: float
    t1_out: float
    t2_in: float
    t2_out: float
    pet: float
    first: int                 # 1 = track i reached the conflict zone first, 2 = track j


def _candidate(a: Track, b: Track, fps: float, cfg: ConflictConfig) -> bool:
    if np.intersect1d(a.frames, b.frames).size > 0:
        return True
    if cfg.pet_max_gap_s <= 0:
        return False
    gap_frames = max(int(b.frames[0]) - int(a.frames[-1]), int(a.frames[0]) - int(b.frames[-1]))
    return gap_frames / fps <= cfg.pet_max_gap_s


def _entry_exit(track: Track, point: np.ndarray, zone: float, fps: float) -> Tuple[float, float]:
    inside = np.linalg.norm(track.xy_m - point, axis=1) <= zone
    frames = track.frames[inside]
    return float(frames.min()) / fps, float(frames.max()) / fps


def compute_pet(tracks: Sequence[Track], fps: float, cfg: ConflictConfig) -> Tuple[PetRow, ...]:
    rows = []
    for i, j in combinations(range(len(tracks)), 2):
        a, b = tracks[i], tracks[j]
        if len(a.frames) == 0 or len(b.frames) == 0 or not _candidate(a, b, fps, cfg):
            continue
        dist, nearest = cKDTree(b.xy_m).query(a.xy_m)
        ka = int(np.argmin(dist))
        if dist[ka] > cfg.pet_zone_m:
            continue
        point = (a.xy_m[ka] + b.xy_m[int(nearest[ka])]) / 2.0
        t1_in, t1_out = _entry_exit(a, point, cfg.pet_zone_m, fps)
        t2_in, t2_out = _entry_exit(b, point, cfg.pet_zone_m, fps)
        pet = max(0.0, max(t1_in, t2_in) - min(t1_out, t2_out))
        first = 1 if t1_in <= t2_in else 2
        rows.append(PetRow(i, j, point, float(dist[ka]), t1_in, t1_out, t2_in, t2_out, pet, first))
    return tuple(rows)
