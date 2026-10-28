"""
Same-frame track pairs, conflict angle and conflict type (spec「同幀配對」「衝突角度與類型」).

Every frame pairs all tracks whose velocity is defined, skipping class combinations outside
`class_pairs`, pairs farther apart than `pair_radius_m`, and pairs where both tracks are
slower than `min_speed_mps`. The angle is between velocity vectors with the cosine clipped to
[-1, 1]; conflict types use continuous bins (the reference implementation left integer gaps).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import ConflictConfig
from .kinematics import Kinematics

UNDEFINED = "Undefined"


@dataclass(frozen=True)
class PairFrames:
    frame: np.ndarray          # (M,) int
    i: np.ndarray              # (M,) index into the kinematics list (i < j)
    j: np.ndarray
    ri: np.ndarray             # (M,) row index inside track i / j
    rj: np.ndarray
    angle_deg: np.ndarray      # (M,) NaN when a heading is undefined
    conflict_type: np.ndarray  # (M,) object array of labels

    def __len__(self) -> int:
        return len(self.frame)


def conflict_angle_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", v1, v2)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def classify_angle(angles: np.ndarray, bins: Tuple[Tuple[float, str], ...]) -> np.ndarray:
    labels = np.full(len(angles), UNDEFINED, dtype=object)
    lower = -np.inf
    for upper, label in bins:
        labels[(angles > lower) & (angles <= upper)] = label
        lower = upper
    return labels


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_pairs(kins: Sequence[Kinematics], cfg: ConflictConfig) -> PairFrames:
    by_frame = defaultdict(list)
    for k, kin in enumerate(kins):
        defined = ~np.isnan(kin.vel[:, 0])
        for r in np.flatnonzero(defined):
            by_frame[int(kin.track.frames[r])].append((k, int(r)))

    rows = []
    for frame in sorted(by_frame):
        members = by_frame[frame]
        if len(members) < 2:
            continue
        pos = np.array([kins[k].pos[r] for k, r in members])
        for a, b in sorted(map(tuple, cKDTree(pos).query_pairs(cfg.pair_radius_m, output_type="ndarray"))):
            (ka, ra), (kb, rb) = sorted((members[a], members[b]))
            kin_a, kin_b = kins[ka], kins[kb]
            if _pair_key(kin_a.track.class_name, kin_b.track.class_name) not in cfg.class_pairs:
                continue
            if kin_a.speed[ra] < cfg.min_speed_mps and kin_b.speed[rb] < cfg.min_speed_mps:
                continue
            rows.append((frame, ka, kb, ra, rb))

    if not rows:
        empty_i = np.zeros(0, dtype=np.int64)
        return PairFrames(empty_i, empty_i, empty_i, empty_i, empty_i, np.zeros(0), np.zeros(0, dtype=object))

    frame, i, j, ri, rj = (np.array(col, dtype=np.int64) for col in zip(*rows))
    v1 = np.array([kins[a].vel[r] for a, r in zip(i, ri)])
    v2 = np.array([kins[b].vel[r] for b, r in zip(j, rj)])
    heading_ok = np.array([
        not (np.isnan(kins[a].heading_deg[x]) or np.isnan(kins[b].heading_deg[y]))
        for a, x, b, y in zip(i, ri, j, rj)
    ])
    angle = np.where(heading_ok, conflict_angle_deg(v1, v2), np.nan)
    return PairFrames(frame, i, j, ri, rj, angle, classify_angle(angle, cfg.angle_types))
