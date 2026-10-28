"""
TTC, MTTC, collision point and leading/following per pair row (spec「TTC 與 MTTC」「前導與跟隨」).

Both are found by sampling 0..ttc_horizon_s in ttc_step_s steps (reference implementation):
TTC extrapolates at constant velocity, MTTC adds the current acceleration and is only
computed when TTC is finite. Only approaching pairs are evaluated; a pair already
closer than the collision distance at t = 0 is flagged `overlap_at_start` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ConflictConfig
from .kinematics import Kinematics
from .pairs import UNDEFINED, PairFrames

_BATCH = 4096


@dataclass(frozen=True)
class TtcResult:
    ttc: np.ndarray               # (M,) seconds; inf = no hit; NaN = overlap_at_start
    mttc: np.ndarray              # (M,) seconds; inf = no hit; NaN = undefined
    overlap_at_start: np.ndarray  # (M,) bool
    collision_xy_m: np.ndarray    # (M, 2) midpoint of both positions at TTC; NaN when TTC is not finite
    leading: np.ndarray           # (M,) 1 = track i leads, 2 = track j leads, 0 = undefined


def gather(kins: Sequence[Kinematics], idx: np.ndarray, rows: np.ndarray, attr: str) -> np.ndarray:
    return np.array([getattr(kins[k], attr)[r] for k, r in zip(idx, rows)], dtype=np.float64).reshape(len(idx), -1)


def _first_hit(d0: np.ndarray, dv: np.ndarray, da: np.ndarray, thr: np.ndarray, times: np.ndarray) -> np.ndarray:
    """First sampled time where |d0 + dv t + da t²/2| < thr, or inf."""
    out = np.full(len(d0), np.inf)
    for s in range(0, len(d0), _BATCH):
        e = s + _BATCH
        rel = (d0[s:e, None, :] + dv[s:e, None, :] * times[None, :, None]
               + 0.5 * da[s:e, None, :] * (times ** 2)[None, :, None])
        hit = np.linalg.norm(rel, axis=2) < thr[s:e, None]
        any_hit = hit.any(axis=1)
        chunk = out[s:e]
        chunk[any_hit] = times[hit[any_hit].argmax(axis=1)]
    return out


def ttc_for_pairs(kins: Sequence[Kinematics], pairs: PairFrames, cfg: ConflictConfig) -> TtcResult:
    m = len(pairs)
    if m == 0:
        empty = np.zeros(0)
        return TtcResult(empty, empty, np.zeros(0, bool), np.zeros((0, 2)), np.zeros(0, np.int64))

    p1, p2 = gather(kins, pairs.i, pairs.ri, "pos"), gather(kins, pairs.j, pairs.rj, "pos")
    v1, v2 = gather(kins, pairs.i, pairs.ri, "vel"), gather(kins, pairs.j, pairs.rj, "vel")
    a1, a2 = gather(kins, pairs.i, pairs.ri, "acc"), gather(kins, pairs.j, pairs.rj, "acc")
    thr = np.array([cfg.collision_distance_m(kins[a].track.class_name, kins[b].track.class_name)
                    for a, b in zip(pairs.i, pairs.j)])
    d0, dv, da = p1 - p2, v1 - v2, a1 - a2
    overlap = np.linalg.norm(d0, axis=1) < thr
    approaching = np.einsum("ij,ij->i", d0, dv) < 0
    times = np.arange(0.0, cfg.ttc_horizon_s + cfg.ttc_step_s / 2, cfg.ttc_step_s)

    ttc = np.full(m, np.inf)
    sel = approaching & ~overlap
    ttc[sel] = _first_hit(d0[sel], dv[sel], np.zeros_like(dv[sel]), thr[sel], times)
    ttc[overlap] = np.nan

    finite = np.isfinite(ttc)
    acc_ok = ~np.isnan(da).any(axis=1)
    mttc = np.where(overlap, np.nan, np.inf)
    mttc[finite & ~acc_ok] = np.nan
    sel_m = finite & acc_ok
    mttc[sel_m] = _first_hit(d0[sel_m], dv[sel_m], da[sel_m], thr[sel_m], times)

    t_hit = np.where(finite, ttc, 0.0)[:, None]
    at1, at2 = p1 + v1 * t_hit, p2 + v2 * t_hit
    collision = np.where(finite[:, None], (at1 + at2) / 2.0, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        proj1 = np.einsum("ij,ij->i", collision - at1, v1 / np.linalg.norm(v1, axis=1, keepdims=True))
        proj2 = np.einsum("ij,ij->i", collision - at2, v2 / np.linalg.norm(v2, axis=1, keepdims=True))
    can_lead = finite & (pairs.conflict_type != UNDEFINED)
    leading = np.where(can_lead & (proj1 < proj2), 1, np.where(can_lead & (proj2 < proj1), 2, 0)).astype(np.int64)
    return TtcResult(ttc, mttc, overlap, collision, leading)
