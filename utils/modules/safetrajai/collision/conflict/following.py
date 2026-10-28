"""
Front/rear relations and rear-end metrics (最近前車與後車、追撞後車指標、DRAC 超限、Crash Index).

Front/rear of a rear-end pair follows the reference implementation: project the vector from
track i to track j onto track i's heading; positive means j is in front. ΔS and DRAC use the
centre distance; DRAC uses the closing speed of the rear track towards the front one.
DRAC exceedance is what the reference implementation called PICUD (DRAC > the rear track's
largest observed deceleration). Crash Index uses speed and acceleration magnitudes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .config import ConflictConfig
from .kinematics import Kinematics
from .pairs import PairFrames
from .ttc import TtcResult, gather

REAR_END = "Rear-end"


@dataclass(frozen=True)
class RearEndMetrics:
    rear_side: np.ndarray          # (M,) 1 = track i is the rear track, 2 = track j, 0 = not a rear-end row
    delta_s: np.ndarray            # (M,) centre distance, metres
    delta_v: np.ndarray            # (M,) |v_i − v_j|, m/s
    dr: np.ndarray                 # (M,) rear deceleration along its heading, m/s²
    maxd: np.ndarray               # (M,) largest DR over the rear track
    drac: np.ndarray               # (M,) closing speed² / (2 ΔS)
    drac_exceeds: np.ndarray       # (M,) 1.0 / 0.0 / NaN
    drac_excess: np.ndarray        # (M,) DRAC − MaxD
    drac_exceed_ratio: np.ndarray  # (M,) share of exceeding rows within the same track pair
    ci: np.ndarray                 # (M,) Crash Index
    ci_m: np.ndarray               # (M,) mass-weighted Crash Index


def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def deceleration_along_heading(vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """max(0, −a·v̂); NaN where acceleration or velocity is undefined."""
    along = np.einsum("ij,ij->i", acc, _unit(vel))
    return np.where(np.isnan(along), np.nan, np.maximum(0.0, -along))


def drac_exceedance(drac: np.ndarray, maxd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    undefined = np.isnan(drac) | np.isnan(maxd)
    exceeds = np.where(undefined, np.nan, (drac > maxd).astype(np.float64))
    excess = np.where(undefined, np.nan, drac - maxd)
    return exceeds, excess


def crash_index(v_follow: float, v_lead: float, a_follow: float, a_lead: float, mttc: float,
                m_follow: float = 1.0, m_lead: float = 1.0) -> float:
    """((V_F + a_F·T)² − (V_L + a_L·T)²) / (2T), optionally mass weighted (CI_m)."""
    return (m_follow * (v_follow + a_follow * mttc) ** 2 - m_lead * (v_lead + a_lead * mttc) ** 2) / (2.0 * mttc)


def _max_deceleration(kin: Kinematics) -> float:
    dr = deceleration_along_heading(kin.vel, kin.acc)
    return float(np.nanmax(dr)) if np.any(~np.isnan(dr)) else np.nan


def closest_front_rear(kins: Sequence[Kinematics], pairs: PairFrames) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """{(track index, frame): (nearest front index, nearest rear index)}; -1 when there is none."""
    best: Dict[Tuple[int, int], list] = defaultdict(lambda: [(np.inf, -1), (np.inf, -1)])
    rear_end = pairs.conflict_type == REAR_END
    for frame, a, b, ra, rb in zip(pairs.frame[rear_end], pairs.i[rear_end], pairs.j[rear_end],
                                   pairs.ri[rear_end], pairs.rj[rear_end]):
        for self_k, self_r, other_k, other_r in ((a, ra, b, rb), (b, rb, a, ra)):
            offset = kins[other_k].pos[other_r] - kins[self_k].pos[self_r]
            proj = float(np.dot(offset, _unit(kins[self_k].vel[self_r])))
            dist = float(np.linalg.norm(offset))
            slot = best[(int(self_k), int(frame))]
            side = 0 if proj > 0 else 1 if proj < 0 else None
            if side is not None and dist < slot[side][0]:
                slot[side] = (dist, int(other_k))
    return {key: (slot[0][1], slot[1][1]) for key, slot in sorted(best.items())}


def rear_end_metrics(kins: Sequence[Kinematics], pairs: PairFrames, ttc: TtcResult, cfg: ConflictConfig) -> RearEndMetrics:
    m = len(pairs)
    nan = np.full(m, np.nan)
    if m == 0:
        return RearEndMetrics(np.zeros(0, np.int64), *([nan] * 10))

    p1, p2 = gather(kins, pairs.i, pairs.ri, "pos"), gather(kins, pairs.j, pairs.rj, "pos")
    v1, v2 = gather(kins, pairs.i, pairs.ri, "vel"), gather(kins, pairs.j, pairs.rj, "vel")
    a1, a2 = gather(kins, pairs.i, pairs.ri, "acc"), gather(kins, pairs.j, pairs.rj, "acc")

    proj = np.einsum("ij,ij->i", p2 - p1, _unit(v1))
    front_is_j = np.where(proj > 0, True, np.where(proj < 0, False, ttc.leading == 2))
    is_rear_end = (pairs.conflict_type == REAR_END) & ((proj != 0) | (ttc.leading != 0))
    rear_side = np.where(is_rear_end, np.where(front_is_j, 1, 2), 0).astype(np.int64)

    rear_p, front_p = np.where(front_is_j[:, None], p1, p2), np.where(front_is_j[:, None], p2, p1)
    rear_v, front_v = np.where(front_is_j[:, None], v1, v2), np.where(front_is_j[:, None], v2, v1)
    rear_a, front_a = np.where(front_is_j[:, None], a1, a2), np.where(front_is_j[:, None], a2, a1)
    rear_k = np.where(front_is_j, pairs.i, pairs.j)
    front_k = np.where(front_is_j, pairs.j, pairs.i)

    delta_s = np.linalg.norm(front_p - rear_p, axis=1)
    delta_v = np.linalg.norm(v1 - v2, axis=1)
    closing = np.einsum("ij,ij->i", rear_v - front_v, _unit(front_p - rear_p))
    with np.errstate(invalid="ignore", divide="ignore"):
        drac = np.where((closing > 0) & (delta_s > 0), closing ** 2 / (2.0 * delta_s), np.nan)

    max_dec = {k: _max_deceleration(kins[k]) for k in set(rear_k[is_rear_end].tolist())}
    dr = deceleration_along_heading(rear_v, rear_a)
    maxd = np.array([max_dec.get(int(k), np.nan) for k in rear_k])
    exceeds, excess = drac_exceedance(drac, maxd)

    mttc = ttc.mttc
    ci_ok = is_rear_end & np.isfinite(mttc) & (mttc > 0) & ~np.isnan(rear_a).any(axis=1) & ~np.isnan(front_a).any(axis=1)
    vf, vl = np.linalg.norm(rear_v, axis=1), np.linalg.norm(front_v, axis=1)
    af, al = np.linalg.norm(rear_a, axis=1), np.linalg.norm(front_a, axis=1)
    mf = np.array([cfg.object_mass_kg[kins[k].track.class_name] for k in rear_k])
    ml = np.array([cfg.object_mass_kg[kins[k].track.class_name] for k in front_k])
    t_safe = np.where(ci_ok, mttc, 1.0)
    ci = np.where(ci_ok, crash_index(vf, vl, af, al, t_safe), np.nan)
    ci_m = np.where(ci_ok, crash_index(vf, vl, af, al, t_safe, mf, ml), np.nan)

    masked = [np.where(is_rear_end, x, np.nan) for x in (delta_s, delta_v, dr, maxd, drac, exceeds, excess)]
    delta_s, delta_v, dr, maxd, drac, exceeds, excess = masked

    ratio = np.full(m, np.nan)
    groups: Dict[Tuple[int, int], list] = defaultdict(list)
    for r in np.flatnonzero(~np.isnan(exceeds)):
        groups[(int(pairs.i[r]), int(pairs.j[r]))].append(r)
    for rows in groups.values():
        ratio[rows] = float(np.mean(exceeds[rows]))

    return RearEndMetrics(rear_side, delta_s, delta_v, dr, maxd, drac, exceeds, excess, ratio, ci, ci_m)
