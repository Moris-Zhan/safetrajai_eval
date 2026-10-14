"""
Conflict events: consecutive same-pair frames whose TTC or MTTC is below `event_threshold_s`
(spec「衝突事件彙總」, design §9). Frames separated by at most `event_gap_frames` missing frames
belong to the same event. `overlap_at_start` rows never form events; they are counted per pair.
Threshold statistics count events, not frame rows (the reference implementation counted rows).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import ConflictConfig
from .following import RearEndMetrics
from .kinematics import Kinematics
from .pairs import PairFrames
from .ttc import TtcResult


@dataclass(frozen=True)
class ConflictEvent:
    event_id: str
    i: int
    j: int
    start_frame: int
    end_frame: int
    n_frames: int
    key_row: int                  # row index of the minimum-TTC frame
    min_ttc: float
    min_ttc_frame: int
    min_mttc: float
    conflict_type: str
    leading: int                  # 1 = track i, 2 = track j, 0 = undefined
    max_drac: float
    drac_ever_exceeded: Optional[bool]
    max_drac_excess: float
    max_ci: float
    max_ci_m: float


def _nanmax(values: np.ndarray) -> float:
    finite = values[~np.isnan(values)]
    return float(finite.max()) if finite.size else float("nan")


def _event_id(kins: Sequence[Kinematics], i: int, j: int, start: int) -> str:
    a, b = sorted((kins[i].track.key, kins[j].track.key))
    return f"{a[0]}:{a[1]}-{b[0]}:{b[1]}@{start}"


def build_events(kins: Sequence[Kinematics], pairs: PairFrames, ttc: TtcResult, metrics: RearEndMetrics,
                 cfg: ConflictConfig) -> Tuple[ConflictEvent, ...]:
    hit = ~ttc.overlap_at_start & ((ttc.ttc < cfg.event_threshold_s) | (ttc.mttc < cfg.event_threshold_s))
    by_pair: Dict[Tuple[int, int], list] = defaultdict(list)
    for r in np.flatnonzero(hit):
        by_pair[(int(pairs.i[r]), int(pairs.j[r]))].append(int(r))

    events = []
    for (i, j), rows in sorted(by_pair.items()):
        rows.sort(key=lambda r: pairs.frame[r])
        groups = [[rows[0]]]
        for r in rows[1:]:
            if pairs.frame[r] - pairs.frame[groups[-1][-1]] - 1 <= cfg.event_gap_frames:
                groups[-1].append(r)
            else:
                groups.append([r])
        for group in groups:
            idx = np.array(group)
            key = int(idx[np.argmin(ttc.ttc[idx])])
            exceeds = metrics.drac_exceeds[idx]
            exceeds = exceeds[~np.isnan(exceeds)]
            events.append(ConflictEvent(
                event_id=_event_id(kins, i, j, int(pairs.frame[idx[0]])),
                i=i, j=j,
                start_frame=int(pairs.frame[idx[0]]), end_frame=int(pairs.frame[idx[-1]]), n_frames=len(idx),
                key_row=key, min_ttc=float(ttc.ttc[key]), min_ttc_frame=int(pairs.frame[key]),
                min_mttc=float(np.min(ttc.mttc[idx])) if np.any(~np.isnan(ttc.mttc[idx])) else float("nan"),
                conflict_type=str(pairs.conflict_type[key]), leading=int(ttc.leading[key]),
                max_drac=_nanmax(metrics.drac[idx]),
                drac_ever_exceeded=bool(exceeds.max() > 0) if exceeds.size else None,
                max_drac_excess=_nanmax(metrics.drac_excess[idx]),
                max_ci=_nanmax(metrics.ci[idx]), max_ci_m=_nanmax(metrics.ci_m[idx]),
            ))
    return tuple(events)


def overlap_at_start_pairs(pairs: PairFrames, ttc: TtcResult) -> int:
    rows = np.flatnonzero(ttc.overlap_at_start)
    return len({(int(pairs.i[r]), int(pairs.j[r])) for r in rows})


def threshold_counts(items: Sequence[object], thresholds: Sequence[float], attr: str) -> Mapping[str, int]:
    """Number of items whose `attr` is below each threshold, keyed like '< 1.5 s'."""
    values = np.array([getattr(item, attr) for item in items], dtype=float)
    return {f"< {t} s": int(np.sum(values < t)) for t in thresholds}
