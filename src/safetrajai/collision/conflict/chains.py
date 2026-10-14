"""
Rear-end chains and travel direction (spec「追撞鏈與行進方向」, design §6).

Follows the reference `floyd_warshall_soft_groups` and `assign_direction_and_rank`: tracks
joined by rear-end pairs get an edge cost
    max(0, min angle − chain_angle_deg) + chain_dist_weight · max(0, min ΔS − chain_dist_m),
all-pairs shortest paths decide connectivity (cost ≤ chain_eps), and connected components
become chains numbered 1..N. ΔS is in metres here (the reference compared pixels to a
"metre" threshold). Each track's direction is the E/S/W/N sector of its median velocity, and
tracks are ranked rear-to-front within (chain, direction) by the projection of their median
position on that direction.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ConflictConfig
from .following import REAR_END, RearEndMetrics
from .kinematics import Kinematics
from .pairs import PairFrames

_DIRECTION_UNIT = {"E": (1.0, 0.0), "S": (0.0, 1.0), "W": (-1.0, 0.0), "N": (0.0, -1.0)}


@dataclass(frozen=True)
class ChainRow:
    track_index: int
    chain_id: int          # 1..N
    direction: str         # "E" / "S" / "W" / "N" / "" when the median speed is too low
    heading_deg: float
    projection_m: float
    rank: int              # 1 = rearmost within (chain, direction); 0 when there is no direction


def travel_direction(heading_deg: float) -> str:
    if math.isnan(heading_deg):
        return ""
    h = heading_deg % 360.0
    if h >= 315.0 or h < 45.0:
        return "E"
    if h < 135.0:
        return "S"
    if h < 225.0:
        return "W"
    return "N"


def _edge_costs(pairs: PairFrames, metrics: RearEndMetrics, cfg: ConflictConfig) -> Dict[Tuple[int, int], float]:
    best: Dict[Tuple[int, int], Tuple[float, float]] = {}
    rear_end = (pairs.conflict_type == REAR_END) & (metrics.rear_side > 0)
    for i, j, angle, ds in zip(pairs.i[rear_end], pairs.j[rear_end], pairs.angle_deg[rear_end], metrics.delta_s[rear_end]):
        key = (int(i), int(j))
        a0, d0 = best.get(key, (np.inf, np.inf))
        best[key] = (min(a0, float(angle)), min(d0, float(ds)))
    return {
        key: max(0.0, angle - cfg.chain_angle_deg) + cfg.chain_dist_weight * max(0.0, ds - cfg.chain_dist_m)
        for key, (angle, ds) in best.items()
    }


def _components(nodes: List[int], costs: Dict[Tuple[int, int], float], eps: float) -> Dict[int, int]:
    index = {n: k for k, n in enumerate(nodes)}
    dist = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(dist, 0.0)
    for (a, b), c in costs.items():
        ia, ib = index[a], index[b]
        dist[ia, ib] = dist[ib, ia] = min(dist[ia, ib], c)
    for k in range(len(nodes)):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    chain: Dict[int, int] = {}
    next_id = 0
    for start in range(len(nodes)):
        if nodes[start] in chain:
            continue
        next_id += 1
        queue = deque([start])
        chain[nodes[start]] = next_id
        while queue:
            u = queue.popleft()
            for v in np.flatnonzero(dist[u] <= eps):
                if nodes[v] not in chain:
                    chain[nodes[v]] = next_id
                    queue.append(int(v))
    return chain


def build_chains(kins: Sequence[Kinematics], pairs: PairFrames, metrics: RearEndMetrics,
                 cfg: ConflictConfig) -> Tuple[ChainRow, ...]:
    costs = _edge_costs(pairs, metrics, cfg)
    nodes = sorted({n for key in costs for n in key})
    if not nodes:
        return ()
    chain = _components(nodes, costs, cfg.chain_eps)

    info = {}
    for k in nodes:
        kin = kins[k]
        defined = ~np.isnan(kin.vel[:, 0])
        vx, vy = (float(np.median(kin.vel[defined, c])) for c in (0, 1))
        heading = math.degrees(math.atan2(vy, vx)) % 360.0 if math.hypot(vx, vy) >= cfg.min_speed_mps else math.nan
        direction = travel_direction(heading)
        median_pos = np.median(kin.pos[defined], axis=0)
        unit = _DIRECTION_UNIT.get(direction)
        projection = float(np.dot(median_pos, unit)) if unit else math.nan
        info[k] = (direction, heading, projection)

    rows = []
    for k in nodes:
        direction, heading, projection = info[k]
        rank = 0
        if direction:
            peers = sorted({info[o][2] for o in nodes if chain[o] == chain[k] and info[o][0] == direction})
            rank = peers.index(projection) + 1
        rows.append(ChainRow(k, chain[k], direction, heading, projection, rank))
    return tuple(rows)
