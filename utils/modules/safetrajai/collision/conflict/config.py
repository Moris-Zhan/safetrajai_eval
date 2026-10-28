"""
Parameters of the stage3 conflict-metrics analysis, read from the scenario block
`conflict_analysis:`.

Defaults follow the reference implementation (Sub_Module/trajectoryanalyst2): a fixed
2 m TTC distance, three conflict types, PET only for co-present track pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from itertools import combinations_with_replacement
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from safetrajai.classes import CLASS_NAMES

CLASSES: Tuple[str, ...] = tuple(CLASS_NAMES[i] for i in sorted(CLASS_NAMES))

# Reference implementation OBJECT_SIZE (traj_config.py).
DEFAULT_OBJECT_LENGTH_M: Mapping[str, float] = {
    "person": 0.4, "car": 5.0, "scooter": 1.8, "small_truck": 6.0, "bus": 12.0, "truck": 7.0,
}
DEFAULT_OBJECT_MASS_KG: Mapping[str, float] = {
    "person": 70.0, "car": 2150.0, "scooter": 200.0, "small_truck": 5000.0, "bus": 16000.0, "truck": 17000.0,
}
DEFAULT_ANGLE_TYPES: Tuple[Tuple[float, str], ...] = ((30.0, "Rear-end"), (85.0, "Lane-change"), (180.0, "Crossing"))
TTC_DISTANCE_MODES = ("fixed", "class_length")


def _pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


DEFAULT_CLASS_PAIRS: FrozenSet[Tuple[str, str]] = frozenset(
    _pair(a, b) for a, b in combinations_with_replacement(CLASSES, 2) if (a, b) != ("person", "person")
)


@dataclass(frozen=True)
class ConflictConfig:
    enabled: bool = False
    # kinematics
    smooth_window_s: float = 0.6
    max_gap_frames: int = 3
    min_speed_mps: float = 0.5
    min_fps_for_acceleration: float = 10.0
    # pairing and conflict type
    pair_radius_m: float = 50.0
    class_pairs: FrozenSet[Tuple[str, str]] = DEFAULT_CLASS_PAIRS
    angle_types: Tuple[Tuple[float, str], ...] = DEFAULT_ANGLE_TYPES
    # TTC / MTTC
    ttc_distance_m: float = 2.0
    ttc_distance_mode: str = "fixed"
    ttc_horizon_s: float = 5.0
    ttc_step_s: float = 0.02
    # rear-end chains
    chain_angle_deg: float = 30.0
    chain_dist_m: float = 10.0
    chain_dist_weight: float = 1.0
    chain_eps: float = 3.0
    # PET
    pet_zone_m: float = 2.0
    pet_max_gap_s: float = 0.0
    # events
    event_threshold_s: float = 2.5
    event_gap_frames: int = 2
    stat_thresholds_s: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    # class properties
    object_length_m: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECT_LENGTH_M))
    object_mass_kg: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECT_MASS_KG))

    def collision_distance_m(self, class_a: str, class_b: str) -> float:
        if self.ttc_distance_mode == "class_length":
            return (self.object_length_m[class_a] + self.object_length_m[class_b]) / 2.0
        return self.ttc_distance_m


_POSITIVE = (
    "smooth_window_s", "min_fps_for_acceleration", "pair_radius_m", "ttc_distance_m", "ttc_horizon_s",
    "ttc_step_s", "pet_zone_m", "event_threshold_s",
)
_NON_NEGATIVE = (
    "max_gap_frames", "min_speed_mps", "chain_angle_deg", "chain_dist_m", "chain_dist_weight",
    "chain_eps", "pet_max_gap_s", "event_gap_frames",
)


def _class_map(value: Any, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(CLASSES):
        raise ValueError(f"conflict_analysis.{name} 必須為 6 個類別都給值：{CLASSES}")
    if any(not isinstance(v, (int, float)) or v <= 0 for v in value.values()):
        raise ValueError(f"conflict_analysis.{name} 的值必須為正數")
    return {k: float(v) for k, v in value.items()}


def _angle_types(value: Any) -> Tuple[Tuple[float, str], ...]:
    bins = tuple((float(upper), str(label)) for upper, label in value)
    uppers = [u for u, _ in bins]
    if not bins or any(b <= a for a, b in zip([0.0] + uppers, uppers)) or uppers[-1] != 180.0:
        raise ValueError("conflict_analysis.angle_types 的上限必須由小到大、連續，且最後一段到 180 度")
    return bins


def _class_pairs(value: Any) -> FrozenSet[Tuple[str, str]]:
    pairs = frozenset(_pair(str(a), str(b)) for a, b in value)
    unknown = sorted({c for pair in pairs for c in pair} - set(CLASSES))
    if unknown:
        raise ValueError(f"conflict_analysis.class_pairs 含未知類別：{unknown}")
    return pairs


def load_conflict_config(block: Optional[Mapping[str, Any]]) -> ConflictConfig:
    """Validate the `conflict_analysis:` block; a missing block means disabled."""
    if not block:
        return ConflictConfig()
    known = {f.name for f in fields(ConflictConfig)}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(f"conflict_analysis 有未知欄位：{unknown}")

    values = dict(block)
    for name in _POSITIVE:
        if name in values and not values[name] > 0:
            raise ValueError(f"conflict_analysis.{name} 必須大於 0（目前 {values[name]!r}）")
    for name in _NON_NEGATIVE:
        if name in values and not values[name] >= 0:
            raise ValueError(f"conflict_analysis.{name} 不可為負（目前 {values[name]!r}）")
    if values.get("ttc_distance_mode", "fixed") not in TTC_DISTANCE_MODES:
        raise ValueError(f"conflict_analysis.ttc_distance_mode 必須是 {TTC_DISTANCE_MODES} 之一")
    if "angle_types" in values:
        values["angle_types"] = _angle_types(values["angle_types"])
    if "class_pairs" in values:
        values["class_pairs"] = _class_pairs(values["class_pairs"])
    if "stat_thresholds_s" in values:
        values["stat_thresholds_s"] = tuple(float(t) for t in values["stat_thresholds_s"])
    for name in ("object_length_m", "object_mass_kg"):
        if name in values:
            values[name] = _class_map(values[name], name)
    values["enabled"] = bool(values.get("enabled", False))
    return ConflictConfig(**values)
