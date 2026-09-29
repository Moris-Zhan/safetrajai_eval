"""外觀距離分布與融合門檻候選（spec: reid-fusion「門檻校準只使用調參序列」，design D4）。

外觀距離＝兩個 L2 正規化特徵的 cosine 距離 / 2（0～1，越小越像），與追蹤器相同。
- 正樣本：同一條 GT 軌跡、時間間隔 ≥ 1 秒的兩個框。
- 硬負樣本：同一幀、同類別、中心距離 ≤ 2×該類別 norm_factor 的不同 GT 軌跡；少於 30 對時改用同類別任意
  不同 GT 軌跡的配對（隨機抽樣），並標示 negative_kind = "any_fallback"。
只接受調參序列（C000025 的 145630、150634 時段）；保留序列或其他序列一律拒絕。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from safetrajai.classes import CLASS_NAMES

TUNING_STAMPS = ("20251209145630", "20251209150634")
MIN_HARD_NEGATIVES = 30
MAX_SAMPLED_PAIRS = 20000
PERCENTILES = (1, 5, 50, 95, 99, 99.9)


def check_tuning_sequences(seq_names: list[str]) -> None:
    """Raises ValueError unless every sequence belongs to the tuning time slots."""
    if not seq_names:
        raise ValueError("沒有任何序列")
    rejected = [s for s in seq_names if not any(f"_{stamp}_" in s for stamp in TUNING_STAMPS)]
    if rejected:
        raise ValueError(f"只接受調參序列（{', '.join(TUNING_STAMPS)} 時段），拒絕：{rejected}")


def read_mot_gt(path: Path, seq: str) -> list[dict]:
    """MOT gt.txt: frame,id,left,top,w,h,conf,class(1-based),vis -> box dicts with pixel centers."""
    boxes = []
    with Path(path).open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            left, top, w, h = (float(v) for v in row[2:6])
            boxes.append({
                "seq": seq, "frame": int(row[0]), "gt_id": int(row[1]), "class": CLASS_NAMES[int(row[7]) - 1],
                "cx": left + w / 2, "cy": top + h / 2, "w": w, "h": h,
            })
    return boxes


def positive_pairs(boxes: list[dict], fps_by_seq: dict[str, float], min_gap_s: float = 1.0) -> list[tuple[int, int]]:
    by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        by_track[(b["seq"], b["gt_id"])].append(i)
    pairs = []
    for (seq, _), idx in by_track.items():
        min_gap = min_gap_s * fps_by_seq[seq]
        pairs.extend((a, b) for a, b in combinations(idx, 2) if abs(boxes[a]["frame"] - boxes[b]["frame"]) >= min_gap)
    return pairs


def hard_negative_pairs(boxes: list[dict], radius_by_class: dict[str, float]) -> list[tuple[int, int]]:
    by_frame: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        by_frame[(b["seq"], b["frame"], b["class"])].append(i)
    pairs = []
    for (_, _, cls), idx in by_frame.items():
        radius = radius_by_class[cls]
        for a, b in combinations(idx, 2):
            if boxes[a]["gt_id"] != boxes[b]["gt_id"] and \
                    np.hypot(boxes[a]["cx"] - boxes[b]["cx"], boxes[a]["cy"] - boxes[b]["cy"]) <= radius:
                pairs.append((a, b))
    return pairs


def any_negative_pairs(indices: list[int], track_keys: list, seed: int, limit: int = MAX_SAMPLED_PAIRS) -> list[tuple[int, int]]:
    """Random pairs among indices whose track keys differ (sampled with a fixed seed)."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices)
    if len(idx) < 2:
        return []
    a, b = rng.integers(0, len(idx), (2, limit * 2))
    keep = [(int(idx[i]), int(idx[j])) for i, j in zip(a, b) if track_keys[idx[i]] != track_keys[idx[j]]]
    return keep[:limit]


def distances(features: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0)
    a, b = np.asarray(pairs).T
    return np.clip((1.0 - np.sum(features[a] * features[b], axis=1)) / 2.0, 0.0, 1.0)


def summarize(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"pairs": 0}
    return {"pairs": int(len(values)), **{f"p{p:g}": round(float(np.percentile(values, p)), 4) for p in PERCENTILES}}


def class_stats(features: np.ndarray, boxes: list[dict], fps_by_seq: dict[str, float],
                radius_by_class: dict[str, float], seed: int) -> dict:
    positives = positive_pairs(boxes, fps_by_seq)
    hard = hard_negative_pairs(boxes, radius_by_class)
    track_keys = [(b["seq"], b["gt_id"]) for b in boxes]
    out = {}
    for cls in sorted({b["class"] for b in boxes}):
        pos = [p for p in positives if boxes[p[0]]["class"] == cls]
        neg = [p for p in hard if boxes[p[0]]["class"] == cls]
        kind = "hard"
        if len(neg) < MIN_HARD_NEGATIVES:
            neg = any_negative_pairs([i for i, b in enumerate(boxes) if b["class"] == cls], track_keys, seed)
            kind = "any_fallback"
        out[cls] = {"positive": summarize(distances(features, pos)), "negative": summarize(distances(features, neg)),
                    "negative_kind": kind}
    return out


def candidates(stats: dict) -> dict:
    """Candidate thresholds per class (design D4); all in similarity direction (larger = stricter)."""
    out = {}
    for cls, s in stats.items():
        pos, neg = s["positive"], s["negative"]
        if not pos.get("pairs") or not neg.get("pairs"):
            continue
        wide = round(1 - pos["p95"], 4)
        strict = max(round(1 - neg["p5"], 4), wide)
        out[cls] = {"appearance_thresh_wide": wide, "appearance_thresh_strict": strict,
                    "appearance_veto_sim_p99": round(1 - pos["p99"], 4),
                    "appearance_veto_sim_p99_9": round(1 - pos["p99.9"], 4),
                    "negative_kind": s["negative_kind"]}
    return out
