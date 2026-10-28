"""外觀距離分布、外觀分離度與融合門檻候選（spec: reid-fusion「門檻校準只使用調參序列」、
reid-model「候選模型先通過外觀分離度關卡」；change align-reid-training-domain D4）。

外觀距離＝兩個 L2 正規化特徵的 cosine 距離 / 2（0～1，越小越像），與追蹤器相同。所有間隔以幀數表示。
- 正樣本（positive）：同一條 GT 軌跡、間隔在 MIN_GAP_FRAMES ～ 該類別保留期限幀數（設定檔的 track_buffer，
  不換算）之間的兩個框；候選門檻與關卡用這組。positive_1s：間隔 ≥ 1 秒（依幀率換算成幀數），
  只供與先前的量測比較。
- 干擾物（negative）：同一幀、同類別、中心距離 ≤ 2×該類別 norm_factor 的不同 GT 軌跡；少於 30 對時改用
  同類別任意不同 GT 軌跡的配對（隨機抽樣），並標示 negative_kind = "any_fallback"。
  negative_object_pairs＝干擾物來自的不同 GT 物體配對數（同一組物體的多幀只算一組；退回時為 0），關卡以它
  判斷證據是否足夠（change separability-gate-object-pairs）。
- GT id < 0 的框沒有身分，量測前以 drop_unlabeled 排除，不參與任何配對。
- 外觀分離度：AUC（正樣本距離小於干擾物距離的機率，相同算一半）與 TPR@FPR5%（正樣本距離小於干擾物 P5 的
  比例）；另依配對中較小的框的最短邊分 <16、16-32、>=32 回報，只供參考。
只接受調參序列（C000025 的 145630、150634 時段）；保留序列或其他序列一律拒絕。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from safetrajai.classes import CLASS_NAMES

TUNING_STAMPS = ("20251209145630", "20251209150634")
MIN_HARD_NEGATIVES = 30
MAX_SAMPLED_PAIRS = 20000
PERCENTILES = (1, 5, 50, 95, 99, 99.9)
MIN_GAP_FRAMES = 5
SIZE_BINS = ((16.0, "<16"), (32.0, "16-32"), (float("inf"), ">=32"))
GATE_CLASSES = ("car", "scooter")
MIN_IMPROVEMENT = {"auc": 0.01, "tpr_at_fpr5": 0.02}
MAX_OTHER_AUC_DROP = 0.02
MIN_OBJECT_PAIRS = 10
EPS = 1e-9


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


def drop_unlabeled(boxes: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Removes GT boxes without an identity (gt_id < 0); returns kept boxes and skipped counts per class."""
    kept = [b for b in boxes if b["gt_id"] >= 0]
    skipped: dict[str, int] = defaultdict(int)
    for b in boxes:
        if b["gt_id"] < 0:
            skipped[b["class"]] += 1
    return kept, dict(skipped)


def object_pair_count(boxes: list[dict], pairs: list[tuple[int, int]]) -> int:
    return len({(boxes[a]["seq"], *sorted((boxes[a]["gt_id"], boxes[b]["gt_id"]))) for a, b in pairs})


def positive_pairs(boxes: list[dict], fps_by_seq: dict[str, float], min_gap_s: float = 1.0) -> list[tuple[int, int]]:
    by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        by_track[(b["seq"], b["gt_id"])].append(i)
    pairs = []
    for (seq, _), idx in by_track.items():
        min_gap = min_gap_s * fps_by_seq[seq]
        pairs.extend((a, b) for a, b in combinations(idx, 2) if abs(boxes[a]["frame"] - boxes[b]["frame"]) >= min_gap)
    return pairs


def retention_frames(config: dict) -> dict[str, int]:
    """Per-class retention in frames = the scenario's track_buffer as is (no tracker/fps conversion)."""
    params = config["class_tracking"]["class_tracking_params"]
    out = {}
    for cls, p in params.items():
        value = p.get("track_buffer")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{cls} 的 track_buffer 必須是正整數幀數：{value!r}")
        out[cls] = value
    return out


def positive_pairs_in_window(boxes: list[dict], min_gap_frames: int,
                             retention_by_class: dict[str, int]) -> list[tuple[int, int]]:
    by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        by_track[(b["seq"], b["gt_id"])].append(i)
    pairs = []
    for idx in by_track.values():
        max_gap = retention_by_class[boxes[idx[0]]["class"]]
        pairs.extend((a, b) for a, b in combinations(idx, 2)
                     if min_gap_frames <= abs(boxes[a]["frame"] - boxes[b]["frame"]) <= max_gap)
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


def auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """P(positive distance < negative distance), ties count half (Mann-Whitney U via average ranks)."""
    if len(positive) == 0 or len(negative) == 0:
        return None
    ranks = rankdata(np.concatenate([positive, negative]))
    u_negative = ranks[len(positive):].sum() - len(negative) * (len(negative) + 1) / 2
    return round(float(u_negative / (len(positive) * len(negative))), 4)


def tpr_at_fpr5(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if len(positive) == 0 or len(negative) == 0:
        return None
    return round(float(np.mean(positive < np.percentile(negative, 5))), 4)


def size_bin(boxes: list[dict], pair: tuple[int, int]) -> str:
    shortest = min(min(boxes[i]["w"], boxes[i]["h"]) for i in pair)
    return next(name for upper, name in SIZE_BINS if shortest < upper)


def _separability(features: np.ndarray, pos: list[tuple[int, int]], neg: list[tuple[int, int]]) -> dict:
    pos_d, neg_d = distances(features, pos), distances(features, neg)
    return {"auc": auc(pos_d, neg_d), "tpr_at_fpr5": tpr_at_fpr5(pos_d, neg_d),
            "positive_pairs": len(pos), "negative_pairs": len(neg)}


def class_stats(features: np.ndarray, boxes: list[dict], fps_by_seq: dict[str, float],
                radius_by_class: dict[str, float], seed: int, retention_by_class: dict[str, int]) -> dict:
    positives_1s = positive_pairs(boxes, fps_by_seq)
    positives = positive_pairs_in_window(boxes, MIN_GAP_FRAMES, retention_by_class)
    hard = hard_negative_pairs(boxes, radius_by_class)
    track_keys = [(b["seq"], b["gt_id"]) for b in boxes]
    out = {}
    for cls in sorted({b["class"] for b in boxes}):
        pos = [p for p in positives if boxes[p[0]]["class"] == cls]
        pos_1s = [p for p in positives_1s if boxes[p[0]]["class"] == cls]
        neg = [p for p in hard if boxes[p[0]]["class"] == cls]
        kind = "hard"
        object_pairs = object_pair_count(boxes, neg)
        if len(neg) < MIN_HARD_NEGATIVES:
            neg = any_negative_pairs([i for i, b in enumerate(boxes) if b["class"] == cls], track_keys, seed)
            kind = "any_fallback"
            object_pairs = 0
        pos_d, pos_1s_d, neg_d = distances(features, pos), distances(features, pos_1s), distances(features, neg)
        by_size = {}
        for _, name in SIZE_BINS:
            bin_pos = [p for p in pos if size_bin(boxes, p) == name]
            bin_neg = [p for p in neg if size_bin(boxes, p) == name]
            if bin_pos or bin_neg:
                by_size[name] = _separability(features, bin_pos, bin_neg)
        out[cls] = {
            "positive": summarize(pos_d), "positive_1s": summarize(pos_1s_d), "negative": summarize(neg_d),
            "negative_kind": kind, "negative_object_pairs": object_pairs,
            "retention_frames": retention_by_class[cls], "min_gap_frames": MIN_GAP_FRAMES,
            "auc": auc(pos_d, neg_d), "tpr_at_fpr5": tpr_at_fpr5(pos_d, neg_d),
            "auc_1s": auc(pos_1s_d, neg_d), "tpr_at_fpr5_1s": tpr_at_fpr5(pos_1s_d, neg_d),
            "by_size": by_size,
        }
    return out


def noise_floor(seed_model: dict, base_model: dict) -> dict:
    """Per-class |difference| of AUC and TPR@FPR5% between two models trained with only the seed changed."""
    return {cls: {m: round(abs(seed_model[cls][m] - base_model[cls][m]), 4) for m in MIN_IMPROVEMENT}
            for cls in GATE_CLASSES}


def separability_gate(candidate: dict, reference: dict, noise: dict) -> dict:
    """Spec reid-model gate; inputs are per-class tuning stats (class -> {auc, tpr_at_fpr5, negative_object_pairs}).

    Evidence is counted in distinct GT object pairs: car/scooter need MIN_OBJECT_PAIRS in both reference and
    candidate; other classes below it are reported in unchecked_classes instead of being judged.
    """
    reasons = []
    unchecked = {}
    min_improvement = {cls: {m: round(max(base, 2 * noise[cls][m]), 4) for m, base in MIN_IMPROVEMENT.items()}
                       for cls in GATE_CLASSES}
    for cls in GATE_CLASSES:
        pairs = min(reference[cls].get("negative_object_pairs", 0), candidate[cls].get("negative_object_pairs", 0))
        if pairs < MIN_OBJECT_PAIRS:
            reasons.append(f"{cls} 干擾物物體配對 {pairs} 組 < {MIN_OBJECT_PAIRS}，證據不足")
        for metric, need in min_improvement[cls].items():
            gain = candidate[cls][metric] - reference[cls][metric]
            if gain + EPS < need:
                reasons.append(f"{cls} {metric} 改善 {gain:+.4f} < 最小改善量 {need:.4f}")
    for cls, ref in reference.items():
        if cls in GATE_CLASSES or cls not in candidate:
            continue
        pairs = min(ref.get("negative_object_pairs", 0), candidate[cls].get("negative_object_pairs", 0))
        if pairs < MIN_OBJECT_PAIRS:
            change = (round(candidate[cls]["auc"] - ref["auc"], 4)
                      if ref.get("auc") is not None and candidate[cls].get("auc") is not None else None)
            unchecked[cls] = {"object_pairs": pairs, "auc_change": change}
            continue
        drop = ref["auc"] - candidate[cls]["auc"]
        if drop > MAX_OTHER_AUC_DROP + EPS:
            reasons.append(f"{cls} AUC 下降 {drop:.4f} > {MAX_OTHER_AUC_DROP}")
    return {"passed": not reasons, "reasons": reasons, "min_improvement": min_improvement,
            "unchecked_classes": unchecked,
            "mean_auc": round(float(np.mean([candidate[c]["auc"] for c in GATE_CLASSES])), 4)}


def select_best(results: dict[str, dict]) -> str | None:
    """Among passing candidates, the one with the highest car/scooter mean AUC; None when nobody passes."""
    passed = {name: r for name, r in results.items() if r["passed"]}
    return max(passed, key=lambda name: passed[name]["mean_auc"]) if passed else None


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
