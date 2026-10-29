"""污染身分偵測的判斷邏輯（見 CONTEXT.md「污染身分」，spec: reid-dataset）。

輸入是每張裁切的 L2 正規化特徵；只產生標記，不改動任何裁切。
- 類別中心點比對：每類以「身分平均特徵」再平均為中心點，以 cosine 相似度取最近的類別。
  不做逐維變異數正規化：身分之間變化很小的維度會被放大成雜訊，讓判定近乎隨機。
- 軌跡內外觀離群值：每張裁切到所屬身分平均特徵的 cosine 距離，超過同類別的百分位門檻即為離群。
- 類別一致性只在粗分組（人／二輪／四輪）之間不一致時標記：審查重點是身分純度，
  四輪子類別之間的誤標（car↔small_truck、truck↔bus）不影響同一身分是否為同一實體。
- 靜止身分與運動跳動只讀索引的框與時間，不讀影像（design D6）。
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

COARSE_GROUPS = {
    "person": "person",
    "scooter": "two_wheel",
    "car": "four_wheel",
    "small_truck": "four_wheel",
    "truck": "four_wheel",
    "bus": "four_wheel",
}


def _identity_groups(rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["class"], row["identity"])].append(i)
    return groups


def class_consistency(
    rows: list[dict],
    features: np.ndarray,
    min_agreement: float = 0.5,
) -> tuple[list[dict], list[str]]:
    """回傳 (身分層級紀錄, 每張裁切的判定類別)。coarse_agreement_rate < min_agreement 的身分標記；
    agreement_rate（細分類）只供參考。"""
    groups = _identity_groups(rows)
    identity_means = {key: features[idx].mean(axis=0) for key, idx in groups.items()}
    classes = sorted({cls for cls, _ in groups})
    centroids = np.stack([
        np.stack([m for (c, _), m in identity_means.items() if c == cls]).mean(axis=0) for cls in classes
    ])
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    predicted = [classes[i] for i in np.argmax(features @ centroids.T, axis=1)]
    records = []
    for (cls, identity), idx in sorted(groups.items()):
        agreement = sum(predicted[i] == cls for i in idx) / len(idx)
        coarse_agreement = sum(COARSE_GROUPS[predicted[i]] == COARSE_GROUPS[cls] for i in idx) / len(idx)
        records.append({
            "class": cls, "identity": identity, "n_crops": len(idx),
            "agreement_rate": round(agreement, 4), "coarse_agreement_rate": round(coarse_agreement, 4),
            "flagged": coarse_agreement < min_agreement,
        })
    return records, predicted


def motion_flags(
    rows: list[dict],
    static_max_displacement: float = 0.5,
    static_min_duration_s: float = 5.0,
    jump_threshold: float = 1.5,
) -> list[dict]:
    """由索引的框與時間標記靜止身分與運動跳動（距離都以框尺寸＝面積平方根正規化）。

    - static：首末兩張的中心點位移 < static_max_displacement，且持續 ≥ static_min_duration_s。
    - motion_jump：相鄰兩段的速度向量差 × 後段時間間隔 > jump_threshold。
    """
    samples: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        size = math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
        samples[(row["class"], row["identity"])].append(
            (float(row["time_s"]), (x1 + x2) / 2, (y1 + y2) / 2, max(size, 1.0)))

    records = []
    for (cls, identity), points in sorted(samples.items()):
        points.sort()
        median_size = sorted(p[3] for p in points)[len(points) // 2]
        duration = points[-1][0] - points[0][0]
        displacement = math.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2]) / median_size

        segments = []
        for a, b in zip(points, points[1:]):
            dt = max(b[0] - a[0], 1e-6)
            segments.append(((b[1] - a[1]) / dt, (b[2] - a[2]) / dt, dt, max(a[3], b[3])))
        velocity_change = max(
            (math.hypot(s2[0] - s1[0], s2[1] - s1[1]) * s2[2] / s2[3] for s1, s2 in zip(segments, segments[1:])),
            default=0.0,
        )
        records.append({
            "class": cls, "identity": identity, "n_crops": len(points),
            "duration_s": round(duration, 3), "displacement": round(displacement, 4),
            "max_velocity_change": round(velocity_change, 4),
            "static": displacement < static_max_displacement and duration >= static_min_duration_s,
            "motion_jump": velocity_change > jump_threshold,
        })
    return records


def appearance_outliers(
    rows: list[dict],
    features: np.ndarray,
    percentile: float = 99.0,
    min_outliers: int = 2,
    min_fraction: float = 0.2,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """回傳 (身分層級紀錄, 每張裁切到身分平均的 cosine 距離, 每張裁切是否離群)。"""
    groups = _identity_groups(rows)
    distances = np.zeros(len(rows))
    for idx in groups.values():
        mean = features[idx].mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        distances[idx] = 1.0 - features[idx] @ mean

    thresholds = {
        cls: float(np.percentile(distances[[i for (c, _), idx in groups.items() if c == cls for i in idx]], percentile))
        for cls in {c for c, _ in groups}
    }
    is_outlier = np.array([distances[i] > thresholds[row["class"]] for i, row in enumerate(rows)], dtype=bool)

    records = []
    for (cls, identity), idx in sorted(groups.items()):
        n_out = int(is_outlier[idx].sum())
        fraction = n_out / len(idx)
        records.append({
            "class": cls, "identity": identity, "n_crops": len(idx), "n_outliers": n_out,
            "outlier_fraction": round(fraction, 4), "flagged": n_out >= min_outliers or fraction >= min_fraction,
        })
    return records, distances, is_outlier
