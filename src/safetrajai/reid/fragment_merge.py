"""遮擋碎片合併（見 CONTEXT.md「碎片」「碎片合併」、spec: reid-dataset）。

同一序列、同一類別內：
1. 候選：A 結束後「保留期限幀數＋margin_frames」幀內 B 出現（保留期限依追蹤器與序列幀率換算，
   見 ADR 0008），且 B 起點在 A 末段每幀速度外插位置的 position_factor 倍框尺寸（面積平方根）以內。
2. 外觀：A 最後 n 張與 B 最前 n 張的特徵平均取 cosine。
3. 一對一：相似度 ≥ 門檻者由高到低貪婪接受，每段最多一個前段、一個後段；以 union-find 串成鏈，
   合併後的身分沿用鏈中首張時間最早的軌跡。

只產生合併紀錄，不動任何裁切、索引與排除清單。
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import yaml

from safetrajai.paths import PROJECT_ROOT

MERGE_FIELDS = ("class", "identity", "merged_identity", "sequence", "score")
SAMPLE_FIELDS = ("sample_id", "stratum", "class", "camera", "sequence", "identity_a", "identity_b", "score")
STRATA = ("low", "mid", "high")


class CropSample(NamedTuple):
    crop_path: str
    frame: int
    time_s: float
    cx: float
    cy: float
    size: float


class Track(NamedTuple):
    cls: str
    identity: str
    sequence: str
    camera: str
    crops: tuple[CropSample, ...]


class Candidate(NamedTuple):
    cls: str
    sequence: str
    camera: str
    identity_a: str
    identity_b: str
    gap_frames: int
    distance_ratio: float


class ScoredCandidate(NamedTuple):
    candidate: Candidate
    score: float


def load_merge_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "time_margin_s" in raw:
        raise ValueError(f"{path}: time_margin_s 已不再支援，請改用 margin_frames（以保留期限幀數計算，見 ADR 0008）")
    threshold = raw.get("threshold")
    sample = raw["sample"]
    return {
        "track_config": PROJECT_ROOT / raw["track_config"],
        "margin_frames": int(raw["margin_frames"]),
        "position_factor": float(raw["position_factor"]),
        "endpoint_crops": int(raw["endpoint_crops"]),
        "threshold": None if threshold is None else float(threshold),
        "sample": {
            "strata_quantiles": [float(q) for q in sample["strata_quantiles"]],
            "counts": {s: int(sample["counts"][s]) for s in STRATA},
            "major_cameras": [str(c) for c in sample["major_cameras"]],
            "min_other": int(sample["min_other"]),
            "target_accuracy": float(sample["target_accuracy"]),
            "min_above": int(sample["min_above"]),
        },
    }


def load_retention(track_config: Path) -> dict[str, int]:
    """回傳逐類別的保留期限幀數，不依追蹤器或序列幀率換算。"""
    with Path(track_config).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    params = raw["class_tracking"]["class_tracking_params"]
    return {cls: int(p["track_buffer"]) for cls, p in params.items()}


def collect_tracks(index_rows: list[dict], exclusions: list[dict]) -> dict[tuple[str, str], Track]:
    """依 (class, identity) 整理軌跡；整段排除的身分不列入，單張排除的裁切不列入端點。"""
    removed = [e for e in exclusions if e.get("action", "remove") == "remove"]
    whole = {(e["class"], e["identity"]) for e in removed if not e.get("image")}
    images = {(e["class"], e["identity"], e["image"]) for e in removed if e.get("image")}

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in index_rows:
        key = (row["class"], row["identity"])
        if key in whole or (*key, PurePosixPath(row["crop_path"]).name) in images or (*key, row["crop_path"]) in images:
            continue
        grouped[key].append(row)

    tracks = {}
    for key, rows in grouped.items():
        samples = []
        for r in rows:
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            size = max(math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)), 1.0)
            samples.append(CropSample(r["crop_path"], int(r["frame"]), float(r["time_s"]),
                                      (x1 + x2) / 2, (y1 + y2) / 2, size))
        samples.sort(key=lambda s: (s.frame, s.crop_path))
        sequence = rows[0]["sequence"]
        tracks[key] = Track(key[0], key[1], sequence, rows[0]["camera"],
                            tuple(samples))
    return tracks


def _extrapolate(track: Track, gap_frames: int) -> tuple[float, float]:
    last = track.crops[-1]
    if len(track.crops) < 2:
        return last.cx, last.cy
    prev = track.crops[-2]
    df = max(last.frame - prev.frame, 1)
    return last.cx + (last.cx - prev.cx) / df * gap_frames, last.cy + (last.cy - prev.cy) / df * gap_frames


def find_candidates(
    tracks: Mapping[tuple[str, str], Track],
    track_buffers: Mapping[str, int],
    margin_frames: int,
    position_factor: float,
) -> list[Candidate]:
    groups: dict[tuple[str, str], list[Track]] = defaultdict(list)
    for track in tracks.values():
        groups[(track.sequence, track.cls)].append(track)

    candidates = []
    for (sequence, cls), members in sorted(groups.items()):
        if cls not in track_buffers:
            raise KeyError(f"track config 缺少類別 {cls!r} 的 track_buffer")
        if len(members) < 2:
            continue
        max_gap = track_buffers[cls] + margin_frames
        members.sort(key=lambda t: (t.crops[0].frame, t.identity))
        starts = [t.crops[0].frame for t in members]
        for a in members:
            end = a.crops[-1]
            lo = bisect.bisect_right(starts, end.frame)
            hi = bisect.bisect_right(starts, end.frame + max_gap)
            for b in members[lo:hi]:
                first = b.crops[0]
                gap = first.frame - end.frame
                px, py = _extrapolate(a, gap)
                ratio = math.hypot(first.cx - px, first.cy - py) / max(end.size, first.size)
                if gap > 0 and ratio <= position_factor:
                    candidates.append(Candidate(cls, sequence, a.camera, a.identity, b.identity, gap, ratio))
    return sorted(candidates, key=lambda c: (c.cls, c.sequence, c.identity_a, c.identity_b))


def tail_paths(track: Track, n: int) -> list[str]:
    return [c.crop_path for c in track.crops[-n:]]


def head_paths(track: Track, n: int) -> list[str]:
    return [c.crop_path for c in track.crops[:n]]


def endpoint_crop_paths(
    candidates: list[Candidate], tracks: Mapping[tuple[str, str], Track], n: int
) -> list[str]:
    paths = set()
    for c in candidates:
        paths.update(tail_paths(tracks[(c.cls, c.identity_a)], n))
        paths.update(head_paths(tracks[(c.cls, c.identity_b)], n))
    return sorted(paths)


def _mean_unit(paths: list[str], features: Mapping[str, np.ndarray]) -> np.ndarray:
    mean = np.mean([features[p] for p in paths], axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def score_candidates(
    candidates: list[Candidate],
    tracks: Mapping[tuple[str, str], Track],
    features: Mapping[str, np.ndarray],
    n: int,
) -> list[ScoredCandidate]:
    return [
        ScoredCandidate(c, float(np.dot(
            _mean_unit(tail_paths(tracks[(c.cls, c.identity_a)], n), features),
            _mean_unit(head_paths(tracks[(c.cls, c.identity_b)], n), features),
        )))
        for c in candidates
    ]


def select_pairs(scored: list[ScoredCandidate], threshold: float) -> list[ScoredCandidate]:
    """一對一貪婪配對：相似度由高到低，同分依 (class, sequence, A, B) 排序。"""
    ordered = sorted(
        (s for s in scored if s.score >= threshold),
        key=lambda s: (-s.score, s.candidate.cls, s.candidate.sequence, s.candidate.identity_a, s.candidate.identity_b),
    )
    has_next, has_prev, accepted = set(), set(), []
    for s in ordered:
        a, b = (s.candidate.cls, s.candidate.identity_a), (s.candidate.cls, s.candidate.identity_b)
        if a in has_next or b in has_prev:
            continue
        has_next.add(a)
        has_prev.add(b)
        accepted.append(s)
    return accepted


def merge_records(accepted: list[ScoredCandidate], tracks: Mapping[tuple[str, str], Track]) -> list[dict]:
    """以 union-find 串接配對；只列出被改名的原始身分（score 為接到前段的相似度），依 class、identity 排序。"""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(key: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def order(key: tuple[str, str]) -> tuple[int, str]:
        return tracks[key].crops[0].frame, key[1]

    link_score = {}
    for s in accepted:
        a, b = (s.candidate.cls, s.candidate.identity_a), (s.candidate.cls, s.candidate.identity_b)
        ra, rb = find(a), find(b)
        if ra != rb:
            first, second = sorted((ra, rb), key=order)
            parent[second] = first
        link_score[b] = s.score

    records = []
    for key in parent:
        root = find(key)
        if root != key:
            records.append({"class": key[0], "identity": key[1], "merged_identity": root[1],
                            "sequence": tracks[key].sequence, "score": f"{link_score[key]:.6f}"})
    return sorted(records, key=lambda r: (r["class"], r["identity"]))


def merge_map(records: list[dict]) -> dict[tuple[str, str], str]:
    return {(r["class"], r["identity"]): r["merged_identity"] for r in records}


def allocate_proportional(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """依大小比例分配名額（最大餘數法，依名稱排序），不超過各組大小。"""
    pool = sum(sizes.values())
    if not pool:
        return {g: 0 for g in sizes}
    quotas = {g: min(sizes[g], total * sizes[g] / pool) for g in sorted(sizes)}
    alloc = {g: int(q) for g, q in quotas.items()}
    left = min(total, pool) - sum(alloc.values())
    for g in sorted(quotas, key=lambda g: (-(quotas[g] - int(quotas[g])), g)):
        if left <= 0:
            break
        if alloc[g] < sizes[g]:
            alloc[g] += 1
            left -= 1
    return alloc


def stratified_sample(scored: list[ScoredCandidate], sample_cfg: dict, seed: int) -> list[dict]:
    """依相似度分位數分成低／中／高三層，每層內主要鏡頭與其他鏡頭依候選數比例分配，其他鏡頭合計至少 min_other 組。"""
    ordered = sorted(scored, key=lambda s: (s.candidate.cls, s.candidate.sequence,
                                            s.candidate.identity_a, s.candidate.identity_b))
    scores = np.array([s.score for s in ordered])
    cuts = np.quantile(scores, sample_cfg["strata_quantiles"]) if len(scores) else np.zeros(2)
    strata = {name: [] for name in STRATA}
    for s in ordered:
        strata[STRATA[int(np.searchsorted(cuts, s.score, side="right"))]].append(s)

    major = set(sample_cfg["major_cameras"])
    groups = {name: {"major": [s for s in members if s.candidate.camera in major],
                     "other": [s for s in members if s.candidate.camera not in major]}
              for name, members in strata.items()}
    alloc = {name: allocate_proportional(sample_cfg["counts"][name], {g: len(v) for g, v in groups[name].items()})
             for name in STRATA}
    for name in ("high", "mid", "low"):
        short = sample_cfg["min_other"] - sum(a["other"] for a in alloc.values())
        spare = len(groups[name]["other"]) - alloc[name]["other"]
        move = max(0, min(short, spare, alloc[name]["major"]))
        alloc[name]["other"] += move
        alloc[name]["major"] -= move

    rng = np.random.default_rng(seed)
    picks = []
    for name in STRATA:
        for group in ("major", "other"):
            members = groups[name][group]
            chosen = rng.choice(len(members), size=alloc[name][group], replace=False) if members else []
            picks += [(name, members[i]) for i in sorted(chosen)]
    return [
        {"sample_id": i, "stratum": name, "class": s.candidate.cls, "camera": s.candidate.camera,
         "sequence": s.candidate.sequence, "identity_a": s.candidate.identity_a,
         "identity_b": s.candidate.identity_b, "score": f"{s.score:.6f}"}
        for i, (name, s) in enumerate(picks, start=1)
    ]


def calibrate_threshold(scores_and_labels: list[tuple[float, bool]], target_accuracy: float) -> dict | None:
    """回傳讓「相似度 ≥ 門檻」的樣本準確率 ≥ target 的最低門檻（候選值為樣本相似度），達不到時回傳 None。"""
    for threshold in sorted({score for score, _ in scores_and_labels}):
        above = [same for score, same in scores_and_labels if score >= threshold]
        accuracy = sum(above) / len(above)
        if accuracy >= target_accuracy:
            return {"threshold": threshold, "accuracy": accuracy, "n_above": len(above),
                    "n_wrong_above": len(above) - sum(above)}
    return None


def supplementary_sample(
    scored: list[ScoredCandidate],
    min_score: float,
    exclude_pairs: set[tuple[str, str, str]],
    n: int,
    seed: int,
    start_id: int,
) -> list[dict]:
    """門檻以上樣本不足時的補抽：從相似度 ≥ min_score、未抽過的候選 (class, A, B) 中以固定種子抽 n 組。"""
    pool = sorted(
        (s for s in scored if s.score >= min_score
         and (s.candidate.cls, s.candidate.identity_a, s.candidate.identity_b) not in exclude_pairs),
        key=lambda s: (s.candidate.cls, s.candidate.sequence, s.candidate.identity_a, s.candidate.identity_b),
    )
    picks = sorted(np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False))
    return [
        {"sample_id": start_id + k, "stratum": "extra", "class": s.candidate.cls, "camera": s.candidate.camera,
         "sequence": s.candidate.sequence, "identity_a": s.candidate.identity_a,
         "identity_b": s.candidate.identity_b, "score": f"{s.score:.6f}"}
        for k, s in enumerate(pool[i] for i in picks)
    ]
