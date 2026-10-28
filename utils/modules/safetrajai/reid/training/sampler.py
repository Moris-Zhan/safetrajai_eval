"""每批 P 個身分 × K 張的取樣（design D1）；身分張數不足 K 時重複取樣。

DistractorPKBatchSampler（change align-reid-training-domain D3）：每批以本 epoch 未用過的一個身分為種子，最多
round(P × fraction) − 1 個名額從種子的干擾物中選（先選未用過的，不足再重複選入已用過的；同一批不重複），
其餘從尚未用過的身分補滿；每個 epoch 的批數不變。干擾物＝同序列、同類別，
裁切幀範圍（不擴張）至少重疊 cooccur_min_frames 幀，且有一對幀差 ≤ pair_max_frame_gap 的裁切中心距離
≤ 半徑；幀範圍不重疊的碎片因此不會被當成負樣本推開。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import yaml
from torch.utils.data import Sampler

from safetrajai.reid.dataset_index import index_path

EVAL_IMAGE_WIDTH = 640


class PKBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: Sequence[int], p: int, k: int, seed: int):
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_label[int(label)].append(index)
        if len(self.by_label) < p:
            raise ValueError(f"身分數 {len(self.by_label)} 少於每批身分數 P={p}")
        self.p, self.k, self.seed = p, k, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.by_label) // self.p

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(sorted(self.by_label))
        for start in range(0, len(self) * self.p, self.p):
            batch = []
            for label in order[start:start + self.p]:
                pool = self.by_label[int(label)]
                batch.extend(int(i) for i in rng.choice(pool, self.k, replace=len(pool) < self.k))
            yield batch


@dataclass(frozen=True)
class CropPoint:
    frame: int
    cx: float
    cy: float


def distractor_radius(norm_factor: float, image_width: int) -> float:
    return 2.0 * norm_factor * image_width / EVAL_IMAGE_WIDTH


def _has_close_pair(a: list[CropPoint], b: list[CropPoint], radius: float, max_gap: int) -> bool:
    a_frames = np.array([p.frame for p in a])[:, None]
    b_frames = np.array([p.frame for p in b])[None, :]
    a_xy = np.array([(p.cx, p.cy) for p in a])[:, None, :]
    b_xy = np.array([(p.cx, p.cy) for p in b])[None, :, :]
    close_in_time = np.abs(a_frames - b_frames) <= max_gap
    close_in_space = np.linalg.norm(a_xy - b_xy, axis=2) <= radius
    return bool(np.any(close_in_time & close_in_space))


def find_distractors(tracks: Mapping[int, tuple[str, str, list[CropPoint]]], radius: Mapping[int, float],
                     cooccur_min_frames: int, pair_max_frame_gap: int) -> dict[int, frozenset[int]]:
    """tracks: label -> (sequence, class, crop points); returns label -> its distractor labels."""
    groups: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for label, (sequence, cls, points) in tracks.items():
        frames = [p.frame for p in points]
        groups[(sequence, cls)].append((min(frames), max(frames), label))

    found: dict[int, set[int]] = {label: set() for label in tracks}
    for spans in groups.values():
        spans.sort()
        for i, (start_a, end_a, label_a) in enumerate(spans):
            for start_b, end_b, label_b in spans[i + 1:]:
                if start_b > end_a - cooccur_min_frames + 1:
                    break
                if min(end_a, end_b) - start_b + 1 < cooccur_min_frames:
                    continue
                if _has_close_pair(tracks[label_a][2], tracks[label_b][2], radius[label_a], pair_max_frame_gap):
                    found[label_a].add(label_b)
                    found[label_b].add(label_a)
    return {label: frozenset(others) for label, others in found.items()}


def sequence_from_crop_path(crop_path: str) -> str:
    return crop_path.split("/")[1].rsplit("_track", 1)[0]


def load_crop_points(rows: Sequence[dict], crop_root: Path) -> dict[str, tuple[str, CropPoint]]:
    """crop_path -> (sequence, point) from the per-sequence crop indexes (metadata only, no images)."""
    wanted = {r["crop_path"] for r in rows}
    points = {}
    for sequence in sorted({sequence_from_crop_path(p) for p in wanted}):
        with index_path(crop_root, sequence).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["crop_path"] in wanted:
                    cx = (float(row["x1"]) + float(row["x2"])) / 2
                    cy = (float(row["y1"]) + float(row["y2"])) / 2
                    points[row["crop_path"]] = (sequence, CropPoint(int(row["frame"]), cx, cy))
    missing = wanted - set(points)
    if missing:
        raise ValueError(f"{len(missing)} 張 manifest 裁切在裁切索引中找不到，例如 {sorted(missing)[0]}")
    return points


def camera_widths(cameras: set[str], roi_dir: Path) -> dict[str, int]:
    widths = {}
    for camera in sorted(cameras):
        roi = yaml.safe_load((Path(roi_dir) / f"{camera}.yaml").read_text(encoding="utf-8"))
        widths[camera] = int(roi["image_size"][0])
    return widths


def build_label_tracks(rows: Sequence[dict], labels: Sequence[int], points: Mapping[str, tuple[str, CropPoint]],
                       ) -> dict[int, tuple[str, str, list[CropPoint]]]:
    tracks: dict[int, tuple[str, str, list[CropPoint]]] = {}
    for row, label in zip(rows, labels):
        sequence, point = points[row["crop_path"]]
        if label not in tracks:
            tracks[label] = (sequence, row["class"], [])
        elif tracks[label][0] != sequence:
            raise ValueError(f"身分 {row['identity']} 跨越多個序列：{tracks[label][0]}、{sequence}")
        tracks[label][2].append(point)
    return tracks


class DistractorPKBatchSampler(PKBatchSampler):
    """fraction=0 yields exactly the same batches as PKBatchSampler with the same seed."""

    def __init__(self, labels: Sequence[int], p: int, k: int, seed: int,
                 distractors: Mapping[int, frozenset[int]], fraction: float):
        super().__init__(labels, p, k, seed)
        self.distractors = distractors
        self.slots = max(0, round(p * fraction) - 1)
        self.epoch_stats = {"distractor_identities": 0, "distractor_share": 0.0, "reused_distractors": 0,
                            "seen_identity_share": 0.0}

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = [int(label) for label in rng.permutation(sorted(self.by_label))]
        used: set[int] = set()
        cursor = 0
        picked_total = 0
        reused_total = 0

        def next_unused() -> int:
            nonlocal cursor
            while order[cursor] in used:
                cursor += 1
            return order[cursor]

        for _ in range(len(self)):
            seed_label = next_unused()
            used.add(seed_label)
            chosen = [seed_label]
            if self.slots:
                others = self.distractors.get(seed_label, frozenset())
                fresh = sorted(others - used)
                if fresh:
                    picks = [int(c) for c in rng.permutation(fresh)[:self.slots]]
                    chosen.extend(picks)
                    used.update(picks)
                    picked_total += len(picks)
                missing = self.slots - (len(chosen) - 1)
                repeat = sorted(others - set(chosen)) if missing > 0 else []
                if repeat:
                    picks = [int(c) for c in rng.permutation(repeat)[:missing]]
                    chosen.extend(picks)
                    picked_total += len(picks)
                    reused_total += len(picks)
            while len(chosen) < self.p:
                label = next_unused()
                used.add(label)
                chosen.append(label)
            batch = []
            for label in chosen:
                pool = self.by_label[label]
                batch.extend(int(i) for i in rng.choice(pool, self.k, replace=len(pool) < self.k))
            yield batch

        self.epoch_stats = {
            "distractor_identities": picked_total,
            "distractor_share": round(picked_total / max(1, len(self) * self.p), 4),
            "reused_distractors": reused_total,
            "seen_identity_share": round(len(used) / max(1, len(self.by_label)), 4),
        }
