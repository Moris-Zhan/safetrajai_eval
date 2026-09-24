"""Re-ID crop collection used by stage1 when save_reid_crops is enabled (spec: reid-dataset)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name
from safetrajai.reid.dataset_index import date_from_seq_name, index_path, write_index
from safetrajai.roi import is_center_in_roi
from safetrajai.tracking import CLASS_NAMES

MIN_CROP_SIDE_PX = 32
SAMPLE_INTERVAL_S = 0.5


def frame_time_s(pos_msec: float, frame_id: int, fps: float, prev_time_s: float | None) -> float:
    """影片時間：優先用解碼器時間戳；回傳 0 或沒有遞增時退回 frame_id / fps。"""
    if pos_msec and pos_msec > 0:
        t = pos_msec / 1000.0
        if prev_time_s is None or t > prev_time_s:
            return t
    return frame_id / fps


class ReidCropCollector:
    """蒐集單一序列的 Re-ID 裁切：ROI 內、最短邊 >= 32px、每條軌跡每 0.5 秒最多一張。

    裁切存到 crop_dir/{class}/{seq}_track{track_id}/{frame:06d}.jpg，序列結束時由 finalize()
    一次寫出 crop_dir/_index/{seq}.csv（每張裁切一列）與略過統計 {seq}.skipped.json。
    序列的索引已存在時報錯，避免重複寫入。
    """

    def __init__(
        self,
        crop_dir: Path,
        roi: np.ndarray,
        seq_name: str,
        min_side: int = MIN_CROP_SIDE_PX,
        interval_s: float = SAMPLE_INTERVAL_S,
    ):
        self.crop_dir = Path(crop_dir)
        self.roi = roi
        self.seq_name = seq_name
        self.camera = camera_id_from_seq_name(seq_name)
        self.date = date_from_seq_name(seq_name)
        self.min_side = min_side
        self.interval_s = interval_s
        self.index_file = index_path(self.crop_dir, seq_name)
        if self.index_file.exists():
            raise FileExistsError(f"序列的裁切索引已存在，請先移走舊輸出再重跑：{self.index_file}")
        self._rows: list[dict] = []
        self._last_sampled: dict[tuple[int, int], float] = {}
        self._skipped_small: Counter = Counter()
        self._skipped_outside: Counter = Counter()

    def collect(
        self,
        tracks,
        clean_frame: np.ndarray,
        frame_id: int,
        time_s: float,
        width: int,
        height: int,
    ) -> None:
        for track in tracks:
            if len(track) < 7:
                continue
            x1, y1, x2, y2 = (float(v) for v in track[:4])
            track_id = int(track[4])
            score = float(track[5])
            cls_id = int(track[6])
            cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

            if not is_center_in_roi(self.roi, (x1 + x2) / 2, (y1 + y2) / 2):
                self._skipped_outside[cls_name] += 1
                continue

            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(width, int(x2)), min(height, int(y2))
            if min(xi2 - xi1, yi2 - yi1) < self.min_side:
                self._skipped_small[cls_name] += 1
                continue

            key = (cls_id, track_id)
            last = self._last_sampled.get(key)
            if last is not None and time_s - last < self.interval_s:
                continue

            identity = f"{self.seq_name}_track{track_id}"
            crop_rel = Path(cls_name) / identity / f"{frame_id:06d}.jpg"
            crop_abs = self.crop_dir / crop_rel
            crop_abs.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_abs), clean_frame[yi1:yi2, xi1:xi2])
            self._last_sampled[key] = time_s
            self._rows.append({
                "crop_path": crop_rel.as_posix(), "identity": identity, "class": cls_name,
                "camera": self.camera, "sequence": self.seq_name, "date": self.date,
                "frame": frame_id, "time_s": f"{time_s:.3f}",
                "x1": xi1, "y1": yi1, "x2": xi2, "y2": yi2, "score": f"{score:.4f}",
            })

    def finalize(self) -> dict:
        write_index(self.index_file, self._rows)
        stats = {
            "crops": len(self._rows),
            "identities": len({(r["class"], r["identity"]) for r in self._rows}),
            "too_small": dict(sorted(self._skipped_small.items())),
            "outside_roi": dict(sorted(self._skipped_outside.items())),
        }
        self.index_file.with_suffix(".skipped.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return stats
