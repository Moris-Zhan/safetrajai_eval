"""Re-ID crop collection used by stage1 when save_reid_crops is enabled."""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from safetrajai.tracking import CLASS_NAMES


def make_min_size_quality_filter(min_width: int, min_height: int) -> Callable[[np.ndarray], bool]:
    """Tier 1 品質過濾器範例：裁切寬或高小於門檻值就捨棄。

    quality_filters 掛載點吃的是 Callable[[np.ndarray], bool]（True=保留），
    用法：ReidCropCollector(crop_dir, det_corner, quality_filters=[make_min_size_quality_filter(20, 20)])
    預設（不傳 quality_filters）行為與現行完全相同，見 openspec/changes/reid-training-data-pipeline/。
    """
    def _filter(crop: np.ndarray) -> bool:
        h, w = crop.shape[:2]
        return w >= min_width and h >= min_height
    return _filter


class ReidCropCollector:
    """蒐集 Re-ID 訓練用裁切圖（save_reid_crops=True 時啟用）。

    存到 crop_dir/{class}/{seq_stem}_track{track_id}/{frame_id:06d}.jpg，
    用「seq_stem_track_id」當身分資料夾，避免不同影片的 track_id 重來
    （tracker 每支影片重建一次）互相撞名。
    """

    def __init__(
        self,
        crop_dir: Path,
        det_corner: np.ndarray | None,
        quality_filters: list[Callable[[np.ndarray], bool]] | None = None,
    ):
        self.crop_dir = crop_dir
        self.det_corner = det_corner
        self.quality_filters = quality_filters

    def collect(
        self,
        tracks,
        clean_frame: np.ndarray,
        seq_stem: str,
        frame_id: int,
        width: int,
        height: int,
    ) -> None:
        """對每個 track：多邊形過濾（與 draw/annotate 邏輯一致）→ 裁切 → 品質過濾（可選）→ 存檔。"""
        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                cls_id   = int(track[6])

                if self.det_corner is not None and len(self.det_corner) > 0:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if cv2.pointPolygonTest(self.det_corner, (float(cx), float(cy)), False) < 0:
                        continue

                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(width, int(x2)), min(height, int(y2))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue

                crop = clean_frame[yi1:yi2, xi1:xi2]

                if self.quality_filters and not all(f(crop) for f in self.quality_filters):
                    continue

                cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                identity_dir = self.crop_dir / cls_name / f"{seq_stem}_track{track_id}"
                identity_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(identity_dir / f"{frame_id:06d}.jpg"), crop)
