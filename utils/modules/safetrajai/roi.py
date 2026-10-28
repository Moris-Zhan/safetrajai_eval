"""以鏡頭為鍵的 ROI（關注區域，見 CONTEXT.md、docs/adr/0005）。

ROI 定義在 config/roi/<camera_id>.yaml：
    camera: C000025
    image_size: [640, 480]
    polygon:
      - [210, 90]
      - ...
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name
from safetrajai.paths import CONFIG_DIR

ROI_DIR = CONFIG_DIR / "roi"


def full_frame_polygon(image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)


def resolve_roi(seq_name: str, image_size: tuple[int, int], roi_dir: Path = ROI_DIR) -> np.ndarray:
    """回傳序列所屬鏡頭的 ROI 多邊形（float32, shape (N, 2)）。

    沒有 ROI 檔時回傳整張影像並提示；ROI 記錄的 image_size 與實際尺寸不符時報錯，
    不自動縮放，避免解析度不同時悄悄套用錯誤的區域。
    """
    camera_id = camera_id_from_seq_name(seq_name)
    roi_path = Path(roi_dir) / f"{camera_id}.yaml"
    if not roi_path.exists():
        print(f"  [ROI] 鏡頭 {camera_id} 沒有 ROI 檔（{roi_path}），以整張影像為 ROI")
        return full_frame_polygon(image_size)

    with roi_path.open("r", encoding="utf-8") as f:
        roi_def = yaml.safe_load(f) or {}

    declared_size = tuple(int(v) for v in roi_def.get("image_size", ()))
    actual_size = (int(image_size[0]), int(image_size[1]))
    if declared_size != actual_size:
        raise ValueError(
            f"ROI {roi_path} 的 image_size={list(declared_size)} 與序列 {seq_name!r} "
            f"的實際影像尺寸 {list(actual_size)} 不符"
        )
    return np.array(roi_def["polygon"], dtype=np.float32)


def is_center_in_roi(polygon: np.ndarray, center_x: float, center_y: float) -> bool:
    """框中心點是否在 ROI 內（落在邊界上視為在內）。"""
    return cv2.pointPolygonTest(polygon, (float(center_x), float(center_y)), False) >= 0
