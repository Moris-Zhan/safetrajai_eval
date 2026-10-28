"""DINOv2 裁切特徵（letterbox 224、L2 正規化），供污染標記與碎片合併共用。

DINOv2 權重第一次使用時從 HF Hub 下載（只下載，不上傳）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = 224


def letterbox(image_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def embed_crops(crop_root: Path, rows: list[dict], batch_size: int) -> np.ndarray:
    import timm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE).eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    features = []
    for start in range(0, len(rows), batch_size):
        batch = np.stack([letterbox(cv2.imread(str(crop_root / r["crop_path"]))) for r in rows[start:start + batch_size]])
        tensor = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div(255.0)
        with torch.no_grad():
            feats = model((tensor - mean) / std)
        features.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
        print(f"  embedded {min(start + batch_size, len(rows))}/{len(rows)}")
    return np.concatenate(features) if features else np.zeros((0, 1))
