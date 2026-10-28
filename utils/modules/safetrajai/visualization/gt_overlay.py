"""GT overlay drawing shared by scripts/visualization/visualize_gt_on_video.py and visualize_gt_on_class_video.py."""

import configparser
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ── 每類別固定顏色（BGR）────────────────────────────────────────────────────
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0:  (255, 200,  50),   # person     → 黃
    1:  ( 50, 200, 255),   # car        → 天藍
    2:  ( 50, 255, 130),   # scooter    → 綠
    3:  (255,  80, 200),   # bus        → 紫
    4:  ( 80, 130, 255),   # truck      → 藍紫
    5:  (255, 150,  30),   # small_truck→ 橘
    6:  (255,  50,  50),   # bicycle    → 紅
    7:  (150, 255,  50),   # motorcycle → 黃綠
    -1: (200, 200, 200),   # unknown
}


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """依 track_id 決定 bbox 顏色（id 內唯一顏色）。"""
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]
    return palette[track_id % len(palette)]


def parse_seqinfo(seqinfo_path: Path) -> Dict:
    """解析 seqinfo.ini，回傳 dict（含 imWidth, imHeight, frameRate, seqLength 等）。"""
    cfg = configparser.ConfigParser()
    cfg.read(str(seqinfo_path), encoding="utf-8")
    sec = cfg["Sequence"] if "Sequence" in cfg else {}
    return {
        "name":        sec.get("name", seqinfo_path.parent.name),
        "imWidth":     int(sec.get("imWidth",  1920)),
        "imHeight":    int(sec.get("imHeight", 1080)),
        "frameRate":   float(sec.get("frameRate", 30)),
        "seqLength":   int(sec.get("seqLength", 0)),
        "imDir":       sec.get("imDir", "img1"),
        "imExt":       sec.get("imExt", ".jpg"),
    }


def draw_gt_on_frame(
    frame: np.ndarray,
    gt_annots: List[dict],
    show_id: bool = True,
    show_class: bool = True,
    show_conf: bool = False,
    color_by: str = "class",        # "class" | "id"
    alpha: float = 0.0,             # 填色透明度（0=不填色）
) -> np.ndarray:
    """
    在 frame 上繪製 GT bbox。

    Parameters
    ----------
    frame     : BGR numpy array
    gt_annots : 當幀的 annotation list（來自 load_gt_txt）
    show_id   : 是否顯示 track_id
    show_class: 是否顯示 class_name
    show_conf : 是否顯示 conf 分數
    color_by  : 顏色依據 "class" 或 "id"
    alpha     : bbox 填色透明度（0 = 只畫框線）
    """
    overlay = frame.copy() if alpha > 0 else None

    for ann in gt_annots:
        x1 = int(ann["x"])
        y1 = int(ann["y"])
        x2 = int(ann["x"] + ann["w"])
        y2 = int(ann["y"] + ann["h"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

        if color_by == "id":
            color = color_for_id(ann["id"])
        else:
            color = CLASS_COLORS.get(ann["class_id"], CLASS_COLORS[-1])

        # 填色
        if alpha > 0 and overlay is not None:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # 框線（GT 用虛線樣式模擬：畫兩個矩形，外白內色）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # 標籤文字
        parts = []
        if show_class:
            parts.append(ann["class_name"])
        if show_id:
            parts.append(f"#{ann['id']}")
        if show_conf:
            parts.append(f"{ann['conf']:.2f}")
        label = " ".join(parts)

        if label:
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness  = 1
            (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
            # 標籤背景
            lx1 = x1
            ly1 = max(0, y1 - th - 4)
            lx2 = x1 + tw + 4
            ly2 = y1
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                frame, label,
                (lx1 + 2, ly2 - 2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )

    if alpha > 0 and overlay is not None:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame
