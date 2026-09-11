"""
Homography check points 人工點選輔助圖：CCTV 第一幀與衛星底圖並排 + 座標格線，
方便人工目視估算/量測對應點座標，填進 config/homography_checkpoints/<tag>.yaml。

不是互動選點工具（本專案 headless 環境沒有 $DISPLAY，無法用 cv2/matplotlib 滑鼠事件），
只是把兩張圖排好、加上座標格線輔助讀數，實際點位仍由人工判讀後手動填 yaml，
沿用第一輪決定的「存圖 + 手寫座標」流程（見 module/collision/homography_points.py）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _draw_coord_grid(img: np.ndarray, step: int = 100, color=(0, 255, 255)) -> np.ndarray:
    """在圖上疊加每 step 像素一條的座標格線 + 座標數字，方便讀出點選座標。"""
    img = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
    for x in range(0, w, step):
        for y in range(0, h, step):
            cv2.putText(img, f"{x},{y}", (x + 3, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def _pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=img.dtype)
    return np.vstack([img, pad])


def save_pick_aid_image(
    cctv_frame: np.ndarray,
    satellite_bgr: np.ndarray,
    output_path: Path,
    grid_step: int = 100,
) -> None:
    """CCTV 第一幀（左）與衛星底圖（右）並排、各自疊上座標格線後存檔。

    Args:
        cctv_frame: CCTV 來源影片第一幀，BGR
        satellite_bgr: 對應路口的衛星底圖（models/homography/satellite_images/<tag>.png），BGR
        output_path: 輸出 PNG 路徑
        grid_step: 格線間距（像素）
    """
    left = _draw_coord_grid(cctv_frame, grid_step)
    right = _draw_coord_grid(satellite_bgr, grid_step)

    h = max(left.shape[0], right.shape[0])
    left = _pad_to_height(left, h)
    right = _pad_to_height(right, h)

    gap = np.full((h, 4, 3), 255, dtype=np.uint8)
    canvas = np.hstack([left, gap, right])

    cv2.putText(canvas, "CCTV (source)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Satellite (target)", (left.shape[1] + 14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Pick-aid 並排圖已存到: {output_path}")


def get_first_frame(video_path: Path) -> np.ndarray:
    """讀取影片第一幀（BGR）。"""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"無法讀取影片第一幀: {video_path}")
    return frame


# =============================================================================
# 批次產生：本專案實際有 CCTV 影片可用的 6 個路口
# =============================================================================

# (tag, 來源影片路徑) —— tag 需與 models/homography/matrices/H_<tag>.npy 的檔名一致
USABLE_INTERSECTIONS = [
    (
        "C000025 中華三路 民生二路(東北角)-2-20-1280-1280",
        "dataset/中華三路_民生二路(東北角)yolo驗證用_all8/annotations_all8/"
        "annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1/"
        "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1.mp4",
    ),
    (
        "C000043 民族一路 九如一路-2-20-1280-1280",
        "video/C000043 民族一路 九如一路_20250514071352.mp4",
    ),
    (
        "C000054 民族一路 天祥二路-2-20-1280-1280",
        "video/C000054 民族一路 天祥二路_20250627071742.mp4",
    ),
    (
        "C000085 中正路 三多路(建軍)-2-20-1280-1280",
        "video/C000085 中正路 三多路(建軍)_20250514130803.mp4",
    ),
    (
        "C000294 一心凱旋西南角-2-20-1280-1280",
        "video/C000294 一心凱旋西南角_20250514072037.mp4",
    ),
    (
        "C000297 一心 凱旋(西北角)-2-20-1280-1280",
        "video/C000297 一心 凱旋(西北角)_20250514071438.mp4",
    ),
]


def generate_all_pick_aids(
    satellite_dir: Path = Path("models/homography/satellite_images"),
    output_dir: Path = Path("models/homography/pick_aid"),
) -> None:
    """對 USABLE_INTERSECTIONS 逐一產生並排點選輔助圖。"""
    for tag, video_path in USABLE_INTERSECTIONS:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"⚠️  {tag}: 找不到來源影片 {video_path}，略過")
            continue

        sat_path = Path(satellite_dir) / f"{tag}.png"
        satellite_bgr = cv2.imread(str(sat_path))
        if satellite_bgr is None:
            print(f"⚠️  {tag}: 找不到衛星底圖 {sat_path}，略過")
            continue

        cctv_frame = get_first_frame(video_path)
        save_pick_aid_image(cctv_frame, satellite_bgr, Path(output_dir) / f"{tag}_pick_aid.png")


if __name__ == "__main__":
    generate_all_pick_aids()
