"""
Homography 逐幀 warp 比對影片：把整支來源影片逐幀 warp 到衛星圖座標系跟底圖疊合，
跟原始影片並排輸出成一支影片，比 `homography_validation.save_reprojection_overlay()`
的單張靜態疊圖更能看出「對齊狀況隨時間變化」（例如遠處車道因為透視角度大，warp 後
對齊精準度是否比近處差、有沒有隨影片晃動等）。

比照舊版參考結果（已不在 repo 內）
C000025 中華三路 民生二路(東北角)-2-20-1280-1280_compare.mp4 的既有產出（左：原始 CCTV
+ check point 標記；右：逐幀 warp 疊底圖 + GT/預測點 + 統計文字），是重新設計、獨立實作的
版本，原始產生程式碼已經找不到。

⚠️ 這是相對耗時的操作（要重新讀整支影片逐幀 warp），只在明確需要「整支影片時間軸上的
對齊檢查」時才呼叫；純粹要知道矩陣準不準，`homography_validation.compute_reprojection_stats()`
的數值統計已經足夠。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _draw_points(
    img: np.ndarray, sat_points: np.ndarray, pred_sat_points: Optional[np.ndarray] = None,
) -> None:
    """在 img（原地修改）上畫 GT（綠）與（若提供）預測（紅）check point，標號從 1 開始。"""
    for i, (x, y) in enumerate(sat_points):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(img, pt, 5, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (pt[0] + 6, pt[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
    if pred_sat_points is not None:
        for x, y in pred_sat_points:
            pt = (int(round(x)), int(round(y)))
            cv2.circle(img, pt, 4, (0, 0, 255), -1)


def save_reprojection_compare_video(
    video_path: Path,
    satellite_bgr: np.ndarray,
    H: np.ndarray,
    cctv_points: np.ndarray,
    sat_points: np.ndarray,
    pred_sat_points: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    """左：原始 CCTV 影片（疊 check point 標記，固定位置） + 右：逐幀 warp 到衛星座標系
    疊底圖（疊 GT/預測點 + 誤差統計文字），並排輸出成一支 mp4。

    Args:
        video_path: 來源 CCTV 影片路徑（建議用 stage1 已複製到 <seq>/ 底下的那份，
            跟 homography_pick_aid.py 抓第一幀的來源一致）
        satellite_bgr: 衛星底圖，numpy array (H,W,3)，BGR
        H: 3x3 Homography 矩陣
        cctv_points, sat_points: check points 的 CCTV / 衛星圖座標，(N,2) array
        pred_sat_points: cctv_points 套用 H 投影後的預測衛星座標，(N,2) array
            （對應 homography_validation.compute_reprojection_stats() 的回傳值）
        output_path: 輸出 mp4 路徑
        alpha: 疊圖透明度，0.45 表示 warp 後的 CCTV 內容佔 45% 權重
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sat_h, sat_w = satellite_bgr.shape[:2]
    left_w = max(1, round(src_w * (sat_h / src_h)))  # CCTV 面板等比縮放到跟衛星圖同高
    canvas_w = left_w + sat_w

    dist = np.linalg.norm(pred_sat_points - sat_points, axis=1)
    stats_text = f"mean={float(np.mean(dist)):.2f}px  std={float(np.std(dist)):.2f}px  N={len(dist)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, sat_h))

    H = H.astype(np.float64)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        left = frame.copy()
        _draw_points(left, cctv_points)
        left = cv2.resize(left, (left_w, sat_h), interpolation=cv2.INTER_AREA)

        warped = cv2.warpPerspective(frame, H, (sat_w, sat_h))
        right = cv2.addWeighted(satellite_bgr, 1 - alpha, warped, alpha, 0)
        _draw_points(right, sat_points, pred_sat_points)
        cv2.putText(right, stats_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        canvas = np.hstack([left, right])
        writer.write(canvas)

    cap.release()
    writer.release()
    print(f"Homography 比對影片已存到: {output_path}（{frame_count} 幀）")
