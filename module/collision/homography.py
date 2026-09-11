"""
Homography（單應性變換）矩陣計算與座標轉換核心運算。

移植自 markdown/reference_code/homography_core.py（SafeTrajAI 專案原始碼萃取版），
邏輯未修改。互動選點 GUI（原程式用 cv2 滑鼠事件）與目標標註流程高度相關，不易
通用，因此不包含在這裡——本專案改用「人工存圖 + 手寫座標到 yaml」的方式取得對應點，
見 module/collision/homography_points.py。

本專案目前沒有真實地圖底圖，"map" 側點位是使用者自行在同一張 CCTV 第一幀畫面上
定義的校正後座標（見 markdown/HOMOGRAPHY.md），因此這裡的轉換實質上是手動透視校正，
不是地理座標轉換，詳見 markdown/TRAJECTORY_FORMAT.md 第 3 節。

原始程式位置：
- utils/map_utils.py 的 class HomographyMapping
- utils/csv_utils.py 的 transform_point() / transform_box() / transform_pos_bbox()
- utils/coll_pointer/ttc_analyzer.py 內對 H_inv 的使用（碰撞點反投影回 CCTV 視角）

相依套件：numpy、opencv-python（cv2，用於 findHomography / perspectiveTransform /
warpPerspective）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


# =============================================================================
# 1. 計算 Homography 矩陣
# =============================================================================

def compute_homography(src_pts: Sequence[Point], dst_pts: Sequence[Point]) -> np.ndarray:
    """由來源影像（如 CCTV）與目標影像（如地圖底圖）上的對應點，計算 3x3 Homography 矩陣。

    Args:
        src_pts: 來源影像上的對應點座標，至少 4 組，如 [(x1,y1), (x2,y2), ...]
        dst_pts: 目標影像上的對應點座標，需與 src_pts 一一對應、順序相同

    Returns:
        3x3 numpy array，將 src 座標系轉換到 dst 座標系的 Homography 矩陣

    Note:
        - 只有 4 組點時 `cv2.findHomography()` 是「恰好可解」的 DLT 運算，沒有
          RANSAC 容錯機制，點位精準度會直接反映在轉換結果上。
        - 若提供 4 組以上的點，`cv2.findHomography()` 預設會改用最小平方法，
          建議搭配 `method=cv2.RANSAC` 以提升對雜訊點的容錯度（此處保留原始行為
          為預設，如需更穩健的結果可自行加上
          `method=cv2.RANSAC, ransacReprojThreshold=3.0` 等參數）。
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    if len(src) < 4 or len(dst) < 4:
        raise ValueError("Homography 計算至少需要 4 組對應點")
    if len(src) != len(dst):
        raise ValueError("src_pts 與 dst_pts 的點數必須相同且一一對應")

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Homography 計算失敗，請確認提供的對應點是否共線或重複")
    return H


def save_homography(H: np.ndarray, path: str) -> None:
    """儲存 Homography 矩陣為 .npy 檔案。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, H)


def load_homography(path: str) -> np.ndarray:
    """載入 Homography 矩陣。"""
    return np.load(path, allow_pickle=True)


# =============================================================================
# 2. 套用 Homography 轉換座標
# =============================================================================

def transform_point(x: float, y: float, H: np.ndarray) -> Point:
    """使用 Homography 矩陣轉換單一點座標。"""
    point = np.array([x, y, 1.0])
    transformed = H @ point
    transformed /= transformed[2]  # 齊次座標歸一化
    return float(transformed[0]), float(transformed[1])


def transform_points_batch(points: Sequence[Point], H: np.ndarray) -> List[Point]:
    """向量化版本，一次轉換多個點（比逐點呼叫 transform_point() 更有效率）。

    內部使用 cv2.perspectiveTransform，數學上與 transform_point() 逐點運算
    結果相同。
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return [(float(p[0][0]), float(p[0][1])) for p in transformed]


def transform_box(top_left: Point, bottom_right: Point, H: np.ndarray) -> List[List[List[float]]]:
    """將 bounding box 的四個角點分別轉換，回傳透視變形後的四邊形（非矩形）。

    由於透視變換不保持矩形形狀，僅轉換左上/右下兩點再連直線會失真，因此需將
    四個角點分別轉換。
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    box_pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    transformed_pts = cv2.perspectiveTransform(box_pts, H)
    return transformed_pts.tolist()


def transform_positions_and_bboxes(
    positions: Dict[int, Tuple[float, float, float]],
    bboxes: Dict[int, Tuple[Point, Point]],
    H: np.ndarray,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[Point, Point]], Dict[int, list]]:
    """批次轉換一條軌跡的 Positions / Bboxs（對應 TRAJECTORY_FORMAT.md 第 3 節格式）。

    Args:
        positions: {frame: (x, y, z)}，z 不會被轉換，原值保留
        bboxes: {frame: ((x1,y1), (x2,y2))}
        H: 3x3 Homography 矩陣

    Returns:
        (new_positions, new_bboxes, new_bboxes_polygon)
            new_positions: {frame: (x', y', z)}
            new_bboxes: {frame: ((x1',y1'), (x2',y2'))}  # 僅轉換左上右下兩點(簡化版，不考慮透視變形)
            new_bboxes_polygon: {frame: [[x,y],[x,y],[x,y],[x,y]]}  # 四角分別轉換後的四邊形(精確版)
    """
    new_positions = {}
    for frame, (x, y, z) in positions.items():
        new_x, new_y = transform_point(x, y, H)
        new_positions[frame] = (new_x, new_y, z)

    new_bboxes = {}
    new_bboxes_polygon = {}
    for frame, (pt1, pt2) in bboxes.items():
        new_x1, new_y1 = transform_point(pt1[0], pt1[1], H)
        new_x2, new_y2 = transform_point(pt2[0], pt2[1], H)
        new_bboxes[frame] = ((new_x1, new_y1), (new_x2, new_y2))
        new_bboxes_polygon[frame] = transform_box(pt1, pt2, H)[0]

    return new_positions, new_bboxes, new_bboxes_polygon


def invert_homography(H: np.ndarray) -> np.ndarray:
    """回傳反向矩陣，用於將目標座標系（如地圖）的點反投影回來源座標系（如 CCTV 影像）。"""
    return np.linalg.inv(H)


# =============================================================================
# 3. 驗證工具（無 GUI 版本）
# =============================================================================

def warp_and_overlay(
    src_img: np.ndarray, dst_img: np.ndarray, H: np.ndarray, alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """將來源影像依 H 轉換到目標影像座標系，並與目標影像疊合，用於人工目視檢查對齊精準度。

    不呼叫 cv2.imshow()（headless 環境可用），改為直接回傳影像陣列，由呼叫端
    自行決定要顯示、儲存或進一步處理。

    Args:
        src_img: 來源影像（如 CCTV 畫面），numpy array (H, W, 3)
        dst_img: 目標影像（如地圖底圖），numpy array (H, W, 3)
        H: 3x3 Homography 矩陣（src -> dst 方向）
        alpha: 疊合透明度，0.5 表示來源/目標影像各佔一半權重

    Returns:
        (warped, overlay)：
            warped: src_img 依 H 轉換到 dst_img 座標系與尺寸後的結果
            overlay: warped 與 dst_img 依 alpha 疊合後的結果，可用於目視比對是否對齊
    """
    h, w = dst_img.shape[:2]
    warped = cv2.warpPerspective(src_img, H, (w, h))
    overlay = cv2.addWeighted(dst_img, 1 - alpha, warped, alpha, 0)
    return warped, overlay
