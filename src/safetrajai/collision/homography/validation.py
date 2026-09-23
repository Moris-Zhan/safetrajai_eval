"""
Homography 矩陣驗證：拿獨立於訓練點之外的 check points，套用矩陣做 reprojection，
量測誤差（像素），評估某個路口的 Homography 矩陣準不準。

移植自 markdown/reference_results/homongraphy_val/homography_validator.py，本專案版本
跟原始程式的差異：
- 只做像素誤差。原始程式的公尺換算需要每個路口的緯度（Web Mercator 公式），
  來源的 GPS mapping Excel 沒有包在這份快照裡，目前拿不到，見
  markdown/ARCHITECTURE.md「Homography 資料」章節。
- 拿掉互動式選點 fallback（原始 pick_pairs_simple()，cv2/matplotlib 滑鼠事件，
  本專案 headless 環境（無 $DISPLAY）無法用）——check points 一律從
  config/homography_checkpoints/<tag>.yaml 讀，找不到就回傳「未驗證」，不會跳出視窗。

check points 必須是「獨立於訓練點之外」的驗證用點位（不是拿來算 H 的那 4 個訓練點），
才是有意義的 out-of-sample 驗證；用訓練點驗證會循環論證（誤差趨近於 0，見
markdown/ARCHITECTURE.md 相關章節的說明）。

⚠️ 本專案能真的產生 check points 的路口只有 6 個（有 CCTV 來源影片可看：C000025、
C000043、C000054、C000085、C000294、C000297），其餘 20 個既有矩陣路口沒有影片可點選，
永遠是「未驗證」狀態，不代表矩陣有問題，只是缺乏驗證資料。

yaml 格式（config/homography_checkpoints/<tag>.yaml，cctv_points / sat_points
需一一對應、順序相同，建議至少 6-8 組，越多統計越穩定，且不能跟算 H 用的訓練點重複）：
    cctv_points:
      - [x1, y1]
      ...
    sat_points:
      - [x1, y1]
      ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from . import load_homography


# =============================================================================
# 1. check points 載入
# =============================================================================

def checkpoints_path(checkpoints_dir: Path, tag: str) -> Path:
    """<checkpoints_dir>/<tag>.yaml 的完整路徑。"""
    return Path(checkpoints_dir) / f"{tag}.yaml"


def load_checkpoints(checkpoints_dir: Path, tag: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """讀 <checkpoints_dir>/<tag>.yaml，回傳 (cctv_points, sat_points)（皆為 (N,2) array）。

    找不到檔案，或欄位缺漏/數量不一致，回傳 None（呼叫端視為「該路口尚未提供驗證點」）。
    """
    path = checkpoints_path(checkpoints_dir, tag)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cctv_points = raw.get("cctv_points")
    sat_points = raw.get("sat_points")
    if not cctv_points or not sat_points:
        return None
    if len(cctv_points) != len(sat_points):
        print(f"⚠️  {path}: cctv_points/sat_points 數量不一致，略過驗證")
        return None

    return (
        np.asarray(cctv_points, dtype=np.float64),
        np.asarray(sat_points, dtype=np.float64),
    )


# =============================================================================
# 2. reprojection 誤差統計
# =============================================================================

def _apply_homography(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out[:, 0, :]


def compute_reprojection_stats(
    H: np.ndarray, cctv_points: np.ndarray, sat_points: np.ndarray, topk: int = 10,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """把 cctv_points 套用 H 投影到衛星座標系，跟 sat_points（GT）比較誤差（像素）。

    Returns:
        (stats, pred_sat_points, dist)：
            stats 含 N、dist_mean/std/median/rmse/p95/max_px、dx/dy_mean/std_px、
            worst_points（誤差最大的 topk 筆，依 dist 由大到小排序）。
            pred_sat_points：(N,2) array，cctv_points 投影後的預測座標，供畫 overlay 用。
            dist：(N,) array，每個點的誤差距離（像素），供 homography_heatmap.py 內插用。
    """
    pred_sat = _apply_homography(H, cctv_points)
    err = pred_sat - sat_points
    dx, dy = err[:, 0], err[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    order = np.argsort(-dist)
    k = min(topk, len(dist))
    worst_points = [
        {"index": int(i), "dist_px": round(float(dist[i]), 2),
         "dx_px": round(float(dx[i]), 2), "dy_px": round(float(dy[i]), 2)}
        for i in order[:k]
    ]

    stats = {
        "N": int(len(dist)),
        "dist_mean_px": float(np.mean(dist)),
        "dist_std_px": float(np.std(dist)),
        "dist_median_px": float(np.median(dist)),
        "dist_rmse_px": float(np.sqrt(np.mean(dist ** 2))),
        "dist_p95_px": float(np.percentile(dist, 95)),
        "dist_max_px": float(np.max(dist)),
        "dx_mean_px": float(np.mean(dx)),
        "dx_std_px": float(np.std(dx)),
        "dy_mean_px": float(np.mean(dy)),
        "dy_std_px": float(np.std(dy)),
        "worst_points": worst_points,
    }
    return stats, pred_sat, dist


# =============================================================================
# 3. 目視驗證圖（warp CCTV 疊到衛星圖，標 GT vs 預測）
# =============================================================================

def save_reprojection_overlay(
    cctv_frame: np.ndarray,
    satellite_bgr: np.ndarray,
    H: np.ndarray,
    sat_points: np.ndarray,
    pred_sat_points: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    """把 CCTV 影格用 H warp 到衛星圖座標系疊合，並標出 GT（綠）與預測（紅）check point，
    方便目視判斷誤差集中在畫面的哪個區域。
    """
    h, w = satellite_bgr.shape[:2]
    warped = cv2.warpPerspective(cctv_frame, H.astype(np.float64), (w, h))
    overlay = cv2.addWeighted(satellite_bgr, 1 - alpha, warped, alpha, 0)

    for i in range(len(sat_points)):
        gx, gy = sat_points[i]
        px, py = pred_sat_points[i]
        cv2.circle(overlay, (int(round(gx)), int(round(gy))), 5, (0, 255, 0), -1)  # GT
        cv2.circle(overlay, (int(round(px)), int(round(py))), 4, (0, 0, 255), -1)  # Pred
        cv2.putText(overlay, str(i + 1), (int(round(gx)) + 6, int(round(gy)) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


# =============================================================================
# 3b. 統計數字輸出成 Excel
# =============================================================================

def export_stats_excel(
    cctv_points: np.ndarray,
    sat_points: np.ndarray,
    pred_sat: np.ndarray,
    dist: np.ndarray,
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """把 compute_reprojection_stats() 的結果輸出成 Excel（summary + per_point 兩個 sheet），
    比照 markdown/reference_results/.../_stats.xlsx 的結構。

    ⚠️ 沒有公尺欄位——原始檔案的公尺換算需要每個路口的緯度（Web Mercator 公式），
    來源的 GPS mapping Excel 沒有包在這份快照裡，目前拿不到，見本檔案開頭模組說明。
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "summary"
    ws_summary.append(["metric", "value"])
    for key in (
        "N", "dist_mean_px", "dist_std_px", "dist_median_px", "dist_rmse_px",
        "dist_p95_px", "dist_max_px", "dx_mean_px", "dx_std_px", "dy_mean_px", "dy_std_px",
    ):
        ws_summary.append([key, stats[key]])

    ws_points = wb.create_sheet("per_point")
    ws_points.append([
        "pair_no", "cctv_x", "cctv_y", "gt_x", "gt_y", "pred_x", "pred_y",
        "dx_px", "dy_px", "dist_px", "rank_desc",
    ])
    order = np.argsort(-dist)
    rank_of = {int(idx): rank + 1 for rank, idx in enumerate(order)}
    for i in range(len(dist)):
        dx = float(pred_sat[i, 0] - sat_points[i, 0])
        dy = float(pred_sat[i, 1] - sat_points[i, 1])
        ws_points.append([
            i + 1,
            float(cctv_points[i, 0]), float(cctv_points[i, 1]),
            float(sat_points[i, 0]), float(sat_points[i, 1]),
            float(pred_sat[i, 0]), float(pred_sat[i, 1]),
            dx, dy, float(dist[i]), rank_of[i],
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))


# =============================================================================
# 4. 單一路口的完整驗證流程
# =============================================================================

def validate_tag(
    tag: str,
    matrices_dir: Path,
    satellite_dir: Path,
    checkpoints_dir: Path,
    cctv_frame: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    cctv_video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """對單一路口 tag 跑完整驗證：讀矩陣 + check points → 算誤差統計 → （可選）畫 overlay/
    輸出 Excel/產生逐幀 warp 比對影片。

    Args:
        cctv_frame: 該路口任一支來源影片的第一幀（BGR）。沒有提供（例如既有 26 路口裡
            沒有 CCTV 影片的 20 個）就不畫 overlay，只回傳數值統計（若有 check points 的話）。
        output_dir: 提供時把 overlay/熱力圖/Excel/比對影片存到這個目錄下。
        cctv_video_path: 該路口來源影片的完整路徑。提供時（且 output_dir 也提供）會額外產生
            `<tag>_compare.mp4`（見 module/collision/homography_compare_video.py），逐幀把
            整支影片 warp 到衛星座標系跟底圖疊合，跟原始影片並排存成影片，比單張 overlay
            更能看出整支影片時間軸上的對齊狀況。這支影片較耗時，沒有影片路徑就跳過。

    Returns:
        {"tag": tag, "validated": bool, "reason": str（validated=False 時才有）, ...統計欄位}
        "reason" 可能是 "no_matrix"（matrices_dir 沒有這個 tag 的 H）或
        "no_checkpoints"（沒有 config/homography_checkpoints/<tag>.yaml，多半是那 20 個
        沒有 CCTV 影片可點選的路口）。

    Side effect:
        output_dir 提供時會輸出：overlay 圖、`<tag>_stats.xlsx`（summary + per_point 兩個
        sheet）、IDW/Linear/RBF 誤差熱力圖各兩種風格（`save_error_heatmaps()`，check points
        少於 4 個會自動跳過）、（若也給了 cctv_video_path）逐幀 warp 比對影片。
    """
    H_path = Path(matrices_dir) / f"H_{tag}.npy"
    if not H_path.exists():
        return {"tag": tag, "validated": False, "reason": "no_matrix"}
    H = load_homography(str(H_path))

    checkpoints = load_checkpoints(checkpoints_dir, tag)
    if checkpoints is None:
        return {"tag": tag, "validated": False, "reason": "no_checkpoints"}

    cctv_points, sat_points = checkpoints
    stats, pred_sat, dist = compute_reprojection_stats(H, cctv_points, sat_points)
    stats["tag"] = tag
    stats["validated"] = True

    if output_dir is not None:
        sat_path = Path(satellite_dir) / f"{tag}.png"
        satellite_bgr = cv2.imread(str(sat_path))
        if satellite_bgr is not None:
            if cctv_frame is not None:
                save_reprojection_overlay(
                    cctv_frame, satellite_bgr, H, sat_points, pred_sat,
                    Path(output_dir) / f"{tag}_overlay.png",
                )
            export_stats_excel(
                cctv_points, sat_points, pred_sat, dist, stats,
                Path(output_dir) / f"{tag}_stats.xlsx",
            )
            from .heatmap import save_error_heatmaps
            save_error_heatmaps(tag, sat_points, dist, satellite_bgr, output_dir)

            if cctv_video_path is not None and Path(cctv_video_path).exists():
                from .compare_video import save_reprojection_compare_video
                save_reprojection_compare_video(
                    cctv_video_path, satellite_bgr, H, cctv_points, sat_points, pred_sat,
                    Path(output_dir) / f"{tag}_compare.mp4",
                )
        else:
            print(f"⚠️  {sat_path} 讀取失敗，略過 overlay/熱力圖/Excel/比對影片")

    return stats
