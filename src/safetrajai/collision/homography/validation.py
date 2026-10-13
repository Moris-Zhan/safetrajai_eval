"""
Homography 矩陣驗證：拿獨立於訓練點之外的 check points，套用矩陣做 reprojection，
量測誤差（像素），評估某個路口的 Homography 矩陣準不準。

移植自 Sub_Module/homongraphy_val/homography_validator.py，本專案版本
跟原始程式的差異：
- 以鏡頭為鍵：check points 從 config/homography_checkpoints/<camera>.yaml 讀，
  矩陣與比例尺資訊取自 config/homography/<camera>.yaml（spec `camera-homography`）。
- 公尺欄位只在鏡頭的 m_per_pixel 可用時輸出（見 georef.py：需要緯度，且比例尺檢查通過）；
  原始程式的換算漏乘 cos(緯度)。
- 另外輸出訓練點（算 H 用的點）的擬合殘差，欄位以 `train_residual_` 開頭；
  只有 4 點時 DLT 恰好可解，殘差接近 0 是預期結果，不代表矩陣準確。
- 拿掉互動式選點 fallback（原始 pick_pairs_simple()，cv2/matplotlib 滑鼠事件，
  本專案 headless 環境（無 $DISPLAY）無法用），找不到 check points 就回傳「未驗證」。

check points 必須是「獨立於訓練點之外」的驗證用點位，才是有意義的 out-of-sample 驗證；
與訓練點 CCTV 座標相同的 check point 會被排除，並在報表的 `check_excluded` 計數。
check points 少於 4 點時視為點數不足（`validated=False`），但仍輸出統計。

⚠️ 本專案能真的產生 check points 的路口只有 4 個（有 CCTV 來源影片可看：C000025、
C000043、C000054、C000085），其餘既有矩陣路口沒有影片可點選，
是「未驗證」狀態，不代表矩陣有問題，只是缺乏驗證資料。

yaml 格式（config/homography_checkpoints/<camera>.yaml，cctv_points / sat_points
需一一對應、順序相同，至少 4 組，建議 8 組以上）：
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
from .camera_config import CameraHomographyConfig
from .georef import meters_per_pixel


# =============================================================================
# 1. check points 載入
# =============================================================================

def checkpoints_path(checkpoints_dir: Path, camera: str) -> Path:
    """<checkpoints_dir>/<camera>.yaml 的完整路徑。"""
    return Path(checkpoints_dir) / f"{camera}.yaml"


def load_checkpoints(checkpoints_dir: Path, camera: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """讀 <checkpoints_dir>/<camera>.yaml，回傳 (cctv_points, sat_points)（皆為 (N,2) array）。

    找不到檔案，或欄位缺漏/數量不一致，回傳 None（呼叫端視為「該鏡頭尚未提供驗證點」）。
    """
    path = checkpoints_path(checkpoints_dir, camera)
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

MIN_CHECKPOINTS = 4
_STAT_KEYS = ("mean", "std", "median", "rmse", "p95", "max")


def _stat_columns(prefix: str, stats: Dict[str, Any], mpp: Optional[float], with_dist: bool) -> Dict[str, Any]:
    """Flatten reprojection stats into summary columns: `<prefix>_[dist_]<stat>_px` and `_m`."""
    stem = f"{prefix}_dist" if with_dist else prefix
    columns: Dict[str, Any] = {f"{prefix}_N": stats["N"]}
    for key in _STAT_KEYS:
        px = stats[f"dist_{key}_px"]
        columns[f"{stem}_{key}_px"] = px
        columns[f"{stem}_{key}_m"] = px * mpp if mpp is not None else None
    return columns


def validate_camera(
    camera_cfg: CameraHomographyConfig,
    matrices_dir: Path,
    satellite_dir: Path,
    checkpoints_dir: Path,
    cctv_frame: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    cctv_video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """對單一鏡頭跑完整驗證：讀矩陣、訓練點與 check points → 算誤差統計 → （可選）畫 overlay/
    輸出 Excel/產生逐幀 warp 比對影片。

    Args:
        camera_cfg: config/homography/<camera>.yaml（camera_config.load_camera_config()）。
        cctv_frame: 該鏡頭任一支來源影片的第一幀（BGR）。沒有提供（例如既有 26 路口裡
            沒有 CCTV 影片的 22 個）就不畫 overlay，只回傳數值統計（若有 check points 的話）。
        output_dir: 提供時把 overlay/熱力圖/Excel/比對影片存到這個目錄下。
        cctv_video_path: 該路口來源影片的完整路徑。提供時（且 output_dir 也提供）會額外產生
            `<tag>_compare.mp4`（見 safetrajai/collision/homography/compare_video.py），逐幀把
            整支影片 warp 到衛星座標系跟底圖疊合，跟原始影片並排存成影片，比單張 overlay
            更能看出整支影片時間軸上的對齊狀況。這支影片較耗時，沒有影片路徑就跳過。

    Returns:
        一列彙總：camera、tag、validated、reason（validated=False 時才有）、scale_check、
        m_per_pixel、check_N、check_excluded、check_dist_<stat>_px／_m、
        train_residual_N、train_residual_<stat>_px／_m（<stat> 為 mean/std/median/rmse/p95/max；
        _m 欄位在 m_per_pixel 不可用時為 None）。
        "reason" 可能是 "no_matrix"、"no_checkpoints" 或 "too_few_checkpoints"（少於 4 點）。

    Side effect:
        output_dir 提供時會輸出：overlay 圖、`<tag>_stats.xlsx`（summary + per_point 兩個
        sheet）、IDW/Linear/RBF 誤差熱力圖各兩種風格（`save_error_heatmaps()`，check points
        少於 4 個會自動跳過）、（若也給了 cctv_video_path）逐幀 warp 比對影片。
    """
    tag = camera_cfg.matrix_tag
    row: Dict[str, Any] = {
        "camera": camera_cfg.camera, "tag": tag, "scale_check": camera_cfg.scale_check,
        "m_per_pixel": meters_per_pixel(camera_cfg),
    }
    H_path = Path(matrices_dir) / f"H_{tag}.npy"
    if tag is None or not H_path.exists():
        return {**row, "validated": False, "reason": "no_matrix"}
    H = load_homography(str(H_path))
    mpp = row["m_per_pixel"]

    train_cctv_path = Path(matrices_dir) / f"pts_cctv_{tag}.npy"
    train_sat_path = Path(matrices_dir) / f"pts_sate_{tag}.npy"
    train_cctv = None
    if train_cctv_path.exists() and train_sat_path.exists():
        train_cctv = np.load(train_cctv_path).reshape(-1, 2).astype(np.float64)
        train_sat = np.load(train_sat_path).reshape(-1, 2).astype(np.float64)
        train_stats, _, _ = compute_reprojection_stats(H, train_cctv, train_sat)
        row.update(_stat_columns("train_residual", train_stats, mpp, with_dist=False))

    checkpoints = load_checkpoints(checkpoints_dir, camera_cfg.camera)
    if checkpoints is None:
        return {**row, "validated": False, "reason": "no_checkpoints"}

    cctv_points, sat_points = checkpoints
    keep = np.ones(len(cctv_points), dtype=bool)
    if train_cctv is not None:
        keep = ~np.array([np.any(np.all(np.isclose(p, train_cctv, atol=1e-6), axis=1)) for p in cctv_points])
    row["check_excluded"] = int((~keep).sum())
    cctv_points, sat_points = cctv_points[keep], sat_points[keep]
    if len(cctv_points) == 0:
        return {**row, "validated": False, "reason": "no_checkpoints"}

    stats, pred_sat, dist = compute_reprojection_stats(H, cctv_points, sat_points)
    row.update(_stat_columns("check", stats, mpp, with_dist=True))
    if len(cctv_points) < MIN_CHECKPOINTS:
        row.update({"validated": False, "reason": "too_few_checkpoints"})
    else:
        row["validated"] = True

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

    return row
