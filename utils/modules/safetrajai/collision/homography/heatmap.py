"""
Homography 誤差熱力圖：把 check points 的 reprojection 誤差（像素距離）在衛星圖上做
空間內插，視覺化「這個路口的哪個區域誤差比較大」，是 homography_validation.py
數值統計以外的空間診斷輔助。

比照舊版參考結果（已不在 repo 內）的既有產出風格（IDW／
Linear／RBF 三種內插法各一張，疊在衛星底圖上、標示 check point 位置與誤差值、
convex hull 外遮罩不上色）。⚠️ 那份的原始產生程式碼已經找不到，這裡是依圖面觀察重新設計、獨立實作的版本，
數學細節（配色、遮罩方式、字型排版）不保證跟原始程式完全一致，只有「三種內插法 + 疊圖 +
convexhull 遮罩」這個核心概念相同。

三種內插法：
- IDW（Inverse Distance Weighting，反距離加權）：自行實作，
  value = Σ(w_i·v_i)/Σw_i，w_i = 1/dist(query, point_i)^power
- Linear：scipy.interpolate.griddata(method="linear")，基於 Delaunay 三角剖分的分段線性內插
- RBF：scipy.interpolate.RBFInterpolator（thin-plate spline kernel），徑向基函數內插，
  比 Linear 平滑，但外插區域可能震盪——這也是原圖標題強調「inside convex hull」的原因，
  本模組只在 check points 的 convex hull 範圍內著色，範圍外直接顯示底圖，不內插。

check points 少於 4 個時無法定義有意義的 convex hull，會跳過熱力圖產生。

每種內插法都輸出兩種風格（比照參考範例的兩種檔名慣例）：
- `<tag>_heatmap_{method}.png`（plain）：背景整體調暗、無色階圖例，標題是燒進圖片的單行文字
  （方法名 + mean/std px）。
- `<tag>_pretty_heatmap_{method}.png`（pretty）：背景全亮度、右側有色階圖例、標題有深色底。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import ConvexHull, Delaunay


# =============================================================================
# 內插
# =============================================================================

def _idw_interpolate(
    points: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, power: float = 2.0,
) -> np.ndarray:
    """反距離加權內插（scipy 沒有現成函式，自行實作）。"""
    gx = grid_x.ravel()
    gy = grid_y.ravel()
    result = np.empty_like(gx, dtype=np.float64)

    for i in range(gx.shape[0]):
        dx = points[:, 0] - gx[i]
        dy = points[:, 1] - gy[i]
        dist = np.sqrt(dx * dx + dy * dy)
        exact = dist < 1e-9
        if np.any(exact):
            result[i] = values[np.argmax(exact)]
            continue
        w = 1.0 / (dist ** power)
        result[i] = np.sum(w * values) / np.sum(w)

    return result.reshape(grid_x.shape)


def _interpolate_field(
    method: str, points: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
) -> np.ndarray:
    if method == "idw":
        return _idw_interpolate(points, values, grid_x, grid_y)
    if method == "linear":
        field = griddata(points, values, (grid_x, grid_y), method="linear")
        return np.nan_to_num(field, nan=float(np.mean(values)))
    if method == "rbf":
        rbf = RBFInterpolator(points, values, kernel="thin_plate_spline")
        query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        return rbf(query).reshape(grid_x.shape)
    raise ValueError(f"Unknown interpolation method: {method!r}（只支援 idw/linear/rbf）")


def _convex_hull_mask(hull_points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """回傳與 grid_x/grid_y 同 shape 的布林遮罩，True 代表該格點落在 hull_points 的凸包內。"""
    tri = Delaunay(hull_points)
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    inside = tri.find_simplex(query) >= 0
    return inside.reshape(grid_x.shape)


# =============================================================================
# 繪製
# =============================================================================

def _render_legend(height: int, width: int, vmin: float, vmax: float, margin: int = 10) -> np.ndarray:
    """畫一條垂直色階圖例（JET colormap，上大下小），回傳 (height, width, 3) BGR 陣列。"""
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    bar_w = 24
    bar_h = height - 2 * margin

    gradient = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_w, axis=1)
    gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    legend[margin:margin + bar_h, margin:margin + bar_w] = gradient_color

    cv2.putText(legend, f"{vmax:.1f}px", (margin + bar_w + 6, margin + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, f"{vmin:.1f}px", (margin + bar_w + 6, margin + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return legend


def render_error_heatmap(
    method: str,
    sat_points: np.ndarray,
    error_dist_px: np.ndarray,
    background_bgr: np.ndarray,
    output_path: Path,
    title: str = "",
    subtitle: str = "outside convex hull = masked / low confidence zone",
    legend_width: int = 160,
    style: str = "pretty",
) -> None:
    """把 error_dist_px（sat_points 位置上的 reprojection 誤差量值）在衛星圖 convex hull
    範圍內做空間內插，畫成熱力圖疊在背景圖上存檔。

    Args:
        method: "idw" | "linear" | "rbf"
        sat_points: (N,2) 衛星圖座標系下的 check point 座標
        error_dist_px: (N,) 每個點對應的誤差距離（像素），來自
            homography_validation.compute_reprojection_stats() 回傳的逐點距離
        background_bgr: 背景圖（衛星底圖），numpy array (H,W,3)，BGR
        output_path: 輸出 PNG 路徑
        title: 圖標題（左上角文字）
        style: "pretty"（預設，全亮度底圖 + 右側色階圖例 + 深色標題底）或
            "plain"（底圖整體調暗、無色階圖例、標題直接印在暗底上，比照
            舊版參考結果的 heatmap_{idw,linear,rbf}.png 那組風格）
    """
    height, width = background_bgr.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    grid_x = grid_x.astype(np.float64)
    grid_y = grid_y.astype(np.float64)

    field = _interpolate_field(method, sat_points, error_dist_px, grid_x, grid_y)
    mask = _convex_hull_mask(sat_points, grid_x, grid_y)

    vmin, vmax = float(np.min(error_dist_px)), float(np.max(error_dist_px))
    if vmax <= vmin:
        vmax = vmin + 1.0
    field_norm = np.clip((field - vmin) / (vmax - vmin), 0.0, 1.0)
    field_color = cv2.applyColorMap((field_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    background_for_blend = background_bgr.astype(np.float64)
    if style == "plain":
        background_for_blend = background_for_blend * 0.45  # 整體調暗，比照 plain 風格的參考圖

    alpha = np.where(mask, 0.6, 0.0)[..., None]
    blended = (background_for_blend * (1 - alpha) + field_color.astype(np.float64) * alpha)
    blended = blended.astype(np.uint8)

    hull = ConvexHull(sat_points)
    hull_pts = sat_points[hull.vertices].astype(np.int32)
    cv2.polylines(blended, [hull_pts.reshape(-1, 1, 2)], isClosed=True, color=(255, 255, 255), thickness=2)

    for i, (x, y) in enumerate(sat_points):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(blended, pt, 4, (0, 0, 0), -1)
        cv2.circle(blended, pt, 4, (255, 255, 255), 1)
        cv2.putText(blended, f"#{i + 1} {error_dist_px[i]:.1f}px", (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if style == "plain":
        canvas = blended
    else:
        canvas = np.zeros((height, width + legend_width, 3), dtype=np.uint8)
        canvas[:, :width] = blended
        canvas[:, width:] = _render_legend(height, legend_width, vmin, vmax)

    if title:
        if style == "pretty":
            header_region = canvas[0:56, 0:width].copy()
            black = np.zeros_like(header_region)
            canvas[0:56, 0:width] = cv2.addWeighted(header_region, 0.45, black, 0.55, 0)
            cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, subtitle, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, subtitle, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


_METHOD_LABELS = {"idw": "IDW", "linear": "Linear", "rbf": "RBF (thin-plate spline)"}


def save_error_heatmaps(
    tag: str,
    sat_points: np.ndarray,
    error_dist_px: np.ndarray,
    background_bgr: np.ndarray,
    output_dir: Path,
) -> None:
    """依序輸出 IDW/Linear/RBF 三種內插法、各兩種風格的誤差熱力圖：
    `<tag>_heatmap_{idw,linear,rbf}.png`（plain）與
    `<tag>_pretty_heatmap_{idw,linear,rbf}.png`（pretty），比照參考範例的檔名慣例。

    check points 少於 4 個時（凸包/內插無意義）直接跳過，不產生檔案。
    """
    if len(sat_points) < 4:
        print(f"  [{tag}] check points 少於 4 個，跳過誤差熱力圖（凸包/內插至少需要 4 點）")
        return

    mean_px = float(np.mean(error_dist_px))
    std_px = float(np.std(error_dist_px))

    for method in ("idw", "linear", "rbf"):
        label = _METHOD_LABELS[method]

        render_error_heatmap(
            method, sat_points, error_dist_px, background_bgr,
            Path(output_dir) / f"{tag}_pretty_heatmap_{method}.png",
            title=f"Homography Error Heatmap ({method.upper()} inside convex hull)",
            style="pretty",
        )
        render_error_heatmap(
            method, sat_points, error_dist_px, background_bgr,
            Path(output_dir) / f"{tag}_heatmap_{method}.png",
            title=f"{label} | mean={mean_px:.2f}px std={std_px:.2f}px",
            style="plain",
        )
