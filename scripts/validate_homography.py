"""
批次驗證 models/homography/matrices/ 下所有既有 Homography 矩陣，彙整成一份總表，
方便快速檢視哪些路口已驗證、誤差多大、哪些還缺 check points。

用法：
    python -m module.collision.run_homography_validation

輸出（models/homography/validation_output/）：
    <tag>_overlay.png                     CCTV warp 疊到衛星圖 + GT/預測點（有 check points 才會產生）
    <tag>_stats.xlsx                      summary + per_point 兩個 sheet 的數值統計
    <tag>_heatmap_{idw,linear,rbf}.png       誤差空間內插熱力圖，plain 風格（check points >= 4 個才會產生）
    <tag>_pretty_heatmap_{idw,linear,rbf}.png  同上，pretty 風格（含色階圖例）
    <tag>_compare.mp4                     逐幀 warp 比對影片（只有本專案實際有來源影片的
                                           6 個路口會產生，且預設開啟，see generate_videos 參數）
    summary.csv                           26 個路口的驗證狀態彙總（validated 與否、reprojection 誤差統計）

沒有 check points 的路口（見 config/homography_checkpoints/README.md）在 summary.csv
裡 validated 欄位會是 False，reason 會標示原因（no_checkpoints / no_matrix），
不代表矩陣有問題，只是還沒有驗證資料。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import cv2

from safetrajai.collision.homography.pick_aid import USABLE_INTERSECTIONS
from safetrajai.collision.homography.points import list_available_tags
from safetrajai.collision.homography.validation import validate_tag

MATRICES_DIR = Path("models/homography/matrices")
SATELLITE_DIR = Path("models/homography/satellite_images")
CHECKPOINTS_DIR = Path("config/homography_checkpoints")
OUTPUT_DIR = Path("models/homography/validation_output")

# 只有這幾個路口有 CCTV 來源影片可以讀第一幀畫 overlay，其餘 tag 會是 cctv_frame=None
_TAG_TO_VIDEO = dict(USABLE_INTERSECTIONS)

_SUMMARY_FIELDS = [
    "tag", "validated", "reason", "N",
    "dist_mean_px", "dist_std_px", "dist_median_px",
    "dist_rmse_px", "dist_p95_px", "dist_max_px",
]


def run_all(generate_videos: bool = True) -> List[Dict[str, Any]]:
    """
    Args:
        generate_videos: 是否連帶產生逐幀 warp 比對影片（`<tag>_compare.mp4`，只有實際有
            來源影片的 6 個路口會產生）。影片相對耗時，反覆測試 stats/heatmap 時可設 False
            跳過，只重算數值與圖片。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for tag in list_available_tags(MATRICES_DIR):
        video_path = _TAG_TO_VIDEO.get(tag)
        cctv_frame = None
        if video_path is not None and Path(video_path).exists():
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                cctv_frame = frame

        result = validate_tag(
            tag, MATRICES_DIR, SATELLITE_DIR, CHECKPOINTS_DIR,
            cctv_frame=cctv_frame, output_dir=OUTPUT_DIR,
            cctv_video_path=video_path if generate_videos else None,
        )
        results.append(result)

        if result.get("validated"):
            print(f"[{tag}] validated  dist_mean={result['dist_mean_px']:.2f}px  "
                  f"dist_max={result['dist_max_px']:.2f}px")
        else:
            print(f"[{tag}] skipped ({result.get('reason')})")

    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n彙總表已存到: {summary_path}")

    return results


if __name__ == "__main__":
    run_all()
