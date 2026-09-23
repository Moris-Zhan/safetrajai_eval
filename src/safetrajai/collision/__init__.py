"""
module/collision — 軌跡碰撞偵測（TTC / PET）對外入口。

流程（見 track_eval.py::TrackingEvalPipeline.run_stage3()）：
    1. trajectory_adapter.load_sequence_tracks()：讀 stage1 軌跡檔（Layer 1 CSV）
       → Layer 2 track dict 清單
    2. homography_points.load_homography_for_sequence()：該序列有沒有人工填的對應點檔，
       有就算出 Homography 矩陣 H，並用 trajectory_adapter.add_homography_fields()
       附加 Homo_* 欄位；沒有就退回原始像素座標
    3. detection_core.compute_ttc_for_all_pairs() / compute_pet_for_all_pairs()：
       兩兩配對算 TTC / PET 事件
    4. output.py 寫三種輸出：乾淨 CSV/JSON、比照原始欄位風格的 log CSV、熱區圖 PNG

對外只用 run_collision_detection_for_sequence()，其餘函式各自模組內部使用
（比照 module/tracker 只曝露 create_tracker() 的慣例）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from . import detection_core
from . import homography as homography_mod
from . import homography_points
from . import output as collision_output
from .trajectory_adapter import add_homography_fields, load_sequence_tracks


def _reproject_point(pt: Tuple[float, float], H_inv) -> Tuple[float, float]:
    return homography_mod.transform_point(pt[0], pt[1], H_inv)


def _reproject_box(box, H_inv):
    return tuple(_reproject_point(pt, H_inv) for pt in box)


def _reproject_ttc_events(events: List[Dict[str, Any]], H_inv) -> List[Dict[str, Any]]:
    """把 TTC 事件裡所有座標欄位（collision_point/future_box*/arrow*）反投影回原始
    CCTV 像素座標。事件本身是用 compute_ttc_for_all_pairs() 算出來的，use_homography=True
    時內部座標系是 Homo_*（衛星圖／地圖座標），若不轉回來，輸出檔案裡的座標會跟本專案
    「CCTV 像素座標」慣例不一致，也跟熱區圖（疊在 CCTV 第一幀上）對不起來。
    H_inv 為 None（沒用 homography）時原樣回傳。
    """
    if H_inv is None:
        return events
    reprojected = []
    for e in events:
        e2 = dict(e)
        e2["collision_point"] = _reproject_point(e["collision_point"], H_inv)
        e2["future_box1"] = _reproject_box(e["future_box1"], H_inv)
        e2["future_box2"] = _reproject_box(e["future_box2"], H_inv)
        e2["arrow1"] = _reproject_box(e["arrow1"], H_inv)
        e2["arrow2"] = _reproject_box(e["arrow2"], H_inv)
        reprojected.append(e2)
    return reprojected


def _reproject_pet_events(events: List[Dict[str, Any]], H_inv) -> List[Dict[str, Any]]:
    """同 `_reproject_ttc_events()`，PET 事件只有 cross_point 需要反投影。"""
    if H_inv is None:
        return events
    reprojected = []
    for e in events:
        e2 = dict(e)
        e2["cross_point"] = _reproject_point(e["cross_point"], H_inv)
        reprojected.append(e2)
    return reprojected


def _resolve_ttc_bounds(tag: Optional[str], satellite_dir: Path, cctv_bounds: Tuple[int, int]) -> Tuple[int, int]:
    """`detection_core.find_ttc()` 用來判斷碰撞點是否落在畫面內的有效範圍。

    ⚠️ 這個 bounds 必須跟實際計算座標系一致：use_homography=True 且用既有矩陣時，
    座標是衛星圖座標系（最大到衛星圖尺寸，通常 1280x1280），不是原始 CCTV 影片解析度
    （通常 640x480）——如果誤用 CCTV 解析度當 bounds，find_ttc() 會把所有超出 640x480
    範圍的合法衛星圖座標點誤判為「落在畫面外」而全部丟棄，導致 TTC 事件被靜默漏檢。
    人工點選 fallback（沒有 tag、沒有真正底圖尺寸可讀）目前仍退回 cctv_bounds，這是已知限制。
    """
    if tag is None:
        return cctv_bounds
    sat_path = homography_points.prebuilt_satellite_image_path(satellite_dir, tag)
    sat_img = cv2.imread(str(sat_path))
    if sat_img is None:
        return cctv_bounds
    return (sat_img.shape[0], sat_img.shape[1])


def run_collision_detection_for_sequence(
    seq_output_dir: Path,
    cfg,
    fps: float,
    bounds: Tuple[int, int],
) -> Dict[str, Any]:
    """對單一序列跑完整的軌跡碰撞偵測流程，輸出寫到 cfg.collision_output_dir/<seq_name>/ 底下。

    Args:
        seq_output_dir: annotation_<seq> 目錄（stage1 輸出）
        cfg: AppConfig，需要 collision_* 系列參數與 homography_points_dir（見
             evaluation/config_loader.py）
        fps: 該序列影片的幀率
        bounds: (height, width)——該序列影片解析度。use_homography=False 時直接當
            TTC 有效碰撞區域範圍；use_homography=True 且用既有矩陣時會改用衛星圖尺寸
            （見 `_resolve_ttc_bounds()`），這個參數在那種情況下只是備援值。
            ⚠️ detection_core.find_ttc() 內部用 `0<=x<height and 0<=y<width` 判斷碰撞點
            是否落在畫面內，x/y 對應的界線是反過來的（跟 rects_iou 用 (x,y)=(寬,高)方向
            不一致），這是移植自參考程式碼的既有寫法，本次未修改，這裡不用特別處理，只在此註記。

    Returns:
        {"seq_name": str, "num_tracks": int, "ttc_events": [...], "pet_events": [...]}
        兩個事件清單裡的座標欄位（collision_point/future_box*/arrow*/cross_point）一律是
        原始 CCTV 像素座標——即使運算是在 Homography 轉換後座標系下進行的，輸出前都會
        反投影回來，跟本專案的座標慣例一致，也跟熱區圖疊圖背景一致。
    """
    seq_name = seq_output_dir.stem

    tracks = load_sequence_tracks(seq_output_dir, cfg, fps)

    tag = homography_points.find_tag_for_sequence(seq_name, cfg.homography_matrices_dir)
    H = homography_points.resolve_homography(
        seq_name, cfg.homography_matrices_dir, cfg.homography_points_dir,
    )
    use_homography = H is not None
    H_inv = homography_mod.invert_homography(H) if use_homography else None

    if use_homography and tag is not None:
        add_homography_fields(tracks, H, fps)
        print(f"  [{seq_name}] 用既有已算好的矩陣（tag={tag}），碰撞偵測改用轉換後座標系")
    elif use_homography:
        add_homography_fields(tracks, H, fps)
        print(f"  [{seq_name}] 找到人工點選的 homography 對應點檔，碰撞偵測改用轉換後座標系")
    else:
        print(
            f"  [{seq_name}] 沒有可用的 homography 矩陣（既有 26 路口不含此序列，"
            f"也沒有人工點選對應點檔"
            f"（{cfg.homography_points_dir}/{seq_name}.yaml），碰撞偵測使用原始像素座標"
        )

    max_time_diff_frames = int(round(cfg.collision_lookback_sec * fps))
    ttc_bounds = _resolve_ttc_bounds(tag, cfg.homography_satellite_dir, bounds) if use_homography else bounds

    ttc_events = detection_core.compute_ttc_for_all_pairs(
        tracks, fps=fps, max_ttc_frames=cfg.collision_max_ttc_frames, bounds=ttc_bounds,
        use_homography=use_homography, iou_threshold=cfg.collision_iou_threshold,
    )
    pet_events = detection_core.compute_pet_for_all_pairs(
        tracks, fps=fps, max_time_diff_frames=max_time_diff_frames,
        overlap_threshold=cfg.collision_overlap_threshold, use_homography=use_homography,
    )

    # 輸出前一律反投影回原始 CCTV 像素座標（H_inv 為 None 時原樣回傳，等同沒用 homography）
    ttc_events = _reproject_ttc_events(ttc_events, H_inv)
    pet_events = _reproject_pet_events(pet_events, H_inv)

    seq_out_dir = Path(cfg.collision_output_dir) / seq_name

    collision_output.write_events_json(ttc_events, seq_out_dir / "ttc_events.json")
    collision_output.write_events_csv(ttc_events, seq_out_dir / "ttc_events.csv")
    collision_output.write_ttc_legacy_log(ttc_events, seq_name, fps, seq_out_dir / "collision_ttc_log.csv")

    collision_output.write_events_json(pet_events, seq_out_dir / "pet_events.json")
    collision_output.write_events_csv(pet_events, seq_out_dir / "pet_events.csv")
    collision_output.write_pet_legacy_log(pet_events, seq_name, fps, seq_out_dir / "collision_pet_log.csv")

    first_frame_path = seq_output_dir / "first_frame_with_box.jpg"
    background = cv2.imread(str(first_frame_path)) if first_frame_path.exists() else None
    if background is not None:
        ttc_points = [e["collision_point"] for e in ttc_events]
        pet_points = [e["cross_point"] for e in pet_events]
        collision_output.draw_collision_heatmap(
            ttc_points, background, seq_out_dir / "ttc_collision_heatmap.png",
            title="TTC Collision Heatmap",
        )
        collision_output.draw_collision_heatmap(
            pet_points, background, seq_out_dir / "pet_collision_heatmap.png",
            title="PET Collision Heatmap",
        )
    else:
        print(f"  [{seq_name}] 找不到 {first_frame_path}，略過熱區圖輸出")

    print(
        f"  [{seq_name}] tracks={len(tracks)}  TTC 事件={len(ttc_events)}  "
        f"PET 事件={len(pet_events)}  輸出目錄={seq_out_dir}"
    )

    return {
        "seq_name": seq_name,
        "num_tracks": len(tracks),
        "ttc_events": ttc_events,
        "pet_events": pet_events,
    }


__all__ = ["run_collision_detection_for_sequence"]
