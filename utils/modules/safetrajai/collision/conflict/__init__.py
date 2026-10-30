"""
safetrajai.collision.conflict — stage3 衝突指標分析（change `conflict-metrics`）。

在有地理對位（m_per_pixel 可用）的鏡頭上，以公尺為單位，從 stage1 軌跡計算逐幀配對的
TTC／MTTC、衝突角度與類型、前導與跟隨、最近前後車、DRAC 與 DRAC 超限、Crash Index、
追撞鏈、PET，並合併成衝突事件。預設關閉，由情境 config 的 `conflict_analysis:` 區塊開啟。

輸出寫到 cfg.collision_output_dir/<seq>/conflict/（spec「輸出檔與座標」）；既有 stage3 的
TTC／PET 輸出不經過這裡，也不受影響（design 第 2 節）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from safetrajai.collision import homography as homography_mod
from safetrajai.collision import output as collision_output
from safetrajai.collision.homography import points as homography_points
from safetrajai.collision.homography.camera_config import load_camera_config_for_sequence
from safetrajai.collision.homography.georef import meters_per_pixel
from safetrajai.collision.homography.points import prebuilt_satellite_image_path
from safetrajai.collision.counting.crossing import find_crossings
from safetrajai.collision.counting.labels import resolve_labels
from safetrajai.collision.counting.lines import COUNTING_LINES_DIR, resolve_counting_lines
from safetrajai.collision.counting.tracks import EVERYWHERE_POLYGON, load_counting_tracks
from safetrajai.collision.homography.satellite import image_tag
from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name
from safetrajai.roi import resolve_roi

from .blind_area import compute_blind_area, polygon_mask
from .bev import BevBlindArea, BevCounting, BevCountingLine, BevEvent, bev_tracks, write_bev_video
from .chains import build_chains
from .config import ConflictConfig
from .events import build_events, overlap_at_start_pairs, threshold_counts
from .following import closest_front_rear, rear_end_metrics
from .kinematics import Kinematics, compute_kinematics, smoothing_window_frames
from .output import point_columns, records, write_csv, write_json
from .pairs import build_pairs
from .pet import compute_pet
from .tracks import Track, load_tracks, project_points
from .ttc import ttc_for_pairs

SMOOTHING_METHOD = "centred_moving_average"


def _key(track: Track) -> str:
    return f"{track.class_name}:{track.track_id}"


def _side_key(kins: Sequence[Kinematics], side: int, i: int, j: int) -> str:
    return _key(kins[i].track) if side == 1 else _key(kins[j].track) if side == 2 else ""


def _params(cfg: ConflictConfig) -> Dict[str, Any]:
    params = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    for name in ("bev_video", "blind_area", "object_width_m"):
        params.pop(name)  # optional outputs: not part of the metric parameters, keeps summary.json unchanged
    params["class_pairs"] = [list(p) for p in sorted(cfg.class_pairs)]
    params["angle_types"] = [list(t) for t in cfg.angle_types]
    params["object_length_m"] = dict(sorted(cfg.object_length_m.items()))
    params["object_mass_kg"] = dict(sorted(cfg.object_mass_kg.items()))
    return params


def _limitations(cfg: ConflictConfig, fps: float, acceleration_available: bool) -> List[Dict[str, str]]:
    if cfg.ttc_distance_mode == "fixed":
        distance = f"碰撞距離固定為兩條軌跡中心距離 {cfg.ttc_distance_m} m：對大型車偏寬鬆（車身早已重疊），對行人偏嚴格"
    else:
        distance = "碰撞距離為兩條軌跡類別長度一半的和（class_length 模式），不考慮寬度與朝向"
    items = [
        {"code": "collision_distance", "detail": distance},
        {"code": "centre_distance", "detail": "ΔS 與 DRAC 使用中心距離，不是車頭到車尾的間距，DRAC 偏低"},
        {"code": "ci_magnitude", "detail": "CI 與 CI_m 的速度與加速度取向量大小，無法區分加速與減速"},
        {"code": "mttc_requires_ttc", "detail": "MTTC 只在 TTC 命中時計算，只靠加速度才會接近的情況不列入"},
        {"code": "pet_single_point", "detail": "PET 每對軌跡只取一個衝突點"},
    ]
    if cfg.pet_max_gap_s <= 0:
        items.append({"code": "pet_co_presence", "detail": "PET 只納入至少同時出現過一幀的軌跡配對，漏掉前一方離開後另一方才進入的情況"})
    if not acceleration_available:
        items.append({
            "code": "no_acceleration_low_fps",
            "detail": f"序列 {fps} fps 低於 min_fps_for_acceleration={cfg.min_fps_for_acceleration}："
                      "MTTC、DR、MaxD、DRAC 超限、CI、CI_m 未定義",
        })
    items.append({"code": "not_calibrated", "detail": "未經實測校正，結果只能當作技術訊號，不能當成正式的安全結論"})
    return items


def _skip(out_dir: Path, seq_name: str, camera: str, tag: Optional[str], reason: str) -> Dict[str, Any]:
    summary = {
        "sequence": seq_name, "camera": camera, "matrix_tag": tag,
        "m_per_pixel": None, "skipped_reason": reason,
    }
    write_json(summary, out_dir / "summary.json")
    print(f"  [{seq_name}] 衝突指標分析略過（{reason}），只輸出 {out_dir / 'summary.json'}")
    return summary


def _bev_counting(cfg, seq_output_dir: Path, bounds: Tuple[int, int], H: np.ndarray) -> Optional[BevCounting]:
    if not cfg.counting.enabled:
        return None
    lines = resolve_counting_lines(
        seq_output_dir.name, (bounds[1], bounds[0]), getattr(cfg, "counting_lines_dir", COUNTING_LINES_DIR))
    if lines is None:
        return None
    crossings = find_crossings(load_counting_tracks(seq_output_dir, cfg), lines)
    labelled = [(line, resolve_labels(line, H)) for line in lines]
    projected = tuple(
        BevCountingLine(line.name, *(tuple(p) for p in project_points(np.array([line.p1, line.p2]), H)),
                        labels.forward, labels.backward) for line, labels in labelled)
    return BevCounting(projected, crossings)


def _satellite_background(cfg, camera_cfg) -> Optional[np.ndarray]:
    """The camera's prebuilt satellite image, or None when it has not been downloaded."""
    image_path = prebuilt_satellite_image_path(Path(cfg.homography_satellite_dir), image_tag(camera_cfg))
    return cv2.imread(str(image_path)) if image_path.exists() else None


def _blind_area(out_dir: Path, cfg, seq_output_dir: Path, ccfg: ConflictConfig, fps: float, roi: np.ndarray,
                H: np.ndarray, mpp: float,
                ) -> Tuple[Dict[str, Any], Optional[BevBlindArea], Optional[Dict[int, float]]]:
    """Summary block, the shadows to draw when the bird's-eye video needs them, and the ratio of every frame."""
    camera_cfg = load_camera_config_for_sequence(cfg.homography_camera_dir, seq_output_dir.name)
    if camera_cfg.camera_view_point is None:
        return {"written": False, "skipped_reason": "no_camera_view_point"}, None, None
    background = _satellite_background(cfg, camera_cfg)
    if background is None:
        return {"written": False, "skipped_reason": "no_satellite_image"}, None, None
    viewpoint = (float(camera_cfg.camera_view_point[0]), float(camera_cfg.camera_view_point[1]))
    # Occluders are every tracked vehicle, not only those inside the ROI: a large vehicle outside it still casts a shadow in.
    tracks = load_tracks(seq_output_dir, cfg.benchmark_name, cfg.split_name, cfg.tracker_name, EVERYWHERE_POLYGON, H, mpp)
    all_frames = [int(f) for t in tracks for f in t.frames]
    frames = range(min(all_frames), max(all_frames) + 1) if all_frames else range(0)
    occluders = [compute_kinematics(t, fps, ccfg) for t in tracks if t.class_name in ccfg.blind_area.occluder_classes]
    roi_mask = polygon_mask(project_points(roi, H), background.shape[:2])
    result = compute_blind_area(
        occluders, viewpoint, roi_mask, mpp, fps, frames, ccfg.object_length_m, ccfg.object_width_m,
        ccfg.max_gap_frames, keep_shadows=ccfg.bev_video.enabled)
    write_csv(result.columns, out_dir / "blind_area.csv")
    ratios = np.array(result.columns["blind_ratio"], dtype=np.float64)
    drawn = BevBlindArea(viewpoint, roi_mask, result.shadows) if ccfg.bev_video.enabled else None
    return {
        "written": True,
        "viewpoint_px": list(viewpoint),
        "occluder_classes": list(ccfg.blind_area.occluder_classes),
        "frames": len(ratios),
        "mean_blind_ratio": float(np.nanmean(ratios)) if len(ratios) else None,
        "max_blind_ratio": float(np.nanmax(ratios)) if len(ratios) else None,
        "viewpoint_inside_skips": result.viewpoint_inside_skips,
        "occluders_without_heading": result.occluders_without_heading,
        "frames_without_heading": result.frames_without_heading,
    }, drawn, dict(zip(result.columns["frame"], result.columns["blind_ratio"]))


def _bev_video(out_dir: Path, cfg, seq_output_dir: Path, kins, events, ccfg: ConflictConfig, fps: float,
               bounds: Tuple[int, int], H: np.ndarray, blind: Optional[BevBlindArea] = None) -> Dict[str, Any]:
    camera_cfg = load_camera_config_for_sequence(cfg.homography_camera_dir, seq_output_dir.name)
    background = _satellite_background(cfg, camera_cfg)
    if background is None:
        return {"written": False, "skipped_reason": "no_satellite_image"}
    links = tuple(BevEvent(e.start_frame, e.end_frame, e.i, e.j) for e in events)
    counting = _bev_counting(cfg, seq_output_dir, bounds, H)
    return write_bev_video(out_dir / "bev.mp4", background, bev_tracks(kins), ccfg.bev_video, fps, links, counting, blind)


def _pair_frame_columns(kins, pairs, ttc, metrics, neighbours, fps, mpp, H_inv) -> Dict[str, Any]:
    defined = (
        np.isfinite(ttc.ttc) | np.isfinite(ttc.mttc) | ttc.overlap_at_start
        | ~np.isnan(metrics.delta_s) | ~np.isnan(metrics.drac)
    )
    rows = np.flatnonzero(defined)
    rows = rows[np.lexsort((pairs.j[rows], pairs.i[rows], pairs.frame[rows]))]
    i, j, frame = pairs.i[rows], pairs.j[rows], pairs.frame[rows]

    def neighbour(k: int, f: int, side: int) -> str:
        other = neighbours.get((int(k), int(f)), (-1, -1))[side]
        return _key(kins[other].track) if other >= 0 else ""

    exceeds = metrics.drac_exceeds[rows]
    columns: Dict[str, Any] = {
        "frame": frame,
        "time_s": frame / fps,
        "class1": [kins[k].track.class_name for k in i],
        "id1": [kins[k].track.track_id for k in i],
        "class2": [kins[k].track.class_name for k in j],
        "id2": [kins[k].track.track_id for k in j],
        "angle_deg": pairs.angle_deg[rows],
        "conflict_type": pairs.conflict_type[rows],
        "overlap_at_start": ttc.overlap_at_start[rows],
        "ttc_s": ttc.ttc[rows],
        "mttc_s": ttc.mttc[rows],
        "leading": [_side_key(kins, s, a, b) for s, a, b in zip(ttc.leading[rows], i, j)],
    }
    columns.update(point_columns("collision", ttc.collision_xy_m[rows], mpp, H_inv))
    columns.update({
        "rear": [_side_key(kins, s, a, b) for s, a, b in zip(metrics.rear_side[rows], i, j)],
        "delta_s_m": metrics.delta_s[rows],
        "delta_v_mps": metrics.delta_v[rows],
        "dr_mps2": metrics.dr[rows],
        "maxd_mps2": metrics.maxd[rows],
        "drac_mps2": metrics.drac[rows],
        "drac_exceeds_maxd": [None if np.isnan(x) else bool(x) for x in exceeds],
        "drac_excess_mps2": metrics.drac_excess[rows],
        "drac_exceed_ratio": metrics.drac_exceed_ratio[rows],
        "ci": metrics.ci[rows],
        "ci_m": metrics.ci_m[rows],
        "front_of_1": [neighbour(k, f, 0) for k, f in zip(i, frame)],
        "rear_of_1": [neighbour(k, f, 1) for k, f in zip(i, frame)],
        "front_of_2": [neighbour(k, f, 0) for k, f in zip(j, frame)],
        "rear_of_2": [neighbour(k, f, 1) for k, f in zip(j, frame)],
    })
    return columns


def _event_columns(kins, events, ttc, fps, mpp, H_inv) -> Dict[str, Any]:
    events = sorted(events, key=lambda e: (e.start_frame, e.event_id))
    columns: Dict[str, Any] = {
        "event_id": [e.event_id for e in events],
        "class1": [kins[e.i].track.class_name for e in events],
        "id1": [kins[e.i].track.track_id for e in events],
        "class2": [kins[e.j].track.class_name for e in events],
        "id2": [kins[e.j].track.track_id for e in events],
        "start_frame": [e.start_frame for e in events],
        "end_frame": [e.end_frame for e in events],
        "n_frames": [e.n_frames for e in events],
        "start_time_s": [e.start_frame / fps for e in events],
        "end_time_s": [e.end_frame / fps for e in events],
        "min_ttc_s": [e.min_ttc for e in events],
        "min_ttc_frame": [e.min_ttc_frame for e in events],
        "min_mttc_s": [e.min_mttc for e in events],
        "conflict_type": [e.conflict_type for e in events],
        "leading": [_side_key(kins, e.leading, e.i, e.j) for e in events],
    }
    key_xy = ttc.collision_xy_m[[e.key_row for e in events]] if events else np.zeros((0, 2))
    columns.update(point_columns("collision", key_xy, mpp, H_inv))
    columns.update({
        "max_drac_mps2": [e.max_drac for e in events],
        "drac_ever_exceeded": [e.drac_ever_exceeded for e in events],
        "max_drac_excess_mps2": [e.max_drac_excess for e in events],
        "max_ci": [e.max_ci for e in events],
        "max_ci_m": [e.max_ci_m for e in events],
    })
    return columns


def _mean_blind_ratio(ratio_by_frame: Dict[int, float], start: int, end: int) -> float:
    values = [ratio_by_frame[f] for f in range(start, end + 1) if f in ratio_by_frame]
    return float(np.mean(values)) if values else float("nan")


def _pet_columns(tracks, pet_rows, mpp, H_inv) -> Dict[str, Any]:
    columns: Dict[str, Any] = {
        "class1": [tracks[r.i].class_name for r in pet_rows],
        "id1": [tracks[r.i].track_id for r in pet_rows],
        "class2": [tracks[r.j].class_name for r in pet_rows],
        "id2": [tracks[r.j].track_id for r in pet_rows],
    }
    xy = np.array([r.conflict_xy_m for r in pet_rows]) if pet_rows else np.zeros((0, 2))
    columns.update(point_columns("conflict", xy, mpp, H_inv))
    columns.update({
        "min_distance_m": [r.min_distance_m for r in pet_rows],
        "t1_in_s": [r.t1_in for r in pet_rows],
        "t1_out_s": [r.t1_out for r in pet_rows],
        "t2_in_s": [r.t2_in for r in pet_rows],
        "t2_out_s": [r.t2_out for r in pet_rows],
        "pet_s": [r.pet for r in pet_rows],
        "first": [_key(tracks[r.i] if r.first == 1 else tracks[r.j]) for r in pet_rows],
    })
    return columns


def _chain_columns(kins, chains) -> Dict[str, Any]:
    chains = sorted(chains, key=lambda c: (c.chain_id, c.direction, c.rank, kins[c.track_index].track.key))
    return {
        "class": [kins[c.track_index].track.class_name for c in chains],
        "track_id": [kins[c.track_index].track.track_id for c in chains],
        "chain_id": [c.chain_id for c in chains],
        "direction": [c.direction for c in chains],
        "heading_deg": [c.heading_deg for c in chains],
        "projection_m": [c.projection_m for c in chains],
        "rank": [c.rank for c in chains],
    }


def run_conflict_analysis_for_sequence(
    seq_output_dir: Path,
    cfg,
    fps: float,
    bounds: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """Conflict analysis of one stage1 sequence; returns the summary, or None when disabled.

    Args:
        seq_output_dir: stage1 序列目錄（含 trackers/ 與 first_frame_with_box.jpg）
        cfg: AppConfig，需要 conflict_analysis、homography_camera_dir、homography_matrices_dir、
             collision_output_dir、benchmark_name、split_name、tracker_name
        fps: 序列影片幀率
        bounds: (height, width) 影片解析度，用於 ROI
    """
    ccfg: ConflictConfig = cfg.conflict_analysis
    if not ccfg.enabled:
        return None

    seq_output_dir = Path(seq_output_dir)
    seq_name = seq_output_dir.stem
    camera = camera_id_from_seq_name(seq_name)
    out_dir = Path(cfg.collision_output_dir) / seq_name / "conflict"

    H, tag = homography_points.resolve_camera_homography(
        seq_name, cfg.homography_camera_dir, cfg.homography_matrices_dir,
    )
    if H is None:
        return _skip(out_dir, seq_name, camera, tag, "no_homography")
    mpp = meters_per_pixel(load_camera_config_for_sequence(cfg.homography_camera_dir, seq_name))
    if mpp is None:
        return _skip(out_dir, seq_name, camera, tag, "no_meters_per_pixel")
    H_inv = homography_mod.invert_homography(H)

    roi = resolve_roi(seq_name, (bounds[1], bounds[0]))
    tracks = load_tracks(seq_output_dir, cfg.benchmark_name, cfg.split_name, cfg.tracker_name, roi, H, mpp)
    kins = tuple(compute_kinematics(t, fps, ccfg) for t in tracks)
    pairs = build_pairs(kins, ccfg)
    ttc = ttc_for_pairs(kins, pairs, ccfg)
    metrics = rear_end_metrics(kins, pairs, ttc, ccfg)
    neighbours = closest_front_rear(kins, pairs)
    pet_rows = compute_pet(tracks, fps, ccfg)
    chains = build_chains(kins, pairs, metrics, ccfg)
    events = build_events(kins, pairs, ttc, metrics, ccfg)
    acceleration_available = fps >= ccfg.min_fps_for_acceleration

    pair_columns = _pair_frame_columns(kins, pairs, ttc, metrics, neighbours, fps, mpp, H_inv)
    blind_summary, blind_drawn, blind_ratios = (
        _blind_area(out_dir, cfg, seq_output_dir, ccfg, fps, roi, H, mpp) if ccfg.blind_area.enabled else (None, None, None))
    event_columns = _event_columns(kins, events, ttc, fps, mpp, H_inv)
    if blind_ratios is not None:
        event_columns["mean_blind_ratio"] = [
            _mean_blind_ratio(blind_ratios, start, end)
            for start, end in zip(event_columns["start_frame"], event_columns["end_frame"])]
    write_csv(pair_columns, out_dir / "pair_frames.csv")
    write_csv(event_columns, out_dir / "conflict_events.csv")
    write_json(records(event_columns), out_dir / "conflict_events.json")
    write_csv(_pet_columns(tracks, pet_rows, mpp, H_inv), out_dir / "pet.csv")
    write_csv(_chain_columns(kins, chains), out_dir / "chains.csv")

    first_frame_path = seq_output_dir / "first_frame_with_box.jpg"
    background = cv2.imread(str(first_frame_path)) if first_frame_path.exists() else None
    if background is not None:
        points = [
            (x, y) for x, y in zip(event_columns["collision_x_cctv_px"], event_columns["collision_y_cctv_px"])
            if np.isfinite(x) and np.isfinite(y)
        ]
        collision_output.draw_collision_heatmap(
            points, background, out_dir / "conflict_heatmap.png", title="Conflict Event Heatmap",
        )
    else:
        print(f"  [{seq_name}] 找不到 {first_frame_path}，略過衝突事件熱區圖")

    chain_sizes = Counter(Counter(c.chain_id for c in chains).values())
    summary = {
        "sequence": seq_name,
        "camera": camera,
        "matrix_tag": tag,
        "m_per_pixel": mpp,
        "skipped_reason": None,
        "fps": fps,
        "acceleration_available": acceleration_available,
        "smoothing": {"method": SMOOTHING_METHOD, "window_frames": smoothing_window_frames(ccfg.smooth_window_s, fps)},
        "params": _params(ccfg),
        "num_tracks": len(tracks),
        "num_pair_frames": len(pairs),
        "num_pair_frames_written": len(pair_columns["frame"]),
        "num_events": len(events),
        "events_by_type": dict(sorted(Counter(e.conflict_type for e in events).items())),
        "ttc_event_counts": threshold_counts(events, ccfg.stat_thresholds_s, "min_ttc"),
        "mttc_event_counts": threshold_counts(events, ccfg.stat_thresholds_s, "min_mttc") if acceleration_available else None,
        "pet_pair_counts": threshold_counts(pet_rows, ccfg.stat_thresholds_s, "pet"),
        "num_pet_pairs": len(pet_rows),
        "overlap_at_start_pairs": overlap_at_start_pairs(pairs, ttc),
        "chain_size_counts": {str(size): n for size, n in sorted(chain_sizes.items())},
        "limitations": _limitations(ccfg, fps, acceleration_available),
    }
    if ccfg.bev_video.enabled:
        summary["bev_video"] = _bev_video(out_dir, cfg, seq_output_dir, kins, events, ccfg, fps, bounds, H, blind_drawn)
        summary["limitations"].append({
            "code": "bev_video_fps",
            "detail": "bev.mp4 的輸出 fps 固定為 5（既有 VideoWriterWithFfmpeg 行為，未修正）；序列不是 5 fps 時，播放速度與實際時間不同。",
        })
    if ccfg.blind_area.enabled:
        summary["blind_area"] = blind_summary
        summary["limitations"].append({
            "code": "blind_area_2d",
            "detail": "盲區把車輛視為沒有高度的矩形，陰影是從視點投影到地面的平面區域，所以會高估盲區（真實車輛只遮住視線高度以下的範圍）；視點與朝向的誤差也直接影響結果。",
        })
    write_json(summary, out_dir / "summary.json")
    print(
        f"  [{seq_name}] 衝突指標：tracks={len(tracks)}  配對列={len(pairs)}  事件={len(events)}  "
        f"PET 配對={len(pet_rows)}  輸出目錄={out_dir}"
    )
    return summary


__all__ = ["run_conflict_analysis_for_sequence"]
