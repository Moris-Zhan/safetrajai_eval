"""Optional outputs of the conflict analysis: bird's-eye video and blind area (both off by default)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from safetrajai.collision.counting.crossing import find_crossings
from safetrajai.collision.counting.labels import resolve_labels
from safetrajai.collision.counting.lines import COUNTING_LINES_DIR, resolve_counting_lines
from safetrajai.collision.counting.tracks import EVERYWHERE_POLYGON, load_counting_tracks
from safetrajai.collision.homography.camera_config import load_camera_config_for_sequence
from safetrajai.collision.homography.points import prebuilt_satellite_image_path
from safetrajai.collision.homography.satellite import image_tag

from .bev import BevBlindArea, BevCounting, BevCountingLine, BevEvent, bev_tracks, write_bev_video
from .blind_area import compute_blind_area, polygon_mask
from .config import ConflictConfig
from .kinematics import compute_kinematics
from .output import write_csv
from .tracks import load_tracks, project_points


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


@dataclass(frozen=True)
class BlindAreaOutput:
    summary: Dict[str, Any]                                   # the `blind_area` block of summary.json
    drawn: Optional[BevBlindArea] = None                      # shadows for the video; only when it is enabled
    ratio_by_frame: Optional[Dict[int, float]] = None         # None when the blind area was skipped


def blind_area_output(out_dir: Path, cfg, seq_output_dir: Path, ccfg: ConflictConfig, fps: float, roi: np.ndarray,
                      H: np.ndarray, mpp: float) -> BlindAreaOutput:
    camera_cfg = load_camera_config_for_sequence(cfg.homography_camera_dir, seq_output_dir.name)
    if camera_cfg.camera_view_point is None:
        return BlindAreaOutput({"written": False, "skipped_reason": "no_camera_view_point"})
    background = _satellite_background(cfg, camera_cfg)
    if background is None:
        return BlindAreaOutput({"written": False, "skipped_reason": "no_satellite_image"})
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
    summary = {
        "written": True,
        "viewpoint_px": list(viewpoint),
        "occluder_classes": list(ccfg.blind_area.occluder_classes),
        "frames": len(ratios),
        "mean_blind_ratio": float(np.nanmean(ratios)) if len(ratios) else None,
        "max_blind_ratio": float(np.nanmax(ratios)) if len(ratios) else None,
        "viewpoint_inside_skips": result.viewpoint_inside_skips,
        "occluders_without_heading": result.occluders_without_heading,
        "frames_without_heading": result.frames_without_heading,
    }
    return BlindAreaOutput(summary, drawn, dict(zip(result.columns["frame"], result.columns["blind_ratio"])))


def mean_blind_ratio(ratio_by_frame: Dict[int, float], start: int, end: int) -> float:
    values = [ratio_by_frame[f] for f in range(start, end + 1) if f in ratio_by_frame]
    return float(np.mean(values)) if values else float("nan")


def bev_video_output(out_dir: Path, cfg, seq_output_dir: Path, kins, events, ccfg: ConflictConfig, fps: float,
                     bounds: Tuple[int, int], H: np.ndarray, blind: Optional[BevBlindArea] = None) -> Dict[str, Any]:
    camera_cfg = load_camera_config_for_sequence(cfg.homography_camera_dir, seq_output_dir.name)
    background = _satellite_background(cfg, camera_cfg)
    if background is None:
        return {"written": False, "skipped_reason": "no_satellite_image"}
    links = tuple(BevEvent(e.start_frame, e.end_frame, e.i, e.j) for e in events)
    counting = _bev_counting(cfg, seq_output_dir, bounds, H)
    return write_bev_video(out_dir / "bev.mp4", background, bev_tracks(kins), ccfg.bev_video, fps, links, counting, blind)
