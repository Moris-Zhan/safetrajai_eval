"""Stage3 counting-line statistics of one sequence (spec F2).

Reads the stage1 MOT tracks (bottom-centre of every box, no ROI filtering), counts crossings of
the camera's counting lines and writes `collision_output/<seq>/counting/{counts.csv,summary.json}`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from safetrajai.collision.conflict.output import write_csv, write_json
from safetrajai.collision.counting.crossing import find_crossings
from safetrajai.collision.counting.labels import ResolvedLabels, resolve_labels
from safetrajai.collision.counting.lines import COUNTING_LINES_DIR, CountingLine, resolve_counting_lines
from safetrajai.collision.counting.tracks import CLASSES, load_counting_tracks
from safetrajai.collision.homography import points as homography_points
from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name

DIRECTIONS = ("forward", "backward")

LIMITATIONS: List[Dict[str, str]] = [
    {"code": "id_switches",
     "detail": "ID 切換（身分中斷後換號）會讓同一物體被計多次，遺漏追蹤則少計；數字只當技術訊號，不是正式流量。"},
    {"code": "not_roi_filtered",
     "detail": "stage1 軌跡不依 ROI 過濾（ADR 0005），計數線在 ROI 外的穿越也會被計入。"},
]


def _line_record(line: CountingLine, labels: ResolvedLabels) -> Dict[str, Any]:
    return {"name": line.name, "points": [list(line.p1), list(line.p2)],
            "forward_label": labels.forward, "backward_label": labels.backward,
            "label_source": {"forward": labels.forward_source, "backward": labels.backward_source}}


def run_counting_for_sequence(
    seq_output_dir: Path,
    cfg,
    fps: float,
    bounds: Tuple[int, int],
    lines_dir: Path = COUNTING_LINES_DIR,
) -> Optional[Dict[str, Any]]:
    """Counting of one stage1 sequence; returns the summary, or None when disabled.

    Args:
        seq_output_dir: stage1 序列目錄（含 trackers/）
        cfg: AppConfig，需要 counting、collision_output_dir、benchmark_name、split_name、tracker_name、
             homography_camera_dir、homography_matrices_dir（有 Homography 時標籤用羅盤方向）
        fps: 序列影片幀率（時間欄位為 frame / fps，與衝突指標一致）
        bounds: (height, width) 影片解析度，用來核對 counting_lines 的 image_size
        lines_dir: counting_lines 設定檔目錄
    """
    if not cfg.counting.enabled:
        return None

    seq_output_dir = Path(seq_output_dir)
    seq_name = seq_output_dir.stem
    out_dir = Path(cfg.collision_output_dir) / seq_name / "counting"
    summary: Dict[str, Any] = {"sequence": seq_name, "camera": camera_id_from_seq_name(seq_name), "fps": fps}

    lines = resolve_counting_lines(seq_name, (bounds[1], bounds[0]), lines_dir)
    if lines is None:
        summary["skipped_reason"] = "no_counting_lines"
        write_json(summary, out_dir / "summary.json")
        print(f"  [{seq_name}] 計數線統計略過（no_counting_lines），只輸出 {out_dir / 'summary.json'}")
        return summary

    tracks = load_counting_tracks(seq_output_dir, cfg)
    homography, _ = homography_points.resolve_camera_homography(
        seq_name, cfg.homography_camera_dir, cfg.homography_matrices_dir)
    resolved = {line.name: resolve_labels(line, homography) for line in lines}
    label = {(name, "forward"): labels.forward for name, labels in resolved.items()}
    label.update({(name, "backward"): labels.backward for name, labels in resolved.items()})
    running: Counter = Counter()
    columns: Dict[str, list] = {name: [] for name in (
        "frame", "time_s", "line", "direction", "label", "class", "track_id", "cumulative")}
    totals = {line.name: {d: {c: 0 for c in (*CLASSES, "all")} for d in DIRECTIONS} for line in lines}
    for crossing in find_crossings(tracks, lines):
        running[(crossing.line, crossing.direction, crossing.class_name)] += 1
        totals[crossing.line][crossing.direction][crossing.class_name] += 1
        totals[crossing.line][crossing.direction]["all"] += 1
        for name, value in zip(columns, (
                crossing.frame, crossing.frame / fps, crossing.line, crossing.direction,
                label[(crossing.line, crossing.direction)], crossing.class_name, crossing.track_id,
                running[(crossing.line, crossing.direction, crossing.class_name)])):
            columns[name].append(value)
    write_csv(columns, out_dir / "counts.csv")

    summary.update({
        "skipped_reason": None, "num_tracks": len(tracks), "lines": [_line_record(l, resolved[l.name]) for l in lines],
        "totals": totals, "limitations": LIMITATIONS,
    })
    write_json(summary, out_dir / "summary.json")
    print(f"  [{seq_name}] 計數線統計：tracks={len(tracks)}  穿越={len(columns['frame'])}  輸出目錄={out_dir}")
    return summary
