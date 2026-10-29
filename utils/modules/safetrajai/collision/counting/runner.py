"""Stage3 counting-line statistics of one sequence (spec F2).

Reads the stage1 MOT tracks (bottom-centre of every box, no ROI filtering), counts crossings of
the camera's counting lines and writes `collision_output/<seq>/counting/{counts.csv,summary.json}`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.collision.conflict.output import write_csv, write_json
from safetrajai.collision.counting.crossing import CountingTrack, find_crossings
from safetrajai.collision.counting.lines import COUNTING_LINES_DIR, CountingLine, resolve_counting_lines
from safetrajai.collision.trajectory_adapter import _read_class_trajectory_file
from safetrajai.evaluation.dataset_layout import camera_id_from_seq_name

CLASSES: Tuple[str, ...] = tuple(CLASS_NAMES[i] for i in sorted(CLASS_NAMES))
DIRECTIONS = ("forward", "backward")
# Stage1 tracks are not ROI-filtered; the reader wants a polygon, so give it one that holds every box centre.
_EVERYWHERE = np.array([[-1e9, -1e9], [1e9, -1e9], [1e9, 1e9], [-1e9, 1e9]], dtype=np.float32)

LIMITATIONS: List[Dict[str, str]] = [
    {"code": "id_switches",
     "detail": "ID 切換（身分中斷後換號）會讓同一物體被計多次，遺漏追蹤則少計；數字只當技術訊號，不是正式流量。"},
    {"code": "not_roi_filtered",
     "detail": "stage1 軌跡不依 ROI 過濾（ADR 0005），計數線在 ROI 外的穿越也會被計入。"},
]


def _load_tracks(seq_output_dir: Path, cfg) -> Tuple[CountingTrack, ...]:
    traj_root = seq_output_dir / "trackers" / "mot_challenge" / f"{cfg.benchmark_name}-{cfg.split_name}"
    tracks = []
    for class_name in CLASSES:
        path = traj_root / f"{cfg.tracker_name}__{class_name}" / "data" / f"{seq_output_dir.name}.txt"
        for track_id, rows in sorted(_read_class_trajectory_file(path, _EVERYWHERE).items()):
            rows = sorted(rows, key=lambda r: r[0])
            frames = np.array([r[0] for r in rows], dtype=np.int64)
            points = np.array([(left + w / 2.0, top + h) for _, left, top, w, h, _ in rows], dtype=np.float64)
            tracks.append(CountingTrack(class_name, int(track_id), frames, points))
    return tuple(tracks)


def _line_record(line: CountingLine) -> Dict[str, Any]:
    return {"name": line.name, "points": [list(line.p1), list(line.p2)],
            "forward_label": line.forward_label, "backward_label": line.backward_label}


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
        cfg: AppConfig，需要 counting、collision_output_dir、benchmark_name、split_name、tracker_name
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

    tracks = _load_tracks(seq_output_dir, cfg)
    label = {(line.name, "forward"): line.forward_label for line in lines}
    label.update({(line.name, "backward"): line.backward_label for line in lines})
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
        "skipped_reason": None, "num_tracks": len(tracks), "lines": [_line_record(l) for l in lines],
        "totals": totals, "limitations": LIMITATIONS,
    })
    write_json(summary, out_dir / "summary.json")
    print(f"  [{seq_name}] 計數線統計：tracks={len(tracks)}  穿越={len(columns['frame'])}  輸出目錄={out_dir}")
    return summary
