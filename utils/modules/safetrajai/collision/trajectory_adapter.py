"""
Layer 1（本專案實際輸出）→ Layer 2（碰撞偵測核心運算要的格式）轉換層。

Layer 1：evaluation/mot_format.py::save_mot_format() 產生的逐類別 MOT CSV
（格式見 CLAUDE.md「軌跡格式契約」）：
    annotation_<seq>/trackers/mot_challenge/<benchmark>-<split>/<tracker_name>__<class>/data/<seq>.txt
    每行：frame,track_id,left,top,w,h,score（像素座標，無 header）

Layer 2：safetrajai/collision/detection_core.py 的 compute_ttc_for_pair()/
compute_pet_for_pair() 需要的單一物件軌跡 dict（定義與衍生規則如下）：
    {
        "Track ID": int,
        "class_id": int, "class_name": str,        # 本專案擴充欄位
        "Positions": {frame: (x, y, z)},            # bbox 中心點，z 固定 0.0
        "Bboxs":     {frame: ((x1,y1),(x2,y2))},    # bbox 左上/右下
        "Velocity":  {frame: (vx, vy)},              # 中心差分法，見 _center_diff_velocity()
        "Direction": {frame: (vx, vy)},              # 與 Velocity 同值：get_collision_type() 只取
                                                       # 夾角，兩者归一化後方向相同，不需另外維護
        "Size":      {frame: (w, h)},
    }
Homography 版（"Homo_" 前綴）由 add_homography_fields() 另外附加。

⚠️ track_id 跨類別合併的前提：本專案的 class-aware tracker 對每支影片只建立一個
tracker 實例，track_id 計數器是該實例共用的單一計數器（class-aware 只影響配對邏輯，
不影響 ID 配發），因此同一支影片內、不同類別的軌跡檔之間 track_id 不會重複，可以直接
合併成同一份清單比對。跨影片仍會重複，因此本模組一次只處理一個序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.roi import is_center_in_roi


def _read_class_trajectory_file(path: Path, roi: np.ndarray) -> Dict[int, List[tuple]]:
    """讀一份逐類別 MOT CSV，回傳 {track_id: [(frame, left, top, w, h, score), ...]}。

    框中心點在 ROI 外的列在分組前就捨棄，軌跡的排列順序因此等於「軌跡在 ROI 內第一次出現」
    的順序；下游 PET 計算與軌跡順序有關，必須保持這個順序。
    """
    by_track: Dict[int, List[tuple]] = {}
    if not path.exists():
        return by_track

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left, top, w, h = (float(x) for x in parts[2:6])
            if not is_center_in_roi(roi, left + w / 2, top + h / 2):
                continue
            score = float(parts[6])
            by_track.setdefault(track_id, []).append((frame, left, top, w, h, score))

    return by_track


def _center_diff_velocity(
    sorted_frames: List[int],
    center_by_frame: Dict[int, Tuple[float, float]],
    fps: float,
) -> Dict[int, Tuple[float, float]]:
    """對單一物件的中心點序列，用中心差分法算每個 frame 的速度向量 (vx, vy)。

    - 中間幀：若前一幀、後一幀在 frame 編號上都恰好相鄰，用中心差分
      v = (pos[f+1] - pos[f-1]) * fps / 2。
    - 軌跡全域首/尾幀：只有單側相鄰幀存在時，用單邊差分（前向/後向），dt = 1/fps。
    - 其餘情況（相鄰幀因追蹤中斷缺幀、或孤立的單幀偵測）：該幀不計算速度，
      不會出現在回傳的 dict 裡（呼叫端據此判斷該幀是否可用於 TTC 比對）。
    """
    velocity: Dict[int, Tuple[float, float]] = {}
    n = len(sorted_frames)
    frame_set = set(sorted_frames)

    for i, f in enumerate(sorted_frames):
        has_prev_adjacent = (f - 1) in frame_set
        has_next_adjacent = (f + 1) in frame_set
        cx, cy = center_by_frame[f]

        if has_prev_adjacent and has_next_adjacent:
            cx_prev, cy_prev = center_by_frame[f - 1]
            cx_next, cy_next = center_by_frame[f + 1]
            vx = (cx_next - cx_prev) * fps / 2.0
            vy = (cy_next - cy_prev) * fps / 2.0
        elif i == 0 and has_next_adjacent:
            cx_next, cy_next = center_by_frame[f + 1]
            vx = (cx_next - cx) * fps
            vy = (cy_next - cy) * fps
        elif i == n - 1 and has_prev_adjacent:
            cx_prev, cy_prev = center_by_frame[f - 1]
            vx = (cx - cx_prev) * fps
            vy = (cy - cy_prev) * fps
        else:
            continue  # 缺幀相鄰、或孤立單幀：不計算速度

        velocity[f] = (vx, vy)

    return velocity


def build_track(track_id: int, class_id: int, rows: List[tuple], fps: float) -> Dict[str, Any]:
    """把單一 track_id 的逐幀 (frame,left,top,w,h,score) 列表轉成 Layer 2 track dict。"""
    rows_sorted = sorted(rows, key=lambda r: r[0])
    sorted_frames = [r[0] for r in rows_sorted]

    positions: Dict[int, tuple] = {}
    bboxs: Dict[int, tuple] = {}
    sizes: Dict[int, tuple] = {}
    center_by_frame: Dict[int, Tuple[float, float]] = {}

    for frame, left, top, w, h, _score in rows_sorted:
        cx = left + w / 2.0
        cy = top + h / 2.0
        positions[frame] = (cx, cy, 0.0)
        bboxs[frame] = ((left, top), (left + w, top + h))
        sizes[frame] = (w, h)
        center_by_frame[frame] = (cx, cy)

    velocity = _center_diff_velocity(sorted_frames, center_by_frame, fps)

    return {
        "Track ID": track_id,
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, f"cls_{class_id}"),
        "Positions": positions,
        "Bboxs": bboxs,
        "Velocity": velocity,
        "Direction": velocity,
        "Size": sizes,
    }


def load_sequence_tracks(seq_output_dir: Path, cfg, fps: float, roi: np.ndarray) -> List[Dict[str, Any]]:
    """讀取一個序列 stage1 輸出的六個逐類別軌跡檔，合併成跨類別的 track dict 清單。

    stage1 軌跡不做 ROI 過濾（docs/adr/0005），這裡以框中心點硬過濾：ROI 外的列
    直接捨棄，過濾後沒有任何列的軌跡也一併捨棄。

    Args:
        seq_output_dir: annotation_<seq> 目錄（stage1 輸出，det_output_dir 下）
        roi: 序列所屬鏡頭的 ROI 多邊形（safetrajai.roi.resolve_roi()）
        cfg: AppConfig，需要 benchmark_name / split_name / tracker_name。
             ⚠️ evaluation/mot_format.py::save_mot_format() 目前把 "CCTV-val" /
             "YOLO11x" 寫死在路徑字串裡，沒有讀 cfg——本專案所有情境檔的
             benchmark_name/split_name/tracker_name 剛好都是這三個預設值，
             現況下路徑會一致；若之後任一邊改了設定值但沒同步修，這裡會讀到空結果
             （逐檔案，不會報錯）。
        fps: 影片幀率，用於速度換算

    Returns:
        list[dict]，每個元素是一個 track 的 Layer 2 格式（見模組 docstring）。
    """
    seq_name = seq_output_dir.stem
    traj_root = seq_output_dir / "trackers" / "mot_challenge" / f"{cfg.benchmark_name}-{cfg.split_name}"

    tracks: List[Dict[str, Any]] = []
    for class_id, class_name in sorted(CLASS_NAMES.items()):
        traj_file = traj_root / f"{cfg.tracker_name}__{class_name}" / "data" / f"{seq_name}.txt"
        by_track = _read_class_trajectory_file(traj_file, roi)
        for track_id, rows in by_track.items():
            tracks.append(build_track(track_id, class_id, rows, fps))

    return tracks


def add_homography_fields(
    tracks: List[Dict[str, Any]], H, fps: float,
) -> List[Dict[str, Any]]:
    """對每個 track 附加 Homo_Positions / Homo_Bboxs / Homo_Velocity / Homo_Direction / Homo_Size。

    Homo_Velocity/Homo_Direction 是對 Homo_Positions（轉換後座標）重新跑一次中心差分法算出來
    的，不是把原始座標系的 Velocity 直接套用矩陣轉換——因為 Homography 是非線性的透視變換，
    速度向量不能像座標點一樣直接乘矩陣，必須在目標座標系下重新計算差分。Homo_Size 取
    Homo_Bboxs（簡化版，僅轉換左上右下兩點）的寬高，而非精確的四邊形版本，
    對齊 detection_core.py::rects_iou() 假設的「軸對齊矩形」用法。

    Args:
        tracks: load_sequence_tracks() 回傳的 track dict 清單，會被原地修改並回傳
        H: 3x3 Homography 矩陣（safetrajai/collision/homography/__init__.py::compute_homography()）
        fps: 影片幀率，用於 Homo_Velocity 換算

    Returns:
        同一份 tracks list（原地附加欄位），方便鏈式呼叫。
    """
    from . import homography as homography_mod

    for track in tracks:
        homo_positions, homo_bboxs, _homo_bboxs_polygon = homography_mod.transform_positions_and_bboxes(
            track["Positions"], track["Bboxs"], H,
        )

        homo_center_by_frame = {f: (p[0], p[1]) for f, p in homo_positions.items()}
        homo_sorted_frames = sorted(homo_center_by_frame.keys())
        homo_velocity = _center_diff_velocity(homo_sorted_frames, homo_center_by_frame, fps)

        homo_sizes = {}
        for frame, (pt1, pt2) in homo_bboxs.items():
            homo_sizes[frame] = (abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1]))

        track["Homo_Positions"] = homo_positions
        track["Homo_Bboxs"] = homo_bboxs
        track["Homo_Velocity"] = homo_velocity
        track["Homo_Direction"] = homo_velocity
        track["Homo_Size"] = homo_sizes

    return tracks
