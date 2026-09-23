"""
碰撞偵測核心演算法（純運算，無 cv2 繪圖/影片 I/O 依賴）。

移植自 markdown/reference_code/collision_detection_core.py（SafeTrajAI 專案原始碼
萃取版），邏輯未修改。輸入的 track dict 格式見 safetrajai/collision/trajectory_adapter.py
（Layer 2：後處理單軌跡格式）。

包含兩種交通衝突代理安全指標（Surrogate Safety Measures）:

1. TTC（Time-To-Collision，碰撞時間）
   對兩個物體目前的位置與速度做「等速度線性外推」，模擬未來一段時間內
   是否會發生邊界框重疊（IoU 超過門檻），若會，回傳预估碰撞所需時間、
   碰撞位置與碰撞類型（追撞／對撞／側撞）。

2. PET（Post-Encroachment Time，先後通過時間）
   在允許的最大時間差（時間窗）內，尋找兩條軌跡「空間距離最近」的一對
   時空點；若該最近距離小於重疊門檻，代表兩物體先後通過了幾乎同一個
   空間位置，兩者的時間差即為 PET。

相依套件：numpy（必要）、numba（選用，加速 PET 最近距離搜尋，未安裝時自動退回
純 Python 迴圈，結果相同、速度較慢）。
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba 為選用加速套件
    _HAS_NUMBA = False

    def njit(func=None, **kwargs):
        """numba 不存在時的空裝飾器，讓程式仍可正常運作（僅較慢）。"""
        if func is None:
            return lambda f: f
        return func


# =============================================================================
# 1. TTC（Time-To-Collision）
# =============================================================================

def rects_iou(
    pos1: Sequence[float], size1: Sequence[float],
    pos2: Sequence[float], size2: Sequence[float],
) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
    """計算兩個「以中心點表示」的矩形之間的 IoU（交集面積 / 聯集面積）。

    Args:
        pos1, pos2: 矩形中心座標 (x, y)
        size1, size2: 矩形尺寸 (width, height)

    Returns:
        (iou, box1, box2)：box1/box2 為對應矩形的 ((x1,y1),(x2,y2)) 表示法。
    """
    x1_min = pos1[0] - size1[0] / 2
    x1_max = pos1[0] + size1[0] / 2
    y1_min = pos1[1] - size1[1] / 2
    y1_max = pos1[1] + size1[1] / 2

    x2_min = pos2[0] - size2[0] / 2
    x2_max = pos2[0] + size2[0] / 2
    y2_min = pos2[1] - size2[1] / 2
    y2_max = pos2[1] + size2[1] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = size1[0] * size1[1]
    area2 = size2[0] * size2[1]
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    box1 = ((int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)))
    box2 = ((int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)))
    return iou, box1, box2


def get_collision_type(dir1: Sequence[float], dir2: Sequence[float]) -> str:
    """依兩物體的行進方向向量夾角，判斷碰撞類型。

    Returns:
        "Right-angle collision"（側撞，約略垂直，70~110 度）
        "Rear-end collision"（追撞，方向幾乎相同）
        "Head-on collision"（對撞，方向幾乎相反）
        "Other"（其他角度）
    """
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v

    d1 = normalize(np.asarray(dir1, dtype=np.float64))
    d2 = normalize(np.asarray(dir2, dtype=np.float64))

    cos_theta = float(np.dot(d1, d2))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))

    if 70 <= angle_deg <= 110:
        return "Right-angle collision"
    elif cos_theta > 0.9:
        return "Rear-end collision"
    elif cos_theta < -0.9:
        return "Head-on collision"
    else:
        return "Other"


def find_ttc(
    pos1: np.ndarray, vel1: np.ndarray, size1: np.ndarray,
    pos2: np.ndarray, vel2: np.ndarray, size2: np.ndarray,
    fps: float, max_ttc_frames: int,
    bounds: Tuple[int, int],
    max_time: float = 10.0, dt: float = 0.1, iou_threshold: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """對兩物體目前的位置/速度做等速度線性外推模擬，尋找是否會發生碰撞。

    以 dt 為步進間隔，逐步將兩物體沿各自速度方向推進，只要外推後的邊界框
    IoU 超過 iou_threshold，即視為「預估會碰撞」，回傳第一個滿足條件的時間點。

    Args:
        pos1, pos2: 目前中心座標 (x, y)（numpy array）
        vel1, vel2: 目前速度 (vx, vy)（單位：座標單位/秒）
        size1, size2: 邊界框尺寸 (w, h)
        fps: 影片幀率，用於將秒數換算為 frame 數
        max_ttc_frames: 最大模擬 frame 數上限（模擬時間上限 = max_ttc_frames / fps）
        bounds: (height, width)，碰撞點需落在此範圍內才視為有效
        max_time: 模擬時間上限（秒），與 max_ttc_frames 換算後取較小值
        dt: 模擬時間步進（秒）
        iou_threshold: 判定為碰撞的 IoU 門檻

    Returns:
        None，或包含以下欄位的 dict：
            ttc_frames: 預估碰撞所需 frame 數
            ttc_time: 預估碰撞所需秒數
            collision_point: 碰撞位置 (x, y)（兩物體外推位置中點）
            iou: 碰撞當下的 IoU 值
            arrows: ((起點,終點), (起點,終點)) 兩物體從目前位置到碰撞位置的位移
            future_box: (box1, box2) 碰撞當下兩物體的外推邊界框
    """
    height, width = bounds
    max_sim_time = min(max_time, max_ttc_frames / fps)

    for t in np.arange(0, max_sim_time, dt):
        future_pos1 = pos1 + vel1 * t
        future_pos2 = pos2 + vel2 * t

        iou, future_box1, future_box2 = rects_iou(future_pos1, size1, future_pos2, size2)
        if iou >= iou_threshold:
            ttc_frames = int(round(t * fps))
            collision_point = (future_pos1 + future_pos2) / 2
            x, y = collision_point
            if (0 <= int(x) < height) and (0 <= int(y) < width) and (ttc_frames != 0):
                return {
                    "ttc_frames": ttc_frames,
                    "ttc_time": float(t),
                    "collision_point": collision_point,
                    "iou": iou,
                    "arrows": (
                        (tuple(map(int, pos1.tolist())), tuple(map(int, future_pos1.tolist()))),
                        (tuple(map(int, pos2.tolist())), tuple(map(int, future_pos2.tolist()))),
                    ),
                    "future_box": (future_box1, future_box2),
                }
    return None


def compute_ttc_for_pair(
    track1: Dict[str, Any], track2: Dict[str, Any],
    fps: float, max_ttc_frames: int, bounds: Tuple[int, int],
    use_homography: bool = True, iou_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """對一組物體軌跡配對，逐一比對其共同出現的 frame，計算 TTC 事件。

    找到第一個發生碰撞的 frame 後即停止比對該配對（同一配對只回報一次）。

    Args:
        track1, track2: 軌跡 dict（見 trajectory_adapter.py）
        fps: 影片幀率
        max_ttc_frames: 最大模擬 frame 數
        bounds: (height, width) 有效碰撞區域範圍
        use_homography: True 時使用 "Homo_*" 欄位（地圖座標系），
                         False 時使用原始影像座標系欄位
        iou_threshold: 判定為碰撞的 IoU 門檻（原程式預設 0.8）

    Returns:
        list[dict]，每筆為一次 TTC 碰撞事件：
            {
                "ID1", "ID2": 兩物體的 Track ID,
                "collision_frame": 預估碰撞發生的 frame,
                "collision_occur_sec": 預估碰撞發生的秒數（相對影片起點）,
                "collision_point": (x, y),
                "collision_type": "Right-angle collision" 等,
                "ttc_seconds": 從目前 frame 到碰撞所需秒數,
                "iou": 碰撞當下 IoU,
                "future_box1", "future_box2": 兩物體碰撞當下外推邊界框,
                "arrow1", "arrow2": 兩物體位移向量（起點, 終點）,
            }
        通常最多 1 筆（找到即停止），無碰撞則為空 list。
    """
    prefix = "Homo_" if use_homography else ""
    keys = [f"{prefix}{k}" for k in ("Positions", "Bboxs", "Velocity", "Size", "Direction")]

    shared_frames = sorted(set(track1.get(f"{prefix}Bboxs", {}).keys())
                            & set(track2.get(f"{prefix}Bboxs", {}).keys()))
    if not shared_frames:
        return []

    events: List[Dict[str, Any]] = []

    # 從第二個共同 frame 開始比對（需要至少一個先前點才有意義）
    for i in range(1, len(shared_frames)):
        curr_frame = shared_frames[i]

        if not all(k in track1 and curr_frame in track1[k] and
                   k in track2 and curr_frame in track2[k] for k in keys):
            continue

        pos1 = np.array(track1[f"{prefix}Positions"][curr_frame][:2], dtype=np.float64)
        pos2 = np.array(track2[f"{prefix}Positions"][curr_frame][:2], dtype=np.float64)
        vel1 = np.array(track1[f"{prefix}Velocity"][curr_frame], dtype=np.float64)
        vel2 = np.array(track2[f"{prefix}Velocity"][curr_frame], dtype=np.float64)
        dir1 = np.array(track1[f"{prefix}Direction"][curr_frame], dtype=np.float64)
        dir2 = np.array(track2[f"{prefix}Direction"][curr_frame], dtype=np.float64)
        size1 = np.array(track1[f"{prefix}Size"][curr_frame], dtype=np.float64)
        size2 = np.array(track2[f"{prefix}Size"][curr_frame], dtype=np.float64)

        result = find_ttc(pos1, vel1, size1, pos2, vel2, size2,
                           fps=fps, max_ttc_frames=max_ttc_frames, bounds=bounds,
                           iou_threshold=iou_threshold)

        if result is not None:
            events.append({
                "ID1": track1["Track ID"],
                "ID2": track2["Track ID"],
                "collision_frame": curr_frame + result["ttc_frames"],
                "collision_occur_sec": curr_frame / fps + result["ttc_time"],
                "collision_point": tuple(result["collision_point"][:2]),
                "collision_type": get_collision_type(dir1, dir2),
                "ttc_seconds": result["ttc_time"],
                "iou": result["iou"],
                "future_box1": result["future_box"][0],
                "future_box2": result["future_box"][1],
                "arrow1": result["arrows"][0],
                "arrow2": result["arrows"][1],
            })
            break  # 同一配對找到一次碰撞事件即停止

    return events


def compute_ttc_for_all_pairs(
    tracks: List[Dict[str, Any]],
    fps: float, max_ttc_frames: int, bounds: Tuple[int, int],
    use_homography: bool = True, iou_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """對整批軌跡做兩兩配對（itertools.combinations），彙整所有 TTC 碰撞事件。"""
    all_events: List[Dict[str, Any]] = []
    for track1, track2 in combinations(tracks, 2):
        all_events.extend(
            compute_ttc_for_pair(track1, track2, fps, max_ttc_frames, bounds,
                                  use_homography=use_homography, iou_threshold=iou_threshold)
        )
    return all_events


# =============================================================================
# 2. PET（Post-Encroachment Time）
# =============================================================================

@njit
def compute_min_distance_pair(
    path1_arr: np.ndarray, path2_arr: np.ndarray,
    time1_arr: np.ndarray, time2_arr: np.ndarray,
    max_time_diff: int, overlap_threshold: float,
):
    """在時間差限制內，窮舉搜尋兩條軌跡座標點中「空間距離最小」的一對。

    若未安裝 numba，本函式會自動以純 Python 執行（見檔案頂部的 njit 空裝飾器
    fallback），結果相同、速度較慢。

    Args:
        path1_arr, path2_arr: shape=(N, 2) 的座標陣列 (x, y)
        time1_arr, time2_arr: 對應座標的 frame index 陣列
        max_time_diff: 允許比對的最大時間差（frame 數），超過則略過該組合
        overlap_threshold: 視為「同一空間位置」的距離門檻（座標單位，如像素）

    Returns:
        (t1, t2, idx1, idx2, dist)：
            t1, t2: 兩軌跡各自最近點的 frame index
            idx1, idx2: 對應在輸入陣列中的索引
            dist: 該對點之間的歐氏距離
        若找不到符合門檻的配對，回傳 (-1, -1, -1, -1, -1.0)
    """
    min_dist = 1e10
    best_pair = (-1, -1, -1, -1, -1.0)
    for i in range(path1_arr.shape[0]):
        for j in range(path2_arr.shape[0]):
            dt = abs(time1_arr[i] - time2_arr[j])
            if dt > max_time_diff:
                continue
            dx = path1_arr[i, 0] - path2_arr[j, 0]
            dy = path1_arr[i, 1] - path2_arr[j, 1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= overlap_threshold and dist < min_dist:
                min_dist = dist
                best_pair = (time1_arr[i], time2_arr[j], i, j, dist)
    return best_pair


def compute_pet_for_pair(
    track1: Dict[str, Any], track2: Dict[str, Any],
    fps: float, max_time_diff_frames: int, overlap_threshold: float,
    use_homography: bool = True,
) -> Optional[Dict[str, Any]]:
    """計算一組軌跡配對的 PET（先後通過時間）事件。

    對兩條軌跡各自完整的 (位置, 時間) 陣列做一次全域最近點搜尋
    （compute_min_distance_pair），適合離線批次處理（已有完整軌跡）。

    Args:
        track1, track2: 軌跡 dict（見 trajectory_adapter.py）
        fps: 影片幀率，用於將 frame 差換算為秒數
        max_time_diff_frames: 允許比對的最大時間差（frame 數）
        overlap_threshold: 視為「同一空間位置」的距離門檻（座標單位）
        use_homography: True 時使用 "Homo_Positions"（地圖座標系）

    Returns:
        None，或包含以下欄位的 dict：
            {
                "ID1", "ID2": 兩物體的 Track ID,
                "cross_point": (x, y)  兩物體最近點的中點座標,
                "pet_seconds": PET 值（秒）,
                "frame1", "frame2": 兩物體各自通過該最近點的 frame,
                "distance": 最近距離,
            }
    """
    prefix = "Homo_" if use_homography else ""
    pos_key = f"{prefix}Positions"

    pos1_dict = track1.get(pos_key, {})
    pos2_dict = track2.get(pos_key, {})
    if not pos1_dict or not pos2_dict:
        return None

    frames1 = sorted(pos1_dict.keys())
    frames2 = sorted(pos2_dict.keys())

    pos1_arr = np.array([pos1_dict[f][:2] for f in frames1], dtype=np.float32)
    pos2_arr = np.array([pos2_dict[f][:2] for f in frames2], dtype=np.float32)
    time1_arr = np.array(frames1, dtype=np.int32)
    time2_arr = np.array(frames2, dtype=np.int32)

    t1, t2, idx1, idx2, dist = compute_min_distance_pair(
        pos1_arr, pos2_arr, time1_arr, time2_arr, max_time_diff_frames, overlap_threshold
    )

    if idx1 == -1 or idx2 == -1:
        return None

    p1 = pos1_arr[idx1]
    p2 = pos2_arr[idx2]
    cross_point = (float((p1[0] + p2[0]) / 2), float((p1[1] + p2[1]) / 2))
    pet_seconds = abs(t2 - t1) / fps

    return {
        "ID1": track1["Track ID"],
        "ID2": track2["Track ID"],
        "cross_point": cross_point,
        "pet_seconds": float(pet_seconds),
        "frame1": int(t1),
        "frame2": int(t2),
        "distance": round(float(dist), 2),
    }


def compute_pet_for_all_pairs(
    tracks: List[Dict[str, Any]],
    fps: float, max_time_diff_frames: int, overlap_threshold: float,
    use_homography: bool = True,
) -> List[Dict[str, Any]]:
    """對整批軌跡做兩兩配對，彙整所有 PET 事件（distance 未超過門檻的才會有結果）。"""
    events: List[Dict[str, Any]] = []
    for track1, track2 in combinations(tracks, 2):
        event = compute_pet_for_pair(track1, track2, fps, max_time_diff_frames,
                                      overlap_threshold, use_homography=use_homography)
        if event is not None:
            events.append(event)
    return events
