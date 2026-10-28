"""
TTC/PET 碰撞事件輸出：乾淨 CSV/JSON、比照原始欄位風格的 log CSV、熱區圖 PNG。

三種輸出：
1. 乾淨 CSV/JSON——欄位直接對應 detection_core.py::compute_ttc_for_pair() /
   compute_pet_for_pair() 回傳的 dict key，跨專案交換建議優先用這份。
2. log CSV——欄位風格比照原始 SafeTrajAI 的 collision_ttc_log.csv（中文表頭），
   但修正了原始格式裡 PET `colision_occur_sec` 未除以 fps、被誤當秒數使用的問題，
   這裡一律是正確的秒數；
   另外本專案每次執行覆寫檔案（不用原始的 append 累加模式），避免跨執行結果混雜。
3. 熱區圖 PNG——改寫自 SafeTrajAI 原始碼的 draw_utils.py::draw_collision_heatmap()，
   拿掉即時逐幀繪圖/影片疊圖的部分，只保留「一批碰撞點 + 一張背景圖 → 熱區圖」的核心邏輯，
   背景圖用 stage1 已存的 <seq>/first_frame_with_box.jpg。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# =============================================================================
# 1. 乾淨 CSV / JSON
# =============================================================================

def write_events_json(events: List[Dict[str, Any]], output_path: Path) -> None:
    """把事件 list 原樣寫成 JSON（tuple 會被序列化成 JSON array）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def write_events_csv(events: List[Dict[str, Any]], output_path: Path) -> None:
    """把事件 list 寫成 CSV，欄位就是每個 dict 的 key；tuple/list 欄位序列化成 JSON 字串存放。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not events:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(events[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = {
                k: (json.dumps(v) if isinstance(v, (tuple, list)) else v)
                for k, v in event.items()
            }
            writer.writerow(row)


# =============================================================================
# 2. 比照原始欄位風格的 log CSV
# =============================================================================

def write_ttc_legacy_log(
    events: List[Dict[str, Any]], video_name: str, fps: float, output_path: Path,
) -> None:
    """輸出比照 collision_ttc_log.csv 欄位風格的 TTC log（中文表頭，每次執行覆寫）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "FrameID", "Video", "車輛1", "車輛2", "預估碰撞時間點",
            "TTC碰撞時長", "碰撞點", "碰撞類型", "TTC標籤", "指標",
        ])
        for e in events:
            ttc_seconds = e["ttc_seconds"]
            # collision_frame = curr_frame + ttc_frames，反推 curr_frame（觸發偵測當下的幀）
            frame_id = int(round(e["collision_frame"] - ttc_seconds * fps))
            # detection_core.find_ttc() 的 collision_point 是 numpy array 相加後直接 tuple() 出來的，
            # 元素是 np.float64（非 float）；numpy>=2 的 repr 會印成 "np.float64(200.0)"，
            # 用 csv.writer 直接寫 tuple 會呼叫 repr() 顯示這種寫法，這裡轉成 float 避免 log 難讀
            collision_point = tuple(float(v) for v in e["collision_point"])
            writer.writerow([
                frame_id,
                video_name,
                e["ID1"],
                e["ID2"],
                round(e["collision_occur_sec"], 2),
                round(ttc_seconds, 2),
                collision_point,
                e["collision_type"],
                f"TTC@{ttc_seconds:.2f}",
                "TTC",
            ])


def write_pet_legacy_log(
    events: List[Dict[str, Any]], video_name: str, fps: float, output_path: Path,
) -> None:
    """輸出比照原始 PET 欄位風格的 log CSV（原始程式沒有對應的結構化 CSV，這裡仿 TTC
    格式補上，"預估碰撞時間點" 修正為 frame2/fps 的正確秒數，每次執行覆寫）。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "FrameID", "Video", "車輛1", "車輛2", "預估碰撞時間點",
            "PET碰撞時長", "碰撞點", "PET標籤", "指標",
        ])
        for e in events:
            frame_id = min(e["frame1"], e["frame2"])
            pet_seconds = e["pet_seconds"]
            collision_time_sec = e["frame2"] / fps
            writer.writerow([
                frame_id,
                video_name,
                e["ID1"],
                e["ID2"],
                round(collision_time_sec, 2),
                round(pet_seconds, 2),
                e["cross_point"],
                f"PET@{pet_seconds:.2f}",
                "PET",
            ])


# =============================================================================
# 3. 熱區圖 PNG
# =============================================================================

def draw_collision_heatmap(
    points: Sequence[Tuple[float, float]],
    background_bgr: np.ndarray,
    output_path: Path,
    radius: int = 15,
    alpha: float = 0.5,
    dpi: int = 100,
    title: str = "Collision Heatmap",
) -> bool:
    """把一批碰撞點畫成熱區圖，與背景圖疊合後存檔。

    改寫自 draw_utils.py::draw_collision_heatmap()：拿掉 ttc_collision_point 那種
    「frame -> event list」的巢狀結構與逐幀疊圖，只留「(x,y) 點列表 + 背景圖」的核心邏輯，
    直接存檔（原函式回傳 Figure，呼叫端要自己 savefig）。

    Returns:
        True 表示有畫出圖檔；False 表示 points 為空，沒有產生檔案（沒有碰撞事件時不需要熱區圖）。
    """
    if not points:
        # 沒有碰撞事件就不產生檔案；但若前一次執行（例如不同參數）留下了舊檔，
        # 要主動清掉，否則舊的熱區圖會被誤以為是本次執行的結果
        output_path.unlink(missing_ok=True)
        return False

    height, width = background_bgr.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for x, y in points:
        if 0 <= int(y) < height and 0 <= int(x) < width:
            cv2.circle(heatmap, (int(x), int(y)), radius, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=radius)
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(background_rgb, 1 - alpha, heatmap_color, alpha, 0)

    figsize = (width / dpi, height / dpi)
    fig = Figure(figsize=figsize, dpi=dpi)
    fig.set_facecolor("black")
    canvas = FigureCanvas(fig)  # noqa: F841 - 建立 Agg canvas 供 savefig 使用

    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(overlay)
    ax.axis("off")

    cax = fig.add_axes([0.95, 0.05, 0.03, 0.9])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Collision Count (color intensity)", fontsize=10, color="white", labelpad=10)
    cbar.ax.tick_params(labelsize=10, colors="white")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    ax.set_title(title, color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, facecolor=fig.get_facecolor())
    return True
