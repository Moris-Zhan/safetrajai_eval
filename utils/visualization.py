import re
import cv2
import numpy as np
from pathlib import Path
from argostranslate import translate


def normalize_string(s: str) -> str:
    s2 = re.sub(r'[^A-Za-z0-9/]', '_', s)
    s2 = re.sub(r'_+', '_', s2)
    print(s)
    return s2


def translate_path(video_path: str, from_code: str = "zh", to_code: str = "en") -> str:
    return normalize_string(translate.translate(video_path, from_code, to_code))


def draw_polygon_and_save(frame, output_path: Path, pts=None, show_coord: bool = True, save_img: bool = True):
    if pts is not None:
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        if show_coord:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, (x, y) in enumerate(pts[:, 0, :]):
                txt = f"P{i+1}({x},{y})"
                cv2.putText(frame, txt, (x - 100, y - 5), font, 0.5, (255, 255, 255), 4)
                cv2.putText(frame, txt, (x - 100, y - 5), font, 0.5, (0, 0, 0), 2)

    if save_img:
        save_path = output_path / "first_frame_with_box.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"Saved first frame with box to: {save_path}")


def draw_track_box(
    annotator,
    box,
    det_corner,
    color,
    *,
    embed_label: str | None = None,
    overlay_label: str | None = None,
    overlay_anchor: tuple[int, int] | None = None,
) -> None:
    """
    統一的「多邊形過濾 → 畫框 → 標註」邏輯，取代 VideoSequenceTracker.run() 內
    matched/unmatched/removed 三段重複程式碼。

    - det_corner 為 None 時不做多邊形過濾（全部畫）；不為 None 時，只在框中心點落在
      多邊形內才畫（沿用現行 cv2.pointPolygonTest 判斷）。
    - embed_label：文字直接嵌入 box_label()（matched track 目前的畫法：框內嵌 track_id）。
    - overlay_label + overlay_anchor：額外用 annotator.text() 疊加文字（unmatched/removed
      目前的畫法：紅框/黑框 + 框外文字標籤）。兩種模式互斥，由呼叫方依三種情境各自決定。

    Parameters
    ----------
    box : tuple[float, float, float, float]
        (x1, y1, x2, y2)，像素座標。
    """
    x1, y1, x2, y2 = box

    if det_corner is not None and len(det_corner) > 0:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if cv2.pointPolygonTest(det_corner, (float(cx), float(cy)), False) < 0:
            return

    annotator.box_label([x1, y1, x2, y2], embed_label or "", color=color)

    if overlay_label is not None and overlay_anchor is not None:
        annotator.text(
            overlay_anchor, overlay_label,
            txt_color=(255, 255, 255), anchor="top",
            box_color=color,
        )


def save_fst_frame(frame, output_path: Path, show_coord: bool = True):
    pts = np.array([
        [0, 0],
        [640, 0],
        [640, 480],
        [0, 480],
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    if show_coord:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (x, y) in enumerate(pts[:, 0, :]):
            txt = f"P{i+1}({x},{y})"
            cv2.putText(frame, txt, (x - 100, y - 5), font, 0.5, (255, 255, 255), 4)
            cv2.putText(frame, txt, (x - 100, y - 5), font, 0.5, (0, 0, 0), 2)

        save_path = output_path / "first_frame.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"Saved first frame with box to: {save_path}")
