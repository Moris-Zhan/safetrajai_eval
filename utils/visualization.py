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
