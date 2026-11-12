#!/usr/bin/env python3
"""
Pilot 階段：對 `detection_source`（`config/datasets.yaml`）9 支鏡頭裡挑出的 10 個檔案，
依固定秒數間隔取樣畫面，用目前在用的偵測權重（`models/yolo/yolo11x_dennis.pt`）產生偽標籤，
供之後人工覆核用。對應 `training_eval_audit.md`「未來計畫：以 yolo11x_dennis.pt 為起始模型
精煉並製作偵測資料集」（2026-11-09 `/grill-me` 定案的 10 點共識）第 1、4、5 點。

這支腳本只做「取樣＋偽標籤」，不做人工覆核、不做資料集切分（切分留到真正要訓練時再決定）。

Pilot 檔案清單（PILOT_FILES）：9 支鏡頭裡 8 支各只有 1 個原始檔，直接全取；
`C000054` 有 31 個檔案，取其中相隔最遠的 2 個（第一個與最後一個時間戳記），
盡量涵蓋不同日期／時段的車流樣態，符合共識第 4 點「各取 1～2 個檔案」。

輸出（dataset/detection/，images/、labels/ 是版本無關的累積池，splits/<version>/ 才是
版本化的清單，仿照 dataset/reid/crops 的慣例）：
    dataset/detection/images/<seq>/<frame_idx:06d>.jpg
    dataset/detection/labels/<seq>/<frame_idx:06d>.txt   (YOLO 格式：cls xc yc w h，正規化；
                                                           沒有偵測到任何框的幀也會寫 0-byte 空檔，
                                                           明確標示「已取樣、已推論、背景幀」)
    dataset/detection/splits/<version>/manifest.csv

用法：
    python scripts/detection_data/sample_and_pseudolabel.py
    python scripts/detection_data/sample_and_pseudolabel.py --interval-sec 1.0 --version pilot_v2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from safetrajai.classes import CLASS_NAMES
from safetrajai.detection.detector import Detector
from safetrajai.evaluation.config_loader import load_config
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.tracking.config import load_track_config

TRACK_CONFIG_PATH = CONFIG_DIR / "tracking" / "botsort_reid_timm.yaml"

DETECTION_ROOT = PROJECT_ROOT / "dataset" / "detection"
VIDEO_ROOT = PROJECT_ROOT / "dataset" / "raw" / "videos"

# (camera, 檔名)；C000054 刻意取時間跨度最大的頭尾兩個檔案，其餘 8 支鏡頭各只有 1 個原始檔。
PILOT_FILES = [
    ("C000011", "C000011 中正路交流道南下入口_20250512065725.mp4"),
    ("C000012", "C000012 中正路交流道北上入口_20250512065724.mp4"),
    ("C000033", "C000033 大中一路 民族一路_20250512065631.mp4"),
    ("C000043", "C000043 民族一路 九如一路_20250514071352.mp4"),
    ("C000048", "C000048 翠華路 明潭路_20250512070138.mp4"),
    ("C000054", "C000054 民族一路 天祥二路_20250623065608.mp4"),
    ("C000054", "C000054 民族一路 天祥二路_20250629082435.mp4"),
    ("C000056", "C000056 左楠路 後昌路_20250512065829.mp4"),
    ("C000085", "C000085 中正路 三多路(建軍)_20250514130803.mp4"),
    ("C000256", "C000256 中山三路 光華三路_20250512070015.mp4"),
]


def sample_video(detector: Detector, iou: float, imgsz: int, camera: str, filename: str, interval_sec: float):
    video_path = VIDEO_ROOT / camera / filename
    seq = Path(filename).stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * interval_sec))

    images_dir = DETECTION_ROOT / "images" / seq
    labels_dir = DETECTION_ROOT / "labels" / seq
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            boxes = detector.detect(frame, iou=iou, imgsz=imgsz)
            cv2.imwrite(str(images_dir / f"{frame_idx:06d}.jpg"), frame)

            class_counts = {name: 0 for name in CLASS_NAMES.values()}
            lines = []
            for xywhn, cls_id in zip(boxes.xywhn.tolist(), boxes.cls.tolist()):
                cls_id = int(cls_id)
                xc, yc, w, h = xywhn
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[CLASS_NAMES[cls_id]] += 1
            (labels_dir / f"{frame_idx:06d}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

            rows.append({
                "image_path": f"{seq}/{frame_idx:06d}.jpg",
                "label_path": f"{seq}/{frame_idx:06d}.txt",
                "camera": camera,
                "seq": seq,
                "frame_idx": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "num_boxes": len(lines),
                **class_counts,
            })
        frame_idx += 1

    cap.release()
    return rows


def write_manifest(rows: list[dict], version: str) -> Path:
    manifest_dir = DETECTION_ROOT / "splits" / version
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    fieldnames = (
        ["image_path", "label_path", "camera", "seq", "frame_idx", "timestamp_sec", "num_boxes"]
        + list(CLASS_NAMES.values())
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval-sec", type=float, default=1.5, help="每隔幾秒取樣一張畫面")
    parser.add_argument("--version", default="pilot_v1", help="splits/<version>/manifest.csv 的版本名稱")
    parser.add_argument("--track-config", default=str(TRACK_CONFIG_PATH))
    args = parser.parse_args()

    load_track_config(args.track_config)
    cfg = load_config(args.track_config)
    detector = Detector(cfg.det_model_path)

    all_rows = []
    for camera, filename in PILOT_FILES:
        print(f"[{camera}] 取樣 {filename} ...")
        rows = sample_video(detector, cfg.iou, cfg.imgsz, camera, filename, args.interval_sec)
        print(f"[{camera}]   {len(rows)} 幀，共 {sum(r['num_boxes'] for r in rows)} 個框")
        all_rows.extend(rows)

    manifest_path = write_manifest(all_rows, args.version)

    totals = {name: 0 for name in CLASS_NAMES.values()}
    for row in all_rows:
        for name in CLASS_NAMES.values():
            totals[name] += row[name]

    print(f"\n已寫入 {manifest_path}（{len(all_rows)} 幀）")
    print("逐類別框數：")
    for name, count in totals.items():
        flag = "" if count >= 100 else "  <-- 未達 pilot 門檻（100）"
        print(f"  {name}: {count}{flag}")


if __name__ == "__main__":
    main()
