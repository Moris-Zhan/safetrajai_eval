#!/usr/bin/env python3
"""
對 `PILOT_FILES`（跟 sample_and_pseudolabel.py 完全同一份清單）這 10 支原始影片，
跑完整的逐幀 class-aware tracking（沿用 config/tracking/botsort_reid_timm.yaml，
與專案目前驗收基準一致），輸出到 runs/cvat_pilot_v1_tracking/。

跟 sample_and_pseudolabel.py 的差異：那支只用 Detector 做單幀偽標籤（無 track ID），
每 1.5 秒取樣一張；這支用完整的 VideoSequenceTracker（Detector + ClassAwareBOTSORT）
處理每一幀，產生有 track ID 的逐類別 MOT 檔（trackers/mot_challenge/CCTV-val/
YOLO11x__<class>/data/<seq>.txt），供之後對齊 pilot_v1 manifest 的取樣幀、
把 track ID 一起匯入 CVAT 用（CVAT 的自動標註本身不支援批次多物件追蹤，
所以用這份離線追蹤結果代替，見 Markdown/training_eval_audit.md 的 #5 執行說明）。

用法：
    python scripts/detection_data/track_pilot_videos.py
"""
from __future__ import annotations

from safetrajai.evaluation.config_loader import load_config
from safetrajai.evaluation.mot_format import save_mot_format
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.pipeline.sequence_tracker import VideoSequenceTracker
from safetrajai.evaluation.utils import probe_video
from safetrajai.pipeline.tracking_pipeline import resolve_frame_rate
from safetrajai.tracking.config import load_track_config

TRACK_CONFIG_PATH = CONFIG_DIR / "tracking" / "botsort_reid_timm.yaml"
VIDEO_ROOT = PROJECT_ROOT / "dataset" / "raw" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "cvat_pilot_v1_tracking"

# 必須與 sample_and_pseudolabel.py 的 PILOT_FILES 完全一致（同一批 10 個檔案）。
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


def main() -> None:
    load_track_config(str(TRACK_CONFIG_PATH))
    cfg = load_config(str(TRACK_CONFIG_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for camera, filename in PILOT_FILES:
        video_path = str(VIDEO_ROOT / camera / filename)
        print(f"\n[{camera}] tracking {filename} ...")

        seq_tracker = VideoSequenceTracker(
            model_path=cfg.det_model_path,
            tracker_type=cfg.tracker,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            frame_rate=resolve_frame_rate(cfg.frame_rate, video_path),
            gmc_method=cfg.gmc_method,
        )
        output_path, images_path, labels_path = seq_tracker.run(video_path, OUTPUT_DIR)

        width, height, _ = probe_video(video_path)
        save_mot_format(
            imgs_path=images_path,
            labels_path=labels_path,
            output_file=output_path / f"{output_path.stem}.txt",
            img_width=width,
            img_height=height,
        )
        print(f"  -> {output_path}")


if __name__ == "__main__":
    main()
