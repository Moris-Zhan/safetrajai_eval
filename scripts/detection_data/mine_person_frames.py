#!/usr/bin/env python3
"""
Mine raw videos for frames that contain people, as candidates for extra person training labels.

The base model is run on every --stride-th frame of each source camera's videos (sequential decode only).
Frames already in the pilot_v1 manifest are skipped. Held-out evaluation cameras are never read
(C000025 plus --exclude, default C000012 and C000056 which stay a fair evaluation set).

Output: runs/cvat_review/person_candidates.json
  [{camera, video, frame_idx, persons: [[x1, y1, x2, y2, conf], ...]}]

Usage (conda env track_eval, needs cv2/ultralytics):
    python scripts/detection_data/mine_person_frames.py --seconds 1 --seconds-c000054 3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEOS = PROJECT_ROOT / "dataset" / "raw" / "videos"
MANIFEST = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolo11x_base.pt"
SOURCE_CAMERAS = ("C000011", "C000033", "C000043", "C000048", "C000054", "C000085", "C000256")
BATCH = 16


def already_sampled(manifest: Path) -> Set[Tuple[str, int]]:
    with manifest.open(encoding="utf-8") as handle:
        return {(row["camera"], int(row["frame_idx"])) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=1.0, help="seconds between inspected frames")
    parser.add_argument("--seconds-c000054", type=float, default=3.0, help="same for C000054 (31 long videos)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cameras", nargs="+", default=list(SOURCE_CAMERAS))
    parser.add_argument("--exclude", nargs="+", default=["C000012", "C000056", "C000025"])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "cvat_review" / "person_candidates.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    skip = already_sampled(MANIFEST)
    found: List[Dict[str, object]] = []
    for camera in args.cameras:
        if camera in args.exclude:
            continue
        for video in sorted((VIDEOS / camera).glob("*.mp4")):
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
            stride = max(1, int(round(fps * (args.seconds_c000054 if camera == "C000054" else args.seconds))))
            frames, indexes, index = [], [], 0
            while True:
                ok = capture.grab()
                if not ok:
                    break
                if index % stride == 0 and (camera, index) not in skip:
                    ok, image = capture.retrieve()
                    if ok:
                        frames.append(image)
                        indexes.append(index)
                index += 1
                if len(frames) == BATCH:
                    found.extend(_detect(model, frames, indexes, camera, video.name, args.conf))
                    frames, indexes = [], []
            if frames:
                found.extend(_detect(model, frames, indexes, camera, video.name, args.conf))
            capture.release()
            print(f"{camera} {video.name}: {len(found)} candidate frames so far", flush=True)
    args.out.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(found)} frames -> {args.out}")


def _detect(model, frames, indexes, camera: str, video: str, conf: float) -> List[Dict[str, object]]:
    results = model.predict(frames, imgsz=640, conf=conf, classes=[0], verbose=False)
    out = []
    for result, index in zip(results, indexes):
        boxes = result.boxes
        if len(boxes):
            persons = [[*map(float, xyxy), float(score)] for xyxy, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist())]
            out.append({"camera": camera, "video": video, "frame_idx": index, "persons": persons})
    return out


if __name__ == "__main__":
    main()
