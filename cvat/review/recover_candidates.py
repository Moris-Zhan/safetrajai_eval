#!/usr/bin/env python3
"""
Find possibly missed objects: run the pseudo-label detector on whole frames plus overlapping tiles
(small objects get upscaled) at a low confidence, and keep detections that no current CVAT box
covers (max IoU with any existing box below MAX_IOU). Nothing is written to CVAT; the output is
only a candidate list and overlay images for a by-eye precision check.

Input:  runs/cvat_review/phase2/sample.json (from phase2_frames.py), pilot_v1_annotations.json
Output: runs/cvat_review/phase2/candidates.json and cand_NNN.jpg
        (existing boxes thin gray; candidates thick, colored by class, tagged with confidence)

Usage (conda env track_eval, needs cv2/ultralytics):
    python cvat/review/recover_candidates.py --conf 0.10
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
IMAGES = PROJECT_ROOT / "dataset" / "detection" / "images"
WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolo11x_base.pt"
CLASS_NAMES = ["person", "car", "scooter", "small_truck", "bus", "truck"]
COLORS = {"person": (0, 255, 255), "car": (0, 255, 0), "scooter": (255, 0, 255),
          "small_truck": (255, 128, 0), "bus": (0, 128, 255), "truck": (0, 0, 255)}
MAX_IOU = 0.3
MAX_COVERED = 0.6
TILE_OVERLAP = 0.2
TILE_IMGSZ = 1280

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def covered_fraction(a: Box, b: Box) -> float:
    """Share of box a's area that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def tiles(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    tile_w, tile_h = int(width / (2 - TILE_OVERLAP)), int(height / (2 - TILE_OVERLAP))
    xs, ys = [0, width - tile_w], [0, height - tile_h]
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


def detect(model: YOLO, image: np.ndarray, conf: float, offset: Tuple[int, int]) -> List[Tuple[Box, int, float]]:
    result = model.predict(image, conf=conf, imgsz=TILE_IMGSZ, verbose=False)[0]
    found = []
    for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                               result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        found.append(((x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]), int(cls), float(score)))
    return found


def merge(found: List[Tuple[Box, int, float]]) -> List[Tuple[Box, int, float]]:
    kept: List[Tuple[Box, int, float]] = []
    for item in sorted(found, key=lambda entry: -entry[2]):
        if all(iou(item[0], other[0]) < 0.5 or item[1] != other[1] for other in kept):
            kept.append(item)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = parser.parse_args()

    out_dir = args.review_dir / "phase2"
    sample = json.loads((out_dir / "sample.json").read_text(encoding="utf-8"))
    tracks = json.loads((args.review_dir / "pilot_v1_annotations.json").read_text(encoding="utf-8"))["tracks"]
    wanted = {item["manifest_row"] for item in sample}
    existing: Dict[int, List[Box]] = defaultdict(list)
    for track in tracks:
        for shape in track["shapes"]:
            if not shape["outside"] and shape["frame"] in wanted:
                existing[shape["frame"]].append(tuple(shape["points"]))

    model = YOLO(str(WEIGHTS))
    report = []
    for item in sample:
        image = cv2.imread(str(IMAGES / item["seq"] / f"{item['frame_idx']:06d}.jpg"))
        height, width = image.shape[:2]
        found = detect(model, image, args.conf, (0, 0))
        for x1, y1, x2, y2 in tiles(width, height):
            found += detect(model, image[y1:y2, x1:x2], args.conf, (x1, y1))
        merged = merge(found)
        boxes = existing[item["manifest_row"]]
        candidates = [(box, cls, score) for box, cls, score in merged
                      if max((iou(box, old) for old in boxes), default=0.0) < MAX_IOU
                      and max((covered_fraction(box, old) for old in boxes), default=0.0) < MAX_COVERED]
        canvas = image.copy()
        for old in boxes:
            cv2.rectangle(canvas, (int(old[0]), int(old[1])), (int(old[2]), int(old[3])), (160, 160, 160), 1)
        for box, cls, score in candidates:
            color = COLORS[CLASS_NAMES[cls]]
            cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(canvas, f"{score:.2f}", (int(box[0]), max(10, int(box[1]) - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        name = item["image"].replace("frame_", "cand_")
        cv2.imwrite(str(out_dir / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        report.append({"image": name, "seq": item["seq"], "existing": len(boxes), "candidates": [
            {"box": [round(float(v), 1) for v in box], "cls": CLASS_NAMES[cls], "conf": round(score, 3)}
            for box, cls, score in candidates]})
    (out_dir / "candidates.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    total = sum(len(item["candidates"]) for item in report)
    print(f"{len(report)} frames, {sum(i['existing'] for i in report)} existing boxes, {total} candidates")


if __name__ == "__main__":
    main()
