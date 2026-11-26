#!/usr/bin/env python3
"""
Suggest per-camera "unreliable annotation" regions from the phase-2 missed-object candidates.

For every camera the frame is cut into a grid. A cell is marked when the high-confidence
candidates (detections that no current CVAT box covers, see recover_candidates.py) are a large
share of everything seen in the cell (candidates + existing boxes) across the sampled frames.
Marked cells are dilated by one cell, joined into connected regions, and each region becomes one
polygon. The result is only a suggestion for a human to confirm; it changes nothing in CVAT.

A masked region would later be painted over in the training images and its boxes dropped, so the
detector does not learn unlabeled objects as background. Mask polygons are exclusion zones, the
opposite of the keep-zone polygons in config/roi/<camera>.yaml.

Input:  runs/cvat_review/phase2/{candidates.json,sample.json}, pilot_v1_annotations.json
Output: runs/cvat_review/phase2/masks_suggested/<camera>.json  ({"image_size":[w,h],"polygons":[[[x,y],...]]})
        and <camera>.jpg (first sampled frame with the suggested regions shaded)

Usage (conda env track_eval, needs cv2/numpy):
    python cvat/review/suggest_masks.py --min-conf 0.5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
GRID_COLUMNS = 24
GRID_ROWS = 14
MIN_CANDIDATES = 2
MIN_SHARE = 0.4
MIN_REGION_CELLS = 3


def camera_of(sequence: str) -> str:
    return sequence.split(" ")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = parser.parse_args()

    phase2 = args.review_dir / "phase2"
    sample = {item["image"].replace("frame_", "cand_"): item for item in
              json.loads((phase2 / "sample.json").read_text(encoding="utf-8"))}
    candidates = json.loads((phase2 / "candidates.json").read_text(encoding="utf-8"))
    tracks = json.loads((args.review_dir / "pilot_v1_annotations.json").read_text(encoding="utf-8"))["tracks"]
    existing: Dict[int, List[List[float]]] = defaultdict(list)
    wanted = {item["manifest_row"] for item in sample.values()}
    for track in tracks:
        for shape in track["shapes"]:
            if not shape["outside"] and shape["frame"] in wanted:
                existing[shape["frame"]].append(shape["points"])

    by_camera: Dict[str, List[Dict]] = defaultdict(list)
    for entry in candidates:
        by_camera[camera_of(entry["seq"])].append(entry)

    out_dir = phase2 / "masks_suggested"
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera, entries in sorted(by_camera.items()):
        image = cv2.imread(str(phase2 / entries[0]["image"].replace("cand_", "frame_")))
        height, width = image.shape[:2]
        cand_count = np.zeros((GRID_ROWS, GRID_COLUMNS))
        box_count = np.zeros((GRID_ROWS, GRID_COLUMNS))

        def cell_of(x: float, y: float) -> Tuple[int, int]:
            return (min(GRID_ROWS - 1, max(0, int(y / height * GRID_ROWS))),
                    min(GRID_COLUMNS - 1, max(0, int(x / width * GRID_COLUMNS))))

        for entry in entries:
            for candidate in entry["candidates"]:
                if candidate["conf"] >= args.min_conf:
                    x1, y1, x2, y2 = candidate["box"]
                    row, col = cell_of((x1 + x2) / 2, (y1 + y2) / 2)
                    cand_count[row, col] += 1
            for x1, y1, x2, y2 in existing[sample[entry["image"]]["manifest_row"]]:
                row, col = cell_of((x1 + x2) / 2, (y1 + y2) / 2)
                box_count[row, col] += 1

        share = cand_count / np.maximum(cand_count + box_count, 1)
        marked = ((cand_count >= MIN_CANDIDATES) & (share >= MIN_SHARE)).astype(np.uint8)
        marked = cv2.dilate(marked, np.ones((3, 3), np.uint8))
        count, labels = cv2.connectedComponents(marked)
        polygons: List[List[List[float]]] = []
        shade = image.copy()
        scale = 8
        for label in range(1, count):
            region = (labels == label).astype(np.uint8)
            if region.sum() < MIN_REGION_CELLS:
                continue
            upscaled = cv2.resize(region, (GRID_COLUMNS * scale, GRID_ROWS * scale), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(upscaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                simplified = cv2.approxPolyDP(contour, 3, True).reshape(-1, 2)
                polygon = [[round(float(x) / (GRID_COLUMNS * scale) * width, 1),
                            round(float(y) / (GRID_ROWS * scale) * height, 1)] for x, y in simplified]
                if len(polygon) >= 3:
                    polygons.append(polygon)
                    cv2.fillPoly(shade, [np.array(polygon, np.int32)], (0, 0, 255))
        overlay = cv2.addWeighted(shade, 0.35, image, 0.65, 0)
        for polygon in polygons:
            cv2.polylines(overlay, [np.array(polygon, np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{camera}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        area = float(marked.sum()) / marked.size
        (out_dir / f"{camera}.json").write_text(json.dumps(
            {"camera": camera, "image_size": [width, height], "frames": len(entries),
             "min_conf": args.min_conf, "marked_cell_share": round(area, 3), "polygons": polygons},
            indent=1), encoding="utf-8")
        print(f"{camera}: {len(entries)} frames, {len(polygons)} polygons, {area:.0%} of the frame marked")


if __name__ == "__main__":
    main()
