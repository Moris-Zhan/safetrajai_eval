#!/usr/bin/env python3
"""
Build review montages for the remaining (not yet reviewed) pilot_v1 tracks of one class.

A track counts as reviewed when it has an applied correction, is an appearance-sample
representative, or is an audit-sample item. The rest is the remaining pool.

Each montage cell is the mid-frame crop of one track (red box, context padding) tagged
"<index> <track_id> <W>x<H>". index.json maps (montage, index) to track_id so verdicts can be
written as corrections CSVs.

Output: runs/cvat_review/review_batches/<label>/<label>_NN.jpg and index.json

Usage (conda env track_eval, needs cv2/numpy):
    python scripts/cvat_review/build_review_montage.py --label bus
    python scripts/cvat_review/build_review_montage.py --label small_truck --sample 200 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
MANIFEST = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
IMAGES = PROJECT_ROOT / "dataset" / "detection" / "images"
CELL_W, CELL_H = 240, 180
COLUMNS, ROWS = 5, 4
LABEL_H = 24
SMALL_BOX_PX = 20
SMALL_PAD_RATIO = 0.25


def reviewed_ids(review_dir: Path) -> Set[str]:
    seen: Set[str] = set()
    with (review_dir / "corrections_applied.csv").open(newline="", encoding="utf-8") as handle:
        seen |= {row["track_id"] for row in csv.DictReader(handle)}
    for path in review_dir.glob("*_appearance_sample*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            seen |= {row["representative_track_id"] for row in csv.DictReader(handle)}
    sample = json.loads((review_dir / "audit_sample.json").read_text(encoding="utf-8"))
    seen |= {str(item["track_id"]) for item in sample["items"]}
    return seen


def render_cell(track: Dict, index: int, manifest: List[Dict[str, str]]) -> np.ndarray:
    cell = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    visible = [shape for shape in track["shapes"] if not shape["outside"]]
    if not visible:
        return cell
    shape = visible[len(visible) // 2]
    row = manifest[shape["frame"]]
    image = cv2.imread(str(IMAGES / row["seq"] / f"{int(row['frame_idx']):06d}.jpg"))
    if image is None:
        return cell
    x1, y1, x2, y2 = (int(round(value)) for value in shape["points"])
    height, width = image.shape[:2]
    longest = max(x2 - x1, y2 - y1)
    pad = max(6, int(SMALL_PAD_RATIO * longest)) if min(x2 - x1, y2 - y1) < SMALL_BOX_PX else max(12, int(0.5 * longest))
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(width, x2 + pad), min(height, y2 + pad)
    crop = image[top:bottom, left:right]
    area_h = CELL_H - LABEL_H
    scale = min(CELL_W / crop.shape[1], area_h / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_CUBIC)
    # Draw the box after scaling so the stroke width stays constant; expand it by the stroke
    # so the object itself is not covered, black outer + magenta inner for any background.
    bx1, by1 = int((x1 - left) * scale) - 2, int((y1 - top) * scale) - 2
    bx2, by2 = int((x2 - left) * scale) + 2, int((y2 - top) * scale) + 2
    cv2.rectangle(crop, (bx1 - 1, by1 - 1), (bx2 + 1, by2 + 1), (0, 0, 0), 3)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (255, 0, 255), 1)
    cell[LABEL_H:LABEL_H + crop.shape[0], :crop.shape[1]] = crop
    tag = f"{index}  {track['track_id']}  {x2 - x1}x{y2 - y1}"
    cv2.putText(cell, tag, (4, LABEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), (90, 90, 90), 1)
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample", type=int, default=None, help="random subset size of the remaining pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = parser.parse_args()

    tracks = json.loads((args.review_dir / "pilot_v1_annotations.json").read_text(encoding="utf-8"))["tracks"]
    seen = reviewed_ids(args.review_dir)
    pool = sorted((t for t in tracks if t["label"] == args.label and str(t["track_id"]) not in seen),
                  key=lambda t: t["track_id"])
    sample: Optional[int] = args.sample
    if sample is not None and sample < len(pool):
        pool = sorted(random.Random(args.seed).sample(pool, sample), key=lambda t: t["track_id"])

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    out_dir = args.review_dir / "review_batches" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    per_montage = COLUMNS * ROWS
    index: Dict[str, List[int]] = {}
    for start in range(0, len(pool), per_montage):
        chunk = pool[start:start + per_montage]
        cells = [render_cell(track, i, manifest) for i, track in enumerate(chunk)]
        cells += [np.zeros((CELL_H, CELL_W, 3), np.uint8)] * (per_montage - len(cells))
        grid = np.vstack([np.hstack(cells[r * COLUMNS:(r + 1) * COLUMNS]) for r in range(ROWS)])
        name = f"{args.label}_{start // per_montage:02d}.jpg"
        cv2.imwrite(str(out_dir / name), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
        index[name] = [track["track_id"] for track in chunk]
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{args.label}: {len(pool)} tracks -> {len(index)} montages in {out_dir}")


if __name__ == "__main__":
    main()
