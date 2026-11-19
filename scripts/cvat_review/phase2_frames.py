#!/usr/bin/env python3
"""
Phase-2 measurement frames: stratified random sample of whole frames drawn with every current
CVAT box, used to count missed objects and clearly shifted boxes by eye.

Sampling: the pilot_v1 manifest has one row per frame; frames are grouped by sequence (10 files)
and PER_SEQUENCE frames are drawn from each with a fixed seed, so the sample is reproducible.

Each output image is the full frame with one thin box per visible shape, colored by class, tagged
with the track id. sample.json lists (image name, manifest row, sequence, frame_idx) so verdicts
can be written back by name.

Output: runs/cvat_review/phase2/frame_NN.jpg and sample.json

Usage (conda env track_eval, needs cv2):
    python scripts/cvat_review/phase2_frames.py --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
MANIFEST = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
IMAGES = PROJECT_ROOT / "dataset" / "detection" / "images"
PER_SEQUENCE = 10
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {  # BGR
    "person": (0, 255, 255), "car": (0, 255, 0), "scooter": (255, 0, 255),
    "small_truck": (255, 128, 0), "bus": (0, 128, 255), "truck": (0, 0, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-sequence", type=int, default=PER_SEQUENCE)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = parser.parse_args()

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    by_sequence: Dict[str, List[int]] = defaultdict(list)
    for row_index, row in enumerate(manifest):
        by_sequence[row["seq"]].append(row_index)

    rng = random.Random(args.seed)
    chosen: List[int] = []
    for sequence in sorted(by_sequence):
        rows = by_sequence[sequence]
        chosen += sorted(rng.sample(rows, min(args.per_sequence, len(rows))))

    tracks = json.loads((args.review_dir / "pilot_v1_annotations.json").read_text(encoding="utf-8"))["tracks"]
    boxes_by_frame: Dict[int, List[Tuple[str, int, List[float]]]] = defaultdict(list)
    wanted = set(chosen)
    for track in tracks:
        for shape in track["shapes"]:
            if not shape["outside"] and shape["frame"] in wanted:
                boxes_by_frame[shape["frame"]].append((track["label"], track["track_id"], shape["points"]))

    out_dir = args.review_dir / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = []
    for number, row_index in enumerate(chosen):
        row = manifest[row_index]
        image = cv2.imread(str(IMAGES / row["seq"] / f"{int(row['frame_idx']):06d}.jpg"))
        if image is None:
            continue
        for label, track_id, (x1, y1, x2, y2) in boxes_by_frame[row_index]:
            color = CLASS_COLORS[label]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(image, str(track_id), (int(x1), max(10, int(y1) - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        name = f"frame_{number:03d}.jpg"
        cv2.imwrite(str(out_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        listing.append({"image": name, "manifest_row": row_index, "seq": row["seq"],
                        "frame_idx": int(row["frame_idx"]), "boxes": len(boxes_by_frame[row_index])})
    (out_dir / "sample.json").write_text(json.dumps(listing, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(listing)} frames from {len(by_sequence)} sequences -> {out_dir}")


if __name__ == "__main__":
    main()
