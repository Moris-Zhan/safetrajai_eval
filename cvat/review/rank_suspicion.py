#!/usr/bin/env python3
"""
Rank the not-yet-reviewed pilot_v1 tracks of one class by how suspicious their boxes are, so the
large car/scooter pools are reviewed worst-first in batches.

Two rules (agreed in the review plan), both computed from the CVAT annotation export only:
  group_count    max over the track's frames of the number of OTHER vehicle boxes (car, scooter,
                 small_truck, bus, truck; not person) whose center lies inside this box and whose
                 area is smaller than this box. 2 or more means "suspected group box".
  oversize_ratio median box area of the track divided by the median of the per-track median areas
                 of the same class on the same camera. 3 or more means "suspected oversized box".

Rank order: group_count (desc), then oversize_ratio (desc). The ranking only orders the review;
nothing is deleted automatically.

Output: runs/cvat_review/suspicion_<label>.csv with columns
        rank,batch,track_id,camera,group_count,oversize_ratio,median_area
(batch is 0-based, 100 tracks per batch)

Usage (plain python3, stdlib only):
    python3 cvat/review/rank_suspicion.py --label car
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
MANIFEST = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
VEHICLE_LABELS = {"car", "scooter", "small_truck", "bus", "truck"}
BATCH_SIZE = 100
GROUP_MIN = 2
OVERSIZE_MIN = 3.0

Box = Tuple[int, float, float, float, float]  # track_id, x1, y1, x2, y2


def reviewed_ids(review_dir: Path) -> Set[str]:
    seen: Set[str] = set()
    with (review_dir / "corrections_applied.csv").open(newline="", encoding="utf-8") as handle:
        seen |= {row["track_id"] for row in csv.DictReader(handle)}
    for path in review_dir.glob("*_appearance_sample*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            seen |= {row["representative_track_id"] for row in csv.DictReader(handle)}
    sample = json.loads((review_dir / "audit_sample.json").read_text(encoding="utf-8"))
    seen |= {str(item["track_id"]) for item in sample["items"]}
    kept_path = review_dir / "reviewed_kept.csv"
    if kept_path.exists():
        with kept_path.open(newline="", encoding="utf-8") as handle:
            seen |= {row["track_id"] for row in csv.DictReader(handle)}
    return seen


def box_area(box: Box) -> float:
    return max(0.0, box[3] - box[1]) * max(0.0, box[4] - box[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, choices=["car", "scooter"])
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-area", type=float, default=None,
                        help="only rank tracks whose median box area is below this many px^2 (small-box sweep)")
    args = parser.parse_args()

    tracks = json.loads((args.review_dir / "pilot_v1_annotations.json").read_text(encoding="utf-8"))["tracks"]
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        frame_camera = [row["camera"] for row in csv.DictReader(handle)]

    vehicle_by_frame: Dict[int, List[Box]] = defaultdict(list)
    track_info: Dict[int, Dict] = {}
    for track in tracks:
        if track["label"] not in VEHICLE_LABELS:
            continue
        areas: List[float] = []
        camera = ""
        for shape in track["shapes"]:
            if shape["outside"]:
                continue
            x1, y1, x2, y2 = shape["points"]
            vehicle_by_frame[shape["frame"]].append((track["track_id"], x1, y1, x2, y2))
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            camera = frame_camera[shape["frame"]]
        if areas:
            track_info[track["track_id"]] = {
                "label": track["label"], "camera": camera, "median_area": statistics.median(areas),
                "shapes": [s for s in track["shapes"] if not s["outside"]],
            }

    median_area_by_group: Dict[Tuple[str, str], float] = {}
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for info in track_info.values():
        grouped[(info["camera"], info["label"])].append(info["median_area"])
    for key, values in grouped.items():
        median_area_by_group[key] = statistics.median(values)

    seen = reviewed_ids(args.review_dir)
    ranked: List[Dict] = []
    for track_id, info in track_info.items():
        if info["label"] != args.label or str(track_id) in seen:
            continue
        if args.max_area is not None and info["median_area"] >= args.max_area:
            continue
        group_count = 0
        for shape in info["shapes"]:
            x1, y1, x2, y2 = shape["points"]
            own_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            inside = 0
            for other in vehicle_by_frame[shape["frame"]]:
                if other[0] == track_id or box_area(other) >= own_area:
                    continue
                center_x, center_y = (other[1] + other[3]) / 2, (other[2] + other[4]) / 2
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    inside += 1
            group_count = max(group_count, inside)
        reference = median_area_by_group[(info["camera"], info["label"])]
        ranked.append({
            "track_id": track_id, "camera": info["camera"], "group_count": group_count,
            "oversize_ratio": round(info["median_area"] / reference, 2) if reference > 0 else 0.0,
            "median_area": round(info["median_area"], 1),
        })

    ranked.sort(key=lambda row: (-row["group_count"], -row["oversize_ratio"], row["track_id"]))
    out_path = args.out or args.review_dir / f"suspicion_{args.label}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "batch", "track_id", "camera", "group_count", "oversize_ratio", "median_area"])
        for rank, row in enumerate(ranked):
            writer.writerow([rank, rank // BATCH_SIZE, row["track_id"], row["camera"], row["group_count"],
                             row["oversize_ratio"], row["median_area"]])

    group_flagged = sum(1 for row in ranked if row["group_count"] >= GROUP_MIN)
    oversize_flagged = sum(1 for row in ranked if row["oversize_ratio"] >= OVERSIZE_MIN)
    either = sum(1 for row in ranked if row["group_count"] >= GROUP_MIN or row["oversize_ratio"] >= OVERSIZE_MIN)
    print(f"{args.label}: {len(ranked)} remaining tracks -> {out_path}")
    print(f"  suspected group (>= {GROUP_MIN} inside): {group_flagged}")
    print(f"  suspected oversized (>= {OVERSIZE_MIN}x median): {oversize_flagged}")
    print(f"  either: {either} (~{(either + BATCH_SIZE - 1) // BATCH_SIZE} batches)")


if __name__ == "__main__":
    main()
