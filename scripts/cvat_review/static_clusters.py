#!/usr/bin/env python3
"""
Find groups of tracks of one class whose boxes sit at the same place, again and again, on the same
camera. For scooters this is a strong sign of a static false positive (palm leaves, tree trunks,
road marks) or of a parked rider-less scooter; both are deleted by the labeling guide. It is NOT valid
for cars: parked cars are real and legitimately stay in one place.

Same-place criterion (same as the earlier static-FP propagation): same camera, median box center
within CENTER_TOL px of the cluster seed, width/height within SIZE_RATIO of the seed, and the
track's own center spread (when it has 2+ boxes) at most SPREAD_TOL px.

Output:
  runs/cvat_review/static_clusters_<label>.csv
      cluster_id,camera,size,rep_track_id,member_track_ids,center_x,center_y,width,height,max_spread
  runs/cvat_review/static_reps_<label>_rank.csv   (rank,batch,track_id; one representative per
      cluster of size >= MIN_SIZE, 100 per batch, for build_review_montage.py --rank-csv)

Usage (plain python3, stdlib only):
    python3 scripts/cvat_review/static_clusters.py --label scooter
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
MANIFEST = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
CENTER_TOL = 6.0
SIZE_RATIO = 0.25
SPREAD_TOL = 4.0
MIN_SIZE = 3
BATCH_SIZE = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, choices=["scooter"])
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = parser.parse_args()

    tracks = json.loads((args.review_dir / "pilot_v1_annotations.json").read_text(encoding="utf-8"))["tracks"]
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        frame_camera = [row["camera"] for row in csv.DictReader(handle)]

    by_camera: Dict[str, List[Dict]] = defaultdict(list)
    for track in tracks:
        if track["label"] != args.label:
            continue
        shapes = [s for s in track["shapes"] if not s["outside"]]
        if not shapes:
            continue
        centers = [((s["points"][0] + s["points"][2]) / 2, (s["points"][1] + s["points"][3]) / 2) for s in shapes]
        median_x = statistics.median(c[0] for c in centers)
        median_y = statistics.median(c[1] for c in centers)
        spread = max(math.hypot(c[0] - median_x, c[1] - median_y) for c in centers)
        if spread > SPREAD_TOL:
            continue
        width = statistics.median(s["points"][2] - s["points"][0] for s in shapes)
        height = statistics.median(s["points"][3] - s["points"][1] for s in shapes)
        by_camera[frame_camera[shapes[0]["frame"]]].append({
            "track_id": track["track_id"], "x": median_x, "y": median_y, "w": width, "h": height, "spread": spread,
        })

    clusters: List[Dict] = []
    for camera, items in by_camera.items():
        items.sort(key=lambda item: item["track_id"])
        unassigned = items[:]
        while unassigned:
            seed = unassigned[0]
            members = [item for item in unassigned
                       if math.hypot(item["x"] - seed["x"], item["y"] - seed["y"]) <= CENTER_TOL
                       and abs(item["w"] - seed["w"]) <= SIZE_RATIO * max(seed["w"], 1)
                       and abs(item["h"] - seed["h"]) <= SIZE_RATIO * max(seed["h"], 1)]
            member_ids = {item["track_id"] for item in members}
            unassigned = [item for item in unassigned if item["track_id"] not in member_ids]
            clusters.append({
                "camera": camera, "size": len(members), "rep": seed["track_id"],
                "members": sorted(member_ids), "x": round(seed["x"], 1), "y": round(seed["y"], 1),
                "w": round(seed["w"], 1), "h": round(seed["h"], 1),
                "max_spread": round(max(item["spread"] for item in members), 1),
            })

    clusters.sort(key=lambda cluster: -cluster["size"])
    out_path = args.review_dir / f"static_clusters_{args.label}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "camera", "size", "rep_track_id", "member_track_ids",
                         "center_x", "center_y", "width", "height", "max_spread"])
        for cluster_id, cluster in enumerate(clusters):
            writer.writerow([cluster_id, cluster["camera"], cluster["size"], cluster["rep"],
                             ";".join(str(i) for i in cluster["members"]), cluster["x"], cluster["y"],
                             cluster["w"], cluster["h"], cluster["max_spread"]])

    big = [cluster for cluster in clusters if cluster["size"] >= MIN_SIZE]
    rank_path = args.review_dir / f"static_reps_{args.label}_rank.csv"
    with rank_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "batch", "track_id"])
        for rank, cluster in enumerate(big):
            writer.writerow([rank, rank // BATCH_SIZE, cluster["rep"]])

    covered = sum(cluster["size"] for cluster in big)
    print(f"{args.label}: {len(clusters)} clusters, {len(big)} with >= {MIN_SIZE} members covering {covered} tracks")
    print(f"  top sizes: {[cluster['size'] for cluster in big[:10]]}")
    print(f"  -> {out_path}")
    print(f"  -> {rank_path} ({(len(big) + BATCH_SIZE - 1) // BATCH_SIZE} batch(es) of representatives)")


if __name__ == "__main__":
    main()
