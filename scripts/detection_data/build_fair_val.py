#!/usr/bin/env python3
"""Sample the fair_val frames and build the base-model prefill (Markdown/detection/finetune-plan.md section 3.6).

fair_val is the fair-camera evaluation GT (C000012 + C000056), relabeled in the cctv_all8 style: far small objects are
labeled, no masks, vehicle classes follow the GT convention, riders count as scooter. Frames come only from the
validation sequences of pilot_v1 (never in any training export): --per-camera frames per camera, evenly spaced in time.

The prefill is the base model (models/yolo/yolo11x_base.pt, imgsz 640) at conf >= --conf. Boxes below --low-conf are
flagged so the reviewer can find them; every box is reviewed by hand and a separate missed-object pass follows. The
prefill biases the GT slightly toward the base model; reports must say so.

Output: dataset/detection/fair_val/{images/<stem>.jpg, manifest.csv, prefill.json}

    python scripts/detection_data/build_fair_val.py --per-camera 50
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil

import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.paths import PROJECT_ROOT

DETECTION_ROOT = PROJECT_ROOT / "dataset" / "detection"
MANIFEST = DETECTION_ROOT / "splits" / "pilot_v1" / "manifest.csv"
OUT = DETECTION_ROOT / "fair_val"
WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolo11x_base.pt"
FAIR_CAMERAS = ("C000012", "C000056")


def stem_of(row: dict) -> str:
    return f"{row['camera']}_{row['seq'].rsplit('_', 1)[-1]}_f{int(row['frame_idx']):06d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-camera", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.1, help="lowest base-model confidence kept in the prefill")
    parser.add_argument("--low-conf", type=float, default=0.3, help="prefill boxes below this are flagged low_conf")
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; fair_val is built once (move it aside first)")

    with MANIFEST.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["camera"] in FAIR_CAMERAS]
    picked = []
    for camera in FAIR_CAMERAS:
        frames = sorted((row for row in rows if row["camera"] == camera), key=lambda r: (r["seq"], int(r["frame_idx"])))
        if len(frames) < args.per_camera:
            raise SystemExit(f"{camera}: only {len(frames)} frames")
        picked += [frames[i] for i in np.linspace(0, len(frames) - 1, args.per_camera).round().astype(int)]

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    (OUT / "images").mkdir(parents=True)
    prefill = []
    for row in picked:
        stem = stem_of(row)
        target = OUT / "images" / f"{stem}.jpg"
        shutil.copy2(DETECTION_ROOT / "images" / row["image_path"], target)
        result = model.predict(str(target), imgsz=640, conf=args.conf, verbose=False)[0]
        height, width = result.orig_shape
        boxes = [{"label": CLASS_NAMES[int(c)], "conf": round(float(s), 4), "xyxy": [round(v, 2) for v in b],
                  "low_conf": float(s) < args.low_conf}
                 for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
        prefill.append({"image": f"{stem}.jpg", "width": width, "height": height, "boxes": boxes})
    with (OUT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "camera", "seq", "frame_idx", "source_image"])
        for row in picked:
            writer.writerow([f"{stem_of(row)}.jpg", row["camera"], row["seq"], row["frame_idx"], row["image_path"]])
    (OUT / "prefill.json").write_text(json.dumps({
        "weights": str(WEIGHTS.relative_to(PROJECT_ROOT)), "imgsz": 640, "conf": args.conf, "low_conf": args.low_conf,
        "frames": prefill}, indent=1, ensure_ascii=False), encoding="utf-8")
    n_boxes = sum(len(f["boxes"]) for f in prefill)
    n_low = sum(b["low_conf"] for f in prefill for b in f["boxes"])
    print(f"{len(prefill)} frames, {n_boxes} prefill boxes ({n_low} low_conf) -> {OUT}")


if __name__ == "__main__":
    main()
