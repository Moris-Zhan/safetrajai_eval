#!/usr/bin/env python3
"""Export the reviewed fair_val CVAT task as YOLO labels (finetune-plan.md section 3.6).

Reads the task id from dataset/detection/fair_val/cvat_task.json, fetches the rectangle shapes, and writes
  dataset/detection/fair_val/labels/<stem>.txt   (YOLO: class cx cy w h, normalized; empty file = no objects)
  dataset/detection/fair_val/annotations_<UTC>.json  (raw snapshot of the shapes, for reproduction)
  dataset/detection/fair_val/data.yaml           (Ultralytics val/train both = images/)
Image sizes come from prefill.json. Labels outside the 6 project classes are an error. Run it again after more
corrections; labels/ is rewritten from the latest CVAT state each time.

Run with the CVAT SDK venv:
    cvat/deploy/cvat_client_venv/bin/python cvat/review/export_fair_val.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from cvat_sdk import make_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
FAIR_VAL = PROJECT_ROOT / "dataset" / "detection" / "fair_val"
LABELS = ["person", "car", "scooter", "small_truck", "bus", "truck"]  # utils/modules/safetrajai/classes.py order


def _setting(name: str) -> str:
    value = os.environ.get(name)
    if not value and ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            key, _, raw = line.strip().partition("=")
            if key.strip() == name and raw.strip():
                value = raw.strip().strip('"').strip("'")
    if not value:
        raise SystemExit(f"missing setting {name} (environment or {ENV_PATH})")
    return value


def main() -> None:
    task_id = json.loads((FAIR_VAL / "cvat_task.json").read_text(encoding="utf-8"))["task_id"]
    sizes = {f["image"]: (f["width"], f["height"]) for f in
             json.loads((FAIR_VAL / "prefill.json").read_text(encoding="utf-8"))["frames"]}
    url = urlsplit(_setting("CVAT_URL"))
    with make_client(host=f"{url.scheme}://{url.hostname}", port=url.port,
                     credentials=(_setting("CVAT_USERNAME"), _setting("CVAT_PASSWORD"))) as client:
        task = client.tasks.retrieve(task_id)
        label_names = {label.id: label.name for label in task.get_labels()}
        frame_names = [Path(meta.name).name for meta in task.get_meta().frames]
        annotations = task.get_annotations()
    if annotations.tracks:
        raise SystemExit(f"task {task_id} has {len(annotations.tracks)} tracks; fair_val expects shapes only")
    per_frame = {name: [] for name in frame_names}
    snapshot = []
    for shape in annotations.shapes:
        if shape.type.value != "rectangle":
            raise SystemExit(f"unexpected shape type {shape.type.value} on frame {shape.frame}")
        name = label_names[shape.label_id]
        if name not in LABELS:
            raise SystemExit(f"unknown label {name}")
        image = frame_names[shape.frame]
        x1, y1, x2, y2 = shape.points
        per_frame[image].append((LABELS.index(name), min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        snapshot.append({"image": image, "label": name, "xyxy": [x1, y1, x2, y2]})
    labels_dir = FAIR_VAL / "labels"
    labels_dir.mkdir(exist_ok=True)
    counts = {name: 0 for name in LABELS}
    for image, boxes in per_frame.items():
        width, height = sizes[image]
        lines = []
        for cls, x1, y1, x2, y2 in boxes:
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
            counts[LABELS[cls]] += 1
        (labels_dir / f"{Path(image).stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (FAIR_VAL / f"annotations_{stamp}.json").write_text(json.dumps(
        {"task_id": task_id, "exported_utc": stamp, "shapes": snapshot}, indent=1, ensure_ascii=False), encoding="utf-8")
    (FAIR_VAL / "data.yaml").write_text(
        f"path: {FAIR_VAL}\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(LABELS)), encoding="utf-8")
    print(f"task {task_id}: {len(frame_names)} frames, {sum(counts.values())} boxes {counts}")


if __name__ == "__main__":
    main()
