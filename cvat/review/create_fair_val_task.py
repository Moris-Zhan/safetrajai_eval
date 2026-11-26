#!/usr/bin/env python3
"""Create the fair_val CVAT task and load the base-model prefill as rectangle shapes (finetune-plan.md section 3.6).

Input: dataset/detection/fair_val/{images/, prefill.json} from scripts/detection_data/build_fair_val.py.
Labels are the 6 project classes; each has a checkbox attribute low_conf (true for prefill boxes below the
build_fair_val --low-conf threshold, so they can be filtered in the CVAT UI). Writes the task id to
dataset/detection/fair_val/cvat_task.json.

Run with the CVAT SDK venv (no cv2 / safetrajai needed):
    cvat/deploy/cvat_client_venv/bin/python cvat/review/create_fair_val_task.py

CVAT connection from environment variables or the repo-root .env (CVAT_URL / CVAT_USERNAME / CVAT_PASSWORD);
the password is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from cvat_sdk import make_client, models

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="safetrajai fair_val v1")
    args = parser.parse_args()
    record = FAIR_VAL / "cvat_task.json"
    if record.exists():
        raise SystemExit(f"{record} exists; the task was already created")
    prefill = json.loads((FAIR_VAL / "prefill.json").read_text(encoding="utf-8"))
    frames = prefill["frames"]

    url = urlsplit(_setting("CVAT_URL"))
    with make_client(host=f"{url.scheme}://{url.hostname}", port=url.port,
                     credentials=(_setting("CVAT_USERNAME"), _setting("CVAT_PASSWORD"))) as client:
        spec = models.TaskWriteRequest(name=args.name, labels=[
            models.PatchedLabelRequest(name=name, type="rectangle", attributes=[
                models.AttributeRequest(name="low_conf", mutable=False, input_type="checkbox",
                                        default_value="false", values=["false"])])
            for name in LABELS])
        task = client.tasks.create_from_data(spec=spec, resources=[str(FAIR_VAL / "images" / f["image"]) for f in frames],
                                             data_params={"image_quality": 95, "sorting_method": "predefined"})
        labels = {label.name: label for label in task.get_labels()}
        frame_index = {Path(meta.name).name: i for i, meta in enumerate(task.get_meta().frames)}
        shapes = []
        for frame in frames:
            for box in frame["boxes"]:
                label = labels[box["label"]]
                attr = label.attributes[0]
                shapes.append(models.LabeledShapeRequest(
                    type="rectangle", frame=frame_index[frame["image"]], label_id=label.id, points=box["xyxy"],
                    attributes=[models.AttributeValRequest(spec_id=attr.id, value=str(box["low_conf"]).lower())]))
        task.update_annotations(models.PatchedLabeledDataRequest(shapes=shapes))
        record.write_text(json.dumps({"task_id": task.id, "name": args.name, "frames": len(frames),
                                      "prefill_shapes": len(shapes)}, indent=1), encoding="utf-8")
        print(f"task {task.id}: {len(frames)} frames, {len(shapes)} prefill shapes")


if __name__ == "__main__":
    main()
