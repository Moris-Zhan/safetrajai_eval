#!/usr/bin/env python3
"""
Export the human-corrected CVAT annotations of pilot_v1 as a YOLO fine-tuning dataset.

Rules (agreed in the export-design review):
  * Labels come from the CVAT JSON snapshot (fetch_task_annotations.py), not from the original
    pseudo-label txts. A snapshot copy is stored beside the export so a version can be reproduced.
  * Frames listed in runs/cvat_review/box_issues.csv (group boxes deleted, so the people/vehicles in
    them are unlabeled) are excluded whole.
  * Per-camera mask polygons (config/detection_masks/<camera>.json, exclusion zones) are painted gray
    (114). A box with at least --mask-drop-ratio of its area inside the mask is dropped; boxes that are
    kept are never painted over.
  * Split by sequence: --val takes "camera" or "camera:text-in-sequence-name" entries; every other
    sequence is train. The evaluation camera C000025 must not appear in the manifest.
  * No resampling here; per-class box counts are written to export_record.json.

Output: dataset/detection/exports/<version>/{images,labels}/{train,val}/, data.yaml,
        export_record.json, annotations_snapshot.json

Usage (conda env track_eval, needs cv2/numpy/yaml):
    python scripts/detection_data/export_training_set.py --version ft_v1_trial
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml

from safetrajai.classes import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_ROOT = PROJECT_ROOT / "dataset" / "detection"
MANIFEST = DETECTION_ROOT / "splits" / "pilot_v1" / "manifest.csv"
IMAGES = DETECTION_ROOT / "images"
MASK_DIR = PROJECT_ROOT / "config" / "detection_masks"
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
EVAL_CAMERA = "C000025"
GRAY = 114
DEFAULT_VAL = ("C000012", "C000054:20250629")
CLASS_IDS: Dict[str, int] = {name: class_id for class_id, name in CLASS_NAMES.items()}

Box = Tuple[int, float, float, float, float]  # class_id, x1, y1, x2, y2


def parse_excluded_frames(box_issues_csv: Path) -> Set[int]:
    excluded: Set[int] = set()
    with box_issues_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] == "group_deleted":
                excluded.update(int(frame) for frame in row["manifest_frames"].split(";") if frame)
    return excluded


def is_validation(camera: str, sequence: str, val_specs: Sequence[str]) -> bool:
    for spec in val_specs:
        spec_camera, _, needle = spec.partition(":")
        if camera == spec_camera and needle in sequence:
            return True
    return False


def mask_canvas(polygons: Sequence[Sequence[Sequence[float]]], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.round(np.array(polygon)).astype(np.int32)], 1)
    return canvas


def masked_fraction(canvas: np.ndarray, box: Box) -> float:
    height, width = canvas.shape
    x1, y1 = max(0, int(np.floor(box[1]))), max(0, int(np.floor(box[2])))
    x2, y2 = min(width, int(np.ceil(box[3]))), min(height, int(np.ceil(box[4])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(canvas[y1:y2, x1:x2].mean())


def apply_mask(image: np.ndarray, canvas: np.ndarray, kept: Sequence[Box]) -> np.ndarray:
    painted = image.copy()
    painted[canvas.astype(bool)] = GRAY
    height, width = canvas.shape
    for _, x1, y1, x2, y2 in kept:
        ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        painted[iy1:iy2, ix1:ix2] = image[iy1:iy2, ix1:ix2]
    return painted


def yolo_line(box: Box, width: int, height: int) -> str:
    class_id, x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    return (f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def boxes_by_frame(annotations: dict) -> Dict[int, List[Box]]:
    result: Dict[int, List[Box]] = defaultdict(list)
    for track in annotations["tracks"]:
        class_id = CLASS_IDS[track["label"]]
        for shape in track["shapes"]:
            if not shape["outside"]:
                x1, y1, x2, y2 = shape["points"]
                result[shape["frame"]].append((class_id, x1, y1, x2, y2))
    return result


def frame_stem(camera: str, sequence: str, frame_idx: int) -> str:
    return f"{camera}_{sequence.rsplit('_', 1)[-1]}_{frame_idx:06d}"


def load_masks(mask_dir: Path) -> Dict[str, dict]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(mask_dir.glob("*.json"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="export folder name under dataset/detection/exports/")
    parser.add_argument("--annotations", type=Path, default=REVIEW_DIR / "pilot_v1_annotations.json")
    parser.add_argument("--box-issues", type=Path, default=REVIEW_DIR / "box_issues.csv")
    parser.add_argument("--val", nargs="+", default=list(DEFAULT_VAL))
    parser.add_argument("--mask-drop-ratio", type=float, default=0.5)
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = DETECTION_ROOT / "exports" / args.version
    if out_root.exists():
        if not args.overwrite:
            sys.exit(f"{out_root} exists; versions are never overwritten silently (use --overwrite for a trial)")
        shutil.rmtree(out_root)

    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if any(row["camera"] == EVAL_CAMERA for row in manifest):
        sys.exit(f"{EVAL_CAMERA} is the evaluation camera and must not be exported")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    frame_boxes = boxes_by_frame(annotations)
    excluded = parse_excluded_frames(args.box_issues)
    masks = {} if args.no_masks else load_masks(MASK_DIR)

    canvases: Dict[str, np.ndarray] = {}
    counts = {"train": Counter(), "val": Counter()}
    frames: Counter = Counter()
    dropped_by_mask: Counter = Counter()
    skipped_excluded = 0
    for row_index, row in enumerate(manifest):
        if row_index in excluded:
            skipped_excluded += 1
            continue
        camera, sequence, frame_idx = row["camera"], row["seq"], int(row["frame_idx"])
        split = "val" if is_validation(camera, sequence, args.val) else "train"
        image = cv2.imread(str(IMAGES / row["image_path"]))
        if image is None:
            sys.exit(f"cannot read {row['image_path']}")
        height, width = image.shape[:2]

        mask = masks.get(camera)
        if mask is not None:
            if mask["image_size"] != [width, height]:
                sys.exit(f"{camera}: mask image_size {mask['image_size']} != frame {[width, height]}")
            if camera not in canvases:
                canvases[camera] = mask_canvas(mask["polygons"], width, height)
        kept: List[Box] = []
        for box in frame_boxes.get(row_index, []):
            if camera in canvases and masked_fraction(canvases[camera], box) >= args.mask_drop_ratio:
                dropped_by_mask[CLASS_NAMES[box[0]]] += 1
            else:
                kept.append(box)
        if camera in canvases:
            image = apply_mask(image, canvases[camera], kept)

        stem = frame_stem(camera, sequence, frame_idx)
        for kind in ("images", "labels"):
            (out_root / kind / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "images" / split / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            "".join(yolo_line(box, width, height) + "\n" for box in kept), encoding="utf-8")
        frames[split] += 1
        counts[split].update(CLASS_NAMES[box[0]] for box in kept)

    shutil.copy2(args.annotations, out_root / "annotations_snapshot.json")
    (out_root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out_root), "train": "images/train", "val": "images/val",
        "names": dict(CLASS_NAMES)}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    record = {
        "version": args.version, "frames": dict(frames), "excluded_group_deleted_frames": skipped_excluded,
        "boxes_per_class": {split: dict(counts[split]) for split in counts},
        "boxes_dropped_by_mask": dict(dropped_by_mask), "mask_drop_ratio": args.mask_drop_ratio,
        "masked_cameras": sorted(canvases), "val": list(args.val),
        "annotations_source": str(args.annotations), "box_issues": str(args.box_issues),
    }
    (out_root / "export_record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
