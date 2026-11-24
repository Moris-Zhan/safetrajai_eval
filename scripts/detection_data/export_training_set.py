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
  * --pseudo-labels exports the same frames, masks and split with the original pseudo-labels (control run).
  * --restore-group-deleted SNAPSHOT keeps those frames: each deleted group box is painted gray (ignore region)
    using its original coordinates from the pre-correction snapshot; labeled boxes inside it are not painted over.
  * --mined-person-frames adds frames mined from raw videos (mine_person_frames.py) after the person review
    (serve_person_review.py): reviewed person boxes keep their class (or the corrected box), false detections are
    dropped, "can't tell" boxes are painted gray, frames with an uncorrected partial box or a missed person are
    excluded (listed in partial_pending.json), other classes come from dennis at --other-class-conf. Mined frames
    are train only; frames of a validation sequence are skipped.
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
WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolo11x_dennis.pt"
VIDEOS = PROJECT_ROOT / "dataset" / "raw" / "videos"
GRAY = 114
DEFAULT_VAL = ("C000012", "C000056")
CLASS_IDS: Dict[str, int] = {name: class_id for class_id, name in CLASS_NAMES.items()}

Box = Tuple[int, float, float, float, float]  # class_id, x1, y1, x2, y2


def parse_excluded_frames(box_issues_csv: Path) -> Set[int]:
    excluded: Set[int] = set()
    with box_issues_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] == "group_deleted":
                excluded.update(int(frame) for frame in row["manifest_frames"].split(";") if frame)
    return excluded


def group_boxes_by_frame(snapshot: dict, box_issues_csv: Path) -> Dict[int, List[Box]]:
    """Boxes of the group_deleted tracks, only on the manifest frames the ledger lists for each track."""
    wanted: Dict[int, Set[int]] = {}
    with box_issues_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] == "group_deleted":
                wanted[int(row["track_id"])] = {int(frame) for frame in row["manifest_frames"].split(";") if frame}
    result: Dict[int, List[Box]] = defaultdict(list)
    for track in snapshot["tracks"]:
        frames = wanted.get(track["track_id"])
        if frames is None:
            continue
        for shape in track["shapes"]:
            if not shape["outside"] and shape["frame"] in frames:
                x1, y1, x2, y2 = shape["points"]
                result[shape["frame"]].append((0, float(x1), float(y1), float(x2), float(y2)))
    return dict(result)


def add_boxes_to_canvas(canvas: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    merged = canvas.copy()
    height, width = merged.shape
    for _, x1, y1, x2, y2 in boxes:
        merged[max(0, int(np.floor(y1))):min(height, int(np.ceil(y2))), max(0, int(np.floor(x1))):min(width, int(np.ceil(x2)))] = 1
    return merged


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


def pseudo_boxes(label_file: Path, width: int, height: int) -> List[Box]:
    boxes: List[Box] = []
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id, cx, cy, w, h = line.split()[:5]
                cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
                boxes.append((int(class_id), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def parse_verdict(value: str) -> Tuple[str, str]:
    value = {"not_person": "none"}.get(value, value)
    if value in ("none", "unsure"):
        return value, "person"
    if value == "partial":
        return "partial", "person"
    if value.startswith("partial:"):
        return "partial", value.split(":", 1)[1]
    return "normal", value


def resolve_review(persons: Sequence[Sequence[float]], answer: dict) -> Tuple[List[Box], List[Box], str]:
    """Reviewed labels, boxes to paint gray, and the reason the frame is unusable ('' when it is usable)."""
    verdicts, corrections = answer.get("boxes", {}), answer.get("corrections", {})
    kept: List[Box] = []
    gray: List[Box] = []
    reason = "missed_person" if answer.get("missed") else ""
    for index, person in enumerate(persons):
        status, name = parse_verdict(verdicts.get(str(index), "person"))
        x1, y1, x2, y2 = (float(v) for v in corrections.get(str(index), person[:4]))
        if status == "none":
            continue
        if status == "unsure":
            gray.append((0, x1, y1, x2, y2))
        elif status == "partial" and str(index) not in corrections:
            reason = reason or "partial_uncorrected"
        else:
            kept.append((CLASS_IDS[name], x1, y1, x2, y2))
    return kept, gray, reason


def box_iou(a: Box, b: Box) -> float:
    width = min(a[3], b[3]) - max(a[1], b[1])
    height = min(a[4], b[4]) - max(a[2], b[2])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter)


def fill_other_boxes(detections: Sequence[Box], reviewed: Sequence[Box], iou_threshold: float = 0.5) -> List[Box]:
    return [box for box in detections if all(box_iou(box, kept) < iou_threshold for kept in reviewed)]


def effective_height(box: Box, width: int, height: int) -> float:
    return (box[4] - box[2]) * 640 / max(width, height)


def tiny_person_boxes(detections: Sequence[Box], occupied: Sequence[Box], width: int, height: int,
                      max_height: float) -> List[Box]:
    small = [box for box in detections if box[0] == 0 and effective_height(box, width, height) <= max_height]
    return fill_other_boxes(small, occupied)


def repeated_train_list(image_paths: Sequence[str], has_person: Sequence[bool], repeat: int) -> List[str]:
    lines: List[str] = []
    for path, person in zip(image_paths, has_person):
        lines.extend([path] * (repeat if person else 1))
    return lines


def latest_answers(responses: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for line in responses.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["frame_id"]] = record
    return latest


def read_video_frame(camera: str, video: str, frame_idx: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(VIDEOS / camera / video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, image = capture.read()
    capture.release()
    if not ok:
        sys.exit(f"cannot read frame {frame_idx} of {video}")
    return image


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
    parser.add_argument("--pseudo-labels", action="store_true",
                        help="control export: use the original pseudo-label txts instead of the CVAT-corrected boxes")
    parser.add_argument("--restore-group-deleted", type=Path, default=None, metavar="SNAPSHOT",
                        help="keep the group_deleted frames: their original group boxes (taken from this pre-correction "
                             "snapshot) are painted gray as ignore regions instead of excluding the frame")
    parser.add_argument("--mined-person-frames", action="store_true",
                        help="add the reviewed person frames mined from raw videos (see the docstring)")
    parser.add_argument("--mined-candidates", type=Path, default=REVIEW_DIR / "person_candidates.json")
    parser.add_argument("--mined-responses", type=Path, default=REVIEW_DIR / "person_review_responses.jsonl")
    parser.add_argument("--other-class-conf", type=float, default=0.5,
                        help="dennis confidence for the non-person boxes of mined frames")
    parser.add_argument("--person-repeat", type=int, default=1,
                        help="list train frames that contain a person this many times in train_repeat.txt (data.yaml train)")
    parser.add_argument("--tiny-person-pseudo", action="store_true",
                        help="add dennis person boxes (unreviewed pseudo-labels) to pilot train frames when small enough")
    parser.add_argument("--tiny-person-conf", type=float, default=0.25)
    parser.add_argument("--tiny-person-max-height", type=float, default=32.0,
                        help="effective height (box height scaled to a 640 long side) at or below which a person counts as tiny")
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
    group_boxes: Dict[int, List[Box]] = {}
    if args.restore_group_deleted:
        group_boxes = group_boxes_by_frame(json.loads(args.restore_group_deleted.read_text(encoding="utf-8")), args.box_issues)
    masks = {} if args.no_masks else load_masks(MASK_DIR)

    canvases: Dict[str, np.ndarray] = {}
    counts = {"train": Counter(), "val": Counter()}
    frames: Counter = Counter()
    dropped_by_mask: Counter = Counter()

    def emit(camera: str, stem: str, split: str, image: np.ndarray, source_boxes: Sequence[Box], gray_boxes: Sequence[Box]) -> None:
        height, width = image.shape[:2]
        mask = masks.get(camera)
        if mask is not None:
            if mask["image_size"] != [width, height]:
                sys.exit(f"{camera}: mask image_size {mask['image_size']} != frame {[width, height]}")
            if camera not in canvases:
                canvases[camera] = mask_canvas(mask["polygons"], width, height)
        kept: List[Box] = []
        for box in source_boxes:
            if camera in canvases and masked_fraction(canvases[camera], box) >= args.mask_drop_ratio:
                dropped_by_mask[CLASS_NAMES[box[0]]] += 1
            else:
                kept.append(box)
        painted_canvas = canvases.get(camera)
        if gray_boxes:
            painted_canvas = add_boxes_to_canvas(
                painted_canvas if painted_canvas is not None else mask_canvas([], width, height), gray_boxes)
        if painted_canvas is not None:
            image = apply_mask(image, painted_canvas, kept)
        for kind in ("images", "labels"):
            (out_root / kind / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "images" / split / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            "".join(yolo_line(box, width, height) + "\n" for box in kept), encoding="utf-8")
        frames[split] += 1
        counts[split].update(CLASS_NAMES[box[0]] for box in kept)

    skipped_excluded = 0
    restored_frames = 0
    tiny_model = None
    tiny_added = 0
    for row_index, row in enumerate(manifest):
        if row_index in excluded and not args.restore_group_deleted:
            skipped_excluded += 1
            continue
        camera, sequence, frame_idx = row["camera"], row["seq"], int(row["frame_idx"])
        image = cv2.imread(str(IMAGES / row["image_path"]))
        if image is None:
            sys.exit(f"cannot read {row['image_path']}")
        height, width = image.shape[:2]
        source_boxes = (pseudo_boxes(DETECTION_ROOT / "labels" / row["label_path"], width, height)
                        if args.pseudo_labels else frame_boxes.get(row_index, []))
        restored_frames += row_index in group_boxes
        split = "val" if is_validation(camera, sequence, args.val) else "train"
        if args.tiny_person_pseudo and split == "train":
            if tiny_model is None:
                from ultralytics import YOLO

                tiny_model = YOLO(str(WEIGHTS))
            result = tiny_model.predict(image, imgsz=640, conf=args.tiny_person_conf, verbose=False)[0]
            detections = [(int(cls), *map(float, xyxy)) for cls, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())]
            added = tiny_person_boxes(detections, list(source_boxes) + group_boxes.get(row_index, []), width, height,
                                      args.tiny_person_max_height)
            tiny_added += len(added)
            source_boxes = list(source_boxes) + added
        emit(camera, frame_stem(camera, sequence, frame_idx), split, image, source_boxes, group_boxes.get(row_index, []))

    mined_record: Dict[str, object] = {}
    if args.mined_person_frames:
        from ultralytics import YOLO

        model = YOLO(str(WEIGHTS))
        candidates = json.loads(args.mined_candidates.read_text(encoding="utf-8"))
        answers = latest_answers(args.mined_responses)
        excluded_reasons: Counter = Counter()
        reviewed_counts: Counter = Counter()
        filled_counts: Counter = Counter()
        partial_pending: List[dict] = []
        used = skipped_val = unanswered = 0
        for number, candidate in enumerate(candidates):
            answer = answers.get(f"f{number}")
            camera, video, frame_idx = candidate["camera"], candidate["video"], candidate["frame_idx"]
            sequence = Path(video).stem
            if answer is None:
                unanswered += 1
                continue
            if is_validation(camera, sequence, args.val):
                skipped_val += 1
                continue
            reviewed, gray, reason = resolve_review(candidate["persons"], answer)
            stem = "m_" + frame_stem(camera, sequence, frame_idx)
            if reason:
                excluded_reasons[reason] += 1
                if reason == "partial_uncorrected":
                    partial_pending.append({"frame": stem, "camera": camera, "video": video, "frame_idx": frame_idx,
                                            "boxes": {k: v for k, v in answer["boxes"].items() if v.startswith("partial")}})
                continue
            image = read_video_frame(camera, video, frame_idx)
            result = model.predict(image, imgsz=640, conf=args.other_class_conf, verbose=False)[0]
            detections = [(int(cls), *map(float, xyxy)) for cls, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()) if int(cls) != 0]
            filled = fill_other_boxes(detections, reviewed)
            reviewed_counts.update(CLASS_NAMES[box[0]] for box in reviewed)
            filled_counts.update(CLASS_NAMES[box[0]] for box in filled)
            emit(camera, stem, "train", image, reviewed + filled, gray)
            used += 1
        (out_root / "partial_pending.json").write_text(json.dumps(partial_pending, indent=1, ensure_ascii=False), encoding="utf-8")
        mined_record = {"frames_used": used, "excluded": dict(excluded_reasons), "skipped_validation_sequence": skipped_val,
                        "unanswered": unanswered, "boxes_reviewed": dict(reviewed_counts), "boxes_dennis_filled": dict(filled_counts),
                        "other_class_conf": args.other_class_conf, "responses": str(args.mined_responses)}

    shutil.copy2(args.annotations, out_root / "annotations_snapshot.json")
    train_entry = "images/train"
    if args.person_repeat > 1:
        train_images = sorted((out_root / "images" / "train").glob("*.jpg"))
        has_person = [any(line.startswith("0 ") for line in
                          (out_root / "labels" / "train" / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines())
                      for path in train_images]
        (out_root / "train_repeat.txt").write_text(
            "\n".join(repeated_train_list([str(path) for path in train_images], has_person, args.person_repeat)) + "\n",
            encoding="utf-8")
        train_entry = str(out_root / "train_repeat.txt")
    (out_root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out_root), "train": train_entry, "val": "images/val",
        "names": dict(CLASS_NAMES)}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    record = {
        "version": args.version, "frames": dict(frames), "excluded_group_deleted_frames": skipped_excluded,
        "restored_group_deleted_frames": restored_frames,
        "group_deleted_snapshot": str(args.restore_group_deleted) if args.restore_group_deleted else None,
        "boxes_per_class": {split: dict(counts[split]) for split in counts},
        "boxes_dropped_by_mask": dict(dropped_by_mask), "mask_drop_ratio": args.mask_drop_ratio,
        "masked_cameras": sorted(canvases), "val": list(args.val),
        "mined_person_frames": mined_record or None,
        "person_repeat": args.person_repeat,
        "tiny_person_pseudo": ({"boxes_added": tiny_added, "conf": args.tiny_person_conf,
                                "max_effective_height": args.tiny_person_max_height}
                               if args.tiny_person_pseudo else None),
        "pseudo_labels": args.pseudo_labels, "annotations_source": str(args.annotations), "box_issues": str(args.box_issues),
    }
    (out_root / "export_record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
