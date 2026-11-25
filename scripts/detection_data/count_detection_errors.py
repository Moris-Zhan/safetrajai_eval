#!/usr/bin/env python3
"""
Count detector false positives and misses on the validation split of an export, per camera and class.

mAP hides how many extra boxes a model adds, so pseudo-label experiments (export_training_set.py
--tiny-person-pseudo) also report counts at fixed confidence thresholds. The first default threshold
(0.15) matches the person threshold the tracker uses (config/tracking/bytetrack.yaml class_conf_thresh).

Matching: per image and class, predictions sorted by confidence are matched greedily to the unmatched
ground-truth box with the highest IoU, if that IoU is >= --iou. Unmatched predictions are FP, unmatched
ground truth is FN; FN is also split by effective box height (<= --small-height and larger), where effective
height is the height at the --imgsz network input, the same measure as export_training_set.py --tiny-person-max-height.

Output: <out>/errors.json {"<name>/<camera or combined>/<conf>": {"<class>": {"tp", "fp", "fn", "fn_small", "fn_large"}}}

Usage (conda env track_eval):
    python scripts/detection_data/count_detection_errors.py --export ft_v3 --out runs/finetune/errors_v3 \\
        base=models/yolo/yolo11x_base.pt v3_tiny=runs/finetune/ft_v3_tiny_e0/weights/last.pt
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTS = PROJECT_ROOT / "dataset" / "detection" / "exports"

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds: Sequence[Tuple[Box, float]], gts: Sequence[Box], iou_thresh: float) -> Tuple[int, List[int], List[int]]:
    """Greedy one-to-one matching of one class in one image; returns (tp, indices of unmatched preds, indices of unmatched gt)."""
    unmatched = set(range(len(gts)))
    tp = 0
    false_positives: List[int] = []
    for pred_index in sorted(range(len(preds)), key=lambda i: -preds[i][1]):
        best, best_iou = None, iou_thresh
        for index in unmatched:
            value = iou(preds[pred_index][0], gts[index])
            if value >= best_iou:
                best, best_iou = index, value
        if best is None:
            false_positives.append(pred_index)
        else:
            unmatched.discard(best)
            tp += 1
    return tp, false_positives, sorted(unmatched)


def read_labels(path: Path, width: int, height: int) -> Dict[int, List[Box]]:
    boxes: Dict[int, List[Box]] = defaultdict(list)
    if path.exists():
        for line in path.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
            boxes[cls].append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="export version whose images/val is the evaluation set")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.15, 0.25, 0.5])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--small-height", type=float, default=32.0, help="FN split by effective height (pixels at --imgsz)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dump-fp", default=None, metavar="CLASS",
                        help="also write <out>/fp_<name>_<CLASS>.json: every false positive of this class at the lowest --conf "
                             "(image, camera, box, conf), for spot-checking whether they are ground-truth misses")
    parser.add_argument("weights", nargs="+", help="name=path pairs")
    args = parser.parse_args()

    pairs = []
    for item in args.weights:
        name, sep, path = item.partition("=")
        if not sep or not Path(path).exists():
            raise SystemExit(f"bad weights argument {item!r} (expected name=existing_path)")
        pairs.append((name, Path(path).resolve()))

    export_root = EXPORTS / args.export
    images = sorted((export_root / "images" / "val").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no validation images in {export_root / 'images' / 'val'}")
    names = yaml.safe_load((export_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "errors.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    from ultralytics import YOLO

    lowest = min(args.conf)
    dump_cls = None
    if args.dump_fp is not None:
        by_name = {v: k for k, v in names.items()}
        if args.dump_fp not in by_name:
            raise SystemExit(f"--dump-fp {args.dump_fp!r} is not a class in data.yaml ({sorted(by_name)})")
        dump_cls = by_name[args.dump_fp]
    for name, weights in pairs:
        model = YOLO(str(weights))
        counts: Dict[Tuple[str, float, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        dumped: List[dict] = []
        for image in images:
            camera = image.name.split("_", 1)[0]
            result = model.predict(str(image), imgsz=args.imgsz, conf=lowest, verbose=False)[0]
            height, width = result.orig_shape
            gts = read_labels(export_root / "labels" / "val" / f"{image.stem}.txt", width, height)
            xyxy = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = [int(c) for c in result.boxes.cls.cpu().tolist()]
            for conf in args.conf:
                for cls in names:
                    preds = [(tuple(b), s) for b, s, c in zip(xyxy, confs, classes) if c == cls and s >= conf]
                    tp, fp_indices, missed = match(preds, gts.get(cls, []), args.iou)
                    fp = len(fp_indices)
                    if cls == dump_cls and conf == lowest:
                        dumped.extend({"image": image.name, "camera": camera, "box": [round(v, 1) for v in preds[i][0]],
                                       "conf": round(preds[i][1], 4)} for i in fp_indices)
                    scale = args.imgsz / max(width, height)
                    small = sum(1 for i in missed if (gts[cls][i][3] - gts[cls][i][1]) * scale <= args.small_height)
                    for subset in (camera, "combined"):
                        bucket = counts[(subset, conf, cls)]
                        bucket["tp"] += tp
                        bucket["fp"] += fp
                        bucket["fn"] += len(missed)
                        bucket["fn_small"] += small
                        bucket["fn_large"] += len(missed) - small
        for (subset, conf, cls), bucket in counts.items():
            results.setdefault(f"{name}/{subset}/{conf}", {})[names[cls]] = dict(bucket)
        out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        if dump_cls is not None:
            fp_path = args.out / f"fp_{name}_{args.dump_fp}.json"
            fp_path.write_text(json.dumps(dumped, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{name}: {len(dumped)} {args.dump_fp} false positives at conf {lowest} -> {fp_path}", flush=True)
        print(f"{name}: done", flush=True)


if __name__ == "__main__":
    main()
