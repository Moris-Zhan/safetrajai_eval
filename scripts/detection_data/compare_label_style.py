#!/usr/bin/env python3
"""Label-style gate and pseudo-label threshold check against the cctv_all8 GT (Markdown/detection/finetune-plan.md 3.1, 3.5).

The base model (models/yolo/yolo11x_base.pt, imgsz 640) is the common yardstick: the same statistics are computed for
the cctv_all8 tune GT and for a training export, and the export is compared with the GT.

  style       per class: label count, share of small labels by effective height (box height scaled to a 640 long side:
              <=16, 16-32, >32), median IoU / width / height ratio of labels matched to base predictions, share of labels
              with no same-class base prediction (conf >= 0.1, IoU >= 0.3), and share of confident base predictions
              (conf >= 0.5) with no same-class label (IoU >= 0.3) -- the vehicle-class mismatch metric -- and share of
              small base predictions (conf >= 0.3, effective height <= 16) with no same-class label. The "distance" to the
              GT is the mean absolute difference over the last three shares for every class present in the GT. The small
              label share is descriptive only: it mostly reflects the camera geometry, not the labeling style.
  thresholds  base pseudo-labels on the tune GT images at each threshold t: a box with conf >= t is a label, a box with
              0.1 <= conf < t is a gray (ignore) region. Per class: share of GT boxes missed (no label, no gray box,
              same class, IoU >= 0.5), share covered only by a gray box, and share of labels with no GT (extra).

Only the tune subset of cctv_all8 is read; holdout is reserved for the final acceptance and is refused.

    python scripts/detection_data/compare_label_style.py style --out runs/finetune/style/gt_tune.json
    python scripts/detection_data/compare_label_style.py style --export v5a --split train --max-images 800 \
        --reference runs/finetune/style/gt_tune.json --previous runs/finetune/style/ft_v3_train.json \
        --out runs/finetune/style/v5a_train.json
    python scripts/detection_data/compare_label_style.py thresholds --out runs/finetune/style/thresholds_tune.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from safetrajai.classes import CLASS_NAMES
from safetrajai.paths import PROJECT_ROOT

WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolo11x_base.pt"
TUNE_ROOT = PROJECT_ROOT / "dataset" / "eval" / "cctv_all8_subsets" / "tune" / "annotations_all8"
EXPORTS = PROJECT_ROOT / "dataset" / "detection" / "exports"
LOW_CONF = 0.1
HIGH_CONF = 0.5
STYLE_IOU = 0.3
MATCH_IOU = 0.5
SMALL_BINS = ((0.0, 16.0, "le16"), (16.0, 32.0, "16_32"), (32.0, float("inf"), "gt32"))
SMALL_CONF = 0.3
SHARE_KEYS = ("base_small_without_label", "label_without_base", "base_hi_without_label")

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_labels(path: Path, width: int, height: int) -> Dict[int, List[Box]]:
    boxes: Dict[int, List[Box]] = defaultdict(list)
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        boxes[cls].append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def tune_images() -> List[Path]:
    images = sorted(TUNE_ROOT.glob("*/images/*/*.jpg"))
    if not images:
        raise SystemExit(f"no images under {TUNE_ROOT}")
    return images


def export_images(version: str, split: str, max_images: int, seed: int) -> List[Path]:
    images = sorted((EXPORTS / version / "images" / split).glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {EXPORTS / version / 'images' / split}")
    if max_images and len(images) > max_images:
        random.Random(seed).shuffle(images)
        images = sorted(images[:max_images])
    return images


def predict(images: Sequence[Path], batch: int = 16) -> Iterator[Tuple[Path, int, int, List[Tuple[int, float, Box]]]]:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        for path, result in zip(chunk, model.predict(chunk, imgsz=640, conf=LOW_CONF, verbose=False)):
            height, width = result.orig_shape
            preds = [(int(c), float(s), tuple(b)) for c, s, b in
                     zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
            yield Path(path), width, height, preds


def greedy(preds: Sequence[Tuple[float, Box]], labels: Sequence[Box], thresh: float) -> List[Tuple[int, int, float]]:
    """Confidence-ordered one-to-one matching; returns (pred index, label index, iou)."""
    used, pairs = set(), []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i][0]):
        best, best_li = thresh, None
        for li, label in enumerate(labels):
            if li not in used:
                value = iou(preds[pi][1], label)
                if value >= best:
                    best, best_li = value, li
        if best_li is not None:
            used.add(best_li)
            pairs.append((pi, best_li, best))
    return pairs


def style_stats(images: Sequence[Path]) -> Dict[str, dict]:
    acc: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for image, width, height, preds in predict(images):
        labels = read_labels(label_path(image), width, height)
        scale = 640.0 / max(width, height)
        for cls in CLASS_NAMES:
            cls_labels = labels.get(cls, [])
            cls_preds = [(s, b) for c, s, b in preds if c == cls]
            pairs = greedy(cls_preds, cls_labels, STYLE_IOU)
            matched_labels = {li for _, li, _ in pairs}
            matched_preds = {pi for pi, _, _ in pairs}
            a = acc[cls]
            for li, box in enumerate(cls_labels):
                a["height"].append((box[3] - box[1]) * scale)
                a["label_without_base"].append(li not in matched_labels)
            for pi, li, value in pairs:
                pbox, lbox = cls_preds[pi][1], cls_labels[li]
                a["iou"].append(value)
                a["w_ratio"].append((lbox[2] - lbox[0]) / max(pbox[2] - pbox[0], 1e-6))
                a["h_ratio"].append((lbox[3] - lbox[1]) / max(pbox[3] - pbox[1], 1e-6))
            for pi, (score, box) in enumerate(cls_preds):
                if score >= HIGH_CONF:
                    a["base_hi_without_label"].append(pi not in matched_preds)
                if score >= SMALL_CONF and (box[3] - box[1]) * scale <= 16.0:
                    a["base_small_without_label"].append(pi not in matched_preds)
    out = {}
    for cls, name in CLASS_NAMES.items():
        a = acc[cls]
        heights = np.array(a["height"])
        entry = {"labels": int(len(heights)), "base_hi": len(a["base_hi_without_label"])}
        for low, high, key in SMALL_BINS:
            entry[f"small_{key}"] = float(np.mean((heights > low) & (heights <= high))) if len(heights) else None
        for key in ("iou", "w_ratio", "h_ratio"):
            entry[f"median_{key}"] = float(np.median(a[key])) if a[key] else None
        entry["base_small"] = len(a["base_small_without_label"])
        for key in SHARE_KEYS:
            entry[key] = float(np.mean(a[key])) if a[key] else None
        out[name] = entry
    return out


def distance(stats: Dict[str, dict], reference: Dict[str, dict]) -> Tuple[float, Dict[str, float]]:
    per_class = {}
    for name, ref in reference.items():
        if not ref["labels"]:
            continue
        diffs = [abs(stats[name][k] - ref[k]) for k in SHARE_KEYS
                 if stats.get(name, {}).get(k) is not None and ref.get(k) is not None]
        if diffs:
            per_class[name] = float(np.mean(diffs))
    return (float(np.mean(list(per_class.values()))) if per_class else float("nan")), per_class


def print_style(stats: Dict[str, dict], reference: Dict[str, dict] | None) -> None:
    keys = ("labels", "small_le16", "small_16_32", "median_iou", "median_w_ratio", "median_h_ratio",
            "label_without_base", "base_hi_without_label", "base_small_without_label")
    print("class        " + " ".join(f"{k:>22}" for k in keys))
    for name, entry in stats.items():
        cells = []
        for k in keys:
            value = entry.get(k)
            text = "-" if value is None else (f"{value:d}" if isinstance(value, int) else f"{value:.3f}")
            if reference and k != "labels" and value is not None and reference[name].get(k) is not None:
                text += f" (GT {reference[name][k]:.3f})"
            cells.append(f"{text:>22}")
        print(f"{name:12} " + " ".join(cells))


def run_style(args: argparse.Namespace) -> None:
    if args.export:
        images = export_images(args.export, args.split, args.max_images, args.seed)
        source = {"export": args.export, "split": args.split, "images": len(images), "seed": args.seed}
    else:
        images = tune_images()
        source = {"cctv_all8": "tune", "images": len(images)}
    stats = style_stats(images)
    report = {"source": source, "weights": str(WEIGHTS.relative_to(PROJECT_ROOT)), "stats": stats}
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["stats"] if args.reference else None
    if reference:
        total, per_class = distance(stats, reference)
        report["distance_to_gt"] = {"mean": total, "per_class": per_class}
    print_style(stats, reference)
    if reference:
        print(f"\ndistance to GT (mean |diff| of {', '.join(SHARE_KEYS)}): {report['distance_to_gt']['mean']:.4f}")
        if args.previous:
            previous = json.loads(args.previous.read_text(encoding="utf-8"))
            prev_total = previous.get("distance_to_gt", {}).get("mean")
            if prev_total is None:
                prev_total = distance(previous["stats"], reference)[0]
            verdict = "closer" if report["distance_to_gt"]["mean"] < prev_total else "NOT closer"
            report["previous"] = {"file": str(args.previous), "distance": prev_total, "verdict": verdict}
            print(f"previous ({args.previous.name}): {prev_total:.4f} -> {verdict} to GT")
    write(args.out, report)


def run_thresholds(args: argparse.Namespace) -> None:
    thresholds = sorted(args.thresholds)
    if thresholds[0] < LOW_CONF:
        raise SystemExit(f"thresholds must be >= {LOW_CONF}")
    acc = {t: {cls: defaultdict(int) for cls in CLASS_NAMES} for t in thresholds}
    for image, width, height, preds in predict(tune_images()):
        gts = read_labels(label_path(image), width, height)
        for t in thresholds:
            for cls in CLASS_NAMES:
                cls_gt = gts.get(cls, [])
                labels = [(s, b) for c, s, b in preds if c == cls and s >= t]
                grays = [(s, b) for c, s, b in preds if c == cls and s < t]
                pairs = greedy(labels, cls_gt, MATCH_IOU)
                matched = {gi for _, gi, _ in pairs}
                rest = [g for gi, g in enumerate(cls_gt) if gi not in matched]
                gray_covered = len(greedy(grays, rest, MATCH_IOU))
                a = acc[t][cls]
                a["gt"] += len(cls_gt)
                a["labels"] += len(labels)
                a["extra"] += len(labels) - len(pairs)
                a["gray_only"] += gray_covered
                a["missed"] += len(rest) - gray_covered
    report = {"subset": "tune", "weights": str(WEIGHTS.relative_to(PROJECT_ROOT)), "low_conf": LOW_CONF, "results": {}}
    print(f"{'t':>5} {'class':12} {'gt':>6} {'labels':>7} {'missed':>8} {'gray_only':>10} {'extra':>8}")
    for t in thresholds:
        rows = {}
        for cls, name in CLASS_NAMES.items():
            a = acc[t][cls]
            row = {"gt": a["gt"], "labels": a["labels"],
                   "missed": a["missed"] / a["gt"] if a["gt"] else None,
                   "gray_only": a["gray_only"] / a["gt"] if a["gt"] else None,
                   "extra": a["extra"] / a["labels"] if a["labels"] else None}
            rows[name] = row
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"{t:5.2f} {name:12} {row['gt']:6d} {row['labels']:7d} {fmt(row['missed']):>8} "
                  f"{fmt(row['gray_only']):>10} {fmt(row['extra']):>8}")
        report["results"][f"{t:.2f}"] = rows
    write(args.out, report)


def write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    style = sub.add_parser("style", help="label-style statistics of the tune GT or of an export")
    style.add_argument("--export", help="export version under dataset/detection/exports (omit for the cctv_all8 tune GT)")
    style.add_argument("--split", default="train", choices=["train", "val"])
    style.add_argument("--max-images", type=int, default=800, help="random sample of export images (0 = all)")
    style.add_argument("--seed", type=int, default=0)
    style.add_argument("--reference", type=Path, help="style report of the tune GT, to compute the distance")
    style.add_argument("--previous", type=Path, help="style report of the previous export, to check it got closer")
    style.add_argument("--out", type=Path, required=True)
    style.set_defaults(func=run_style)
    thresholds = sub.add_parser("thresholds", help="base pseudo-label thresholds against the tune GT")
    thresholds.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    thresholds.add_argument("--out", type=Path, required=True)
    thresholds.set_defaults(func=run_thresholds)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
