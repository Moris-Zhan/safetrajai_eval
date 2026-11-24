#!/usr/bin/env python3
"""
Compare detector weights on the validation split of an export, per camera and combined.

Evaluation protocol (Markdown/detection/dataset-build-rules.md section 8): use the weights of the
last epoch (weights/last.pt), never pick a checkpoint on the fair cameras, and report per-class
mAP50-95 for each validation camera and for all of them together.

Output: <out>/results.json ({"<subset>/<name>": {"frames", "map50", "map50_95", "per_class_ap50_95"}}),
<out>/val_<subset>.txt and data_<subset>.yaml (Ultralytics image lists), <out>/runs/ (Ultralytics val folders).

Usage (conda env track_eval):
    python scripts/detection_data/eval_finetune.py --export ft_v3 --out runs/finetune/eval_v3c \\
        dennis=models/yolo/yolo11x_dennis.pt v3=runs/finetune/ft_v3_e0/weights/last.pt
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTS = PROJECT_ROOT / "dataset" / "detection" / "exports"


def camera_of(image: Path) -> str:
    return image.name.split("_", 1)[0]


def write_subsets(export_root: Path, out: Path) -> Dict[str, Path]:
    images = sorted((export_root / "images" / "val").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no validation images in {export_root / 'images' / 'val'}")
    groups: Dict[str, List[Path]] = defaultdict(list)
    for image in images:
        groups[camera_of(image)].append(image)
    groups["combined"] = images
    names = yaml.safe_load((export_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    data_files = {}
    for subset, members in groups.items():
        listing = out / f"val_{subset}.txt"
        listing.write_text("\n".join(str(path) for path in members) + "\n", encoding="utf-8")
        data = out / f"data_{subset}.yaml"
        data.write_text(yaml.safe_dump({"path": str(export_root), "train": "images/train", "val": str(listing),
                                        "names": names}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        data_files[subset] = data
    return data_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="export version whose images/val is the evaluation set")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("weights", nargs="+", help="name=path pairs, e.g. dennis=models/yolo/yolo11x_dennis.pt")
    args = parser.parse_args()

    pairs = []
    for item in args.weights:
        name, sep, path = item.partition("=")
        if not sep or not Path(path).exists():
            raise SystemExit(f"bad weights argument {item!r} (expected name=existing_path)")
        pairs.append((name, Path(path).resolve()))

    args.out.mkdir(parents=True, exist_ok=True)
    data_files = write_subsets(EXPORTS / args.export, args.out.resolve())
    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    from ultralytics import YOLO

    for name, weights in pairs:
        model = YOLO(str(weights))
        for subset, data in data_files.items():
            metrics = model.val(data=str(data), split="val", imgsz=args.imgsz, batch=args.batch, workers=0,
                                verbose=False, plots=False, project=str(args.out.resolve() / "runs"),
                                name=f"{name}_{subset}", exist_ok=True)
            per_class = {model.names[int(cls)]: float(metrics.box.maps[int(cls)]) for cls in metrics.box.ap_class_index}
            results[f"{subset}/{name}"] = {
                "weights": str(weights), "frames": len(data.with_name(f"val_{subset}.txt").read_text().split()),
                "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class_ap50_95": per_class}
            results_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{subset}/{name}: mAP50-95 {metrics.box.map:.4f}", flush=True)


if __name__ == "__main__":
    main()
