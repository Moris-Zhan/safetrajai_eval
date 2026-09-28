#!/usr/bin/env python3
"""
污染身分偵測：以 DINOv2 特徵、每張裁切只讀一次，同時做類別中心點比對與軌跡內外觀離群值；
另外只讀索引的框與時間，標記靜止身分與運動跳動（spec: reid-dataset，design D6）。
只寫標記檔，不搬移、改名或刪除任何裁切。

只有裁切數 ≥ min_crops_per_identity（--rules，預設 config/reid_split.yaml）的身分會被標記；
不足的身分不會進 manifest，不需要審查。類別一致性只在人／二輪／四輪粗分組之間不一致時標記。

輸出（--out-dir，預設 dataset/reports/reid/）：
  - class_consistency.csv       class,identity,n_crops,agreement_rate,coarse_agreement_rate,flagged
  - appearance_outliers.csv     class,identity,n_crops,n_outliers,outlier_fraction,flagged
  - motion_flags.csv            class,identity,n_crops,duration_s,displacement,max_velocity_change,static,motion_jump,flagged
  - contamination_crops.csv     crop_path,class,identity,predicted_class,distance_to_identity_mean,is_outlier

DINOv2 權重第一次使用時從 HF Hub 下載（只下載，不上傳）。

用法：
    python scripts/reid_data/detect_contamination.py
    python scripts/reid_data/detect_contamination.py --percentile 98 --min-agreement 0.6
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.contamination import appearance_outliers, class_consistency, motion_flags
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.dinov2_features import embed_crops
from safetrajai.reid.manifest import load_split_rules


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["class", "identity"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-agreement", type=float, default=0.5)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--min-outliers", type=int, default=2)
    parser.add_argument("--min-fraction", type=float, default=0.2)
    parser.add_argument("--static-displacement", type=float, default=0.5)
    parser.add_argument("--static-duration", type=float, default=5.0)
    parser.add_argument("--jump-threshold", type=float, default=1.5)
    parser.add_argument("--rules", default=str(PROJECT_ROOT / "config" / "reid_split.yaml"))
    args = parser.parse_args()
    min_crops = load_split_rules(Path(args.rules))["min_crops_per_identity"]

    crop_root = Path(args.crop_root)
    rows = read_all_indexes(crop_root)
    if not rows:
        raise SystemExit(f"找不到任何裁切索引：{crop_root / '_index'}")
    features = embed_crops(crop_root, rows, args.batch_size)

    consistency, predicted = class_consistency(rows, features, args.min_agreement)
    outliers, distances, is_outlier = appearance_outliers(
        rows, features, args.percentile, args.min_outliers, args.min_fraction,
    )
    motion = [
        {**r, "flagged": r["static"] or r["motion_jump"]}
        for r in motion_flags(rows, args.static_displacement, args.static_duration, args.jump_threshold)
    ]
    for record in (*consistency, *outliers, *motion):
        record["flagged"] = bool(record["flagged"]) and record["n_crops"] >= min_crops

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "class_consistency.csv", consistency)
    write_csv(out_dir / "appearance_outliers.csv", outliers)
    write_csv(out_dir / "motion_flags.csv", motion)
    write_csv(out_dir / "contamination_crops.csv", [
        {"crop_path": r["crop_path"], "class": r["class"], "identity": r["identity"],
         "predicted_class": predicted[i], "distance_to_identity_mean": round(float(distances[i]), 5),
         "is_outlier": bool(is_outlier[i])}
        for i, r in enumerate(rows)
    ])
    print(f"flagged (identities with >= {min_crops} crops: {sum(r['n_crops'] >= min_crops for r in consistency)}): "
          f"class_consistency={sum(r['flagged'] for r in consistency)}, "
          f"appearance_outliers={sum(r['flagged'] for r in outliers)}, "
          f"static={sum(r['flagged'] and r['static'] for r in motion)}, "
          f"motion_jump={sum(r['flagged'] and r['motion_jump'] for r in motion)} "
          f"/ {len(consistency)} identities -> {out_dir}")


if __name__ == "__main__":
    main()
