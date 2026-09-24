#!/usr/bin/env python3
"""
由裁切索引產生 Re-ID 資料集版本（manifest），不複製任何影像（spec: reid-dataset）。

輸入：
  - <crop-root>/_index/*.csv       裁切蒐集時寫出的索引（每張裁切一列）
  - <exclusions>                   人工審查後的排除清單（class,identity,image,action），可不存在
  - config/reid_split.yaml         切分規則（val 的鏡頭與日期、每身分張數上下限）
輸出（dataset/reid/splits/<version>/，版本已存在時報錯）：
  - manifest.csv                   crop_path,identity,class,camera,date,split
  - summary.json                   各類別／split／鏡頭的身分數與張數
  - split_rules.yaml               本次使用的規則複本

用法：
    python scripts/reid_data/build_reid_split.py --version v1
    python scripts/reid_data/build_reid_split.py --version v2 --exclusions dataset/reid/exclusions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from safetrajai.evaluation.dataset_layout import load_dataset_registry
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.manifest import MANIFEST_FIELDS, build_manifest, load_split_rules, summarize


def read_exclusions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eval_cameras() -> set[str]:
    return {cam for entry in load_dataset_registry().values() if entry.get("role") == "eval" for cam in entry.get("cameras", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="資料集版本名稱，例如 v1")
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "exclusions.csv"))
    parser.add_argument("--rules", default=str(CONFIG_DIR / "reid_split.yaml"))
    parser.add_argument("--splits-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "splits"))
    args = parser.parse_args()

    out_dir = Path(args.splits_root) / args.version
    if out_dir.exists():
        raise SystemExit(f"資料集版本已存在，不覆寫：{out_dir}")

    rules = load_split_rules(Path(args.rules))
    manifest = build_manifest(
        read_all_indexes(Path(args.crop_root)), read_exclusions(Path(args.exclusions)), rules, eval_cameras(),
    )

    out_dir.mkdir(parents=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    summary = summarize(manifest)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(args.rules, out_dir / "split_rules.yaml")
    print(f"{args.version}: {summary['identities']} identities, {summary['crops']} crops -> {out_dir}")


if __name__ == "__main__":
    main()
