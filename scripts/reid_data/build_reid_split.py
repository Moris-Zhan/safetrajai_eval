#!/usr/bin/env python3
"""
由裁切索引產生 Re-ID 資料集版本（manifest），不複製任何影像（spec: reid-dataset）。

輸入：
  - <crop-root>/_index/*.csv       裁切蒐集時寫出的索引（每張裁切一列）
  - <exclusions>                   人工排除清單（class,identity,image,action），可不存在
  - <auto-exclusions>              自動排除清單（class,identity,image,action,rule），可不存在；兩份清單都套用
  - --exclude-cameras              抽查不合格而整支排除的鏡頭（見 docs/adr/0007），其所有裁切不進入版本
  - --merges                       碎片合併清單（class,identity,merged_identity,...，build_fragment_merges.py 產生）；
                                   預設不套用。套用時在排除之後、最少張數之前把原始身分改名為合併後的身分
  - config/reid_split.yaml         切分規則（val 的鏡頭與日期、每身分張數上下限）
輸出（dataset/reid/splits/<version>/，版本已存在時報錯）：
  - manifest.csv                   crop_path,identity,class,camera,date,split
  - summary.json                   各類別／split／鏡頭的身分數與張數、兩份排除清單的筆數、被排除的鏡頭、
                                   合併清單（路徑、被合併的原始身分數、合併後的身分數）
  - split_rules.yaml               本次使用的規則複本

用法：
    python scripts/reid_data/build_reid_split.py --version v1
    python scripts/reid_data/build_reid_split.py --version v2 --exclude-cameras C000056
    python scripts/reid_data/build_reid_split.py --version v3 --merges dataset/reid/fragment_merges.csv
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
    parser.add_argument("--auto-exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "auto_exclusions.csv"))
    parser.add_argument("--exclude-cameras", nargs="*", default=[], help="整支排除的鏡頭，例如 C000056")
    parser.add_argument("--merges", default=None, help="碎片合併清單，例如 dataset/reid/fragment_merges.csv")
    parser.add_argument("--rules", default=str(CONFIG_DIR / "reid_split.yaml"))
    parser.add_argument("--splits-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "splits"))
    args = parser.parse_args()

    out_dir = Path(args.splits_root) / args.version
    if out_dir.exists():
        raise SystemExit(f"資料集版本已存在，不覆寫：{out_dir}")

    rules = load_split_rules(Path(args.rules))
    excluded_cameras = sorted(set(args.exclude_cameras))
    index_rows = [r for r in read_all_indexes(Path(args.crop_root)) if r["camera"] not in excluded_cameras]
    manual, auto = read_exclusions(Path(args.exclusions)), read_exclusions(Path(args.auto_exclusions))
    merge_rows = []
    if args.merges:
        if not Path(args.merges).exists():
            raise SystemExit(f"找不到合併清單：{args.merges}")
        merge_rows = read_exclusions(Path(args.merges))
    merges = {(r["class"], r["identity"]): r["merged_identity"] for r in merge_rows}
    manifest = build_manifest(index_rows, manual + auto, rules, eval_cameras(), merges)

    out_dir.mkdir(parents=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    summary = {**summarize(manifest), "exclusions": {"auto": len(auto), "manual": len(manual)},
               "excluded_cameras": excluded_cameras,
               "merges": {"path": args.merges, "merged_identities": len(merges),
                          "merged_into": len({(c, m) for (c, _), m in merges.items()})}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(args.rules, out_dir / "split_rules.yaml")
    print(f"{args.version}: {summary['identities']} identities, {summary['crops']} crops -> {out_dir}")


if __name__ == "__main__":
    main()
