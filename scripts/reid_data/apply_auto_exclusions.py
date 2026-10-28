#!/usr/bin/env python3
"""
由污染標記產生自動排除清單（見 CONTEXT.md「自動排除」）。

讀取裁切索引與 detect_contamination.py 的 motion_flags.csv、contamination_crops.csv，
依固定規則產生 dataset/reid/auto_exclusions.csv（欄位：class,identity,image,action,rule）：
  - static、motion_jump：整個身分排除
  - class_group：四輪與「人／二輪」之間的類別不一致，整個身分排除
  - appearance_outlier：只排除被標為離群的單張裁切
只處理裁切數 ≥ min_crops_per_identity（--rules）的身分。每次都從標記完整重新產生並覆寫整個檔案；
不寫入人工排除清單（exclusions.csv），不搬移、改名或刪除任何裁切。
預設 dry-run（只列印各規則的數量），需要 --apply 才會寫入。

用法：
    python scripts/reid_data/apply_auto_exclusions.py
    python scripts/reid_data/apply_auto_exclusions.py --apply
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.auto_exclusion import AUTO_EXCLUSION_FIELDS, RULE_ORDER, build_auto_exclusions
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.manifest import load_split_rules


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid"))
    parser.add_argument("--rules", default=str(CONFIG_DIR / "reid" / "reid_split.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dataset" / "reid" / "auto_exclusions.csv"))
    parser.add_argument("--apply", action="store_true", help="實際寫入自動排除清單；不帶此旗標只列印數量（dry-run）")
    args = parser.parse_args()

    index_rows = read_all_indexes(Path(args.crop_root))
    reports = Path(args.reports_dir)
    min_crops = load_split_rules(Path(args.rules))["min_crops_per_identity"]
    rows = build_auto_exclusions(
        index_rows, read_csv(reports / "motion_flags.csv"), read_csv(reports / "contamination_crops.csv"), min_crops,
    )

    crops_per_identity = Counter((r["class"], r["identity"]) for r in index_rows)
    whole = [r for r in rows if not r["image"]]
    for rule in RULE_ORDER:
        hits = [r for r in rows if rule in r["rule"].split("+")]
        n_crops = sum(crops_per_identity[(r["class"], r["identity"])] if not r["image"] else 1 for r in hits)
        print(f"  {rule:20s} {sum(not r['image'] for r in hits):6d} identities (whole) "
              f"{sum(bool(r['image']) for r in hits):6d} single crops, {n_crops:7d} crops")
    removed = sum(crops_per_identity[(r["class"], r["identity"])] for r in whole) + (len(rows) - len(whole))
    print(f"total: {len(whole)} whole identities, {len(rows) - len(whole)} single crops, "
          f"{removed} crops out of {len(index_rows)} (identities with >= {min_crops} crops only)")

    out = Path(args.out)
    if not args.apply:
        print(f"dry-run：未寫入 {out}（加 --apply 才會寫入）")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUTO_EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已寫入 {len(rows)} 筆自動排除 -> {out}；未改動人工排除清單與任何裁切。")


if __name__ == "__main__":
    main()
