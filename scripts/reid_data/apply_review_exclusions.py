#!/usr/bin/env python3
"""
把人工審查匯出的確認清單合併進排除清單（見 CONTEXT.md「排除」、docs/adr/0006）。

讀取審查頁匯出的 CSV（欄位：class,identity,image,action；image 為空代表整個身分），
與既有的 dataset/reid/exclusions.csv 合併、去除重複並依鍵排序。只寫排除清單，
不搬移、改名或刪除任何裁切；產生資料集版本時由 build_reid_split.py 套用。
預設 dry-run（只列印將新增的項目），需要 --apply 才會寫入。

用法：
    python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv
    python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv --apply
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.manifest import EXCLUSION_FIELDS

SUPPORTED_ACTIONS = {"remove"}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (row.get(k) or "").strip() for k in EXCLUSION_FIELDS} for row in csv.DictReader(f)]


def merge_exclusions(existing: list[dict], confirmed: list[dict]) -> tuple[list[dict], list[dict]]:
    """回傳 (合併後的排除清單, 本次新增的項目)。"""
    unknown = sorted({r["action"] for r in confirmed} - SUPPORTED_ACTIONS)
    if unknown:
        raise ValueError(f"不支援的 action：{unknown}（只支援 {sorted(SUPPORTED_ACTIONS)}）")
    key = lambda r: tuple(r[k] for k in EXCLUSION_FIELDS)
    existing_keys = {key(r) for r in existing}
    added = sorted({key(r): r for r in confirmed if key(r) not in existing_keys}.values(), key=key)
    return sorted(existing + added, key=key), added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-csv", required=True)
    parser.add_argument("--exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "exclusions.csv"))
    parser.add_argument("--apply", action="store_true", help="實際寫入排除清單；不帶此旗標只列印計畫（dry-run）")
    args = parser.parse_args()

    exclusions_path = Path(args.exclusions)
    merged, added = merge_exclusions(read_rows(exclusions_path), read_rows(Path(args.confirm_csv)))
    for row in added:
        print(f"  + {row['class']}/{row['identity']}/{row['image'] or '(整個身分)'} {row['action']}")
    if not args.apply:
        print(f"dry-run：將新增 {len(added)} 筆排除，未寫入 {exclusions_path}（加 --apply 才會寫入）")
        return
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    print(f"已新增 {len(added)} 筆，排除清單共 {len(merged)} 筆 -> {exclusions_path}")


if __name__ == "__main__":
    main()
