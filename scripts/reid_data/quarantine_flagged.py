#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 問題身分／影像隔離腳本（Tier 3 移除機制）。

讀取人工確認過的清單 CSV（欄位：class,identity,image,action；image 留空代表整個身分，
非空代表僅該張影像；目前唯一支援的 action 是 "remove"），把對應項目從 crop_dataset/
搬到隔離目錄（保留 {class}/{identity}/ 結構），而非直接刪除，讓誤判可回復。預設
dry-run（只列印將搬移的項目，不動任何檔案），需明確帶 --apply 才實際搬移。每次
--apply 執行後在隔離目錄寫入一份 manifest 記錄本次操作。單張影像搬移後若原身分
資料夾清空，一併移除該空資料夾。對應
openspec/changes/reid-crop-identity-audit/specs/reid/crop-quarantine/spec.md。

confirm CSV 可手動撰寫，或用 generate_identity_review.py 產生的報告頁面勾選後匯出
（欄位 schema 兩者一致）。

用法：
    python scripts/reid_data/quarantine_flagged.py --confirm-csv path/to/quarantine_confirm.csv
    python scripts/reid_data/quarantine_flagged.py --confirm-csv path/to/quarantine_confirm.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def load_confirm_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("action") != "remove":
            raise ValueError(f"不支援的 action: {r.get('action')!r}（目前只支援 'remove'）: {r}")
        if not r.get("class") or not r.get("identity"):
            raise ValueError(f"確認清單缺少 class 或 identity: {r}")
    return rows


def plan_moves(rows: list[dict], crop_root: Path, quarantine_root: Path) -> list[dict]:
    """把確認清單展開成實際的 (src, dst) 搬移計畫，不做任何 I/O。"""
    moves = []
    for r in rows:
        cls, identity, image = r["class"], r["identity"], r.get("image") or ""
        src_identity_dir = crop_root / cls / identity
        dst_identity_dir = quarantine_root / cls / identity
        if image:
            src = src_identity_dir / image
            moves.append({"class": cls, "identity": identity, "image": image, "src": src, "dst": dst_identity_dir / image})
        else:
            if not src_identity_dir.is_dir():
                print(f"警告：找不到身分資料夾，略過：{src_identity_dir}")
                continue
            for img_path in sorted(src_identity_dir.iterdir()):
                if img_path.is_file():
                    moves.append({
                        "class": cls, "identity": identity, "image": img_path.name,
                        "src": img_path, "dst": dst_identity_dir / img_path.name,
                    })
    return moves


def apply_moves(moves: list[dict], quarantine_root: Path, confirm_csv: Path) -> Path:
    quarantined_identity_dirs: set[Path] = set()
    manifest_entries = []
    for m in moves:
        src, dst = m["src"], m["dst"]
        if not src.is_file():
            print(f"警告：來源檔案不存在，略過：{src}")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError as e:
            print(f"警告：搬移失敗，略過：{src} -> {dst}（{e}）")
            continue
        quarantined_identity_dirs.add(src.parent)
        manifest_entries.append({"class": m["class"], "identity": m["identity"], "image": m["image"], "src": str(src), "dst": str(dst)})

    # 清理搬移後變空的原身分資料夾，避免下游 discover_identities() 誤判
    for identity_dir in quarantined_identity_dirs:
        if identity_dir.is_dir() and not any(identity_dir.iterdir()):
            identity_dir.rmdir()

    manifest = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "confirm_csv": str(confirm_csv),
        "n_moved": len(manifest_entries),
        "entries": manifest_entries,
    }
    quarantine_root.mkdir(parents=True, exist_ok=True)
    manifest_path = quarantine_root / f"manifest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-csv", required=True)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument("--quarantine-root", default="dataset/reid/crop_dataset_quarantine")
    parser.add_argument("--apply", action="store_true", help="實際搬移檔案；不帶此旗標只列印計畫（dry-run）")
    args = parser.parse_args()

    confirm_csv = Path(args.confirm_csv)
    crop_root = Path(args.crop_root)
    quarantine_root = Path(args.quarantine_root)

    rows = load_confirm_rows(confirm_csv)
    moves = plan_moves(rows, crop_root, quarantine_root)

    if not args.apply:
        print(f"[dry-run] {len(moves)} 個檔案將被搬移到 {quarantine_root}（未實際執行，加 --apply 才會搬移）：")
        for m in moves[:20]:
            print(f"  {m['src']} -> {m['dst']}")
        if len(moves) > 20:
            print(f"  ...（共 {len(moves)} 筆，僅列出前 20 筆）")
        return

    manifest_path = apply_moves(moves, quarantine_root, confirm_csv)
    print(f"done: 搬移 {len(moves)} 個檔案到 {quarantine_root}，manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
