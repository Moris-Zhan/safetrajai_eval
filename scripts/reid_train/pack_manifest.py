#!/usr/bin/env python3
"""
依資料集版本的 manifest，從掛載磁碟各讀一次裁切，打包成本機快取（design D2）。

輸入：
  - dataset/reid/splits/<version>/manifest.csv   只打包其中列出的裁切（crop_path 相對於 --crop-root）
輸出（--pack-dir，預設 ~/.cache/safetrajai/reid_packs/<version>/；已存在時報錯）：
  - crops.bin、index.csv、pack.json（版本、manifest 列數、打包張數、位元組數、從磁碟讀取張數、耗時）

打包會對 manifest 的每一列讀一次檔案（v2 約 15 萬張、1.1 GB），執行前先確認範圍。
--reuse-version 指定另一個版本的快取時，只從磁碟補讀該快取沒有的裁切（例如 v3 沿用 v2）。

用法：
    python scripts/reid_train/pack_manifest.py --version v2
    python scripts/reid_train/pack_manifest.py --version v3 --reuse-version v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.training.data import CropPack, default_pack_dir, pack_manifest, read_manifest, verify_pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="資料集版本名稱，例如 v2")
    parser.add_argument("--splits-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "splits"))
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--pack-dir", default=None, help="預設 ~/.cache/safetrajai/reid_packs/<version>/")
    parser.add_argument("--reuse-version", default=None, help="沿用此版本的快取，只補讀缺少的裁切")
    parser.add_argument("--reuse-pack-dir", default=None, help="沿用的快取目錄（預設依 --reuse-version）")
    args = parser.parse_args()

    manifest = Path(args.splits_root) / args.version / "manifest.csv"
    pack_dir = Path(args.pack_dir) if args.pack_dir else default_pack_dir(args.version)
    rows = read_manifest(manifest)
    print(f"[pack] {manifest}: {len(rows)} rows -> {pack_dir}")
    reuse_dir = args.reuse_pack_dir or (default_pack_dir(args.reuse_version) if args.reuse_version else None)
    reuse = CropPack(Path(reuse_dir)) if reuse_dir else None
    info = pack_manifest(rows, Path(args.crop_root), pack_dir, args.version, reuse=reuse)
    verify_pack(CropPack(pack_dir), rows)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
