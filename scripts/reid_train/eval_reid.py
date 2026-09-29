#!/usr/bin/env python3
"""
以資料集版本的 val 評估 Re-ID 權重的整體與各類別 mAP、Rank-1（design D4）。

每個身分以固定種子取 1 張作 query，其餘全部（跨類別）作 gallery。val 身分都在同一序列內，
分數偏樂觀，只作參考，不作為追蹤啟用 Re-ID 的依據。

用法：
    python scripts/reid_train/eval_reid.py --weights models/class_reid_weights/dinov2s_v2/model.pt --version v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.training.data import CropPack, default_pack_dir, read_manifest, verify_pack
from safetrajai.reid.training.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="model.pt（safetrajai-reid-v1 格式）")
    parser.add_argument("--version", required=True, help="資料集版本名稱，例如 v2")
    parser.add_argument("--splits-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "splits"))
    parser.add_argument("--pack-dir", default=None, help="預設 ~/.cache/safetrajai/reid_packs/<version>/")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="結果 JSON 路徑（選用）")
    args = parser.parse_args()

    rows = read_manifest(Path(args.splits_root) / args.version / "manifest.csv")
    pack = CropPack(Path(args.pack_dir) if args.pack_dir else default_pack_dir(args.version))
    verify_pack(pack, rows)
    result = evaluate_checkpoint(Path(args.weights), [r for r in rows if r["split"] == args.split], pack,
                                 args.seed, num_workers=args.workers)
    result.update({"manifest_version": args.version, "split": args.split, "seed": args.seed})
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
