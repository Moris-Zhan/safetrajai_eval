#!/usr/bin/env python3
"""
碎片端點特徵：只讀取碎片合併候選對的端點裁切（A 最後 n 張、B 最前 n 張），以 DINOv2 抽特徵
（與 detect_contamination.py 相同的 letterbox 224、L2 正規化）；見 change merge-occlusion-fragments design D2。
只寫特徵檔，不搬移、改名或刪除任何裁切，也不讀取候選以外的裁切。

輸入：
  - <crop-root>/_index/*.csv           裁切索引
  - <auto-exclusions>、<exclusions>    兩份排除清單都套用：整段排除的身分不參與、單張排除的裁切不當端點
  - config/reid/fragment_merge.yaml         候選條件（餘裕幀數、位置倍數、端點張數；track_buffer 幀數取自 track_config）
輸出：
  - <out>（預設 dataset/reports/reid/fragment_features.npz）：crop_path（依字典序）、features（float32）

用法：
    python scripts/reid_data/extract_fragment_features.py
    python scripts/reid_data/extract_fragment_features.py --batch-size 128
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.dataset_index import read_all_indexes
from safetrajai.reid.dinov2_features import embed_crops
from safetrajai.reid.fragment_merge import (
    collect_tracks,
    endpoint_crop_paths,
    find_candidates,
    load_merge_config,
    load_retention,
)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "crops"))
    parser.add_argument("--exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "exclusions.csv"))
    parser.add_argument("--auto-exclusions", default=str(PROJECT_ROOT / "dataset" / "reid" / "auto_exclusions.csv"))
    parser.add_argument("--config", default=str(CONFIG_DIR / "reid" / "fragment_merge.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dataset" / "reports" / "reid" / "fragment_features.npz"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_merge_config(Path(args.config))
    crop_root = Path(args.crop_root)
    rows = read_all_indexes(crop_root)
    if not rows:
        raise SystemExit(f"找不到任何裁切索引：{crop_root / '_index'}")
    tracks = collect_tracks(rows, read_rows(Path(args.exclusions)) + read_rows(Path(args.auto_exclusions)))
    track_buffers = load_retention(cfg["track_config"])
    candidates = find_candidates(tracks, track_buffers, cfg["margin_frames"], cfg["position_factor"])
    paths = endpoint_crop_paths(candidates, tracks, cfg["endpoint_crops"])
    print(f"tracks={len(tracks)} candidates={len(candidates)} endpoint_crops={len(paths)} "
          f"(of {len(rows)} indexed crops)")

    started = time.monotonic()
    features = embed_crops(crop_root, [{"crop_path": p} for p in paths], args.batch_size)
    elapsed = time.monotonic() - started

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, crop_path=np.array(paths, dtype=str), features=features.astype(np.float32))
    print(f"embedded {len(paths)} crops in {elapsed:.1f}s -> {out}")
    print("只讀取候選端點裁切，未改動任何裁切。")


if __name__ == "__main__":
    main()
