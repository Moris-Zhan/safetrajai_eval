"""
把 tag 鍵的 Homography 資料遷移成鏡頭鍵的設定檔（change `camera-homography-georef`，design §2）。

用法：
    python scripts/migrate_homography_config.py --dry-run   # 只列出會產生的檔案
    python scripts/migrate_homography_config.py             # 正式寫入

來源（不會被修改）：
    models/homography/matrices/H_<tag>.npy
    config/homography_checkpoints/<tag>.yaml
    models/homography/checkpoints/check_pts_{cctv,sat}_<tag>.npy
目標：
    config/homography/<camera>.yaml
    config/homography_checkpoints/<camera>.yaml
目標已存在且內容不同時，整個遷移停止，不寫入任何檔案。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from safetrajai.collision.homography.migration import apply_migration, plan_migration
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate tag-keyed homography data to camera-keyed configs.")
    parser.add_argument("--dry-run", action="store_true", help="只列出計畫，不寫入")
    parser.add_argument("--matrices-dir", type=Path, default=PROJECT_ROOT / "models/homography/matrices")
    parser.add_argument("--npy-checkpoints-dir", type=Path, default=PROJECT_ROOT / "models/homography/checkpoints")
    parser.add_argument("--checkpoints-dir", type=Path, default=CONFIG_DIR / "homography_checkpoints")
    parser.add_argument("--camera-dir", type=Path, default=CONFIG_DIR / "homography")
    args = parser.parse_args()

    steps = plan_migration(args.matrices_dir, args.npy_checkpoints_dir, args.checkpoints_dir, args.camera_dir)
    for step in steps:
        print(f"[{step.action:9s}] {step.target}  <- {step.source}")
    print("summary:", dict(Counter(s.action for s in steps)))

    if args.dry_run:
        return
    created = apply_migration(steps)
    print(f"wrote {len(created)} file(s)")


if __name__ == "__main__":
    main()
