# tracker_grid_search.py
# -*- coding: utf-8 -*-
"""
Class-Aware Tracker 參數調優工具（ByteTrack）
================================

使用 sklearn-style GridSearchCV 概念，以 HOTA.AssA_mean 作為 scoring function，
對指定類別的 CLASS_TRACKING_PARAMS 進行窮舉搜尋。

共用邏輯（ParameterGrid、scoring、GridSearchRunner 等）見 module/grid_search/core.py；
本檔只保留 PARAM_GRIDS（此腳本專屬的搜尋空間）與 main()（此腳本專屬的 argparse 定義）。

使用方式：
  python module/grid_search/grid_search_bytetrack.py --target-class scooter --config config/grid_search_bytetrack.yaml

進階：
  python module/grid_search/grid_search_bytetrack.py \\
      --target-class scooter \\
      --max-videos 6 \\
      --resume            # 跳過已有結果的組合
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

# ── 專案根目錄加入 path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module.tracker import CLASS_NAMES
from module.tracker import config as tracker_config_module   # 直接操作模組層級 dict
from module.tracker.config import (
    get_min_conf_thresh,
    get_track_config_path,
    load_track_config,
)
from evaluation.config_loader import load_config
from module.grid_search.core import (
    ParameterGrid,
    GridSearchRunner,
    generate_config_patch,
    discover_grid_search_videos,
)


# =============================================================================
# 1. 參數網格定義
#    每個鍵對應 CLASS_TRACKING_PARAMS 中的同名欄位；
#    值為候選列表，GridSearchRunner 會產生所有排列組合。
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 不同類別推薦的搜尋網格（可依需要增刪候選值）
# ──────────────────────────────────────────────────────────────────────────────
PARAM_GRIDS: dict[str, dict[str, list]] = {

    "scooter": {
        "match_thresh":             [0.6, 0.7, 0.8, 0.9],
        "new_track_thresh":         [0.1, 0.2, 0.3, 0.4],
        "second_match_thresh":      [0.4, 0.5, 0.6, 0.7],
        "unconfirmed_match_thresh": [0.6, 0.7, 0.8],
        "center_dist_weight":       [0.10, 0.15, 0.20, 0.25],
        "norm_factor":              [80.0, 100.0, 120.0],
        "fuse_score":               [True, False],
    },

    "person": {
        "match_thresh":             [0.6, 0.7, 0.80, 0.85, 0.90],
        "new_track_thresh":         [0.15, 0.20, 0.25],
        "second_match_thresh":      [0.40, 0.50, 0.60],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.20, 0.25, 0.30],
        "norm_factor":              [120.0, 150.0, 180.0],
        "fuse_score":               [True, False],
    },

    "car": {
        "match_thresh":             [0.6, 0.7, 0.80, 0.85, 0.90],
        "new_track_thresh":         [0.20, 0.25, 0.30],
        "second_match_thresh":      [0.50, 0.60, 0.70],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.40, 0.50, 0.60],
        "norm_factor":              [60.0, 80.0, 100.0],
        "fuse_score":               [True, False],
    },

    "bus": {
        "match_thresh":             [0.6, 0.7, 0.80, 0.85, 0.90],
        "new_track_thresh":         [0.15, 0.20, 0.25],
        "second_match_thresh":      [0.45, 0.50, 0.55],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.20, 0.25, 0.30],
        "norm_factor":              [100.0, 120.0, 150.0, 200.0],
        "fuse_score":               [True, False],
    },

    "truck": {
        "match_thresh":             [0.6, 0.7, 0.80, 0.85, 0.90],
        "new_track_thresh":         [0.15, 0.20, 0.25],
        "second_match_thresh":      [0.45, 0.50, 0.55],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.20, 0.25, 0.30],
        "norm_factor":              [100.0, 120.0, 150.0, 200.0],
        "fuse_score":               [True, False],
    },

    "small_truck": {
        "match_thresh":             [0.6, 0.7, 0.80, 0.85, 0.90],
        "new_track_thresh":         [0.15, 0.20, 0.25],
        "second_match_thresh":      [0.45, 0.50, 0.55],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.20, 0.25, 0.30],
        "norm_factor":              [100.0, 120.0, 150.0, 200.0],
        "fuse_score":               [True, False],
    },

    # 其餘類別使用通用網格
     "_default": {
        "match_thresh":             [0.6, 0.7, 0.80, 0.85, 0.90],
        "new_track_thresh":         [0.15, 0.20, 0.25],
        "second_match_thresh":      [0.45, 0.50, 0.55],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.20, 0.25, 0.30],
        "norm_factor":              [100.0, 120.0, 150.0, 200.0],
        "fuse_score":               [True, False],
    },
}


# =============================================================================
# 2. CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tracker parameter grid search using HOTA.AssA_mean as scoring function"
    )
    parser.add_argument(
        "--target-class", default="scooter",
        choices=list(CLASS_NAMES.values()),
        help="要調優的類別名稱"
    )
    parser.add_argument("--config", default="config/grid_search_bytetrack.yaml", help="流程用 config yaml（路徑、eval、class_tracking 等）")
    parser.add_argument(
        "--track-config",
        default=None,
        help=(
            "class-aware tracker 設定 YAML（與 track_eval 相同）。"
            "不指定時使用 --config 內建的 class_tracking 章節。"
        ),
    )
    parser.add_argument(
        "--tracker", default="bytetrack",
        choices=["botsort", "bytetrack"], help="Tracker 類型"
    )
    parser.add_argument(
        "--model", default=None,
        help="YOLO 模型路徑（預設從 config.yaml 的 det_model_path 讀取）"
    )
    parser.add_argument(
        "--video-glob", default=None,
        help=(
            "影片路徑 glob pattern（覆蓋 config 的預設掃描邏輯）。\n"
            "例如: './data/source/annotations_all8/annotation_*/**.mp4'"
        )
    )
    parser.add_argument(
        "--max-videos", type=int, default=6,
        help="最多用幾支影片評估（加快搜尋速度，預設 2）"
    )
    parser.add_argument(
        "--output-root", default="./runs/grid_search",
        help="Grid search 暫存與結果根目錄"
    )
    parser.add_argument(
        "--result-csv", default=None,
        help="結果 CSV 路徑（預設自動命名）"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="從已有的 CSV 繼續（跳過已完成的組合）"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="印出逐幀 tracking 進度"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只印出參數組合數量，不實際執行"
    )
    parser.add_argument(
        "--params-subset", default=None,
        help=(
            "只搜尋指定參數（逗號分隔），其餘使用當前 track_config.yaml 值。\n"
            "例如: 'match_thresh,new_track_thresh'"
        )
    )

    args = parser.parse_args()

    load_track_config(args.track_config or args.config)
    print(f"Loaded track config: {get_track_config_path()}")

    # ── 載入 config ───────────────────────────────────────────────────────────
    cfg        = load_config(args.config)
    model_path = cfg.det_model_path

    # ── 確定搜尋網格 ──────────────────────────────────────────────────────────
    grid_def = PARAM_GRIDS.get(args.target_class, PARAM_GRIDS["_default"])

    if args.params_subset:
        subset_keys = [k.strip() for k in args.params_subset.split(",")]
        grid_def = {k: v for k, v in grid_def.items() if k in subset_keys}
        if not grid_def:
            print(f"❌ --params-subset 中的參數 ({subset_keys}) 均不在 PARAM_GRIDS 中")
            sys.exit(1)

    pg = ParameterGrid(grid_def)
    print(f"Parameter grid: {len(pg)} combinations")
    print(f"  Parameters searched: {sorted(grid_def.keys())}")

    if args.dry_run:
        print("\n[dry-run] Combinations:")
        for i, p in enumerate(pg, 1):
            print(f"  [{i:4d}] {p}")
        sys.exit(0)

    # ── 收集影片路徑 ──────────────────────────────────────────────────────────
    video_paths = discover_grid_search_videos(cfg, args.video_glob, args.max_videos)

    if not video_paths:
        print(f"❌ No videos found. Check --video-glob or config.yaml paths.")
        sys.exit(1)

    print(f"\nVideos for evaluation ({len(video_paths)}):")
    for vp in video_paths:
        print(f"  {vp}")

    # ── 執行 Grid Search ──────────────────────────────────────────────────────
    # 現行寫死值（不透過 CLI 開放，維持與重構前完全相同的行為）：
    #   conf/iou/imgsz/frame_rate 沿用 _run_tracking_for_video() 原本的內部常數；
    #   gmc_method="none" 是佔位值——ByteTrack 不使用 GMC，
    #   core.py::_run_tracking_for_video() 只在 tracker_type == "botsort" 時才會用到它。
    result_csv = Path(args.result_csv) if args.result_csv else None
    runner = GridSearchRunner(
        param_grid   = grid_def,
        target_class = args.target_class,
        video_paths  = video_paths,
        model_path   = model_path,
        cfg          = cfg,
        tracker_type = args.tracker,
        conf         = get_min_conf_thresh(),
        iou          = 0.5,
        imgsz        = 1280,
        frame_rate   = 30,
        gmc_method   = "none",
        output_root  = Path(args.output_root),
        result_csv   = result_csv,
        resume       = args.resume,
        verbose      = args.verbose,
    )

    outcome = runner.run()
    runner.print_summary()

    # ── 產生 track_config / Python dict 片段建議 ───────────────────────────────
    original_params = copy.deepcopy(
        tracker_config_module.CLASS_TRACKING_PARAMS.get(args.target_class, {})
    )
    patch_str = generate_config_patch(
        target_class    = args.target_class,
        best_params     = outcome["best_params"],
        original_params = original_params,
    )
    print(f"\n📋 Suggested CLASS_TRACKING_PARAMS patch for '{args.target_class}' (貼到 track_config.yaml):")
    print("-" * 60)
    print(patch_str)
    print("-" * 60)

    # 同時寫入文字檔
    patch_file = Path(args.output_root) / f"best_params_{args.target_class}.txt"
    patch_file.write_text(
        f"# Best params for '{args.target_class}'\n"
        f"# Score (HOTA.AssA_mean) = {outcome['best_score']:.4f}\n"
        f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        f"CLASS_TRACKING_PARAMS update:\n{patch_str}\n",
        encoding="utf-8",
    )
    print(f"\n✅ Patch saved to: {patch_file}")


if __name__ == "__main__":
    main()
