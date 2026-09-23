# tracker_grid_search.py
# -*- coding: utf-8 -*-
"""
Class-Aware Tracker 參數調優工具
================================

使用 sklearn-style GridSearchCV 概念，以 HOTA.AssA_mean 作為 scoring function，
對指定類別的 CLASS_TRACKING_PARAMS 進行窮舉搜尋。

設計原則：
  - 不依賴 sklearn 的 Estimator 介面（tracker 非 sklearn model），
    而是自行實作 ParameterGrid + 平行 / 循序評估的 GridSearchRunner。
  - Scoring function 直接呼叫現有的 run_class_aware_tracking → save_mot_format
    → run_trackeval_mot 流程，從 TrackEval 結果中萃取 HOTA.AssA_mean。
  - 每次試驗：動態 patch CLASS_TRACKING_PARAMS → 執行 tracking → 評估 → 還原。
  - 結果存入 CSV（可中斷續跑），並於結束後印出最佳參數組合。

使用方式：
  python tracker_grid_search.py --target-class scooter --config config.yaml

進階：
  python tracker_grid_search.py \\
      --target-class scooter \\
      --video-dir ./data/source \\
      --output-root ./runs/track_eval \\
      --n-jobs 1 \\
      --resume            # 跳過已有結果的組合
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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

    # Grid search 參數直接與 config_BotSort_0413.py (固定值, 非候選列表) 完全一致設定:
    "car": {
        "match_thresh":             [0.7],
        "new_track_thresh":         [0.25],
        "second_match_thresh":      [0.6],
        "unconfirmed_match_thresh": [0.9],
        "center_dist_weight":       [0.4],
        "norm_factor":              [80.0],
        "fuse_score":               [True],
        "use_reid":                 [True],
        "proximity_thresh":         [0.5],
        "appearance_thresh":        [0.1],
        "model": [
            "models/yolo/yolo11x_dennis.pt"
        ],
    },

    "bus": {
        "match_thresh":             [0.7],
        "new_track_thresh":         [0.15],
        "second_match_thresh":      [0.55],
        "unconfirmed_match_thresh": [0.95],
        "center_dist_weight":       [0.25],
        "norm_factor":              [200.0],
        "fuse_score":               [False],
        "use_reid":                 [True],
        "proximity_thresh":         [0.1],
        "appearance_thresh":        [0.1],
        "model": [
            "models/yolo/yolo11x_dennis.pt"
        ],
    },

    "small_truck": {
        "match_thresh":             [0.6],
        "new_track_thresh":         [0.15],
        "second_match_thresh":      [0.45],
        "unconfirmed_match_thresh": [0.9],
        "center_dist_weight":       [0.2],
        "norm_factor":              [100.0],
        "fuse_score":               [True],
        "use_reid":                 [True],
        "proximity_thresh":         [0.3],
        "appearance_thresh":        [0.1],
        "model": [
            "models/yolo-reid/yolo11x-reid-ultra.pt"
        ],
    },

    "truck": {
        "match_thresh":             [0.6],
        "new_track_thresh":         [0.15],
        "second_match_thresh":      [0.45],
        "unconfirmed_match_thresh": [0.9],
        "center_dist_weight":       [0.2],
        "norm_factor":              [100.0],
        "fuse_score":               [False],
        "use_reid":                 [True],
        "proximity_thresh":         [0.6],
        "appearance_thresh":        [0.1],
        "model": [
            "models/yolo/yolo11x.pt"
        ],
    },

    "person": {
        "match_thresh":             [0.8, 0.85, 0.9, 0.95],           # 聚焦高門檻區間
        "new_track_thresh":         [0.05, 0.1, 0.15],         # 往更低探索
        "second_match_thresh":      [0.2, 0.3, 0.4],           # 往更低探索
        "unconfirmed_match_thresh": [0.6, 0.7, 0.75],          # 往更低探索
        "center_dist_weight":       [0.05, 0.1, 0.15],         # 往更低探索
        "norm_factor":              [60, 80, 100],              # 往更低探索
        "fuse_score":               [True],                     # 固定
        # ReID 相關
        "use_reid":                 [True],                     # 固定
        "proximity_thresh":         [0.05, 0.1, 0.15],         # 往更低探索
        "appearance_thresh":        [0.05, 0.1, 0.15],         # 往更低探索
        "model": [
            # "models/yolo/yolo11x.pt",
            "models/class_reid_weights/person/best_full_model.pt"
        ],
    },

    "scooter": {
        "match_thresh":             [0.9, 1.0],                # 固定高門檻，加入 1.0 試探
        "new_track_thresh":         [0.1, 0.2, 0.3],           # 聚焦低值區間，排除 0.4~0.5
        "second_match_thresh":      [0.4, 0.5, 0.6],           # 在最佳值 0.5 附近精搜
        "unconfirmed_match_thresh": [0.6, 0.7],                # 固定低值，往 0.6 試探
        "center_dist_weight":       [0.1],                     # 固定，兩組一致
        "norm_factor":              [150, 200, 250],            # 在 200 附近加密，補上 150/250
        "fuse_score":               [True],                     # 固定
        "use_reid":                 [True],                     # 固定
        "proximity_thresh":         [0.2, 0.3, 0.4],           # 在最佳值 0.3 附近精搜
        "appearance_thresh":        [0.1],                      # 固定
        "model": [
            # "models/yolo-reid/yolo11x-reid-ultra.pt",
            "models/class_reid_weights/scooter/best_full_model.pt"
        ],
    },


    # 其餘類別使用通用網格
    "_default": {
        "match_thresh":             [0.80, 0.85, 0.90],
        "new_track_thresh":         [0.15, 0.20, 0.25],
        "second_match_thresh":      [0.45, 0.50, 0.55],
        "unconfirmed_match_thresh": [0.85, 0.90, 0.95],
        "center_dist_weight":       [0.20, 0.25, 0.30],
        "norm_factor":              [100.0, 150.0, 200.0],
        "fuse_score":               [True],
        # ReID 相關（僅 use_reid=True 時有效）
        "use_reid":                 [True],
        "proximity_thresh":         [round(x, 2) for x in np.arange(0.1, 0.7, 0.1)],   
        "appearance_thresh":        [round(x, 2) for x in np.arange(0.1, 1.0, 0.2)],      
        "model": [
            'models/yolo/yolo11x_dennis.pt',
            'models/yolo-reid/yolo11x-reid-ultra.pt',
            'models/yolo/yolo11x.pt',        
        ]    
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
    parser.add_argument("--config", default="config/botsort_reid.yaml", help="流程用 config yaml（路徑、eval、class_tracking 等）")
    parser.add_argument(
        "--track-config",
        default=None,
        help=(
            "class-aware tracker 設定 YAML（與 track_eval 相同）。"
            "不指定時使用 --config 內建的 class_tracking 章節。"
        ),
    )
    parser.add_argument(
        "--tracker", default="botsort",
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
    # 現行寫死值（這支腳本現況沒有 --conf/--iou/--imgsz/--frame_rate/--gmc_method CLI flag，
    # 統一簽章後也不新增；gmc_method="sparseOptFlow" 是因為現行 _run_tracking_for_video()
    # 完全沒有把 gmc_method 傳給 create_tracker()，實際落到
    # module/tracker/__init__.py::DEFAULT_TRACKER_CONFIG['gmc_method'] 的值，
    # 不是 "ecc"、也不是 grid_search_botsort.py 的 "sift"）：
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
        gmc_method   = "sparseOptFlow",
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