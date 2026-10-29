"""Command line and driver shared by scripts/grid_search/grid_search_*.py.

Each script keeps only its PARAM_GRIDS, its default config/tracker and the options that are
specific to it; everything else (argument definitions, grid selection, video discovery, running
the search, writing the suggested patch) lives here.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from safetrajai.evaluation.config_loader import load_config
from safetrajai.grid_search.core import (
    GridSearchRunner,
    ParameterGrid,
    discover_grid_search_videos,
    generate_config_patch,
)
from safetrajai.tracking import CLASS_NAMES
from safetrajai.tracking import config as tracker_config_module   # 直接操作模組層級 dict
from safetrajai.tracking.config import get_track_config_path, load_track_config


def build_parser(
    *,
    default_config: str,
    default_tracker: str,
    add_tracker_options: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """Argument parser of a grid search script; add_tracker_options adds its extra options after --model."""
    parser = argparse.ArgumentParser(
        description="Tracker parameter grid search using HOTA.AssA_mean as scoring function"
    )
    parser.add_argument(
        "--target-class", default="scooter",
        choices=list(CLASS_NAMES.values()),
        help="要調優的類別名稱"
    )
    parser.add_argument("--config", default=default_config, help="流程用 config yaml（路徑、eval、class_tracking 等）")
    parser.add_argument(
        "--track-config",
        default=None,
        help=(
            "class-aware tracker 設定 YAML（與 track_eval 相同）。"
            "不指定時使用 --config 內建的 class_tracking 章節。"
        ),
    )
    parser.add_argument(
        "--tracker", default=default_tracker,
        choices=["botsort", "bytetrack"], help="Tracker 類型"
    )
    parser.add_argument(
        "--model", default=None,
        help="目前沒有作用：模型路徑一律讀取 --config 的 models.det_model_path"
    )
    if add_tracker_options is not None:
        add_tracker_options(parser)
    parser.add_argument(
        "--video-glob", default=None,
        help=(
            "影片路徑 glob pattern（覆蓋 config 的預設掃描邏輯）。\n"
            "例如: './data/source/annotations_all8/annotation_*/**.mp4'"
        )
    )
    parser.add_argument(
        "--max-videos", type=int, default=6,
        help="最多用幾支影片評估（加快搜尋速度，預設 6）"
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
    return parser


def run_grid_search(
    args: argparse.Namespace,
    param_grids: dict[str, dict[str, list]],
    runner_options: Callable[[argparse.Namespace], dict[str, Any]],
) -> None:
    """Run one grid search; runner_options(args) gives the script's conf/iou/imgsz/frame_rate/gmc_method.

    runner_options is called only when the runner is built, after the track config has been loaded.
    """
    load_track_config(args.track_config or args.config)
    print(f"Loaded track config: {get_track_config_path()}")

    # ── 載入 config ───────────────────────────────────────────────────────────
    cfg        = load_config(args.config)
    model_path = cfg.det_model_path

    # ── 確定搜尋網格 ──────────────────────────────────────────────────────────
    grid_def = param_grids.get(args.target_class, param_grids["_default"])

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
    result_csv = Path(args.result_csv) if args.result_csv else None
    runner = GridSearchRunner(
        param_grid   = grid_def,
        target_class = args.target_class,
        video_paths  = video_paths,
        model_path   = model_path,
        cfg          = cfg,
        tracker_type = args.tracker,
        **runner_options(args),
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
