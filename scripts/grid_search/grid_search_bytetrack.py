# grid_search_bytetrack.py
# -*- coding: utf-8 -*-
"""
Class-Aware Tracker 參數調優工具（ByteTrack）
================================

使用 sklearn-style GridSearchCV 概念，以 HOTA.AssA_mean 作為 scoring function，
對指定類別的 CLASS_TRACKING_PARAMS 進行窮舉搜尋。

共用邏輯見 safetrajai/grid_search/core.py（ParameterGrid、scoring、GridSearchRunner 等）與 cli.py
（參數定義與執行流程）；本檔只保留 PARAM_GRIDS（此腳本專屬的搜尋空間）與此腳本專屬的預設值。

使用方式：
  python scripts/grid_search/grid_search_bytetrack.py --target-class scooter --config config/grid_search/grid_search_bytetrack.yaml

進階：
  python scripts/grid_search/grid_search_bytetrack.py \\
      --target-class scooter \\
      --max-videos 6 \\
      --resume            # 跳過已有結果的組合
"""

from __future__ import annotations

import argparse

from safetrajai.grid_search.cli import build_parser, run_grid_search
from safetrajai.tracking.config import get_min_conf_thresh


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


def _runner_options(args: argparse.Namespace) -> dict:
    # 現行寫死值（不透過 CLI 開放，維持與重構前完全相同的行為）：
    #   conf/iou/imgsz/frame_rate 沿用 _run_tracking_for_video() 原本的內部常數；
    #   gmc_method="none" 是佔位值——ByteTrack 不使用 GMC，
    #   core.py::_run_tracking_for_video() 只在 tracker_type == "botsort" 時才會用到它。
    return dict(conf=get_min_conf_thresh(), iou=0.5, imgsz=1280, frame_rate=30, gmc_method="none")


def main() -> None:
    parser = build_parser(
        default_config="config/grid_search/grid_search_bytetrack.yaml",
        default_tracker="bytetrack",
    )
    run_grid_search(parser.parse_args(), PARAM_GRIDS, _runner_options)


if __name__ == "__main__":
    main()
