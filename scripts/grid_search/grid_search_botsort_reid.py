# grid_search_botsort_reid.py
# -*- coding: utf-8 -*-
"""
Class-Aware Tracker 參數調優工具
================================

使用 sklearn-style GridSearchCV 概念，以 HOTA.AssA_mean 作為 scoring function，
對指定類別的 CLASS_TRACKING_PARAMS 進行窮舉搜尋。

設計原則：
  - 不依賴 sklearn 的 Estimator 介面（tracker 非 sklearn model），
    而是自行實作 ParameterGrid + 平行 / 循序評估的 GridSearchRunner。
  - Scoring function（safetrajai/grid_search/core.py）依序執行 tracking → save_mot_format
    → run_trackeval_mot，從 TrackEval 結果中萃取 HOTA.AssA_mean。
  - 每次試驗：動態 patch CLASS_TRACKING_PARAMS → 執行 tracking → 評估 → 還原。
  - 結果存入 CSV（可中斷續跑），並於結束後印出最佳參數組合。

使用方式：
  python scripts/grid_search/grid_search_botsort_reid.py --target-class scooter --config config/tracking/botsort_reid.yaml

進階：
  python scripts/grid_search/grid_search_botsort_reid.py \\
      --target-class scooter \\
      --max-videos 6 \\
      --output-root ./runs/grid_search \\
      --resume            # 跳過已有結果的組合
"""

from __future__ import annotations

import argparse

import numpy as np

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
            "models/yolo/yolo11x.pt"
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
            "models/yolo/yolo11x.pt"
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
            "models/yolo/yolo11x.pt"
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
            'models/yolo/yolo11x.pt',
            'models/yolo/yolo11x.pt',        
        ]    
    },
}


# =============================================================================
# 2. CLI 入口
# =============================================================================


def _runner_options(args: argparse.Namespace) -> dict:
    # 現行寫死值（這支腳本現況沒有 --conf/--iou/--imgsz/--frame_rate/--gmc_method CLI flag，
    # 統一簽章後也不新增；gmc_method="sparseOptFlow" 是因為現行 _run_tracking_for_video()
    # 完全沒有把 gmc_method 傳給 create_tracker()，實際落到
    # safetrajai/tracking/__init__.py::DEFAULT_TRACKER_CONFIG['gmc_method'] 的值，
    # 不是 "ecc"、也不是 grid_search_botsort.py 的 "sift"）：
    return dict(conf=get_min_conf_thresh(), iou=0.5, imgsz=1280, frame_rate=30, gmc_method="sparseOptFlow")


def main() -> None:
    parser = build_parser(
        default_config="config/tracking/botsort_reid.yaml",
        default_tracker="botsort",
    )
    run_grid_search(parser.parse_args(), PARAM_GRIDS, _runner_options)


if __name__ == "__main__":
    main()
