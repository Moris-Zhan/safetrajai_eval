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
import csv
import itertools
import json
import os
import sys
import time
import traceback
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import random
import string

# ── 專案根目錄加入 path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module.tracker import create_tracker, CLASS_NAMES
from module.tracker import config as tracker_config_module   # 直接操作模組層級 dict
from module.tracker.config import (
    get_conf_thresh_for_class,
    get_min_conf_thresh,
    get_track_config_path,
    load_track_config,
)
from evaluation import dataset_layout
from evaluation.mot_format import save_mot_format
from evaluation.config_loader import load_config
from evaluation.trackeval_runner import run_trackeval_mot
from evaluation.excel_export import flatten_any

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


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
# 2. ParameterGrid — 產生所有參數組合（等同 sklearn.model_selection.ParameterGrid）
# =============================================================================

class ParameterGrid:
    """
    把 {param: [v1, v2, ...], ...} 展開成所有排列組合的清單。

    Examples
    --------
    >>> list(ParameterGrid({"a": [1, 2], "b": [3, 4]}))
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """

    def __init__(self, param_grid: dict[str, list]):
        self.param_grid = param_grid
        self._keys   = sorted(param_grid.keys())          # 固定順序以利 resume 比對
        self._values = [param_grid[k] for k in self._keys]

    def __iter__(self):
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, combo))

    def __len__(self):
        n = 1
        for v in self._values:
            n *= len(v)
        return n

    def combo_id(self, params: dict) -> str:
        """回傳參數組合的唯一字串 ID（供 CSV resume 比對用）。"""
        return json.dumps({k: params[k] for k in self._keys}, sort_keys=True)


# =============================================================================
# 3. Scoring Function — 從 TrackEval results dict 萃取 HOTA.AssA_mean
# =============================================================================

def extract_assa_mean(
    trackeval_results: dict,
    target_class: str,
    tracker_variant: str,
) -> float | None:
    """
    從 TrackEval evaluator.evaluate() 的原始回傳結果中，
    萃取指定類別的 HOTA.AssA_mean（所有 alpha 門檻的平均值）。

    TrackEval result dict 的巢狀結構：
      results
        [dataset_name]          e.g. "MotChallenge2DBox"
          [tracker_name]        e.g. "YOLO11x__scooter"
            [seq_name]          e.g. "C000025_..."  or "COMBINED_SEQ"
              [class_name]      e.g. "scooter"
                [metric_family] e.g. "HOTA"
                  [metric_key]  e.g. "AssA"  → np.ndarray(shape=(19,)) 各 alpha 分數

    AssA_mean = mean(AssA array over all alpha thresholds)

    Returns
    -------
    float | None
        萃取成功回傳 float；任何步驟失敗回傳 None。
    """
    if trackeval_results is None:
        return None

    # evaluator.evaluate() 回傳 (results_dict, messages_dict) tuple
    # run_trackeval_mot 直接 return 這個 tuple，因此取 [0] 得到 results dict
    if isinstance(trackeval_results, tuple):
        trackeval_results = trackeval_results[0]
 
    if not isinstance(trackeval_results, dict):
        return None

    try:
        # 第一層：dataset（通常只有一個）
        for dataset_name, dataset_val in trackeval_results.items():
            if not isinstance(dataset_val, dict):
                continue

            # 第二層：tracker variant
            trk_val = dataset_val.get(tracker_variant)
            if trk_val is None:
                # fallback: 取第一個 tracker
                for v in dataset_val.values():
                    if isinstance(v, dict):
                        trk_val = v
                        break
            if trk_val is None:
                continue

            # 第三層：sequence（取 COMBINED_SEQ 優先，否則取第一個）
            seq_val = trk_val.get("COMBINED_SEQ")
            if seq_val is None:
                for v in trk_val.values():
                    if isinstance(v, dict):
                        seq_val = v
                        break
            if seq_val is None:
                continue

            # 第四層：class
            cls_val = seq_val.get(target_class)
            if cls_val is None:
                # TrackEval 有時用 cls index string
                for v in seq_val.values():
                    if isinstance(v, dict):
                        cls_val = v
                        break
            if cls_val is None:
                continue

            # 第五層：metric family → HOTA → AssA
            hota_val = cls_val.get("HOTA")
            if hota_val is None:
                continue

            assa = hota_val.get("AssA")
            if assa is None:
                continue

            # AssA 是 shape=(19,) 的 np.ndarray（對應 alpha=0.05~0.95 step 0.05）
            if isinstance(assa, np.ndarray):
                return float(assa.mean())
            return float(assa)

    except Exception as e:
        print(f"  [score] extract_assa_mean failed: {e}")

    return None


def score_params(
    params: dict,
    target_class: str,
    video_paths: list[str],
    model_path: str,
    cfg,                        # AppConfig
    tracker_type: str = "bytetrack",
    output_root: Path = Path("./runs/gs_tmp"),
    verbose: bool = False,
) -> float:
    """
    Scoring function（sklearn 慣例：分數越高越好）。

    流程：
      1. Patch CLASS_TRACKING_PARAMS[target_class] with params
      2. 對所有 video_paths 執行 tracking → save_mot_format
      3. 呼叫 run_trackeval_mot → 取得 HOTA.AssA_mean
      4. 還原 CLASS_TRACKING_PARAMS
      5. 回傳平均分數

    Parameters
    ----------
    params        : 本次試驗的參數字典
    target_class  : 要調優的類別名稱（如 "scooter"）
    video_paths   : 評估用的影片路徑清單
    model_path    : YOLO 模型路徑
    cfg           : AppConfig（含 benchmark_name, split_name 等）
    tracker_type  : "botsort" 或 "bytetrack"
    output_root   : 暫存 tracking 輸出的根目錄
    verbose       : 是否印出逐幀進度

    Returns
    -------
    float
        所有序列的 HOTA.AssA_mean 平均值；若無有效結果回傳 -inf。
    """
    # ── 1. Patch CLASS_TRACKING_PARAMS ───────────────────────────────────────
    original_params = copy.deepcopy(
        tracker_config_module.CLASS_TRACKING_PARAMS.get(target_class, {})
    )
    patched = copy.deepcopy(original_params)
    patched.update(params)
    tracker_config_module.CLASS_TRACKING_PARAMS[target_class] = patched

    scores: list[float] = []

    try:
        model = YOLO(model_path)

        for video_path in video_paths:
            seq_stem  = dataset_layout.seq_name_from_video_path(video_path)
            trial_dir = dataset_layout.stage2_seq_dir(output_root / "mot_annotations_all8", seq_stem)
            trial_dir.mkdir(parents=True, exist_ok=True)

            labels_path = trial_dir / "labels" / seq_stem
            images_path = trial_dir / "images" / seq_stem
            labels_path.mkdir(parents=True, exist_ok=True)
            images_path.mkdir(parents=True, exist_ok=True)

            # ── 2a. Tracking ──────────────────────────────────────────────────
            _run_tracking_for_video(
                model       = model,
                video_path  = video_path,
                labels_path = labels_path,
                images_path = images_path,
                trial_dir   = trial_dir,
                tracker_type= tracker_type,
                verbose     = verbose,
            )

            # ── 2b. MOT format + per-class tracker txt ────────────────────────
            cap    = cv2.VideoCapture(video_path)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            save_mot_format(
                imgs_path   = images_path,
                labels_path = labels_path,
                output_file = trial_dir / f"{trial_dir.stem}.txt",
                img_width   = width,
                img_height  = height,
            )

            # ── 2c. 將 GT 複製進 trial_dir，使 trial_dir 自成完整的 trackeval_root ──
            #
            # run_trackeval_mot 的設計是以單一目錄（trackeval_root）同時讀取：
            #   trackeval_root/gt/gt.txt          ← flat GT（prepare_mot_gt_shadow 讀取）
            #   trackeval_root/seqinfo.ini        ← 序列資訊（可選）
            #   trackeval_root/trackers/...       ← tracker prediction txt
            #
            # save_mot_format 已把 tracker txt 寫入 trial_dir/trackers/...
            # 因此只需把 GT 來源的 gt/gt.txt 和 seqinfo.ini 複製進 trial_dir，
            # 再以 trial_dir 作為 trackeval_root 呼叫 run_trackeval_mot 即可。
            # 這樣每次 trial 完全隔離，不會污染原始 GT 目錄。
            gt_src_root = _resolve_gt_source_root(cfg, seq_stem)
            if gt_src_root is None or not Path(gt_src_root).exists():
                print(f"  [score] GT source root not found for seq={seq_stem}, skip.")
                continue
 
            ok = _setup_trial_gt(
                gt_src_root    = gt_src_root,
                trial_dir      = trial_dir,
            )
            if not ok:
                print(f"  [score] GT setup failed for seq={seq_stem}, skip.")
                continue

            # ── 3. TrackEval（以 trial_dir 作為 trackeval_root）──────────────
            tracker_variant = f"{cfg.tracker_name}__{target_class}"
            out_root        = str(trial_dir / "trackeval_out")
 
            results = run_trackeval_mot(
                cfg,
                trackeval_root         = str(trial_dir),
                benchmark_name         = cfg.benchmark_name,
                split_name             = cfg.split_name,
                seq_name               = seq_stem,
                tracker_names_to_eval  = [tracker_variant],
                out_root               = out_root,
                video_path             = video_path,
                classes_to_eval        = [target_class],
                allow_empty_tracker_file = True,
                verbose                = False,
                DEBUG                  = False,
            )

            score = extract_assa_mean(results, target_class, tracker_variant)
            if score is not None:
                scores.append(score)
                if verbose:
                    print(f"    seq={seq_stem}  AssA_mean={score:.4f}")
            else:
                print(f"  [score] AssA_mean=None for seq={seq_stem}")

    except Exception:
        traceback.print_exc()
    finally:
        # ── 4. 還原 CLASS_TRACKING_PARAMS ────────────────────────────────────
        tracker_config_module.CLASS_TRACKING_PARAMS[target_class] = original_params

    if not scores:
        return float("-inf")
    # return float(np.mean(scores))

    # 回傳所有影片 AssA_mean 的第 25 百分位作為此參數組合的總體分數（較偏保守評估，避免極端高分干擾）
    # 先過濾掉分數為 0 的值
    filtered_scores = [s for s in scores if s != 0]
    # filtered_scores = [s for s in scores if s != 0 and s != float("-inf")]
    if not filtered_scores:
        return float("-inf")
    return float(np.percentile(filtered_scores, 25))


# =============================================================================
# 4. 內部輔助函式
# =============================================================================

def _run_tracking_for_video(
    model,
    video_path: str,
    labels_path: Path,
    images_path: Path,
    trial_dir: Path,
    tracker_type: str,
    verbose: bool,
):
    """
    對單一影片執行 class-aware tracking，輸出 label txt 和 jpg 到指定目錄。
    這是 score_params 的內部函式，刻意精簡（不存影片、不繪框）以加速 grid search。
    """
    from copy import deepcopy
    from module.tracker.config import get_conf_thresh_for_class, get_min_conf_thresh

    conf      = get_min_conf_thresh()
    iou       = 0.5
    imgsz     = 1280
    stem_name = Path(video_path).stem.replace(" 中華三路 民生二路(東北角)_", "_")

    tracker = create_tracker(
        tracker_type     = tracker_type,
        frame_rate       = 30,
        track_high_thresh= conf,
        track_low_thresh = conf * 0.3,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = deepcopy(frame)
        frame_id += 1

        min_conf = get_min_conf_thresh()
        results  = model.predict(frame, conf=min_conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        boxes    = results.boxes

        if len(boxes) > 0:
            keep  = [
                i for i in range(len(boxes))
                if float(boxes.conf[i]) >= get_conf_thresh_for_class(int(boxes.cls[i]))
            ]
            boxes = boxes[keep] if keep else boxes[:0]

        if len(boxes) > 0:
            tracks, _, _ = tracker.update(boxes.cpu().numpy(), frame)
        else:
            tracks = np.array([])

        # 儲存 label
        label_file = labels_path / f"{stem_name}_f{frame_id:06d}.txt"
        img_file   = images_path / f"{stem_name}_f{frame_id:06d}.jpg"
        cv2.imwrite(str(img_file), clean_frame)

        with open(label_file, "w") as f:
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[:4]
                    track_id = int(track[4])
                    score    = track[5]
                    cls_id   = int(track[6])
                    cx = (x1 + x2) / 2 / width
                    cy = (y1 + y2) / 2 / height
                    w  = (x2 - x1) / width
                    h  = (y2 - y1) / height
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {track_id} {score:.4f}\n")

        if verbose and frame_id % 30 == 0:
            print(f"      tracking {frame_id}/{total}", end="\r")

    cap.release()

def _resolve_gt_source_root(cfg, seq_stem: str) -> str | None:
    """
    從 cfg 的 run_track_eval_dir 下，找出指定序列的原始 GT 目錄。

    預期結構（無前綴，見 evaluation/dataset_layout.py）：
      base_dir/mot_annotations_all8/
        <seq_stem>/
          gt/
            gt.txt        ← flat GT（9 columns MOT format）
          seqinfo.ini     ← 可選
    """
    base    = Path(cfg.OUTPUT_ROOT)
    mot_dir = base / cfg.run_track_eval_dir

    candidate = dataset_layout.stage2_seq_dir(mot_dir, seq_stem)
    if candidate.exists():
        return str(candidate)

    # fallback: glob 比對前綴（容許 seq_stem 含特殊字元）
    matches = sorted(mot_dir.glob(f"{seq_stem[:20]}*"))
    if matches:
        return str(matches[0])

    return None
 
 
def _setup_trial_gt(
    gt_src_root: str,
    trial_dir: Path,
) -> bool:
    """
    將 GT 來源目錄中的 flat GT 檔案複製進 trial_dir，
    使 trial_dir 具備 run_trackeval_mot 所需的完整輸入結構：
 
      trial_dir/                         ← 作為 trackeval_root
        gt/
          gt.txt                         ← 從 gt_src_root/gt/gt.txt 複製
        seqinfo.ini                      ← 從 gt_src_root/seqinfo.ini 複製（若存在）
        trackers/...                     ← save_mot_format 已寫入
 
    prepare_mot_gt_shadow 會在執行時自動將 gt/gt.txt 展開成
    gt/mot_challenge/<benchmark>-<split>/<seq>/gt/gt.txt 的 shadow layout，
    因此這裡只需提供 flat 的 gt/gt.txt 即可。
 
    Returns
    -------
    bool
        True = 成功；False = gt.txt 來源不存在，無法繼續。
    """
    src_root = Path(gt_src_root)
 
    # ── GT txt ────────────────────────────────────────────────────────────────
    src_gt = src_root / "gt" / "gt.txt"
    if not src_gt.exists():
        print(f"  [setup_gt] gt.txt not found: {src_gt}")
        return False
 
    dst_gt_dir = trial_dir / "gt"
    dst_gt_dir.mkdir(parents=True, exist_ok=True)
    dst_gt = dst_gt_dir / "gt.txt"
 
    # 每次 trial 都強制覆寫，避免上一次試驗留下被修改過的 GT
    shutil.copy2(src_gt, dst_gt)
 
    # ── seqinfo.ini（可選，prepare_mot_gt_shadow 會從影片自動生成） ───────────
    for candidate in [src_root / "seqinfo.ini", src_root / "gt" / "seqinfo.ini"]:
        if candidate.exists():
            shutil.copy2(candidate, trial_dir / "seqinfo.ini")
            break
 
    return True

# =============================================================================
# 5. GridSearchRunner
# =============================================================================

class GridSearchRunner:
    """
    Tracker 參數 Grid Search 執行器。

    Parameters
    ----------
    param_grid    : 參數搜尋空間 {param_name: [v1, v2, ...]}
    target_class  : 調優目標類別（如 "scooter"）
    video_paths   : 評估用影片路徑清單
    model_path    : YOLO 模型路徑
    cfg           : AppConfig
    tracker_type  : "botsort" 或 "bytetrack"
    output_root   : 暫存 tracking 結果的根目錄
    result_csv    : 結果存檔路徑（可 None）
    resume        : True = 跳過已有結果的組合
    verbose       : 是否印出逐幀進度
    """

    def __init__(
        self,
        param_grid:   dict[str, list],
        target_class: str,
        video_paths:  list[str],
        model_path:   str,
        cfg,
        tracker_type: str = "botsort",
        output_root:  Path = Path("./runs/grid_search/botsort"),
        result_csv:   Path | None = None,
        resume:       bool = False,
        verbose:      bool = False,
    ):
        self.grid         = ParameterGrid(param_grid)
        self.target_class = target_class
        self.video_paths  = video_paths
        self.model_path   = model_path
        self.cfg          = cfg
        self.tracker_type = tracker_type
        self.output_root  = Path(output_root) / tracker_type
        rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

        self.result_csv   = result_csv or (
            self.output_root / f"gs_{target_class}_{datetime.now():%Y%m%d_%H%M%S}_{rand_id}.csv"
        )
        self.resume       = resume
        self.verbose      = verbose

        self.results_: list[dict] = []          # 所有試驗結果
        self.best_score_: float   = float("-inf")
        self.best_params_: dict   = {}

        self.output_root.mkdir(parents=True, exist_ok=True)

        # 讀取已有的 CSV（resume 模式）
        self._done_ids: set[str] = set()
        if resume and self.result_csv.exists():
            self._load_existing_results()

    def _load_existing_results(self):
        """讀取已完成的試驗結果，供 resume 時跳過。"""
        with open(self.result_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                combo_id = row.get("combo_id", "")
                score    = float(row.get("score", float("-inf")))
                params   = json.loads(row.get("params", "{}"))
                self._done_ids.add(combo_id)
                self.results_.append({"combo_id": combo_id, "score": score, "params": params})
                if score > self.best_score_:
                    self.best_score_  = score
                    self.best_params_ = params

        print(f"  [resume] Loaded {len(self._done_ids)} completed trials from {self.result_csv}")

    def _append_csv(self, combo_id: str, score: float, params: dict, elapsed: float):
        """將一筆試驗結果追加寫入 CSV。"""
        is_new = not self.result_csv.exists()
        with open(self.result_csv, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["combo_id", "score", "elapsed_sec", "params"] + sorted(params.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            row = {
                "combo_id":    combo_id,
                "score":       f"{score:.6f}",
                "elapsed_sec": f"{elapsed:.1f}",
                "params":      json.dumps(params, sort_keys=True),
            }
            row.update({k: v for k, v in params.items()})
            writer.writerow(row)

    def run(self) -> dict:
        """
        執行完整的 Grid Search。

        Returns
        -------
        dict
            {"best_params": ..., "best_score": ..., "all_results": ...}
        """
        total  = len(self.grid)
        done   = len(self._done_ids)
        remain = total - done

        print(f"\n{'='*70}")
        print(f"  Grid Search: class={self.target_class}  total={total}  remain={remain}")
        print(f"  tracker_type  : {self.tracker_type}")
        print(f"  videos        : {len(self.video_paths)}")
        print(f"  result_csv    : {self.result_csv}")
        print(f"{'='*70}")

        trial_idx = 0
        for params in self.grid:
            combo_id = self.grid.combo_id(params)

            # Resume: 跳過已完成
            if self.resume and combo_id in self._done_ids:
                continue

            trial_idx += 1
            print(f"\n[{trial_idx}/{remain}] Testing: {params}")

            trial_output = self.output_root / f"trial_{trial_idx:04d}"
            t0 = time.time()

            score = score_params(
                params       = params,
                target_class = self.target_class,
                video_paths  = self.video_paths,
                model_path   = self.model_path,
                cfg          = self.cfg,
                tracker_type = self.tracker_type,
                output_root  = trial_output,
                verbose      = self.verbose,
            )

            elapsed = time.time() - t0
            status  = "✅" if score > float("-inf") else "❌"
            print(f"  {status} score (AssA_mean) = {score:.4f}  ({elapsed:.1f}s)")

            # 記錄結果
            entry = {"combo_id": combo_id, "score": score, "params": copy.deepcopy(params)}
            self.results_.append(entry)
            self._append_csv(combo_id, score, params, elapsed)

            if score > self.best_score_:
                self.best_score_  = score
                self.best_params_ = copy.deepcopy(params)
                print(f"  🏆 New best! AssA_mean={self.best_score_:.4f}")

            # 每次試驗結束後清除暫存（節省磁碟空間）
            if trial_output.exists():
                shutil.rmtree(trial_output, ignore_errors=True)

        return {
            "best_params": self.best_params_,
            "best_score":  self.best_score_,
            "all_results": self.results_,
        }

    def print_summary(self):
        """在 terminal 印出搜尋結果摘要（Top-10 + 最佳參數）。"""
        if not self.results_:
            print("No results to summarize.")
            return

        sorted_results = sorted(
            self.results_, key=lambda r: r["score"], reverse=True
        )

        GREEN = "\033[32m"
        RESET = "\033[0m"
        print(f"\n{GREEN}{'='*70}{RESET}")
        print(f"{GREEN}  Grid Search Summary — class={self.target_class}{RESET}")
        print(f"{GREEN}{'='*70}{RESET}")
        print(f"{GREEN}  Total trials : {len(self.results_)}{RESET}")
        print(f"{GREEN}  Best score   : {self.best_score_:.4f}  (HOTA.AssA_mean){RESET}")
        print(f"{GREEN}  Best params  :{RESET}")
        for k, v in sorted(self.best_params_.items()):
            print(f"{GREEN}    {k:<30} = {v}{RESET}")

        print(f"\n{GREEN}  Top-10 trials:{RESET}")
        header = f"  {'Rank':<5} {'AssA_mean':<12} params"
        print(f"{GREEN}{header}{RESET}")
        print(f"{GREEN}  {'-'*65}{RESET}")
        for i, r in enumerate(sorted_results[:10], 1):
            param_str = "  ".join(f"{k}={v}" for k, v in sorted(r["params"].items()))
            print(f"{GREEN}  {i:<5} {r['score']:<12.4f} {param_str}{RESET}")

        print(f"{GREEN}{'='*70}{RESET}")


# =============================================================================
# 6. 便利函式：產生 config.py 的更新建議
# =============================================================================

def generate_config_patch(
    target_class: str,
    best_params: dict,
    original_params: dict,
) -> str:
    """
    產生可直接貼入 config.py 的 CLASS_TRACKING_PARAMS 更新片段。
    保留原始參數中 grid search 未涉及的欄位（如 use_reid、model 等）。
    """
    merged = copy.deepcopy(original_params)
    merged.update(best_params)

    lines = [f"    '{target_class}': {{"]
    for k, v in sorted(merged.items()):
        if isinstance(v, str):
            lines.append(f"        '{k}': '{v}',")
        elif isinstance(v, bool):
            lines.append(f"        '{k}': {v},")
        else:
            lines.append(f"        '{k}': {v},")
    lines.append("    },")
    return "\n".join(lines)


# =============================================================================
# 7. CLI 入口
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
    # 預設來源是 stage1 已產生的輸出樹（OUTPUT_ROOT/run_det_eval_dir/<seq>/<seq>.mp4），
    # 不是 SOURCE_VIDEO_DIR——grid search 需要 stage1 已經跑過（trackers/ 與 GT 都已就緒）。
    import glob as _glob
    if args.video_glob:
        video_paths = sorted(_glob.glob(args.video_glob, recursive=True))
    else:
        det_output_dir = Path(cfg.OUTPUT_ROOT) / cfg.run_det_eval_dir
        video_paths = sorted(
            str(seq_dir / f"{seq_dir.name}.mp4")
            for seq_dir in dataset_layout.list_stage1_sequences(det_output_dir)
            if (seq_dir / f"{seq_dir.name}.mp4").exists()
        )

    if not video_paths:
        print(f"❌ No videos found. Check --video-glob or config.yaml paths.")
        sys.exit(1)

    video_paths = video_paths[: args.max_videos]
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