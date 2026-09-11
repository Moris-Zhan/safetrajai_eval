"""
module/grid_search/ 三支腳本（bytetrack/botsort/botsort_reid）共用邏輯。

三支腳本各自只保留 PARAM_GRIDS 常數 + main()（各自 argparse 定義與預設值），
其餘 90% 以上逐字重複的邏輯（ParameterGrid、scoring、GT 處理、GridSearchRunner）
全部收斂到這裡。詳見 docs/superpowers/specs/2026-09-03-grid-search-dedup-design.md。

【module/detection/ 整合】score_params()/_run_tracking_for_video() 的偵測邏輯
改呼叫 module/detection/detector.py::Detector，取代原本三支腳本各自手刻的
model.predict() + 逐偵測套用類別門檻過濾（與 Detector.detect() 內容逐字重複）。
詳見 docs/superpowers/specs/2026-09-03-detection-module-design.md。

【路徑組裝】discover_grid_search_videos()/_resolve_gt_source_root()/score_params()
的 trial_dir 組裝一律呼叫 evaluation/dataset_layout.py（dataset-path-refactor 已合併），
OUTPUT 端序列目錄不含前綴，SOURCE 端既有命名不變。
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import random
import shutil
import string
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from module.detection.detector import Detector
from module.tracker import create_tracker
from module.tracker import config as tracker_config_module
from evaluation import dataset_layout
from evaluation.mot_format import save_mot_format
from evaluation.trackeval_runner import run_trackeval_mot


# =============================================================================
# 1. ParameterGrid — 產生所有參數組合（等同 sklearn.model_selection.ParameterGrid）
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
# 2. Scoring Function — 從 TrackEval results dict 萃取 HOTA.AssA_mean
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


# =============================================================================
# 3. GT 處理輔助函式
# =============================================================================

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


def discover_grid_search_videos(
    cfg,
    video_glob_override: str | None,
    max_videos: int,
) -> list[str]:
    """
    收集 grid search 用的影片路徑清單。

    預設來源是 stage1 已產生的輸出樹（OUTPUT_ROOT/run_det_eval_dir/<seq>/<seq>.mp4），
    不是 SOURCE_VIDEO_DIR——grid search 需要 stage1 已經跑過（trackers/ 與 GT 都已就緒）。
    video_glob_override 有值時直接 glob，取代預設掃描邏輯。
    不含「找不到影片」的錯誤處理（呼叫端的 main() 自行檢查回傳的 list 是否為空並印
    錯誤/sys.exit，維持現行行為）。
    """
    import glob as _glob

    if video_glob_override:
        video_paths = sorted(_glob.glob(video_glob_override, recursive=True))
    else:
        det_output_dir = Path(cfg.OUTPUT_ROOT) / cfg.run_det_eval_dir
        video_paths = sorted(
            str(seq_dir / f"{seq_dir.name}.mp4")
            for seq_dir in dataset_layout.list_stage1_sequences(det_output_dir)
            if (seq_dir / f"{seq_dir.name}.mp4").exists()
        )

    return video_paths[:max_videos]
