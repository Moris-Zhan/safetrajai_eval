# module/grid_search/ 去重複（含 Detector 整合）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **2026-09-11 修訂**：`dataset-path-refactor` 分支已合併進 master（`53a4501`）。本計畫原本假設尚未合併、`discover_grid_search_videos()`/`_resolve_gt_source_root()`/`score_params()` 的 `trial_dir` 組裝暫時沿用舊 `annotation_*` 字串邏輯的版本已作廢；三支 `grid_search_*.py` 現在**已經**改用 `evaluation/dataset_layout.py`（dataset-path-refactor 的 Task 8-10），本版本對照合併後的實際程式碼重寫。

**Goal:** 把 `module/grid_search/grid_search_bytetrack.py`（993 行）、`grid_search_botsort.py`（1187 行）、`grid_search_botsort_reid.py`（1042 行）之間 90% 以上逐字重複的邏輯整併進新檔 `module/grid_search/core.py`，三支腳本瘦身為 `PARAM_GRIDS` 常數 + `main()`；同時完成 `2026-09-03-detection-module-design.md` 唯一的行動項目——三支腳本手刻的偵測+信心過濾邏輯改呼叫 `module/detection/detector.py::Detector`，不再與 `Detector.detect()` 重複實作。

**Architecture:** 新增 `module/grid_search/core.py` 承接 `ParameterGrid`、`extract_assa_mean()`、`score_params()`、`generate_config_patch()`、`_resolve_gt_source_root()`、`_setup_trial_gt()`、`GridSearchRunner`、新函式 `discover_grid_search_videos()`；`score_params()`/`_run_tracking_for_video()` 內部改用 `Detector(model_path).detect(frame, iou=iou, imgsz=imgsz)` 取代手刻的 `model.predict()` + 逐偵測套用類別門檻過濾。三支腳本 CLI 入口與旗標不變，只改成 `from module.grid_search.core import ...` 並呼叫共用邏輯。所有路徑組裝（影片探索、GT 解析、trial 序列目錄命名）一律沿用三支腳本現有的 `evaluation/dataset_layout.py` 呼叫，不自行拼字串。

**Tech Stack:** Python 3、既有 `ultralytics`/`opencv-python`/`numpy`（不新增套件）。本專案沒有 pytest（見 `CLAUDE.md`：「無測試」），驗證步驟一律用**可直接執行的 assert 腳本**（`python3` 執行、無例外即通過，放 scratchpad 不進版控）+ 手動 CLI 執行比對。

**Spec:** `docs/superpowers/specs/2026-09-03-grid-search-dedup-design.md`（第 0 節「前提依賴」所描述的前提現已成立：dataset-path-refactor 已合併）、`docs/superpowers/specs/2026-09-03-detection-module-design.md`（本計畫是後者唯一行動項目的實作位置，見該 spec 第 3-4 節）。

## Global Constraints

- 固定 6 類別與 ID 不可變動：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`（`.claude/rules/project-and-pipeline.md`）。
- **不合併三支腳本的 CLI 入口**：三個獨立指令入口（`python module/grid_search/grid_search_bytetrack.py ...` 等）、既有旗標（`--dry-run`/`--resume`/`--max-videos`/`--params-subset`/`--tracker` 等）與輸出（CSV 欄位、patch 檔格式）保持不變（spec 非目標，方案 C 已被使用者排除）。
- **不改變 `PARAM_GRIDS` 搜尋空間或任何腳本目前的預設值**——包含容易被誤判為「該統一」的隱藏差異，見下方三點：
  - `grid_search_bytetrack.py` 不新增 `--conf`/`--iou`/`--imgsz`/`--frame_rate`/`--gmc_method` CLI flag；`_run_tracking_for_video()` 統一簽章後，`main()` 呼叫 `GridSearchRunner(...)` 必須傳入現行寫死值：`conf=get_min_conf_thresh()`、`iou=0.5`、`imgsz=1280`、`frame_rate=30`、`gmc_method="none"`（ByteTrack 不使用 GMC，佔位值，`_run_tracking_for_video()` 只在 `tracker_type=="botsort"` 時才把 `gmc_method` 放進 `tracker_kwargs`）。
  - `grid_search_botsort_reid.py` **現況也沒有** `--conf`/`--iou`/`--imgsz`/`--frame_rate`/`--gmc_method` CLI flag（跟 bytetrack 一樣、不是跟 botsort 一樣），統一簽章後同樣不能新增這些 CLI flag，`main()` 必須傳入寫死值：`conf=get_min_conf_thresh()`、`iou=0.5`、`imgsz=1280`、`frame_rate=30`。**`gmc_method` 尤其容易搞錯**：這支腳本現行的 `_run_tracking_for_video()`（`module/grid_search/grid_search_botsort_reid.py:498-577`）完全沒有把 `gmc_method` 傳給 `create_tracker()`（逐字對照 `grid_search_bytetrack.py`，非對照 `grid_search_botsort.py`），即使 `tracker_type` 實際是 `"botsort"`（`main()` 的 `--tracker` 預設值），`create_tracker()` 因此落到 `module/tracker/__init__.py::DEFAULT_TRACKER_CONFIG['gmc_method']` 的預設值 `'sparseOptFlow'`（見 `module/tracker/__init__.py:32`）。統一後 `main()` 必須明確傳入 `gmc_method="sparseOptFlow"`，不可比照 `grid_search_botsort.py` 用 CLI `args.gmc_method`（該腳本沒有這個 CLI 選項），也不可用 `"none"`（會讓 GMC 被停用，跟現況不同——現況是有 GMC、只是用 `sparseOptFlow` 而非 `ecc`/`sift`）。
  - `grid_search_botsort.py` 的 `--gmc_method` CLI 預設值是 `"sift"`（`argparse` 的 `default="sift"`），跟 `GridSearchRunner.__init__`/`score_params()` 簽章上寫的 class-level 預設值 `"ecc"` 不一致——`main()` 一律明確傳入 `args.gmc_method`，所以 `"ecc"` 這個簽章預設值其實從未被使用到。統一到 `core.py` 後保留這個既有的「死預設值」不做修正（不是本次重構範圍，行為不受影響）。
- `_run_tracking_for_video()` 裡的 `conf`/`iou`/`imgsz` 有兩種完全不同的用途，統一時不可混淆：`conf`（連同其衍生值 `conf * 0.3`）只用於 `create_tracker(track_high_thresh=conf, track_low_thresh=conf*0.3, ...)`（追蹤器的軌跡確認門檻），**不影響**偵測階段；偵測階段的信心門檻一律是 `Detector.detect()` 內部呼叫的 `get_min_conf_thresh()`/`get_conf_thresh_for_class()`，與外部傳入的 `conf` 參數無關（這是現行程式碼本來就有的行為，`Detector.detect()` 簽章也證實了這點：`detect(self, frame, iou, imgsz)` 沒有 `conf` 參數）。`iou`/`imgsz` 兩者才會真的傳進 `Detector.detect(frame, iou=iou, imgsz=imgsz)` 影響偵測。
- `Detector.detect()` 內部的信心過濾依賴 `load_track_config()` 已先被呼叫（`CLASS_TRACKING_PARAMS` 非空）。三支腳本的 `main()` 本來就已經在解析引數後呼叫 `load_track_config()`，順序不變，不新增風險（`2026-09-03-detection-module-design.md` 第 5 節）。
- **路徑組裝一律呼叫 `evaluation/dataset_layout.py`**（dataset-path-refactor 已合併，三支腳本現況已是如此）：`discover_grid_search_videos()` 呼叫 `dataset_layout.list_stage1_sequences()`（列舉 stage1 已產生的 `<seq>/` 目錄，無前綴）；`score_params()` 的 `trial_dir` 用 `dataset_layout.seq_name_from_video_path()` + `dataset_layout.stage2_seq_dir()` 組裝（同樣無前綴）；`_resolve_gt_source_root()` 用 `dataset_layout.stage2_seq_dir()` 找 `cfg.OUTPUT_ROOT/cfg.run_track_eval_dir` 下的 GT（找不到才 fallback 到不含前綴的字首比對 glob）。不可退回舊的 `annotation_*`/`mot_annotation_<seq>` 字串拼接。
- `module/tracker/**/*.py` 的 `ClassAwareBOTSORT`/`ClassAwareBYTETracker` override 方法一字不可更動、不可搬離（`.claude/rules/tracker.md`）——本計畫不觸碰這些檔案。
- 不執行全量 grid search、不覆寫 `runs/` 既有輸出——每個驗證步驟都用 `--dry-run` 或 `--max-videos 1` + `--params-subset` 縮小範圍。
- **已知、本計畫不處理的預先存在問題**（`2026-09-03-grid-search-dedup-design.md` 範圍外，不要順手修）：`score_params()` 的 `trial_dir` 固定用 `output_root / "mot_annotations_all8"` 這個字面值當中繼子目錄名稱，不論實際使用哪個 dataset；grid search 的 tracking 迴圈沒有套用 `det_corner` 偵測區域多邊形過濾（`track_eval.py`/`VideoSequenceTracker` 有，這裡沒有），所以 grid search 的 CSV 分數跟 `track_eval.py` 全管線的 Excel 結果本來就不完全可比——這是既有落差，不在本次去重複範圍內。

---

## File Structure

```
module/grid_search/
├── core.py                          [新增] 承接三支腳本共用邏輯，含 Detector 整合
├── grid_search_bytetrack.py         [修改] 瘦身為 PARAM_GRIDS + main()，993 行 → 約 200 行
├── grid_search_botsort.py           [修改] 同上，1187 行 → 約 230 行
└── grid_search_botsort_reid.py      [修改] 同上，1042 行 → 約 210 行
```

不動 `module/detection/detector.py`（介面已經足夠通用，不調整）、`module/tracker/*`、`module/reid/*`、`module/collision/*`、`evaluation/*`（純呼叫既有的 `evaluation/dataset_layout.py`，不修改它）、`track_eval.py`。

---

### Task 1: `module/grid_search/core.py` — 建立骨架 + 搬移 `ParameterGrid`、`extract_assa_mean()`

**Files:**
- Create: `module/grid_search/core.py`
- Test（暫存，不進版控）: `scratchpad/test_core_grid1.py`

**Interfaces:**
- Consumes: 無新依賴。
- Produces（供 Task 2-6 使用）: `ParameterGrid` class、`extract_assa_mean(trackeval_results, target_class, tracker_variant) -> float | None`。

- [ ] **Step 1: 建立 `core.py`，搬移 `ParameterGrid`（逐字複製自 `grid_search_bytetrack.py:162-189`，三支腳本間完全相同）**

```python
# module/grid_search/core.py
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
```

- [ ] **Step 2: 驗證搬移無誤**

在 scratchpad 建立 `test_core_grid1.py`（`PROJECT_ROOT` 依實際專案路徑調整）：

```python
# scratchpad/test_core_grid1.py — 手動驗證腳本，不進版控
import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/moris/Dcoker_Project/Project/safetrajai_eval")
sys.path.insert(0, str(PROJECT_ROOT))

from module.grid_search.core import ParameterGrid, extract_assa_mean

# ParameterGrid
pg = ParameterGrid({"a": [1, 2], "b": [3, 4]})
combos = list(pg)
assert len(pg) == 4
assert combos == [
    {"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4},
]
assert pg.combo_id({"a": 1, "b": 3}) == '{"a": 1, "b": 3}'

# extract_assa_mean：None/非 dict 輸入應回傳 None，不拋例外
assert extract_assa_mean(None, "scooter", "YOLO11x__scooter") is None
assert extract_assa_mean({}, "scooter", "YOLO11x__scooter") is None
assert extract_assa_mean("not a dict", "scooter", "YOLO11x__scooter") is None

print("ALL OK")
```

Run: `python3 scratchpad/test_core_grid1.py`
Expected: `ALL OK`

- [ ] **Step 3: Commit**

```bash
git add module/grid_search/core.py
git commit -m "Add module/grid_search/core.py: ParameterGrid + extract_assa_mean

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `core.py` — 搬移 `_resolve_gt_source_root()` / `_setup_trial_gt()` / 新增 `discover_grid_search_videos()`

**Files:**
- Modify: `module/grid_search/core.py`

**Interfaces:**
- Consumes: `evaluation.dataset_layout`（已於 Task 1 匯入）、`Path`、`shutil`（已在 Task 1 匯入）。
- Produces（供 Task 5-9 使用）: `_resolve_gt_source_root(cfg, seq_stem) -> str | None`、`_setup_trial_gt(gt_src_root, trial_dir) -> bool`、`discover_grid_search_videos(cfg, video_glob_override, max_videos) -> list[str]`。

- [ ] **Step 1: 搬移 `_resolve_gt_source_root()`（逐字複製自 `grid_search_bytetrack.py:534-557`，三支腳本間完全相同，已使用 `dataset_layout.stage2_seq_dir()`）**

在 `core.py` 的 `extract_assa_mean()` 後面新增：

```python
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
```

- [ ] **Step 2: 新增 `discover_grid_search_videos()`（抽出三支腳本 `main()` 目前重複的影片蒐集邏輯，逐字複製自 `grid_search_bytetrack.py:927-939`，三支腳本間完全相同，已使用 `dataset_layout.list_stage1_sequences()`）**

```python
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
```

- [ ] **Step 3: 驗證**

```python
# scratchpad/test_core_grid2.py — 手動驗證腳本，不進版控
import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/moris/Dcoker_Project/Project/safetrajai_eval")
sys.path.insert(0, str(PROJECT_ROOT))

from module.grid_search.core import _resolve_gt_source_root, discover_grid_search_videos
from evaluation.config_loader import load_config

cfg = load_config(str(PROJECT_ROOT / "config" / "bytetrack.yaml"))

# _resolve_gt_source_root：對不存在的序列回傳 None，不拋例外
assert _resolve_gt_source_root(cfg, "__no_such_sequence__") is None

# discover_grid_search_videos：video_glob_override 有值時直接 glob（用一個一定匹配 0 筆的 pattern）
assert discover_grid_search_videos(cfg, "/tmp/__no_such_dir__/*.mp4", max_videos=6) == []

print("ALL OK")
```

Run: `python3 scratchpad/test_core_grid2.py`
Expected: `ALL OK`

- [ ] **Step 4: Commit**

```bash
git add module/grid_search/core.py
git commit -m "core.py: add _resolve_gt_source_root, _setup_trial_gt, discover_grid_search_videos

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `core.py` — 新增統一版 `_run_tracking_for_video()`（含 Detector 整合）

**Files:**
- Modify: `module/grid_search/core.py`

**Interfaces:**
- Consumes: `module.detection.detector.Detector`（已於 Task 1 匯入）、`module.tracker.create_tracker`。
- Produces（供 Task 4 的 `score_params()` 使用）: `_run_tracking_for_video(detector, video_path, labels_path, images_path, trial_dir, tracker_type, conf, iou, imgsz, frame_rate, gmc_method, verbose) -> None`。

此函式統一現行 `grid_search_bytetrack.py`/`grid_search_botsort_reid.py`（`conf`/`iou`/`imgsz`/`frame_rate` 內部寫死、無 `gmc_method`）與 `grid_search_botsort.py`（全部開放成參數）三個版本，採用 `grid_search_botsort.py` 的完整簽章（`2026-09-03-grid-search-dedup-design.md` 第 3.1 節）；呼叫端行為差異在 Task 7-9 的 `main()` 各自傳入正確常數，本 Task 只建立統一函式本身。

- [ ] **Step 1: 新增函式**

在 `core.py` 的 `_setup_trial_gt()` 後面新增：

```python
# =============================================================================
# 4. Tracking（單一影片）
# =============================================================================

def _run_tracking_for_video(
    detector: Detector,
    video_path: str,
    labels_path: Path,
    images_path: Path,
    trial_dir: Path,
    tracker_type: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_rate: int,
    gmc_method: str,
    verbose: bool,
):
    """
    對單一影片執行 class-aware tracking，輸出 label txt 和 jpg 到指定目錄。
    這是 score_params 的內部函式，刻意精簡（不存影片、不繪框）以加速 grid search。

    統一簽章，一律接受 conf/iou/imgsz/frame_rate/gmc_method；gmc_method 只在
    tracker_type == "botsort" 時加入 tracker_kwargs（沿用現行 grid_search_botsort.py
    的判斷邏輯，行為對兩種 tracker 都不變）。

    conf 只影響 create_tracker() 的 track_high_thresh/track_low_thresh（追蹤器的軌跡
    確認門檻），不影響偵測信心過濾——偵測信心過濾一律由 detector.detect() 內部的
    get_min_conf_thresh()/get_conf_thresh_for_class() 決定，與這裡的 conf 參數無關
    （這是現行程式碼本來就有的行為，非本次變動）。

    【module/detection/ 整合】原本各腳本手刻的 model = YOLO(model_path) +
    model.predict(...) + 逐偵測套用 get_conf_thresh_for_class() 過濾，改成呼叫
    module/detection/detector.py 的 Detector.detect(frame, iou=iou, imgsz=imgsz)
    （與 track_eval.py 共用同一份偵測+信心過濾邏輯，不再各自重寫）。
    """
    from copy import deepcopy

    stem_name = Path(video_path).stem.replace(" 中華三路 民生二路(東北角)_", "_")

    tracker_kwargs = {
        "track_high_thresh": conf,
        "track_low_thresh": conf * 0.3,
    }
    if tracker_type == "botsort":
        tracker_kwargs["gmc_method"] = gmc_method

    tracker = create_tracker(
        tracker_type     = tracker_type,
        frame_rate       = frame_rate,
        **tracker_kwargs,
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

        boxes = detector.detect(frame, iou=iou, imgsz=imgsz)

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
```

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile module/grid_search/core.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: Commit**

```bash
git add module/grid_search/core.py
git commit -m "core.py: add unified _run_tracking_for_video() using Detector

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `core.py` — 新增統一版 `score_params()`（Detector 建立、conf/iou/imgsz/frame_rate/gmc_method 傳遞）

**Files:**
- Modify: `module/grid_search/core.py`

**Interfaces:**
- Consumes: `Detector`（Task 1 匯入）、`_run_tracking_for_video()`（Task 3）、`_resolve_gt_source_root()`/`_setup_trial_gt()`（Task 2）、`extract_assa_mean()`（Task 1）、`evaluation.dataset_layout`、`evaluation.mot_format.save_mot_format`、`evaluation.trackeval_runner.run_trackeval_mot`（皆已於 Task 1 匯入）。
- Produces（供 Task 5 的 `GridSearchRunner.run()` 使用）: `score_params(params, target_class, video_paths, model_path, cfg, tracker_type, conf, iou, imgsz, frame_rate, gmc_method, output_root, verbose) -> float`。

- [ ] **Step 1: 新增函式（逐字沿用 `grid_search_botsort.py:427-587` 的完整參數化版本，只把 `model = YOLO(model_path)` 換成 `detector = Detector(model_path)`；`trial_dir` 組裝已使用 `dataset_layout.seq_name_from_video_path()`/`stage2_seq_dir()`）**

在 `core.py` 的 `_run_tracking_for_video()` 後面新增：

```python
# =============================================================================
# 5. Scoring Function（完整流程）
# =============================================================================

def score_params(
    params: dict,
    target_class: str,
    video_paths: list[str],
    model_path: str,
    cfg,                        # AppConfig
    tracker_type: str = "bytetrack",
    conf: float = 0.15,
    iou: float = 0.50,
    imgsz: int = 640,
    frame_rate: int = 30,
    gmc_method: str = "ecc",
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
    conf/iou/imgsz/frame_rate/gmc_method : 傳給 _run_tracking_for_video()，
        呼叫端（GridSearchRunner.run()）依各腳本現行行為決定實際值。
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
        detector = Detector(model_path)

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
                detector    = detector,
                video_path  = video_path,
                labels_path = labels_path,
                images_path = images_path,
                trial_dir   = trial_dir,
                tracker_type= tracker_type,
                conf        = conf,
                iou         = iou,
                imgsz       = imgsz,
                frame_rate  = frame_rate,
                gmc_method  = gmc_method,
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

    # 回傳所有影片 AssA_mean 的第 25 百分位作為此參數組合的總體分數（較偏保守評估，避免極端高分干擾）
    # 先過濾掉分數為 0 的值
    filtered_scores = [s for s in scores if s != 0]
    if not filtered_scores:
        return float("-inf")
    return float(np.percentile(filtered_scores, 25))
```

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile module/grid_search/core.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: Commit**

```bash
git add module/grid_search/core.py
git commit -m "core.py: add unified score_params() using Detector

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: `core.py` — 搬移 `GridSearchRunner`、`generate_config_patch()`

**Files:**
- Modify: `module/grid_search/core.py`

**Interfaces:**
- Consumes: `ParameterGrid`（Task 1）、`score_params()`（Task 4）。
- Produces（供 Task 7-9 使用）: `GridSearchRunner(param_grid, target_class, video_paths, model_path, cfg, tracker_type, conf, iou, imgsz, frame_rate, gmc_method, output_root, result_csv, resume, verbose)`、`generate_config_patch(target_class, best_params, original_params) -> str`。

- [ ] **Step 1: 搬移 `GridSearchRunner`（逐字沿用 `grid_search_botsort.py:763-978` 的完整參數化版本——三支腳本間 `_load_existing_results()`/`_append_csv()`/`print_summary()` 完全相同，`__init__`/`run()` 採用 botsort 版本的完整簽章）**

在 `core.py` 的 `score_params()` 後面新增：

```python
# =============================================================================
# 6. GridSearchRunner
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
    conf/iou/imgsz/frame_rate/gmc_method : 傳給 score_params()，各呼叫端（腳本 main()）
        依現行行為決定實際值（bytetrack/botsort_reid 傳寫死常數，botsort 傳 CLI 值）。
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
        conf: float = 0.15,
        iou: float = 0.50,
        imgsz: int = 640,
        frame_rate: int = 30,
        gmc_method: str = "ecc",
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
        self.conf         = conf
        self.iou          = iou
        self.imgsz        = imgsz
        self.frame_rate   = frame_rate
        self.gmc_method   = gmc_method
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
        print(f"  conf          : {self.conf}")
        print(f"  iou           : {self.iou}")
        print(f"  imgsz         : {self.imgsz}")
        print(f"  frame_rate    : {self.frame_rate}")
        print(f"  gmc_method    : {self.gmc_method}")
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
                conf         = self.conf,
                iou          = self.iou,
                imgsz        = self.imgsz,
                frame_rate   = self.frame_rate,
                gmc_method   = self.gmc_method,
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
# 7. 便利函式：產生 config.py 的更新建議
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
```

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile module/grid_search/core.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 匯入驗證（確認 core.py 整體可 import，且無循環 import）**

```python
# scratchpad/test_core_grid3.py — 手動驗證腳本，不進版控
import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/moris/Dcoker_Project/Project/safetrajai_eval")
sys.path.insert(0, str(PROJECT_ROOT))

from module.grid_search.core import (
    ParameterGrid,
    extract_assa_mean,
    score_params,
    generate_config_patch,
    GridSearchRunner,
    discover_grid_search_videos,
)

# generate_config_patch：保留 grid search 未涉及的欄位
patch = generate_config_patch(
    target_class="scooter",
    best_params={"match_thresh": 0.8},
    original_params={"match_thresh": 0.6, "use_reid": False},
)
assert "'match_thresh': 0.8," in patch
assert "'use_reid': False," in patch

print("ALL OK")
```

Run: `python3 scratchpad/test_core_grid3.py`
Expected: `ALL OK`

- [ ] **Step 4: Commit**

```bash
git add module/grid_search/core.py
git commit -m "core.py: add GridSearchRunner, generate_config_patch

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: `grid_search_bytetrack.py` — 瘦身為 `PARAM_GRIDS` + `main()`

**Files:**
- Modify: `module/grid_search/grid_search_bytetrack.py`（993 行 → 約 200 行）

**Interfaces:**
- Consumes: `module.grid_search.core` 的全部匯出（Task 1-5）。
- Produces: 無新對外介面；CLI 用法與旗標完全不變。

- [ ] **Step 1: 改寫整檔**

現有 `grid_search_bytetrack.py` 的 `PARAM_GRIDS`（`:83-155`）與 `main()`（`:832-993`）內容不變，只替換掉中間 `ParameterGrid`/`extract_assa_mean`/`score_params`/`_run_tracking_for_video`/`_resolve_gt_source_root`/`_setup_trial_gt`/`GridSearchRunner`/`generate_config_patch`（`:162-825`）與相關 import（`:30-71`，含 `from evaluation import dataset_layout`）。整檔改為：

```python
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
```

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile module/grid_search/grid_search_bytetrack.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 手動驗證——`--dry-run`，確認組合數與重構前相同**

Run: `python module/grid_search/grid_search_bytetrack.py --target-class scooter --dry-run`
Expected: `Parameter grid: 4608 combinations`（4×4×4×3×4×3×2＝4608，與重構前 `PARAM_GRIDS["scooter"]` 內容不變，數字應一致），無 `ImportError`/`NameError`。

- [ ] **Step 4: 手動驗證——`--max-videos 1` + 縮小參數子集跑一組 trial**

Run: `python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1 --params-subset match_thresh`
Expected: 正確找到至少 1 支序列影片（前提：`config/grid_search_bytetrack.yaml` 指到的 `OUTPUT_ROOT/run_det_eval_dir` 下已有 stage1 輸出，若沒有先跑一次 `python track_eval.py --config config/bytetrack.yaml` 限量序列產生輸出）、trial 執行完畢並寫入 CSV，`score (AssA_mean)` 有印出實際數值（非全部 `-inf`）。

- [ ] **Step 5: Commit**

```bash
git add module/grid_search/grid_search_bytetrack.py
git commit -m "grid_search_bytetrack.py: slim down to PARAM_GRIDS + main(), use core.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: `grid_search_botsort.py` — 瘦身為 `PARAM_GRIDS` + `main()`

**Files:**
- Modify: `module/grid_search/grid_search_botsort.py`（1187 行 → 約 230 行）

**Interfaces:**
- Consumes: `module.grid_search.core` 的全部匯出（Task 1-5）。
- Produces: 無新對外介面；CLI 用法、`--conf`/`--iou`/`--imgsz`/`--frame_rate`/`--gmc_method` 旗標與行為完全不變。

- [ ] **Step 1: 改寫整檔**

現有 `grid_search_botsort.py` 的 `PARAM_GRIDS`（`:83-289`）內容不變；`main()`（`:1005-1187`）內容不變，除了把「收集影片路徑」區塊改呼叫 `discover_grid_search_videos()`（同 Task 6 Step 1 的做法）；中間 `ParameterGrid`/`extract_assa_mean`/`score_params`/`_run_tracking_for_video`/`_resolve_gt_source_root`/`_setup_trial_gt`/`GridSearchRunner`/`generate_config_patch`（`:298-976`）與相關 import 全部移除，改成：

```python
from module.tracker import CLASS_NAMES
from module.tracker import config as tracker_config_module   # 直接操作模組層級 dict
from module.tracker.config import (
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
```

檔案結構整體變成：header docstring（更新為說明共用邏輯已搬到 `core.py`，同 Task 6）+ 上述 import + 原 `PARAM_GRIDS` 內容（`:83-289`，一字不改）+ `main()`（`:1005-1187`，只改「收集影片路徑」區塊）。

`main()` 內「收集影片路徑」區塊（現有 `:1115-1127`，用 `grep -n "收集影片路徑"` 定位，已使用 `dataset_layout.list_stage1_sequences()`）：

Read 現有：
```python
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
```

改為：
```python
    # ── 收集影片路徑 ──────────────────────────────────────────────────────────
    video_paths = discover_grid_search_videos(cfg, args.video_glob, args.max_videos)
```

（注意：現有版本沒有套用 `args.max_videos` 切片——切片是在後面 `video_paths = video_paths[: args.max_videos]` 那行做的，維持不變，`discover_grid_search_videos()` 內部已經做了切片，所以下面那行 `video_paths = video_paths[: args.max_videos]` 需要一併刪除，避免對已切片過的清單再切一次；用 `grep -n "video_paths = video_paths\[: args.max_videos\]"` 確認並刪除該行。）

`GridSearchRunner(...)` 呼叫處（現有 `:1140-1156`）的 `conf=args.conf, iou=args.iou, imgsz=args.imgsz, frame_rate=args.frame_rate, gmc_method=args.gmc_method` 不變——這支腳本本來就是 CLI 驅動，直接沿用 `args.*`。

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile module/grid_search/grid_search_botsort.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 手動驗證——`--dry-run`**

Run: `python module/grid_search/grid_search_botsort.py --target-class scooter --dry-run`
Expected: 印出參數組合數量（與重構前相同），無錯誤。

- [ ] **Step 4: 手動驗證——`--max-videos 1` + 縮小參數子集跑一組 trial**

Run: `python module/grid_search/grid_search_botsort.py --target-class scooter --max-videos 1 --params-subset match_thresh`
Expected: 同 Task 6 Step 4；且 console 印出的 `conf`/`iou`/`imgsz`/`frame_rate`/`gmc_method` 應與 `--conf`/`--iou`/`--imgsz`/`--frame_rate`/`--gmc_method` 的預設值（`0.15`/`0.50`/`640`/`30`/`sift`）一致。

- [ ] **Step 5: Commit**

```bash
git add module/grid_search/grid_search_botsort.py
git commit -m "grid_search_botsort.py: slim down to PARAM_GRIDS + main(), use core.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: `grid_search_botsort_reid.py` — 瘦身為 `PARAM_GRIDS` + `main()`（⚠️ 常數值需特別核對）

**Files:**
- Modify: `module/grid_search/grid_search_botsort_reid.py`（1042 行 → 約 210 行）

**Interfaces:**
- Consumes: `module.grid_search.core` 的全部匯出（Task 1-5）。
- Produces: 無新對外介面；CLI 用法完全不變——**這支腳本現況沒有 `--conf`/`--iou`/`--imgsz`/`--frame_rate`/`--gmc_method` CLI flag，統一簽章後也不新增**（見本文件 Global Constraints）。

- [ ] **Step 1: 改寫整檔**

現有 `grid_search_botsort_reid.py` 的 `PARAM_GRIDS`（`:83-187`）內容不變；`main()`（`:881-1042`）內容不變，除了「收集影片路徑」區塊改呼叫 `discover_grid_search_videos()`（同 Task 7 Step 1，含刪除多餘的 `video_paths[: args.max_videos]` 切片行）；中間 `ParameterGrid`/`extract_assa_mean`/`score_params`/`_run_tracking_for_video`/`_resolve_gt_source_root`/`_setup_trial_gt`/`GridSearchRunner`/`generate_config_patch`（`:212-874`）與相關 import 全部移除，改成跟 Task 6/7 相同的 import 區塊。

`main()` 內「收集影片路徑」區塊（現有 `:976-988`，用 `grep -n "收集影片路徑"` 定位，已使用 `dataset_layout.list_stage1_sequences()`）：

Read 現有：
```python
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
```

改為：
```python
    # ── 收集影片路徑 ──────────────────────────────────────────────────────────
    video_paths = discover_grid_search_videos(cfg, args.video_glob, args.max_videos)
```

（同 Task 7 Step 1 的提醒：刪除後面多餘的 `video_paths = video_paths[: args.max_videos]` 切片行，避免對已切片過的清單再切一次。）

`GridSearchRunner(...)` 呼叫處（現有 `:1000-1011`）**⚠️ 這一步是本 Task 最容易出錯的地方**——現況這支腳本完全沒有把 `conf`/`iou`/`imgsz`/`frame_rate`/`gmc_method` 傳給 `GridSearchRunner`（其 `_run_tracking_for_video()` 內部寫死這些值、且完全沒有把 `gmc_method` 傳給 `create_tracker()`，逐字對照的是 `grid_search_bytetrack.py` 的簡化版，不是 `grid_search_botsort.py` 的完整參數化版）。統一簽章後**必須明確補上這幾個參數**，否則會落到 `GridSearchRunner.__init__` 的 class 預設值（`conf=0.15`/`iou=0.50`/`imgsz=640`/`frame_rate=30`/`gmc_method="ecc"`），其中 `imgsz`/`gmc_method` 會偷偷變成跟現行不同的值（現行 `imgsz=1280`、GMC 實際落到 `sparseOptFlow`，見本文件 Global Constraints 的推導過程）：

Read 現有：
```python
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
```

改為：
```python
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
```

因此本檔的 import 區塊也要加入 `get_min_conf_thresh`（同 Task 6 的 import 內容，`from module.tracker.config import get_min_conf_thresh, get_track_config_path, load_track_config`）。

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile module/grid_search/grid_search_botsort_reid.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 手動驗證——`--dry-run`**

Run: `python module/grid_search/grid_search_botsort_reid.py --target-class scooter --dry-run`
Expected: 印出參數組合數量（與重構前相同），無錯誤。

- [ ] **Step 4: 手動驗證——`--max-videos 1` + 縮小參數子集跑一組 trial，並核對 console 印出的常數值**

Run: `python module/grid_search/grid_search_botsort_reid.py --target-class scooter --max-videos 1 --params-subset match_thresh --verbose`
Expected: `GridSearchRunner.run()` 開頭印出的 `conf`/`iou`/`imgsz`/`frame_rate`/`gmc_method` 分別是 `<get_min_conf_thresh() 的值>`/`0.5`/`1280`/`30`/`sparseOptFlow`——**這是本 Task 的關鍵驗證點**，若看到 `imgsz: 640` 或 `gmc_method: ecc` 代表 Step 1 的常數值填錯了，需要修正後重跑。

- [ ] **Step 5: Commit**

```bash
git add module/grid_search/grid_search_botsort_reid.py
git commit -m "grid_search_botsort_reid.py: slim down to PARAM_GRIDS + main(), use core.py

Preserves current behavior exactly: this script has no --conf/--iou/--imgsz/
--frame_rate/--gmc_method CLI flags (unlike grid_search_botsort.py), so main()
now explicitly passes the hardcoded constants it always used
(conf=get_min_conf_thresh(), iou=0.5, imgsz=1280, frame_rate=30,
gmc_method=\"sparseOptFlow\" — the DEFAULT_TRACKER_CONFIG fallback this script
was silently relying on, not \"ecc\"/\"sift\" from the other two scripts).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: 三腳本一致性回歸驗證

**Files:**
- 無程式碼修改，純驗證。

**Interfaces:**
- 無。

- [ ] **Step 1: `--resume` 驗證**

對 `grid_search_bytetrack.py` 跑一組小範圍搜尋產生部分 CSV，中斷（`Ctrl+C` 或等它跑完 1-2 組後直接看 CSV），再用 `--resume` 續跑：

```bash
python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1 --params-subset match_thresh,new_track_thresh
# 中斷或等其中一次跑完，記下已完成的 combo_id
python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1 --params-subset match_thresh,new_track_thresh --resume
```

Expected: 第二次執行印出 `[resume] Loaded N completed trials from ...`，且 `remain` 正確排除已完成的組合數。

- [ ] **Step 2: 三支腳本各跑一組 trial，確認 CSV 欄位與現行格式一致**

```bash
python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1 --params-subset match_thresh
python module/grid_search/grid_search_botsort.py --target-class scooter --max-videos 1 --params-subset match_thresh
python module/grid_search/grid_search_botsort_reid.py --target-class scooter --max-videos 1 --params-subset match_thresh
```

Expected: 三份輸出的 CSV（`runs/grid_search/<tracker>/gs_scooter_*.csv`）欄位一致（`combo_id,score,elapsed_sec,params,match_thresh`），`score` 欄位非全部 `-inf`。

- [ ] **Step 3: 核對 `grid_search_bytetrack.py` 的 `conf`/`iou`/`imgsz`/`frame_rate` 與重構前寫死值完全一致**

Run: `python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1 --params-subset match_thresh --verbose`
Expected: `GridSearchRunner.run()` 開頭印出的 `conf`/`iou`/`imgsz`/`frame_rate` 分別是 `<get_min_conf_thresh() 的值>`/`0.5`/`1280`/`30`（`gmc_method` 印出 `none`，ByteTrack 本來就不使用，不影響追蹤行為）。

- [ ] **Step 4: 清除驗證產生的暫存輸出（`runs/grid_search/` 下本次驗證新增的目錄與 CSV，不動其餘既有內容）**

先列出本次驗證產生的檔案，確認範圍後再刪除：

```bash
find runs/grid_search -newer docs/superpowers/plans/2026-09-11-grid-search-dedup.md
```

確認清單只包含本次驗證產生的內容後才刪除，若不確定則保留、交由使用者自行清理。

- [ ] **Step 5: 更新 `2026-09-03-detection-module-design.md` 驗證計畫的完成註記（可選，若團隊有追蹤 spec 狀態的習慣）**

不修改 spec 內容本身，只在需要追蹤進度時於 commit message 或 PR 描述中註明「`2026-09-03-detection-module-design.md` 已隨本計畫 Task 3-4 完成」。

---

## Self-Review

1. **Spec coverage**：`2026-09-03-grid-search-dedup-design.md` 第 3.1 節（`core.py` 六個共用元件）→ Task 1-5；第 3.2 節（三腳本瘦身結構）→ Task 6-8；第 5 節驗證計畫四點→ Task 6-9 涵蓋（`--dry-run` 一致性、`--max-videos 1` trial、`--resume`、bytetrack 常數值核對）。`2026-09-03-detection-module-design.md` 第 3 節（`Detector` 整合）→ Task 3-4。第 0 節「前提依賴」（dataset 路徑重構）：已合併，本版本全面採用 `evaluation/dataset_layout.py`，不再有暫緩段落。
2. **2026-09-11 修訂記錄**：`discover_grid_search_videos()`（Task 2）、`score_params()` 的 `trial_dir` 組裝（Task 4）、`_resolve_gt_source_root()`（Task 2）已對照合併後的實際三支腳本（`53a4501`）重新核對，全部改為呼叫 `evaluation/dataset_layout.py`；Task 7-8 額外提醒刪除現有版本裡「先呼叫 `discover_grid_search_videos()` 又再切一次 `[: args.max_videos]`」的多餘切片行。
3. **Placeholder scan**：所有 Task 的程式碼區塊均為完整、可直接貼上的實作，無「TODO」/「依需求調整」等佔位文字。
4. **Type consistency**：`_run_tracking_for_video(detector: Detector, ...)`、`score_params(..., model_path: str, ...)` 內部建立 `Detector(model_path)`、`GridSearchRunner.__init__` 的 `conf/iou/imgsz/frame_rate/gmc_method` 參數名稱與 `score_params()`/`_run_tracking_for_video()` 一致，貫穿 Task 3-8 沒有改名。
5. **botsort_reid 常數值**（本計畫新增、spec 未明確處理的細節）：Task 8 已用 Global Constraints 段落與 commit message 明確記錄推導過程（`gmc_method="sparseOptFlow"` 而非 `"ecc"`/`"sift"`/`"none"`），並在 Task 8 Step 4、Task 9 Step 3 安排了可觀察的驗證點（`--verbose` 印出的常數值）。
