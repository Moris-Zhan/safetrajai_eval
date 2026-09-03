# Dataset 路徑結構重構 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `annotation_`/`mot_annotation_` 前綴字串拼接、新增 dataset id 對照表取代散落的中文路徑字面值、整段移除補幀（frame-interpolation compensation）死碼，讓 `track_eval.py`、三支 `module/grid_search/*.py`、`module/collision/` 對序列路徑的組裝全部收斂到 `evaluation/dataset_layout.py` 一份共用實作。

**Architecture:** 新增 `evaluation/dataset_layout.py`（pure function 集合，負責 SOURCE 掃描與 OUTPUT 路徑組裝）與 `config/datasets.yaml`（dataset id → 根路徑/子目錄名稱對照表）。`evaluation/config_loader.py::AppConfig` 在載入 `paths.dataset` 時查表填入預設值，仍允許逐欄位顯式覆蓋（escape hatch）。呼叫端（`track_eval.py`、grid_search、collision）一律改呼叫 `dataset_layout.py` 的函式，不再自行組字串；OUTPUT 端序列目錄從 `annotation_<seq>/` 改成 `<seq>/`，SOURCE 端既有資料完全不動。

**Tech Stack:** Python 3、PyYAML、pathlib、opencv-python（既有專案依賴，不新增套件）。本專案沒有 pytest/自動化測試框架（見 `CLAUDE.md`：「無測試」），因此以下「測試」步驟一律採用**可直接執行的 assert 驗證腳本**（`python3` 執行、無例外即通過），不是 pytest test case；純函式驗證腳本放在 scratchpad 執行、不進版控，管線層級的驗證則依 spec 第 5 節的手動流程（跑 `track_eval.py --config ...` 限量序列、比對輸出）。

## Global Constraints

- 固定 6 類別與 ID 不可變動：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`（`.claude/rules/project-and-pipeline.md`）。
- SOURCE 端既有磁碟資料（影片、人工標註 GT）的實體路徑與命名**不可更動**——本次只改我們自己產生的 OUTPUT 樹命名慣例。
- `python track_eval.py --config <path>` 是穩定對外 CLI 介面，`paths`/`models`/`stages`/`eval_flags`/`tracking_params`/`class_tracking` 等 YAML 區塊名稱不可任意改名（`.claude/rules/project-and-pipeline.md`）。
- 不執行全量影片、不覆寫 `runs/` 既有輸出、不改動模型/資料集路徑——每個驗證步驟都先在限量序列上跑。
- 補幀（frame-interpolation compensation）邏輯與其死碼路徑依賴整段移除，不保留相容分支（已查證目前無任何 config 引用該變體，見 spec 第 1 節）。
- `module/tracker/**/*.py` 的 `ClassAwareBOTSORT`/`ClassAwareBYTETracker` override 方法一字不可更動、不可搬離（`.claude/rules/tracker.md`）——本計畫不觸碰這些檔案，僅供交叉確認不誤觸。

---

## File Structure

```
config/
├── datasets.yaml                          [新增] dataset id 對照表
├── bytetrack.yaml                         [修改] paths: 精簡為 dataset: + 差異欄位
├── botsort_noreid.yaml                    [修改] 同上
├── botsort_reid.yaml                      [修改] 同上
├── botsort_save_reid_crops.yaml           [修改] 同上
├── botsort_save_reid_crops_video.yaml     [修改] 同上（dataset: video_seqs）
├── collision_detect.yaml                  [修改] 同上
├── grid_search_botsort.yaml               [修改] 同上
└── grid_search_bytetrack.yaml             [修改] 同上

evaluation/
├── dataset_layout.py                      [新增] 路徑組裝 pure function + registry loader
└── config_loader.py                       [修改] 解析 paths.dataset

track_eval.py                              [修改] 移除補幀邏輯；stage1/stage3 路徑組裝改呼叫 dataset_layout.py

module/grid_search/
├── grid_search_botsort.py                 [修改] trial_dir/_resolve_gt_source_root/影片蒐集改用 dataset_layout.py
├── grid_search_bytetrack.py               [修改] 同上
└── grid_search_botsort_reid.py            [修改] 同上

module/collision/
├── homography_pick_aid.py                 [修改] 硬編碼中文路徑改用 dataset registry
├── __init__.py                            [修改] 移除 .replace("annotation_", "")
├── trajectory_adapter.py                  [修改] 同上
├── output.py                              [修改] docstring 更新（無邏輯變動）
├── homography_points.py                   [修改] docstring 更新（無邏輯變動）
└── homography_compare_video.py            [修改] docstring 更新（無邏輯變動）
```

---

### Task 1: `config/datasets.yaml` — dataset id 對照表

**Files:**
- Create: `config/datasets.yaml`

**Interfaces:**
- Produces: YAML 結構 `{"datasets": {<id>: {"root": str, "run_det_eval_dir": str, "run_track_eval_dir": str}}}`，供 Task 2 的 `load_dataset_registry()` 讀取。

- [ ] **Step 1: 建立檔案**

```yaml
# config/datasets.yaml
# dataset id → 根路徑/子目錄名稱對照表。
# config/*.yaml 的 paths.dataset 欄位用短 id 取代重複的中文長路徑字面值；
# 詳見 docs/superpowers/specs/2026-09-03-dataset-path-refactor-design.md 第 3.3 節。

datasets:
  cctv_all8:
    root: "./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/"
    run_det_eval_dir: annotations_all8
    run_track_eval_dir: mot_annotations_all8
  video_seqs:
    root: "./dataset/video_seqs/"
    run_det_eval_dir: annotations_video
    run_track_eval_dir: mot_annotations_video
```

- [ ] **Step 2: 驗證 YAML 可正確解析**

Run:
```bash
python3 -c "
import yaml
with open('config/datasets.yaml', encoding='utf-8') as f:
    data = yaml.safe_load(f)
assert 'datasets' in data
assert data['datasets']['cctv_all8']['run_det_eval_dir'] == 'annotations_all8'
assert data['datasets']['video_seqs']['root'] == './dataset/video_seqs/'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add config/datasets.yaml
git commit -m "Add config/datasets.yaml dataset id registry

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `evaluation/dataset_layout.py` — 路徑組裝 pure function

**Files:**
- Create: `evaluation/dataset_layout.py`
- Test（暫存，不進版控）: `scratchpad/test_dataset_layout.py`

**Interfaces:**
- Consumes: 無（只依賴 stdlib + PyYAML + `Path`）。
- Produces（供 Task 3-9 使用）:
  - `load_dataset_registry(path: Path | str = <config/datasets.yaml 絕對路徑>) -> dict[str, dict]`
  - `discover_source_sequences(source_video_dir: Path, run_det_eval_dir: str) -> list[Path]`
  - `stage1_seq_dir(det_output_dir: Path, seq_name: str) -> Path`
  - `stage2_seq_dir(track_output_dir: Path, seq_name: str) -> Path`
  - `source_gt_dir(source_video_dir: Path, seq_name: str) -> Path`（回傳值含尾端 `/gt`）
  - `seq_name_from_video_path(video_path: str | Path) -> str`
  - `list_stage1_sequences(det_output_dir: Path) -> list[Path]`

- [ ] **Step 1: 寫驗證腳本（RED）**

在 scratchpad 建立 `test_dataset_layout.py`（`PROJECT_ROOT` 依實際專案路徑調整）：

```python
# scratchpad/test_dataset_layout.py — 手動驗證腳本，不進版控
import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/moris/Dcoker_Project/Project/safetrajai_eval")
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_layout import (
    load_dataset_registry,
    discover_source_sequences,
    stage1_seq_dir,
    stage2_seq_dir,
    source_gt_dir,
    seq_name_from_video_path,
    list_stage1_sequences,
)

# 1. registry 讀取
registry = load_dataset_registry()
assert registry["cctv_all8"]["run_det_eval_dir"] == "annotations_all8"
assert registry["video_seqs"]["run_track_eval_dir"] == "mot_annotations_video"

# 2. seq_name_from_video_path
assert seq_name_from_video_path("/a/b/C000025_15s_1.mp4") == "C000025_15s_1"
assert seq_name_from_video_path(Path("/a/b/foo.mp4")) == "foo"

# 3. stage1_seq_dir / stage2_seq_dir（純路徑組裝，無前綴）
det_out = Path("/runs/track_eval/annotations_all8")
assert stage1_seq_dir(det_out, "C000025_15s_1") == det_out / "C000025_15s_1"
track_out = Path("/runs/track_eval/mot_annotations_all8")
assert stage2_seq_dir(track_out, "C000025_15s_1") == track_out / "C000025_15s_1"

# 4. source_gt_dir（SOURCE 端既有命名不變，含 mot_annotation_ 前綴 + 尾端 /gt）
src = Path("/dataset/中華三路_民生二路(東北角)yolo驗證用_all8")
expected_gt = src / "mot_annotations_all8" / "mot_annotation_C000025_15s_1" / "gt"
assert source_gt_dir(src, "C000025_15s_1") == expected_gt

# 5. discover_source_sequences：對實際專案資料跑一次，確認找到序列、跳過 Demo.mp4
real_src = PROJECT_ROOT / "dataset" / "中華三路_民生二路(東北角)yolo驗證用_all8"
seqs = discover_source_sequences(real_src, "annotations_all8")
assert len(seqs) > 0, "應該至少找到一支序列影片"
assert all(not p.stem.startswith("Demo") for p in seqs)
assert all(p.suffix == ".mp4" for p in seqs)

# 6. list_stage1_sequences：對一個假的（尚不存在）輸出目錄，回傳空清單而不是報錯
fake_det_out = PROJECT_ROOT / "runs" / "__does_not_exist__"
assert list_stage1_sequences(fake_det_out) == []

print("ALL OK")
```

- [ ] **Step 2: 執行驗證腳本，確認因模組不存在而失敗**

Run: `python3 scratchpad/test_dataset_layout.py`
Expected: `ModuleNotFoundError: No module named 'evaluation.dataset_layout'`

- [ ] **Step 3: 實作 `evaluation/dataset_layout.py`**

```python
# evaluation/dataset_layout.py
"""
Dataset 路徑組裝：集中 SOURCE 掃描與 OUTPUT 序列目錄命名，供
track_eval.py、module/grid_search/*.py、module/collision/ 共用。

命名慣例（見 docs/superpowers/specs/2026-09-03-dataset-path-refactor-design.md）：
    SOURCE（唯讀，既有資料，命名不變）：
        SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_<seq>/<seq>.mp4
        SOURCE_VIDEO_DIR/mot_annotations_all8/mot_annotation_<seq>/gt/
    OUTPUT（本專案產生，無前綴）：
        OUTPUT_ROOT/<run_det_eval_dir>/<seq>/...      （stage1）
        OUTPUT_ROOT/<run_track_eval_dir>/<seq>/...    （stage2）
"""

from __future__ import annotations

import glob
from pathlib import Path

import yaml

_DATASETS_YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "datasets.yaml"


def load_dataset_registry(path: Path | str = _DATASETS_YAML_PATH) -> dict[str, dict]:
    """讀取 config/datasets.yaml，回傳 {dataset_id: {root, run_det_eval_dir, run_track_eval_dir}}。
    檔案不存在時回傳空 dict（呼叫端自行決定是否視為錯誤）。"""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("datasets", {}) or {}


def seq_name_from_video_path(video_path: str | Path) -> str:
    """序列名稱的單一來源：影片檔名（不含副檔名）。"""
    return Path(video_path).stem


def discover_source_sequences(source_video_dir: Path, run_det_eval_dir: str) -> list[Path]:
    """掃描 SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_*/**/*.mp4，跳過 Demo.mp4。
    取代舊版 track_eval.py 的 glob + 補幀相關檔名過濾（該過濾邏輯已隨補幀邏輯一併移除，
    見 spec 第 3.6 節）。"""
    pattern = str(Path(source_video_dir) / run_det_eval_dir / "annotation_*" / "**" / "*.mp4")
    return sorted(
        Path(p) for p in glob.glob(pattern, recursive=True)
        if not Path(p).stem.startswith("Demo")
    )


def stage1_seq_dir(det_output_dir: Path, seq_name: str) -> Path:
    """OUTPUT_ROOT/<run_det_eval_dir>/<seq_name>/"""
    return Path(det_output_dir) / seq_name


def stage2_seq_dir(track_output_dir: Path, seq_name: str) -> Path:
    """OUTPUT_ROOT/<run_track_eval_dir>/<seq_name>/"""
    return Path(track_output_dir) / seq_name


def source_gt_dir(source_video_dir: Path, seq_name: str) -> Path:
    """SOURCE_VIDEO_DIR/mot_annotations_all8/mot_annotation_<seq_name>/gt/
    （SOURCE 端既有命名不變，此函式只是讀取，不產生新命名。回傳值已含尾端 /gt。）"""
    return Path(source_video_dir) / "mot_annotations_all8" / f"mot_annotation_{seq_name}" / "gt"


def list_stage1_sequences(det_output_dir: Path) -> list[Path]:
    """列舉 OUTPUT_ROOT/<run_det_eval_dir>/ 下所有已產生的序列目錄（無前綴，直接是子目錄）。
    目錄不存在時回傳空清單，不拋例外。"""
    det_output_dir = Path(det_output_dir)
    if not det_output_dir.is_dir():
        return []
    return sorted(p for p in det_output_dir.iterdir() if p.is_dir())
```

- [ ] **Step 4: 重新執行驗證腳本，確認全數通過**

Run: `python3 scratchpad/test_dataset_layout.py`
Expected: `ALL OK`

- [ ] **Step 5: Commit**

```bash
git add evaluation/dataset_layout.py
git commit -m "Add evaluation/dataset_layout.py: shared path-building functions

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `evaluation/config_loader.py` — 解析 `paths.dataset`

**Files:**
- Modify: `evaluation/config_loader.py:111-121`（`paths` 區塊解析）

**Interfaces:**
- Consumes: `evaluation.dataset_layout.load_dataset_registry()`（Task 2）。
- Produces: `AppConfig` 對外屬性（`cfg.SOURCE_VIDEO_DIR`、`cfg.run_det_eval_dir`、`cfg.run_track_eval_dir`）行為不變，新增支援 `paths.dataset` 欄位；`dataset` 指向不存在 id 時拋 `ValueError`。

- [ ] **Step 1: 寫驗證腳本（RED）**

在 scratchpad 建立 `test_config_dataset.py`：

```python
# scratchpad/test_config_dataset.py — 手動驗證腳本，不進版控
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path("/mnt/moris/Dcoker_Project/Project/safetrajai_eval")
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config_loader import load_config

tmpdir = Path(tempfile.mkdtemp())

# 情境 1：只給 dataset id，root/run_det_eval_dir/run_track_eval_dir 應查表帶入
cfg_a = tmpdir / "a.yaml"
cfg_a.write_text(
    "paths:\n"
    "  dataset: cctv_all8\n"
    "  OUTPUT_ROOT: ./runs/track_eval\n",
    encoding="utf-8",
)
cfg = load_config(str(cfg_a))
assert cfg.run_det_eval_dir == "annotations_all8"
assert cfg.run_track_eval_dir == "mot_annotations_all8"
assert "中華三路" in str(cfg.SOURCE_VIDEO_DIR)

# 情境 2：dataset id + 顯式覆蓋 run_det_eval_dir，應以顯式值為準（逐欄位覆蓋）
cfg_b = tmpdir / "b.yaml"
cfg_b.write_text(
    "paths:\n"
    "  dataset: cctv_all8\n"
    "  run_det_eval_dir: annotations_custom\n"
    "  OUTPUT_ROOT: ./runs/track_eval\n",
    encoding="utf-8",
)
cfg = load_config(str(cfg_b))
assert cfg.run_det_eval_dir == "annotations_custom"
assert cfg.run_track_eval_dir == "mot_annotations_all8"  # 沒顯式覆蓋，仍查表

# 情境 3：不存在的 dataset id，應明確 ValueError（不可靜默回退預設值）
cfg_c = tmpdir / "c.yaml"
cfg_c.write_text("paths:\n  dataset: no_such_dataset\n", encoding="utf-8")
try:
    load_config(str(cfg_c))
    raise AssertionError("應該要拋 ValueError")
except ValueError as e:
    assert "no_such_dataset" in str(e)

# 情境 4：沒有 dataset、也沒有顯式路徑，回退既有 _DEFAULTS（行為不變）
cfg_d = tmpdir / "d.yaml"
cfg_d.write_text("paths:\n  OUTPUT_ROOT: ./runs/track_eval\n", encoding="utf-8")
cfg = load_config(str(cfg_d))
assert cfg.run_det_eval_dir == "annotations_all8"  # _DEFAULTS 的值

print("ALL OK")
```

- [ ] **Step 2: 執行驗證腳本，確認情境 2、3 失敗（現行程式碼還不認得 `paths.dataset`）**

Run: `python3 scratchpad/test_config_dataset.py`
Expected: `AssertionError`（情境 3 不會拋 `ValueError` 而是 `AssertionError: 應該要拋 ValueError`）

- [ ] **Step 3: 修改 `evaluation/config_loader.py` 的 `paths` 區塊解析**

Read 現有第 111-121 行：
```python
        # paths 區塊
        paths = raw.get("paths", {}) or {}
        d["SOURCE_VIDEO_DIR"]     = paths.get("SOURCE_VIDEO_DIR", d["SOURCE_VIDEO_DIR"])
        d["SOURCE_VIDEO_DIR"]     = Path(d["SOURCE_VIDEO_DIR"])

        d["run_det_eval_dir"]     = paths.get("run_det_eval_dir", d["run_det_eval_dir"])
        d["run_track_eval_dir"]   = paths.get("run_track_eval_dir", d["run_track_eval_dir"])
        d["TRACKEVAL_OUT_DIR"]    = paths.get("TRACKEVAL_OUT_DIR", d["TRACKEVAL_OUT_DIR"])
```

改為：
```python
        # paths 區塊
        paths = raw.get("paths", {}) or {}

        # dataset id 對照表（見 evaluation/dataset_layout.py、config/datasets.yaml）：
        # 有 paths.dataset 時先查表取得 root/run_det_eval_dir/run_track_eval_dir 當起始值，
        # 顯式欄位（下方 paths.get(..., registry_value)）仍可逐欄位覆蓋（escape hatch）。
        dataset_id = paths.get("dataset")
        registry_entry: dict = {}
        if dataset_id is not None:
            from evaluation.dataset_layout import load_dataset_registry
            registry = load_dataset_registry()
            if dataset_id not in registry:
                raise ValueError(
                    f"paths.dataset={dataset_id!r} 不存在於 config/datasets.yaml，"
                    f"可用的 dataset id：{sorted(registry.keys())}"
                )
            registry_entry = registry[dataset_id]

        d["SOURCE_VIDEO_DIR"]     = paths.get("SOURCE_VIDEO_DIR", registry_entry.get("root", d["SOURCE_VIDEO_DIR"]))
        d["SOURCE_VIDEO_DIR"]     = Path(d["SOURCE_VIDEO_DIR"])

        d["run_det_eval_dir"]     = paths.get("run_det_eval_dir", registry_entry.get("run_det_eval_dir", d["run_det_eval_dir"]))
        d["run_track_eval_dir"]   = paths.get("run_track_eval_dir", registry_entry.get("run_track_eval_dir", d["run_track_eval_dir"]))
        d["TRACKEVAL_OUT_DIR"]    = paths.get("TRACKEVAL_OUT_DIR", d["TRACKEVAL_OUT_DIR"])
```

其餘 `paths` 區塊解析（`OUTPUT_ROOT`、`base_dir`、`COLLISION_OUT_DIR` 等）不動。

- [ ] **Step 4: 重新執行驗證腳本，確認全數通過**

Run: `python3 scratchpad/test_config_dataset.py`
Expected: `ALL OK`

- [ ] **Step 5: Commit**

```bash
git add evaluation/config_loader.py
git commit -m "config_loader: resolve paths.dataset via dataset_layout registry

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `track_eval.py` — 移除補幀邏輯

**Files:**
- Modify: `track_eval.py:146-165`（`VideoSequenceTracker.run()` 開頭）、`track_eval.py:248-250`（`should_skip_frame` 呼叫處）

**Interfaces:**
- Consumes: 無新依賴。
- Produces: `VideoSequenceTracker.run()` 對外簽章與回傳值不變；`show_frame_id` 語意變成「逐幀遞增，等於 `frame_id`」（因為不再跳幀，兩者永遠相等，但變數本身保留，Task 5/6 不動這部分）。

- [ ] **Step 1: 確認現行行為基準（RED 對照組）**

因為要刪除的是「跳幀」這個分支，而不是新增功能，這裡沒有傳統意義的「先寫失敗測試」。改用**行為基準對照**：先記錄現行序列（皆無 `_frame_comparison.json`）的探索結果，作為 Task 14 完成後要比對「不變」的基準。

Run：

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from evaluation.config_loader import load_config
from evaluation.dataset_layout import discover_source_sequences
cfg = load_config('config/bytetrack.yaml')
seqs = discover_source_sequences(cfg.SOURCE_VIDEO_DIR, cfg.run_det_eval_dir)
print(len(seqs), seqs[0] if seqs else None)
"
```
Expected: 印出 > 0 的序列數與第一支序列路徑（記下這個數字，Task 14 完成後比對相同）。

- [ ] **Step 2: 刪除補幀讀取邏輯**

Read 現有 `track_eval.py:146-165`：
```python
        orig_frames = None

        # 讀取 frame_comparison.json 取得 interpolated_frame_ids
        comparison_json_path = Path(video_path).with_name(f"{Path(video_path).stem}_frame_comparison.json")
        interpolated_frame_ids = set()
        if comparison_json_path.exists():
            with open(comparison_json_path, "r", encoding="utf-8") as f:
                comparison_data = json.load(f)
                # 根據常見格式，假設 {"interpolated_frame_ids": [id1, id2, ...]}
                interpolated_frame_ids = set(comparison_data["interpolated_frame_ids"])
                orig_frames = comparison_data["summary"]["orig_frames"]
        else:
            print(f"Warning: {comparison_json_path} not found. No frames will be excluded.")

        def should_skip_frame(frame_id: int) -> bool:
            return frame_id in interpolated_frame_ids

        # 用法參考：
        # if should_skip_frame(cur_frame_id): continue # 在處理 frame 時跳過

        cap = cv2.VideoCapture(video_path)
```

改為：
```python
        cap = cv2.VideoCapture(video_path)
```

同時刪除下方依賴 `orig_frames` 的分支：
```python
        print(f"Video        : {video_path}")
        if orig_frames != None:
            total_frames = orig_frames
            print(f"Video        : {width}x{height} @ {fps}fps, {orig_frames} Original frames")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video        : {width}x{height} @ {fps}fps, {total_frames} frames")

        # Video        : 640x480 @ 4fps, 72 frames
        # Video        : 640x480 @ 30fps, 449 frames
        # print(f"Video        : {width}x{height} @ {fps}fps, {total_frames} frames")
```

改為：
```python
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video        : {video_path}")
        print(f"Video        : {width}x{height} @ {fps}fps, {total_frames} frames")
```

- [ ] **Step 3: 刪除主迴圈裡的跳幀判斷**

Read 現有 `track_eval.py:248-253`：
```python
            # 檢查是否跳過當前幀
            if should_skip_frame(frame_id):
                continue
            else:
                avg_track_time.append(tr_st - en)
                show_frame_id = len(avg_track_time)
```

改為：
```python
            avg_track_time.append(tr_st - en)
            show_frame_id = len(avg_track_time)
```

**注意**：這行以下（原本 `else:` 分支下的 label/image 存檔、reid 裁切、標注繪圖、進度輸出區塊）原本是縮排在 `else:` 底下，改動後要**往左縮排一級**，改成跟 `avg_track_time.append(...)` 同層。用編輯器的「移除一層縮排」操作處理整段，不要逐行手動改，避免漏改或改錯行。

- [ ] **Step 4: 移除現在用不到的 `import json`**

`track_eval.py:55` 的 `import json` 只被剛刪除的補幀讀取邏輯使用，確認檔案內其餘地方沒有其他 `json.` 呼叫後移除這行 import。

Run: `grep -n "json\." track_eval.py`
Expected: 沒有任何輸出（確認可以安全刪除 import）

- [ ] **Step 5: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 6: 手動驗證（挑 1 支序列跑 stage1，確認補幀相關程式碼移除後行為不受影響）**

依 spec 第 5 節驗證計畫，對 `config/bytetrack.yaml`（限制在 1 支序列，可暫時修改該檔案的來源或用其他方式限制範圍）跑 `python track_eval.py --config config/bytetrack.yaml`，確認：
- 程式正常執行完畢，不因移除 `import json`/`comparison_json_path` 相關邏輯而報錯。
- 輸出的 `labels/`、`images/` 檔案數量與影片實際 frame 數一致（這些序列本來就沒有 `_frame_comparison.json`，現況已經是「不跳幀」路徑，理論上輸出應與改動前逐位元組相同）。

（此步驟先確認移除補幀邏輯本身沒有引入語法/邏輯錯誤即可；完整輸出目錄結構驗證留到 Task 5 完成後一併做，此時目錄名稱仍是舊的 `annotation_<seq>/`，這是預期的。）

- [ ] **Step 7: Commit**

```bash
git add track_eval.py
git commit -m "track_eval.py: remove dead frame-interpolation-compensation logic

No config currently points at the (compasate) source variant this
logic was written for; removal is behavior-preserving for all
active scenarios per 2026-09-03-dataset-path-refactor-design.md section 3.6.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: `track_eval.py` — stage1 路徑組裝改用 `dataset_layout.py`

**Files:**
- Modify: `track_eval.py:63-73`（import 區塊）、`track_eval.py:186-199`（`VideoSequenceTracker.run()` 序列目錄命名）、`track_eval.py:514-521`（`run_stage1()` 影片探索）、`track_eval.py:575-588`（`run_stage1()` 目錄複製）

**Interfaces:**
- Consumes: `evaluation.dataset_layout.discover_source_sequences()`、`stage1_seq_dir()`、`stage2_seq_dir()`、`source_gt_dir()`、`seq_name_from_video_path()`（Task 2）。
- Produces: `VideoSequenceTracker.run()` 回傳的 `output_path` 現在是 `det_output_dir/<seq_name>/`（無 `annotation_` 前綴），下游 Task 6（`run_stage3()`）與後續模組依賴此新命名。

- [ ] **Step 1: 新增 import**

Read 現有 `track_eval.py:63-73`：
```python
from module.detection.detector import Detector
from module.tracker import create_tracker, CLASS_NAMES
from module.tracker.config import (
    get_track_config_path,
    load_track_config,
)
from module.collision import run_collision_detection_for_sequence
from utils.visualization import draw_polygon_and_save, save_fst_frame
from evaluation.mot_format import save_mot_format, make_det_eval_dataset
from evaluation.pipeline import run_eval_pipeline
from evaluation.config_loader import load_config
```

改為（新增最後一行）：
```python
from module.detection.detector import Detector
from module.tracker import create_tracker, CLASS_NAMES
from module.tracker.config import (
    get_track_config_path,
    load_track_config,
)
from module.collision import run_collision_detection_for_sequence
from utils.visualization import draw_polygon_and_save, save_fst_frame
from evaluation.mot_format import save_mot_format, make_det_eval_dataset
from evaluation.pipeline import run_eval_pipeline
from evaluation.config_loader import load_config
from evaluation import dataset_layout
```

- [ ] **Step 2: `VideoSequenceTracker.run()` 序列目錄命名改用 `dataset_layout.stage1_seq_dir()`**

Read 現有 `track_eval.py:186-199`（Task 4 完成後行號會往前偏移，用 `grep -n "seq_stem    = Path"` 重新定位）：
```python
        seq_stem    = Path(video_path).stem
        output_path = output_dir / f"annotation_{seq_stem}"
        output_path.mkdir(parents=True, exist_ok=True)
```

改為：
```python
        seq_stem    = dataset_layout.seq_name_from_video_path(video_path)
        output_path = dataset_layout.stage1_seq_dir(output_dir, seq_stem)
        output_path.mkdir(parents=True, exist_ok=True)
```

其餘用到 `seq_stem` 的地方（`labels_path`、`images_path`、`stem_name = seq_stem.replace(...)`、reid crop 存檔路徑）不動，因為 `seq_stem` 這個變數本身的值不變（`Path(video_path).stem`），只是取值方式改用集中函式。

- [ ] **Step 3: `run_stage1()` 影片探索改用 `dataset_layout.discover_source_sequences()`**

Read 現有 `track_eval.py:514-521`（用 `grep -n "搜尋來源影片"` 定位）：
```python
        # 搜尋來源影片（跳過 Demo.mp4）
        mp4_pattern = str(cfg.SOURCE_VIDEO_DIR / cfg.run_det_eval_dir / "annotation_*" / "**" / "*.mp4")

        mp4s = sorted(
            p for p in glob.glob(mp4_pattern, recursive=True)
            if not Path(p).stem.startswith("Demo")
            if "_org.mp4" not in p
        )
```

改為：
```python
        # 搜尋來源影片（跳過 Demo.mp4）
        mp4s = [str(p) for p in dataset_layout.discover_source_sequences(cfg.SOURCE_VIDEO_DIR, cfg.run_det_eval_dir)]
```

檔案頂端的 `import glob` 若確認全檔案沒有其他用途則一併移除（用 `grep -n "glob\." track_eval.py` 確認，若還有其他呼叫則保留）。

- [ ] **Step 4: `run_stage1()` 目錄複製改用 `dataset_layout.stage2_seq_dir()` / `source_gt_dir()`**

Read 現有 `track_eval.py:575-588`（用 `grep -n "mot_seq_dir = "` 定位）：
```python
            # 將 trackers/ 由annotations_all8複製到 runs/track_eval/mot_annotations_all8/mot_annotation_<seq>/trackers/
            mot_seq_dir = track_output_dir / f"mot_annotation_{output_path.stem.replace('annotation_', '')}"
            shutil.copytree(
                output_path / "trackers",  # annotations_all8
                mot_seq_dir / "trackers",  # mot_annotations_all8
                dirs_exist_ok=True,
            )
            # 準備GT 驗證資料集
            src_gt_dir = cfg.SOURCE_VIDEO_DIR / "mot_annotations_all8" / f"mot_annotation_{output_path.stem.replace('annotation_', '')}"
            shutil.copytree(
                src_gt_dir / "gt",  # dataset/中華三路_民生二路(東北角)yolo驗證用_all8/mot_annotations_all8/mot_annotation_<seq>
                mot_seq_dir / "gt", # runs/track_eval/mot_annotations_all8/mot_annotation_<seq>
                dirs_exist_ok=True,
            )
            print(f"  trackers → {mot_seq_dir / 'trackers'}")
```

改為：
```python
            # 將 trackers/ 由 annotations_all8 複製到 runs/track_eval/mot_annotations_all8/<seq>/trackers/
            mot_seq_dir = dataset_layout.stage2_seq_dir(track_output_dir, output_path.stem)
            shutil.copytree(
                output_path / "trackers",  # annotations_all8/<seq>
                mot_seq_dir / "trackers",  # mot_annotations_all8/<seq>
                dirs_exist_ok=True,
            )
            # 準備 GT 驗證資料集（SOURCE 端既有命名不變，source_gt_dir() 回傳值已含尾端 /gt）
            shutil.copytree(
                dataset_layout.source_gt_dir(cfg.SOURCE_VIDEO_DIR, output_path.stem),
                mot_seq_dir / "gt",
                dirs_exist_ok=True,
            )
            print(f"  trackers → {mot_seq_dir / 'trackers'}")
```

- [ ] **Step 5: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 6: 手動驗證——挑 1-2 支序列跑完整 stage1**

依 spec 第 5 節，對 `config/bytetrack.yaml`（限制在 1-2 支序列）跑：

```bash
python track_eval.py --config config/bytetrack.yaml
```

確認：
- `runs/track_eval/annotations_all8/<seq>/`（**無** `annotation_` 前綴）確實產生，內含 `images/`、`labels/`、`trackers/`、`data.yaml`、`<seq>.mp4`。
- `runs/track_eval/mot_annotations_all8/<seq>/`（**無** `mot_annotation_` 前綴）確實產生，內含 `trackers/`、`gt/`。
- console 沒有 `FileNotFoundError`/`FileExistsError`（`source_gt_dir()` 的 SOURCE 端路徑是否正確組出來的直接證據）。

- [ ] **Step 7: Commit**

```bash
git add track_eval.py
git commit -m "track_eval.py stage1: use dataset_layout for path assembly, drop annotation_ prefix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: `track_eval.py` — `run_stage3()` 改用 `list_stage1_sequences()`

**Files:**
- Modify: `track_eval.py:626-640`（`run_stage3()` 序列列舉）

**Interfaces:**
- Consumes: `evaluation.dataset_layout.list_stage1_sequences()`（Task 2）、Task 5 產出的無前綴 `<seq>/` 目錄。
- Produces: 無新介面，`run_stage3()` 對外行為（呼叫 `run_collision_detection_for_sequence()` 的參數）不變。

- [ ] **Step 1: 修改序列列舉邏輯**

Read 現有 `track_eval.py:626-640`（用 `grep -n "seq_dirs = sorted"` 定位）：
```python
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        seq_dirs = sorted(det_output_dir.glob("annotation_*"))
        if not seq_dirs:
            print(f"⚠️  No stage1 output found under: {det_output_dir}（stage3 需要先跑過 stage1）")
            return

        print(f"\n{'='*70}")
        print(f"  STAGE 3 — Collision Detection (TTC/PET)")
        print(f"  collision_output   : {cfg.collision_output_dir}")
        print(f"  homography_points  : {cfg.homography_points_dir}")
        print(f"{'='*70}\n")

        for seq_dir in seq_dirs:
            seq_name = seq_dir.stem.replace("annotation_", "")
```

改為：
```python
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        seq_dirs = dataset_layout.list_stage1_sequences(det_output_dir)
        if not seq_dirs:
            print(f"⚠️  No stage1 output found under: {det_output_dir}（stage3 需要先跑過 stage1）")
            return

        print(f"\n{'='*70}")
        print(f"  STAGE 3 — Collision Detection (TTC/PET)")
        print(f"  collision_output   : {cfg.collision_output_dir}")
        print(f"  homography_points  : {cfg.homography_points_dir}")
        print(f"{'='*70}\n")

        for seq_dir in seq_dirs:
            seq_name = seq_dir.stem
```

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 手動驗證——stage3-only rerun 沿用 Task 5 剛產生的 stage1 輸出**

用 `config/collision_detect.yaml`（`stages.run_stage1: false`、`run_stage3: true`，沿用 Task 5 的 `runs/track_eval/annotations_all8/` 輸出，若該 config 的 `OUTPUT_ROOT` 與 `bytetrack.yaml` 不同，先暫時對齊或改用同一個 `OUTPUT_ROOT` 測試）：

```bash
python track_eval.py --config config/collision_detect.yaml
```

確認：
- `list_stage1_sequences()` 正確列出 Task 5 產生的 `<seq>/` 目錄（無前綴）。
- `run_collision_detection_for_sequence()` 對每個序列正常執行（此步驟需 Task 12 的 `module/collision/__init__.py` 修改一併完成才會完全正確——若在此步驟先跑，`seq_name` 因為 `module/collision/__init__.py` 內部仍是舊的 `.replace("annotation_", "")` 邏輯而變成 no-op，不會報錯，可以先驗證 `track_eval.py` 這端本身沒問題，Task 12 完成後在 Task 14 再整體重跑一次完整確認）。

- [ ] **Step 4: Commit**

```bash
git add track_eval.py
git commit -m "track_eval.py run_stage3: use list_stage1_sequences(), drop replace() prefix stripping

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: 8 個 `config/*.yaml` — `paths:` 區塊精簡為 `dataset:`

**Files:**
- Modify: `config/bytetrack.yaml`、`config/botsort_noreid.yaml`、`config/botsort_reid.yaml`、`config/botsort_save_reid_crops.yaml`、`config/collision_detect.yaml`、`config/grid_search_botsort.yaml`、`config/grid_search_bytetrack.yaml`（7 個，`dataset: cctv_all8`）
- Modify: `config/botsort_save_reid_crops_video.yaml`（1 個，`dataset: video_seqs`）

**Interfaces:**
- Consumes: Task 3 的 `paths.dataset` 解析邏輯、Task 1 的 `config/datasets.yaml`。
- Produces: 無（純設定檔精簡，`load_config()` 對外行為不變）。

- [ ] **Step 1: 修改 6 個用 `cctv_all8` 且結構相同的情境檔**

以下 6 個檔案，把現有：
```yaml
paths:
  SOURCE_VIDEO_DIR: ./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/
  TRACKEVAL_OUT_DIR: trackeval_output
  OUTPUT_ROOT: ./runs/track_eval
  run_det_eval_dir: annotations_all8
  run_track_eval_dir: mot_annotations_all8
```
改為：
```yaml
paths:
  dataset: cctv_all8
  TRACKEVAL_OUT_DIR: trackeval_output
  OUTPUT_ROOT: ./runs/track_eval
```

套用到：`config/bytetrack.yaml`、`config/botsort_noreid.yaml`、`config/botsort_reid.yaml`、`config/botsort_save_reid_crops.yaml`、`config/grid_search_botsort.yaml`、`config/grid_search_bytetrack.yaml`（逐一 Edit，`paths:` 區塊其餘無關的註解/空行維持原樣，只替換 `SOURCE_VIDEO_DIR`/`run_det_eval_dir`/`run_track_eval_dir` 三行為 `dataset: cctv_all8`）。

- [ ] **Step 2: 修改 `config/collision_detect.yaml`（多一個 `COLLISION_OUT_DIR` 欄位）**

現有：
```yaml
paths:
  SOURCE_VIDEO_DIR: ./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/
  TRACKEVAL_OUT_DIR: trackeval_output
  COLLISION_OUT_DIR: collision_output
  OUTPUT_ROOT: ./runs/track_eval
  run_det_eval_dir: annotations_all8
  run_track_eval_dir: mot_annotations_all8
```
改為：
```yaml
paths:
  dataset: cctv_all8
  TRACKEVAL_OUT_DIR: trackeval_output
  COLLISION_OUT_DIR: collision_output
  OUTPUT_ROOT: ./runs/track_eval
```

- [ ] **Step 3: 修改 `config/botsort_save_reid_crops_video.yaml`**

現有：
```yaml
paths:
  SOURCE_VIDEO_DIR: ./dataset/video_seqs/
  TRACKEVAL_OUT_DIR: trackeval_output
  OUTPUT_ROOT: ./runs/track_eval
  run_det_eval_dir: annotations_video
  run_track_eval_dir: mot_annotations_video
```
改為：
```yaml
paths:
  dataset: video_seqs
  TRACKEVAL_OUT_DIR: trackeval_output
  OUTPUT_ROOT: ./runs/track_eval
```

- [ ] **Step 4: 逐一驗證每個檔案改動後 `load_config()` 解析結果與改動前一致**

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from evaluation.config_loader import load_config

expected = {
    'config/bytetrack.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/botsort_noreid.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/botsort_reid.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/botsort_save_reid_crops.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/collision_detect.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/grid_search_botsort.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/grid_search_bytetrack.yaml': ('annotations_all8', 'mot_annotations_all8'),
    'config/botsort_save_reid_crops_video.yaml': ('annotations_video', 'mot_annotations_video'),
}
for path, (det, track) in expected.items():
    cfg = load_config(path)
    assert cfg.run_det_eval_dir == det, f'{path}: {cfg.run_det_eval_dir} != {det}'
    assert cfg.run_track_eval_dir == track, f'{path}: {cfg.run_track_eval_dir} != {track}'
    assert '中華三路' in str(cfg.SOURCE_VIDEO_DIR) or 'video_seqs' in str(cfg.SOURCE_VIDEO_DIR)
print('ALL 8 CONFIGS OK')
"
```
Expected: `ALL 8 CONFIGS OK`

- [ ] **Step 5: Commit**

```bash
git add config/*.yaml
git commit -m "Simplify config/*.yaml paths blocks to use dataset registry

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: `module/grid_search/grid_search_botsort.py` — 路徑組裝改用 `dataset_layout.py`

**Files:**
- Modify: `module/grid_search/grid_search_botsort.py`（import 區塊、`_resolve_gt_source_root()` L685-708、`trial_dir` 組裝 L481、`main()` 影片蒐集 L1114-1126）

**Interfaces:**
- Consumes: `evaluation.dataset_layout.stage2_seq_dir()`、`list_stage1_sequences()`、`seq_name_from_video_path()`（Task 2）。
- Produces: 無新對外介面（本檔案內部函式簽章不變，仍是 `_resolve_gt_source_root(cfg, seq_stem)` 等既有名稱，供已核准但尚未實作的 `2026-09-03-grid-search-dedup-design.md` 之後合併進 `core.py` 時沿用）。

- [ ] **Step 1: 新增 import**

在檔案現有 `from evaluation.mot_format import save_mot_format` 附近（用 `grep -n "^from evaluation"` 定位）新增：
```python
from evaluation import dataset_layout
```

- [ ] **Step 2: `_resolve_gt_source_root()` 改用 `dataset_layout.stage2_seq_dir()`，移除前綴**

Read 現有（用 `grep -n "def _resolve_gt_source_root"` 定位，約 L685-708）：
```python
def _resolve_gt_source_root(cfg, seq_stem: str) -> str | None:
    """
    從 cfg 的 run_track_eval_dir 下，找出指定序列的原始 GT 目錄。

    預期結構：
      base_dir/mot_annotations_all8/
        mot_annotation_<seq_stem>/
          gt/
            gt.txt        ← flat GT（9 columns MOT format）
          seqinfo.ini     ← 可選
    """
    base    = Path(cfg.OUTPUT_ROOT)
    mot_dir = base / cfg.run_track_eval_dir

    candidate = mot_dir / f"mot_annotation_{seq_stem}"
    if candidate.exists():
        return str(candidate)

    # fallback: glob 比對前綴（容許 seq_stem 含特殊字元）
    matches = sorted(mot_dir.glob(f"mot_annotation_{seq_stem[:20]}*"))
    if matches:
        return str(matches[0])

    return None
```

改為：
```python
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
```

- [ ] **Step 3: `trial_dir` 組裝移除前綴**

Read 現有（用 `grep -n "trial_dir = output_root"` 定位，約 L481）：
```python
            seq_stem  = Path(video_path).stem
            trial_dir = output_root / "mot_annotations_all8" / f"mot_annotation_{seq_stem}"
```

改為：
```python
            seq_stem  = dataset_layout.seq_name_from_video_path(video_path)
            trial_dir = dataset_layout.stage2_seq_dir(output_root / "mot_annotations_all8", seq_stem)
```

- [ ] **Step 4: `main()` 影片蒐集改用 `list_stage1_sequences()`**

Read 現有（用 `grep -n "收集影片路徑"` 定位，約 L1114-1126）：
```python
    # ── 收集影片路徑 ──────────────────────────────────────────────────────────
    import glob as _glob
    if args.video_glob:
        video_paths = sorted(_glob.glob(args.video_glob, recursive=True))
    else:
        pattern = str(
            Path(cfg.OUTPUT_ROOT) / cfg.run_det_eval_dir
            / "annotation_*" / "**" / "*.mp4"
        )
        video_paths = sorted(
            p for p in _glob.glob(pattern, recursive=True)
            if not Path(p).stem.startswith("Demo")
        )
```

改為：
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

`--video-glob` 的 help text（約 L1046-1052，`"例如: './data/source/annotations_all8/annotation_*/**.mp4'"`）維持不動——那是使用者自行指定 override pattern 時的範例，不受本次預設路徑邏輯改動影響。

- [ ] **Step 5: 語法檢查**

Run: `python3 -m py_compile module/grid_search/grid_search_botsort.py`
Expected: 無輸出、exit code 0

- [ ] **Step 6: 手動驗證——`--dry-run` 確認路徑組裝不報錯**

Run:
```bash
python module/grid_search/grid_search_botsort.py --target-class scooter --dry-run
```
Expected: 印出參數組合數量，無 `NameError`/`AttributeError`（`--dry-run` 在收集影片路徑之前就 `sys.exit(0)`，這一步主要驗證 import 與函式定義沒有語法/名稱錯誤；下一步才驗證影片蒐集邏輯本身）。

- [ ] **Step 7: 手動驗證——`--max-videos 1` 跑一組 trial（需要 Task 5 已產生的 stage1 輸出）**

Run:
```bash
python module/grid_search/grid_search_botsort.py --target-class scooter --max-videos 1 --params-subset match_thresh
```
Expected: 正確找到 Task 5 產生的至少 1 支序列影片、`trial_dir` 正確建立在 `runs/grid_search/botsort/mot_annotations_all8/<seq>/`（無前綴）、`_resolve_gt_source_root()` 正確從 `runs/track_eval/mot_annotations_all8/<seq>/` 找到 GT、trial 執行完畢並寫入 CSV。

- [ ] **Step 8: Commit**

```bash
git add module/grid_search/grid_search_botsort.py
git commit -m "grid_search_botsort.py: use dataset_layout for path assembly, drop annotation_ prefix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: `module/grid_search/grid_search_bytetrack.py` — 同 Task 8

**Files:**
- Modify: `module/grid_search/grid_search_bytetrack.py`（import 區塊、`_resolve_gt_source_root()` L532-555、`trial_dir` 組裝 L340、`main()` 影片蒐集 L925-937）

**Interfaces:**
- Consumes: 同 Task 8。
- Produces: 同 Task 8。

- [ ] **Step 1-4: 套用與 Task 8 完全相同的四處改動**

`module/grid_search/grid_search_bytetrack.py` 與 `grid_search_botsort.py` 在這四處的程式碼**逐字相同**（已於 `2026-09-03-grid-search-dedup-design.md` 確認），只有行號不同（`trial_dir` 在 L340、`_resolve_gt_source_root` 在 L532、影片蒐集在 L925 附近，用 `grep -n` 依 Task 8 Step 2-4 列出的關鍵字重新定位）。依 Task 8 Step 1-4 的**完全相同**改前/改後程式碼套用到本檔案。

- [ ] **Step 5: 語法檢查**

Run: `python3 -m py_compile module/grid_search/grid_search_bytetrack.py`
Expected: 無輸出、exit code 0

- [ ] **Step 6: 手動驗證——`--dry-run`**

Run: `python module/grid_search/grid_search_bytetrack.py --target-class scooter --dry-run`
Expected: 印出參數組合數量，無錯誤。

- [ ] **Step 7: 手動驗證——`--max-videos 1` 跑一組 trial**

Run: `python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1 --params-subset match_thresh`
Expected: 同 Task 8 Step 7。

- [ ] **Step 8: Commit**

```bash
git add module/grid_search/grid_search_bytetrack.py
git commit -m "grid_search_bytetrack.py: use dataset_layout for path assembly, drop annotation_ prefix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: `module/grid_search/grid_search_botsort_reid.py` — 同 Task 8

**Files:**
- Modify: `module/grid_search/grid_search_botsort_reid.py`（import 區塊、`_resolve_gt_source_root()` L581-604、`trial_dir` 組裝 L390、`main()` 影片蒐集 L974-986）

**Interfaces:**
- Consumes: 同 Task 8。
- Produces: 同 Task 8。

- [ ] **Step 1-4: 套用與 Task 8 完全相同的四處改動**

依 Task 8 Step 1-4 的改前/改後程式碼套用到本檔案（行號用 `grep -n` 依關鍵字重新定位：`trial_dir` 約 L390、`_resolve_gt_source_root` 約 L581、影片蒐集約 L974）。

- [ ] **Step 5: 語法檢查**

Run: `python3 -m py_compile module/grid_search/grid_search_botsort_reid.py`
Expected: 無輸出、exit code 0

- [ ] **Step 6: 手動驗證——`--dry-run`**

Run: `python module/grid_search/grid_search_botsort_reid.py --target-class person --dry-run`
Expected: 印出參數組合數量，無錯誤。

- [ ] **Step 7: 手動驗證——`--max-videos 1` 跑一組 trial**

Run: `python module/grid_search/grid_search_botsort_reid.py --target-class person --max-videos 1 --params-subset match_thresh`
Expected: 同 Task 8 Step 7。

- [ ] **Step 8: Commit**

```bash
git add module/grid_search/grid_search_botsort_reid.py
git commit -m "grid_search_botsort_reid.py: use dataset_layout for path assembly, drop annotation_ prefix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: `module/collision/homography_pick_aid.py` — 硬編碼中文路徑改用 dataset registry

**Files:**
- Modify: `module/collision/homography_pick_aid.py:87-94`（`USABLE_INTERSECTIONS` 第一筆）

**Interfaces:**
- Consumes: `evaluation.dataset_layout.load_dataset_registry()`（Task 2）。
- Produces: 無（`USABLE_INTERSECTIONS` 對外結構不變，仍是 `list[tuple[str, str]]`）。

- [ ] **Step 1: 新增 import 與 registry 查詢**

在檔案 `from pathlib import Path` 之後（`import cv2` 之前）新增：
```python
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.dataset_layout import load_dataset_registry
```

**注意**：這個檔案原本沒有把專案根目錄加進 `sys.path`（沿用 `python -m module.collision.homography_pick_aid` 從專案根目錄執行時的隱含機制），但直接 `python module/collision/homography_pick_aid.py` 執行時 `evaluation` 套件會 import 失敗，因此這裡明確加上 `sys.path` 處理，兩種執行方式都能正確 import。

- [ ] **Step 2: 用 registry 值取代硬編碼路徑**

Read 現有 `module/collision/homography_pick_aid.py:87-94`：
```python
# (tag, 來源影片路徑) —— tag 需與 models/homography/matrices/H_<tag>.npy 的檔名一致
USABLE_INTERSECTIONS = [
    (
        "C000025 中華三路 民生二路(東北角)-2-20-1280-1280",
        "dataset/中華三路_民生二路(東北角)yolo驗證用_all8/annotations_all8/"
        "annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1/"
        "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1.mp4",
    ),
```

改為：
```python
_CCTV_ALL8_ROOT = Path(
    load_dataset_registry()
    .get("cctv_all8", {})
    .get("root", "./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/")
)

# (tag, 來源影片路徑) —— tag 需與 models/homography/matrices/H_<tag>.npy 的檔名一致
USABLE_INTERSECTIONS = [
    (
        "C000025 中華三路 民生二路(東北角)-2-20-1280-1280",
        str(
            _CCTV_ALL8_ROOT / "annotations_all8"
            / "annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1"
            / "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1.mp4"
        ),
    ),
```

其餘 5 個 `video/C0000xx...mp4` 項目維持原樣不動（那是獨立的、已經很短的相對路徑，不屬於 `dataset/中華三路...` 這個中文長路徑問題）。

- [ ] **Step 3: 語法檢查**

Run: `python3 -m py_compile module/collision/homography_pick_aid.py`
Expected: 無輸出、exit code 0

- [ ] **Step 4: 手動驗證——確認第一筆路徑組裝結果與改動前逐字元相同**

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from module.collision.homography_pick_aid import USABLE_INTERSECTIONS
expected = (
    'dataset/中華三路_民生二路(東北角)yolo驗證用_all8/annotations_all8/'
    'annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1/'
    'C000025 中華三路 民生二路(東北角)_20251209145630_15s_1.mp4'
)
tag, video_path = USABLE_INTERSECTIONS[0]
import os
assert os.path.normpath(video_path) == os.path.normpath(expected), video_path
print('OK', video_path)
"
```
Expected: `OK <路徑>`

- [ ] **Step 5: Commit**

```bash
git add module/collision/homography_pick_aid.py
git commit -m "homography_pick_aid.py: build cctv_all8 path from dataset registry, not hardcoded literal

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 12: `module/collision/__init__.py` + `trajectory_adapter.py` — 移除 `.replace("annotation_", "")`

**Files:**
- Modify: `module/collision/__init__.py:117`
- Modify: `module/collision/trajectory_adapter.py:152`

**Interfaces:**
- Consumes: Task 5/6 產出的無前綴 `<seq>/` 目錄命名（`seq_output_dir.stem` 現在已經是純 `seq_name`，不再需要 `.replace()`）。
- Produces: 無新介面，`run_collision_detection_for_sequence()`、`load_sequence_tracks()` 對外簽章不變。

- [ ] **Step 1: `module/collision/__init__.py:117`**

Read 現有：
```python
    seq_name = seq_output_dir.stem.replace("annotation_", "")
```

改為：
```python
    seq_name = seq_output_dir.stem
```

- [ ] **Step 2: `module/collision/trajectory_adapter.py:152`**

Read 現有：
```python
    seq_name = seq_output_dir.stem.replace("annotation_", "")
```

改為：
```python
    seq_name = seq_output_dir.stem
```

- [ ] **Step 3: 語法檢查**

Run:
```bash
python3 -m py_compile module/collision/__init__.py module/collision/trajectory_adapter.py
```
Expected: 無輸出、exit code 0

- [ ] **Step 4: 手動驗證——完整跑一次 stage3（沿用 Task 5/6 產生的 stage1 輸出）**

依 spec 第 5 節驗證計畫，對 `config/collision_detect.yaml` 跑：
```bash
python track_eval.py --config config/collision_detect.yaml
```
確認：
- `runs/track_eval/collision_output/<seq_name>/` 用的是正確的無前綴序列名（不是空字串、不是誤刪成其他字串）。
- `ttc_events.json`/`pet_events.json` 等輸出檔案正常產生，內容裡的 `seq_name` 欄位（若有）與目錄名稱一致。

- [ ] **Step 5: Commit**

```bash
git add module/collision/__init__.py module/collision/trajectory_adapter.py
git commit -m "module/collision: drop now-unnecessary annotation_ prefix stripping

stage1 output directories no longer carry the annotation_ prefix
(see track_eval.py stage1 changes), so seq_output_dir.stem is
already the bare seq name.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 13: `module/collision/output.py` / `homography_points.py` / `homography_compare_video.py` — docstring 更新

**Files:**
- Modify: `module/collision/output.py:14`
- Modify: `module/collision/homography_points.py:18`
- Modify: `module/collision/homography_compare_video.py:55`

**Interfaces:**
- Consumes: 無。
- Produces: 無（純文件字串更新，不影響任何邏輯或介面）。

- [ ] **Step 1: `module/collision/output.py:14`**

Read 現有：
```
   背景圖用 stage1 已存的 annotation_<seq>/first_frame_with_box.jpg。
```

改為：
```
   背景圖用 stage1 已存的 <seq>/first_frame_with_box.jpg。
```

- [ ] **Step 2: `module/collision/homography_points.py:18`**

Read 現有：
```
    1. 打開 stage1 已存的 annotation_<seq>/first_frame_with_box.jpg，自己在圖上
```

改為：
```
    1. 打開 stage1 已存的 <seq>/first_frame_with_box.jpg，自己在圖上
```

- [ ] **Step 3: `module/collision/homography_compare_video.py:55`**

Read 現有：
```
        video_path: 來源 CCTV 影片路徑（建議用 stage1 已複製到 annotation_<seq>/ 底下的那份，
```

改為：
```
        video_path: 來源 CCTV 影片路徑（建議用 stage1 已複製到 <seq>/ 底下的那份，
```

- [ ] **Step 4: 語法檢查**

Run:
```bash
python3 -m py_compile module/collision/output.py module/collision/homography_points.py module/collision/homography_compare_video.py
```
Expected: 無輸出、exit code 0

- [ ] **Step 5: Commit**

```bash
git add module/collision/output.py module/collision/homography_points.py module/collision/homography_compare_video.py
git commit -m "module/collision: update stale annotation_<seq> naming references in docstrings

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 14: 完整管線驗證（依 spec 第 5 節）

**Files:**
- 無檔案異動，純驗證任務。

**Interfaces:**
- Consumes: Task 1-13 的全部產出。
- Produces: 驗證結果記錄（供 PR/交付說明引用），不產生新程式碼。

- [ ] **Step 1: `config/bytetrack.yaml` 完整 stage1+stage2，比對 HOTA/AssA/CLEAR 與改動前一致**

先在改動前的 commit（或已知基準，見 `markdown/VERIFY.md`）跑一次同樣的限量序列作為 baseline（若已有 Task 5 Step 6 的紀錄可直接沿用）。

Run:
```bash
python track_eval.py --config config/bytetrack.yaml
```
Expected:
- `runs/track_eval/annotations_all8/<seq>/`、`runs/track_eval/mot_annotations_all8/<seq>/`（皆無前綴）正確產生。
- `runs/track_eval/trackeval_output/` 底下的 Excel 檔案，逐類別 HOTA/AssA/CLEAR 數值與 baseline **完全一致**（純結構重構，不應改變任何計算結果）。

- [ ] **Step 2: `config/botsort_save_reid_crops_video.yaml`（`video_seqs` dataset）跑 stage1**

Run:
```bash
python track_eval.py --config config/botsort_save_reid_crops_video.yaml
```
Expected: `dataset: video_seqs` 正確解析成 `SOURCE_VIDEO_DIR=./dataset/video_seqs/`、`run_det_eval_dir=annotations_video`，`runs/track_eval/annotations_video/<seq>/` 正確產生（驗證 escape hatch 機制：情境檔本身沒有顯式覆蓋任何欄位，全部走 registry 查表）。

- [ ] **Step 3: 三支 `module/grid_search/*.py` 各跑一次 `--dry-run`**

已在 Task 8/9/10 的 Step 6-7 個別驗證過，這裡做一次三支一起的最終確認：
```bash
for script in grid_search_bytetrack grid_search_botsort grid_search_botsort_reid; do
  echo "=== $script ==="
  python module/grid_search/$script.py --target-class scooter --dry-run
done
```
Expected: 三支都正常印出組合數，無錯誤。

- [ ] **Step 4: 確認 Excel 匯出的 `AssA_mean` 與 grid search CSV 記錄的分數一致**

挑 Task 8 Step 7 產生的一筆 grid search trial 結果，對照同樣參數手動跑一次 `track_eval.py --config config/botsort_noreid.yaml`（把 `class_tracking` 章節對應類別的參數改成該筆 trial 的參數），確認 Excel 裡的 `HOTA.AssA_mean` 與 CSV 記錄的分數一致（依 `.claude/rules/grid-search.md` 的驗證要求）。

- [ ] **Step 5: 確認補幀相關程式碼路徑已整段移除後，既有序列行為不受影響**

對照 Task 4 Step 1 記錄的序列數與 Task 14 Step 1 產生的實際輸出序列數一致，且確認補幀相關程式碼已無殘留。

Run: `grep -rn "frame_comparison\|interpolated_frame_ids\|should_skip_frame" track_eval.py`
Expected: 無輸出。

- [ ] **Step 6: 清理 scratchpad 暫存驗證腳本**

刪除 Task 2/3 建立的 `test_dataset_layout.py`/`test_config_dataset.py`（它們只是手動驗證用的暫存腳本，不進版控，可安全刪除）。

---

## Self-Review 摘要（寫作計畫時已完成，記錄於此供交付時核對）

- **Spec 覆蓋**：spec 第 3.1-3.7 節（目錄命名、`dataset_layout.py`、`datasets.yaml`、`config_loader.py`、yaml 精簡、補幀移除、影響檔案清單）逐項對應 Task 1-13；spec 第 5 節「驗證計畫」6 點全部對應 Task 14 的各個 Step。
- **無佔位符**：所有 Step 皆為可直接執行的完整程式碼/指令，無 `TODO`/`實作細節待補`。
- **型別/簽章一致性**：`dataset_layout.py` 的 6 個函式簽章在 Task 2 定義後，Task 3/5/6/8/9/10/11/12 全數依相同簽章呼叫（`stage1_seq_dir(det_output_dir, seq_name)`、`stage2_seq_dir(track_output_dir, seq_name)`、`source_gt_dir(source_video_dir, seq_name)` 回傳值含尾端 `/gt`、`discover_source_sequences(source_video_dir, run_det_eval_dir)`、`list_stage1_sequences(det_output_dir)`、`seq_name_from_video_path(video_path)`），無不一致之處。
