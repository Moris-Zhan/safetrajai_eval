# Proposal

## Why

專案沒有可安裝的套件（installable package），至少 7 支腳本各自用 `sys.path.insert` 與 `Path(__file__).parents[N]` 推算根目錄；主入口 `track_eval.py`（736 行）同時承擔 CLI、Re-ID 裁切蒐集、影片寫出與 pipeline 編排；類別 ID 契約在 5 處重複定義；視覺化分散三處；根目錄混雜與本專案無關的工具腳本。結構的凌亂讓每次搬動或新增模組都有「靜默讀錯路徑」的風險，且目前沒有任何測試能攔住。參考 `專案架構重構.txt` 的分層樣板，在**行為完全不變**的前提下整理結構，並先建立黃金輸出（golden output）比對作為回歸防線。

## What Changes

- **Phase 0 回歸防線**：新增 `tests/golden/`（基準 config＋比對腳本）與 `tests/test_smoke.py`；以舊程式碼跑兩次量測自比容忍度（nondeterminism baseline）。
- **Phase 1 清理**：
  - 新增 `safetrajai/classes.py` 作為類別 ID 單一來源（0-based `CLASS_NAMES`，MOT GT 用的 `id + 1` 由此推導），取代 `evaluation/mot_format.py::NAMES`、`trackeval_runner.py::class_name_to_class_id`、`dataset/reid/` 三份 `CLASSES`。
  - 移除無呼叫者的 `translate_path`／`normalize_string`，連帶移除 `argostranslate` 依賴。
  - `dev_design_ppt_favorites_install_*.sh`、`monitor_finetune.sh` 移至 `tools/`。
- **Phase 2 套件化**：
  - 新增 `pyproject.toml`，程式碼移入 `src/safetrajai/`（`module/`、`evaluation/`、`utils/visualization.py` 併入），於 conda env `track_eval` 以 `pip install -e . --no-deps` 安裝。
  - vendored `trackeval/` 移至 `third_party/trackeval/`（內容不變），由同一份 pyproject 對應安裝。
  - 新增 `safetrajai/paths.py` 作為唯一 `PROJECT_ROOT`，移除所有 `sys.path` hack（含 `track_eval.py` 把 repo 上一層加入 path 的那行）。
  - `module/collision/` 保留 `collision` 名稱，6 支 `homography_*` 收進 `collision/homography/` 子套件。
  - Tracker 檔案整檔搬移至 `safetrajai/tracking/`，**只改 import 行**，override 方法本體一字不動。
  - `utils/model.py` 改為 pickle 相容 shim，實作移至 `safetrajai/reid/model.py`。
  - CLI 腳本集中到 `scripts/`（grid search、homography 驗證、Re-ID 訓練、Re-ID 資料稽核、GT 疊圖）。**BREAKING**：`python module/grid_search/*.py`、`python -m module.collision.run_homography_validation`、`python module/reid/training/train.py`、`python dataset/reid/*.py`、`python visualize/*.py` 等舊路徑失效，改用 `scripts/` 下的新路徑。
- **Phase 3 拆分主入口**：`track_eval.py` 拆為 `safetrajai/pipeline/{tracking_pipeline,sequence_tracker,reid_crops,video_io}.py`，CLI 移至 `scripts/run_pipeline.py`；根目錄 `track_eval.py` 保留為相容 wrapper。`evaluation/pipeline.py::main()` 改為 `scripts/run_evaluation.py`。
- **文件同步**：CLAUDE.md、`.claude/commands/*`、`.claude/settings.local.json`、HANDOFF／flow 文件、各 README、`config/*.yaml` 的用法註解（僅註解，不動區塊與數值）。
- **不做**：config 分層（detector/tracker/reid/pipeline 拆檔）、Re-ID 權重轉存 state_dict、任何演算法或參數調整、修改 vendored trackeval 內容。

## Capabilities

### New Capabilities

- `project-structure`：專案結構的不變式契約——可安裝套件且無 `sys.path` hack、`track_eval.py` 入口相容、`utils.model` pickle 相容、類別 ID 單一來源、重構前後黃金輸出等價、tracker override 方法本體不變。

### Modified Capabilities

（無）

## Impact

- **程式碼**：`track_eval.py`、`module/**`、`evaluation/**`、`utils/**`、`visualize/**`、`dataset/reid/*.py`、`trackeval/`（僅搬移）全數換位置；import 路徑全面改寫。
- **入口**：`python track_eval.py --config ...` 維持可用；其餘 CLI 改到 `scripts/`。
- **環境**：conda env `track_eval` 需執行一次 `pip install -e . --no-deps`；新增 dev 依賴 `pytest`；移除 `argostranslate` 依賴。
- **資料與輸出**：`dataset/`、`models/`、`runs/` 的位置與 stage1／stage2 輸出目錄結構不變；`.gitignore` 移除 `dataset/reid/` 的例外規則。
- **文件與工具設定**：CLAUDE.md、`.claude/commands/{run-pipeline,tracker-change,tune-config}.md`、`.claude/settings.local.json` 需同步更新。
