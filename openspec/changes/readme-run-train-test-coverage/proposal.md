# Proposal

## Why

`README.md` 第 7 節只列出追蹤評估（`track_eval.py`、`run_evaluation.py`）與三支 Re-ID 訓練／驗證腳本。對照 repo 實際內容：`scripts/` 有 26 支腳本，其中 16 支沒有出現在 README；`config/` 有 14 份設定檔，其中 6 份沒有出現；測試只有一行 `pytest`，沒有提到單元測試的分組、golden 比對與測試前提。

新成員或在乾淨機器上還原（HF 下載後）的人，無法只靠 README 知道 grid search、stage3 衝突指標、Homography 鏡頭設定、Re-ID 資料流程、ROI 工具、疊圖工具該怎麼跑，也不知道哪些步驟需要連網、金鑰或原始影片。完整指令目前散在 `CLAUDE.md`、`Markdown/USAGE.md`（部分內容已過時，仍停在 `rebuild-reid-dataset` 進行中）與 `scripts/reid_data/README.md`。

## What Changes

- 在 `README.md` 補齊「執行／訓練／測試」的入口指令，涵蓋目前未列出的模組：
  - **調參**：`scripts/grid_search_bytetrack.py`、`grid_search_botsort.py`、`grid_search_botsort_reid.py`（`--dry-run`、`--max-videos`、`--resume`），以及對應的 `config/grid_search_*.yaml`。
  - **stage3 衝突事件**：`config/conflict_analysis.yaml`（TTC／PET＋公尺單位衝突指標，只限有 m/px 的鏡頭）。
  - **Homography 鏡頭設定**：`validate_homography.py`、`migrate_homography_config.py`、`geocode_cameras.py`、`check_homography_scale.py`、`download_satellite.py`，並標明哪些會連網、需要哪把金鑰。
  - **Re-ID 資料流程**：`scripts/reid_data/` 尚未列出的 `apply_auto_exclusions`、`summarize_audit`、`apply_review_exclusions`、`build_fragment_merges`，並指向 `scripts/reid_data/README.md`；標明哪些在只有 HF 資料時無法執行（README 6.4 已有部分說明）。
  - **Re-ID 裁切蒐集設定**：`botsort_save_reid_crops*.yaml`、`reid_split.yaml`。
  - **工具**：`scripts/roi/make_roi_picker.py`、`visualize_gt_on_video.py`、`visualize_gt_on_class_video.py`、`run_pipeline.py`（與 `track_eval.py` 等價）。
- 新增 README「測試」小節：smoke 測試涵蓋範圍、單元測試分組（Re-ID、Homography／georef、衝突指標、ROI／layout、tracker 保留期限）、golden 比對（`tests/golden/run_all.sh`＋`compare.py`，約 16 分鐘、容忍度 0）、哪些測試需要資料或權重、何時必須跑哪一種。
- 新增一個 pytest 檢查：`scripts/**/*.py` 與 `config/*.yaml` 的每個檔名都必須出現在 README 或明確的豁免清單中，之後新增腳本卻忘了寫入文件時會失敗。
- 在 `design.md` 記錄盤點到的測試缺口（沒有直接單元測試的模組）；本 change 只記錄、不補測試。

## Capabilities

### New Capabilities
- `readme-command-coverage`: README 必須列出每個可執行入口（腳本、情境設定、測試指令）的用途與執行方式，並由自動檢查防止漏列。

### Modified Capabilities
<!-- 無：既有 spec 的需求沒有改變 -->

## Impact

- 修改 `README.md`（第 7 節擴充、新增測試小節）。
- 新增 `tests/test_readme_coverage.py`（只讀 README、`scripts/`、`config/` 的檔名，不讀 `dataset/`、`models/`、`runs/`）。
- 不改動 `src/`、`third_party/`、config 內容、tracker override 方法，也不跑任何 stage 或長時間任務。
- `Markdown/USAGE.md` 的過時內容不在本 change 範圍，僅在 design 中記為後續事項。
