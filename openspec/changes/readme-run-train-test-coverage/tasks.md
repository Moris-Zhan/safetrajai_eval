# Tasks

## 1. 覆蓋檢查（先寫會失敗的測試）

- [x] 1.1 新增 `tests/test_readme_coverage.py`：列出 `scripts/**/*.py` 與 `config/*.yaml`（不含子目錄），檢查每個含副檔名的檔名是否出現在 `README.md`；`EXEMPT` 為 dict（檔名 → 理由），理由為空字串視為錯誤；只讀 `README.md` 與 `scripts/`、`config/` 的檔名，不碰 `dataset/`、`models/`、`runs/`。驗證：`pytest tests/test_readme_coverage.py` 失敗，且訊息恰好列出 16 支腳本與 6 份設定檔（紅燈）。
- [x] 1.2 為 `EXEMPT` 機制加兩個自我測試（空理由會失敗；豁免機制不影響已被 README 提到的檔案）。驗證：這兩個測試通過，其餘覆蓋檢查仍為紅燈。

## 2. 調參與衝突事件／Homography 段落

- [x] 2.1 新增 README 7.2「調參（grid search）」：列 `grid_search_bytetrack.py`、`grid_search_botsort.py`、`grid_search_botsort_reid.py` 與 `config/grid_search_bytetrack.yaml`、`config/grid_search_botsort.yaml`，附 `--target-class`、`--dry-run`、`--max-videos 1`、`--resume`，並註明分數是 `HOTA.AssA_mean`（不等於 HOTA）、全量執行耗時且可能覆寫輸出。驗證：三支腳本各以 `--help` 實跑確認範例參數存在；`--dry-run` 實跑確認可解析。
- [x] 2.2 新增 README 7.3「衝突事件與 Homography」：補 `config/conflict_analysis.yaml`（指令、只限有 m/px 的鏡頭、輸出在 `collision_output/<seq>/conflict/`），並列 `validate_homography.py`、`migrate_homography_config.py`、`geocode_cameras.py`、`check_homography_scale.py`、`download_satellite.py`。前提欄標明連網、`GOOGLE_MAPS_API_KEY`／`GOOGLE_MAPS_DEMO_API_KEY`（只寫變數名）、Demo Key 只能 `--purpose verify`。驗證：五支腳本各以 `--help` 實跑；`grep -E 'AIza|key=' README.md` 無輸出。
- [x] 2.3 驗證本組：`pytest tests/test_readme_coverage.py` 的缺漏清單只剩第 3、4 組負責的項目。

## 3. Re-ID 資料與訓練段落

- [x] 3.1 新增 README 7.4「Re-ID 資料」：依順序列蒐集（`config/reid_crops.yaml`）→ 標記 → 自動排除（`apply_auto_exclusions.py`）→ 抽查（`generate_identity_review.py`、`summarize_audit.py`）→ 人工排除（`apply_review_exclusions.py`）→ 碎片合併（`extract_fragment_features.py`、`build_fragment_merges.py`）→ manifest（`build_reid_split.py`、`config/reid_split.yaml`），指向 `scripts/reid_data/README.md`。前提欄標明哪些需要個別裁切 JPEG／來源影片、HF 資料不足以執行（與 6.4 一致），以及 `config/botsort_save_reid_crops.yaml`、`botsort_save_reid_crops_video.yaml` 的用途。驗證：這些腳本各以 `--help` 實跑；README 的前提欄與 6.4 的描述不矛盾。
- [x] 3.2 將原 7.2「Re-ID 訓練與驗證」改編為 7.5，補上 `pack_manifest.py`（HF 還原後不要再跑，6.4 已說明）與 `acceptance_grid.py`（`configs` → 執行 → `summarize`）；同步更新目錄與所有內部連結。驗證：`grep -n '#7-' README.md` 所列錨點都對得上現有標題；`acceptance_grid.py configs --help` 與 `summarize --help` 實跑。
- [x] 3.3 驗證本組：`pytest tests/test_readme_coverage.py` 的缺漏清單只剩第 4 組負責的項目。

## 4. 工具與測試段落

- [x] 4.1 新增 README 7.6「工具」：`scripts/roi/make_roi_picker.py`（ROI 點選頁）、`visualize_gt_on_video.py`、`visualize_gt_on_class_video.py`（疊圖）、`scripts/run_pipeline.py`（與 `track_eval.py` 等價）。驗證：四支腳本各以 `--help` 實跑；覆蓋檢查完全通過（綠燈）。
- [x] 4.2 新增 README 7.7「測試」，依 design D4 的三層表格（smoke／單元／golden）：標明耗時、前提、何時必跑；說明唯一的 `skipif`（被隔離權重）略過不是失敗；單元測試依領域分組（Re-ID、Homography／georef、衝突指標、ROI／layout、tracker 保留期限）；golden 指向 `tests/golden/TOLERANCE.md` 並註明容忍度 0、約 16 分鐘；指令寫成 `compare.py <基準> <run>`，不寫死基準名稱。驗證：`pytest --collect-only -q` 的測試檔清單與小節的分組敘述一致；`grep -n skipif tests/*.py` 的結果與小節說法相符。不實跑 golden 比對。

## 5. 整合檢查

- [x] 5.1 對照 spec `readme-command-coverage` 逐條核對：腳本涵蓋、設定涵蓋、前提標示、測試分層、檢查不讀大型目錄。驗證：每個 Scenario 都能指出 README 或測試中對應的位置。
- [x] 5.2 在 conda env `track_eval` 執行完整 `pytest`。驗證：全部通過；並以 `grep -nE 'dataset|models|runs' tests/test_readme_coverage.py` 確認該測試沒有存取資料目錄（不得為此移動真實資料）。
- [x] 5.3 檢查 diff：`git diff --stat` 只應有 `README.md`、新增的測試檔與 `openspec/changes/readme-run-train-test-coverage/`；`src/`、`config/`、`third_party/` 無變動。驗證：`git status --short` 輸出符合預期。
- [x] 5.4 向使用者顯示 diff 摘要並等待確認後，以單一 commit 提交 README 與測試（commit message 用英文 conventional commits，例如 `docs(readme): document run/train/test entry points`）。驗證：使用者確認，且提交後 `pytest` 為綠燈。

## Workflow follow-up

- 歸檔本 change 前，確認 `openspec validate readme-run-train-test-coverage` 通過。
- 後續另議：補齊 design「盤點」小節列出的測試缺口（優先 `grid_search/core.py`、`tracking/config.py`）；更新或收斂 `Markdown/USAGE.md` 的過時內容。
