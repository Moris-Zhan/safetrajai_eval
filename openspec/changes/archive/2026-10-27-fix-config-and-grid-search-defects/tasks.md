# Tasks

## 1. G1：載入設定先驗證再套用（`tracking/config.py`）

- [x] 1.1 移除 `test_non_mapping_params_for_unlisted_class_leave_globals_unchanged` 與 `test_non_numeric_confidence_threshold_leaves_globals_unchanged` 的 `xfail`，並新增 `default_conf_thresh: null` 的測試；驗證：`pytest tests/test_track_config.py -k "unlisted or non_numeric or default_conf"` 三個新／改的測試失敗（紅燈）
- [x] 1.2 修改 `load_track_config`：mapping 檢查、`float` 轉型（含 `default_conf_thresh`）全部在第一次 `clear()` 之前完成，轉型失敗包成 `ValueError`；驗證：`pytest tests/test_track_config.py tests/test_tracker_retention.py tests/test_reid_fusion.py` 全數通過

## 2. G2：還原時移除原本不存在的類別（`grid_search/core.py`）

- [x] 2.1 移除 `test_class_without_an_entry_is_removed_again_after_the_trial` 的 `xfail`；驗證：該測試失敗（紅燈）
- [x] 2.2 修改 `score_params` 的還原：原本存在則還原內容，否則 `pop`；驗證：`pytest tests/test_grid_search_core.py -k "restore or removed or without_an_entry"` 全數通過（含既有的成功／跳過／例外三條還原路徑）

## 3. G3：GT 目錄前綴 fallback 只接受唯一候選

- [x] 3.1 移除 `test_resolve_gt_source_root_does_not_return_a_sibling_sequence` 的 `xfail`，新增「前綴只有一個候選時回傳該目錄」的測試；驗證：前者失敗（紅燈）、後者通過
- [x] 3.2 修改 `_resolve_gt_source_root`：精確目錄優先，前綴比對恰好一個候選才回傳，其餘 `None`；驗證：`pytest tests/test_grid_search_core.py -k resolve_gt` 全數通過

## 4. G4：萃取分數不借用其他項目

- [x] 4.1 移除 `test_extract_assa_mean_does_not_borrow_another_entry_when_the_class_is_missing` 的 `xfail`；驗證：該測試失敗（紅燈）
- [x] 4.2 修改 `extract_assa_mean`：序列層取不到指定類別就不再退回第一個 dict；驗證：`pytest tests/test_grid_search_core.py -k extract_assa` 全數通過

## 5. G5：resume 重跑失敗的 trial

- [x] 5.1 移除 `test_resume_retries_a_failed_trial` 的 `xfail`，新增「載入時忽略 `-inf` 列、不影響最佳值、舊失敗列保留在 CSV」的測試；驗證：重跑測試失敗（紅燈）
- [x] 5.2 修改 `GridSearchRunner._load_existing_results`：略過 `-inf`／`nan` 列並在載入訊息顯示忽略組數；驗證：`pytest tests/test_grid_search_core.py -k "resume or runner"` 全數通過

## 6. 整體驗證

- [x] 6.1 執行完整 `pytest`；驗證：全部通過、`pytest -rx` 的 xfail 數為 0；與先前基準 728 passed／6 xfailed 對照，只應增加新測試、減少 6 個 xfail
- [x] 6.2 檢查 diff 只動 `src/safetrajai/tracking/config.py`、`src/safetrajai/grid_search/core.py`、兩個測試檔；驗證：`git diff --stat` 沒有 tracker 檔案、`third_party/`、`config/`
- [x] 6.3 執行 `python scripts/grid_search_bytetrack.py --target-class scooter --dry-run` 只算組合數；驗證：腳本可啟動、印出組合數且不寫入 `runs/`

## Workflow follow-up

- 完成後歸檔 change（`/opsx:archive`）：把 delta 同步進 `openspec/specs/track-config-loading`、`openspec/specs/grid-search-trials`
- 另開 change：更正 `CLAUDE.md` 載入順序措辭、更新 `Markdown/USAGE.md`
- 是否另跑 golden 的 grid search 情境，由使用者決定
