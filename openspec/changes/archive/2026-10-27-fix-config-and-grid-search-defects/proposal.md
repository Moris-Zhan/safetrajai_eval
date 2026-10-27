# Proposal

## Why

change `add-config-and-grid-search-unit-tests` 補測試時，發現 `tracking/config.py` 與 `grid_search/core.py` 有 5 個缺口（G1～G5），已用 `xfail(strict=True)` 記錄期望行為，並逐一用 `--runxfail` 確認確實失敗。它們都是**不會報錯**的那一類：載入失敗卻留下半更新的全域參數、還原參數時多出空鍵、拿到別段序列的 GT、拿到別的類別的分數、失敗的 trial 不再重試。結果是調參或追蹤悄悄用錯參數或錯資料。

## What Changes

- **G1** `load_track_config`：所有驗證與型別轉換（每個 `class_tracking_params` 項目必須是 mapping、`class_conf_thresh` 與 `default_conf_thresh` 必須能轉成 `float`）都在修改任何全域參數之前完成；失敗一律拋 `ValueError`，全域參數與載入路徑維持不變。
- **G2** `score_params`：目標類別在 trial 前不在 `CLASS_TRACKING_PARAMS` 時，trial 結束後把該鍵移除，而不是還原成 `{}`。
- **G3** `_resolve_gt_source_root`：精確的序列目錄不存在時，前綴 fallback 只在恰好一個候選時採用；零個或多個候選回傳 `None`，不再取排序第一個。
- **G4** `extract_assa_mean`：結果中沒有指定類別時回傳 `None`，不再退回「第一個 dict」（可能是 `cls_comb_cls_av` 等彙總項）。
- **G5** `GridSearchRunner`：`--resume` 讀到分數為 `-inf` 的失敗列時，不算已完成、不納入結果與最佳值，該組合會重跑；CSV 維持只追加，失敗紀錄保留。使用者已確認採用此語意。
- 移除 6 個對應的 `xfail` 測試標記，使其成為一般測試；補上必要的邊界測試。
- 不改 tracker 的 override 方法（ADR 0004），不改 `third_party/trackeval/`。

## Capabilities

### New Capabilities
<!-- 無 -->

### Modified Capabilities
- `track-config-loading`: 驗證失敗不改變全域參數的範圍擴大到 G1 的兩種情形。
- `grid-search-trials`: 萃取分數不借用其他項目（G4）、還原時移除原本不存在的類別（G2）、新增 GT 目錄解析不得取到其他序列的要求（G3）、resume 對失敗列的語意（G5）。

## Impact

- 程式碼：`src/safetrajai/tracking/config.py`（`load_track_config`）、`src/safetrajai/grid_search/core.py`（`extract_assa_mean`、`_resolve_gt_source_root`、`score_params`、`GridSearchRunner._load_existing_results`）。
- 測試：`tests/test_track_config.py`、`tests/test_grid_search_core.py` 移除 6 個 `xfail`。
- 行為變化：
  - 既有合法設定的載入結果不變（`config/*.yaml` 與 golden 設定都會被既有測試載入）。
  - 既有 grid search CSV 內分數為 `-inf` 的列，下次 `--resume` 會被重跑；其餘列不變。
  - GT 前綴 fallback 與類別 fallback 變嚴格：過去靠「猜」得到結果的情形會改為跳過或回傳 `None`（trial 的該序列被跳過，並印出訊息）。
- 驗證：完整 pytest 為主；是否另跑 golden 的 grid search 情境，在 apply 時向使用者確認（耗時且可能覆寫輸出）。
