# Design

## Context

G1～G5 的觀察、位置與重現方式記錄在已歸檔的 `add-config-and-grid-search-unit-tests/design.md`；每個缺口已有一個 `strict xfail` 測試描述期望行為，並用 `--runxfail` 確認失敗原因正確。本 change 讓這 6 個測試變成一般測試。

限制：

- `tracking/config.py` 的全域 dict 必須原地更新（其他模組持有參考）；`load_track_config` 在 import 時就會呼叫一次。
- `ClassAwareBOTSORT`／`ClassAwareBYTETracker` 的 override 方法凍結（ADR 0004）；本 change 只動 `config.py` 與 `grid_search/core.py`，不碰它們。
- 修正不得改變合法設定的載入結果（所有 `config/*.yaml` 與 `tests/golden/configs/*.yaml` 都在既有測試中被載入）。

## Goals / Non-Goals

**Goals:**
- 5 個缺口各自的期望行為成立，對應測試轉為一般測試且通過。
- 修正是最小的：只調整驗證順序、還原方式、fallback 條件、resume 載入條件。

**Non-Goals:**
- 不改變 `extract_assa_mean` 在 tracker 層與序列層的 fallback（第一個 tracker、第一個序列）；G4 只處理類別層。
- 不處理 `glob` 特殊字元跳脫、CSV 欄位格式等其他疑慮。
- 不更正 `CLAUDE.md` 對載入順序的措辭、不更新 `Markdown/USAGE.md`（各自另開 change）。

## Decisions

### G1：先算出所有結果，最後才動全域

在 `load_track_config` 修改全域之前，建立「已驗證的新內容」：類別名稱（`_coerce_class_names`）、逐項檢查 mapping 後的類別參數（複本，鍵轉 `str`）、逐項 `float` 的信心門檻，以及預設信心門檻。轉型失敗（`ValueError`／`TypeError`）時重新包成 `ValueError`，訊息帶出設定名稱。通過後才依序 `clear()`／`update()`。

替代方案：先備份舊內容、失敗時回滾。否決——回滾本身可能失敗，且「先驗證後套用」符合現有 `track_buffer` 驗證的作法，一致性較好。

`default_conf_thresh` 目前就在 clear 之前轉型，但 `None` 或清單會拋 `TypeError` 而不是 `ValueError`；一併包成 `ValueError`，規格才能一句話涵蓋。

### G2：以「是否存在」決定還原方式

`score_params` 在修補前記錄目標類別的鍵是否存在與原內容的深拷貝；`finally` 時原本存在就還原內容，否則 `pop(target_class, None)`。

替代方案：永遠 `pop` 再 `update`。否決——會改變 dict 的鍵順序。

### G3：前綴 fallback 只接受唯一候選

精確目錄存在 → 回傳；否則 glob 前綴，恰好一個候選 → 回傳，其餘 → `None`。

替代方案：完全移除 fallback。這最嚴格，但在不大量掃描 `dataset/` 的前提下無法確認現有資料是否依賴 fallback（例如目錄名稱被截斷）；保留「唯一候選」能避免已知的歧義，又不讓可能依賴它的流程直接壞掉。風險：只有一個同前綴的別段序列時仍會誤取（見 Risks）。

### G4：類別層不再退回任意項目

序列層取不到指定類別就視為沒有結果（`None`）。TrackEval 的類別鍵是類別名稱（`third_party/trackeval/eval.py` 的彙總鍵是 `cls_comb_cls_av`、`cls_comb_det_av`、`all`），所以真實結果中指定類別存在時不受影響。移除舊註解「TrackEval 有時用 cls index string」——找不到證據支持這個情形；若 apply 時發現反例，再回報。

### G5：失敗列不算已完成，也不載入結果

`_load_existing_results` 讀到分數為 `-inf`（或 `nan`）的列時略過：不加入已完成集合、不加入結果、不更新最佳值。CSV 維持只追加，所以舊的失敗列保留，重跑成功後新列追加在後；再次 resume 時，同一 `combo_id` 的有效列勝出，失敗列被忽略。

替代方案：重跑時改寫 CSV 刪掉失敗列。否決——CSV 目前是只追加的紀錄，保留失敗歷史對除錯有用，且避免改寫檔案造成的資料遺失風險。

載入訊息同時顯示「已完成 N 組」與「忽略失敗 M 組」，讓使用者知道這些組合會重跑。

## Risks / Trade-offs

- [G3 只有唯一候選時仍可能取到別段序列] → 與現狀相同的風險，範圍縮小到單一候選；規格只要求不在歧義時猜。若日後確認資料不依賴 fallback，可另開 change 移除。
- [G4 變嚴格後，某些原本「碰巧」得到分數的 trial 會得到 `None`，該序列被跳過，分數樣本變少] → 這是預期：先前的分數可能是別的類別或彙總項。`score_params` 已有 `AssA_mean=None` 的處理與訊息。
- [G5 讓 `--resume` 對全部失敗的 CSV 重跑耗時更久] → 這正是要求的語意；若失敗是固定原因（例如缺 GT），重跑會再失敗並追加一列，不會無限迴圈。
- [既有 CSV 中 `-inf` 列在升級後行為改變] → 只影響下次 `--resume`；`CLAUDE.md` 的 Grid search 說明原文就是「只能跳過……已經完成的組合」，不需要改。

## Migration Plan

純程式修正，無資料遷移。回滾 = revert 該 commit；`xfail` 標記隨 commit 一起回復。

## Open Questions

- 是否需要在 apply 完成後另跑 golden 的 grid search 情境（約 16 分鐘、可能覆寫輸出）？預設只跑完整 pytest；golden 由使用者決定，不影響 specs 或任務拆分。
