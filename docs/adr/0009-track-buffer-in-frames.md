# track_buffer 統一為保留期限幀數

本決策取代 [ADR 0008](0008-retention-derived-quantities-in-frames.md)，並修訂 [ADR 0004](0004-tracker-overrides-are-frozen.md) 的凍結例外。

## 決策

BoTSORT 與 ByteTrack 的逐類別 `track_buffer` 都是正整數幀數，不再乘上影片幀率。當目前幀號減去軌跡最後更新幀號大於 `track_buffer` 時移除；等於時仍可重新關聯。同一幀數在不同幀率下代表不同的時間長度，需要相同遮擋時間時應在各情境設定不同的幀數。

使用者授權凍結方法只修改兩個 Assign 敘述：

- BoTSORT `update()`：`cls_max_time_lost = int(cls_buffer_sec * self.frame_rate)` → `cls_max_time_lost = int(cls_buffer_sec)`。
- ByteTrack `update()`：`cls_max_time_lost = int(self.frame_rate / 30.0 * cls_buffer)` → `cls_max_time_lost = int(cls_buffer)`。

`cls_buffer_sec` 是歷史名稱，現在存放幀數。方法內「秒」與換算相關的過時註解保留原文；除了上述各一行外，所有凍結方法、註解與變數名稱不變。`__init__` 的全域 fallback 不改；設定載入器要求每個已宣告類別的有效 `track_buffer` 都是正整數，使這些類別不會進入 fallback。沒有類別設定時才取 `default_tracking_params`，不會逐鍵合併。缺少、非正整數、浮點值（包括 `90.0`）與布林值都在變更全域參數之前報錯。

AST smoke 測試只在 BASE_COMMIT 版本套用上述明列替換，每項必須恰好匹配一次，再比對全部凍結方法。例外以外的敘述改動仍不允許。

## 設定遷移

- 固定 30fps 的 BoTSORT：原 3.0／2.0 → 90／60 幀。
- 原先缺少逐類別設定、使用全域 fallback 的 30fps 情境：逐類別補上 120 幀。
- `botsort_reid_timm.yaml` 的 C000025（auto → 5fps）：15／10 幀。
- `reid_crops.yaml`：以多數來源 10fps 換算成 30／20 幀；C000043（9fps）、C000011／C000056（14fps）、C000012（15fps）不等價，影響之後重新蒐集裁切與重新產生合併候選。既有裁切、`fragment_merges.csv` 與 v3 不重建。
- Golden 設定同步換算，驗收要求與 `reid1` 差異為零。

舊設定中的 `3.0` 會被整數檢查拒絕，但整數 `3` 會被視為 3 幀；外部設定必須先依原追蹤器算式換算，不能直接沿用。

## 衍生量

碎片候選窗直接為 `track_buffer + margin_frames`；移除 `retention_frames()`、序列幀率推算與追蹤器參數。外插維持每幀速度，Re-ID 閘門仍為每幀像素乘遺失幀數。

## 驗證與還原

見 [本變更驗證紀錄](../../openspec/changes/track-buffer-in-frames/verification.md)。還原時需成套還原程式、設定、AST 例外與文件；已提交的變更可用 git revert，資料不需還原。
