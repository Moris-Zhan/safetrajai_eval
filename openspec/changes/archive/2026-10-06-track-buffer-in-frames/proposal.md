# Proposal

## Why

`track_buffer` 在兩個追蹤器的單位不同：BoTSORT 的 `update()` 當成秒（`int(cls_buffer_sec × frame_rate)`），ByteTrack 的 `update()` 當成 30fps 基準的幀數（`int(frame_rate / 30 × cls_buffer)`）。同一個數值的實際保留期限相差 30 倍，ADR 0008 只能在外面寫 `retention_frames()` 逐字複製兩套算式來彌補。使用者決定直接統一：兩個追蹤器的 `track_buffer` 都代表**保留期限幀數**，並為此在凍結方法上開一個明列的例外（取代 ADR 0008、修訂 ADR 0004）。

## What Changes

- **追蹤器（凍結方法的明列例外）**：只改兩個 `update()` 中換算 `cls_max_time_lost` 的那一行，改為 `int(cls_buffer)`（BoTSORT 為 `int(cls_buffer_sec)`，變數名稱不改）；兩個類別的其他凍結方法與 `update()` 的其他敘述一字不動。**BREAKING**：`track_buffer` 的單位改變。
- **AST 檢查**：smoke 測試對照 `BASE_COMMIT` 時，只允許上述兩個敘述不同（先把 `BASE_COMMIT` 版的舊敘述替換成新敘述再比對）；其他差異仍然失敗。
- **設定檢查**：載入追蹤設定時，`class_names` 中每個類別都 SHALL 解析到一個正整數 `track_buffer`，否則報錯。追蹤器 `__init__` 的全域 fallback（`self.max_time_lost`）因此不會被用到，`__init__` 不改。
- **config 數值換算（行為不變）**：依各 config 的 `tracking_params.frame_rate` 換算成等價幀數：
  - `frame_rate: 30` 的 BoTSORT config：3.0 秒 → 90、2.0 秒 → 60。
  - 沒有逐類別 `track_buffer` 的 config（`bytetrack.yaml`、`botsort_reid.yaml`、`grid_search_bytetrack.yaml`，以及對應的 golden config）目前用的是 fallback `int(30 / 30 × 120)`，補上每類別 120。
  - `frame_rate: auto` 的 `botsort_reid_timm.yaml`（C000025，影片 4.876fps → 5）：15、10。
  - `reid_crops.yaml`（`frame_rate: auto`，來源序列 9／10／14／15fps）無法對所有序列等價，以多數的 10fps 換算為 30、20（design 的待確認事項）。
  - `tests/golden/configs/*.yaml` 一併換算；預期 golden 比對差異為 0。
- **衍生量**：`retention_frames()` 移除；碎片合併的候選窗改為 `track_buffer + margin_frames`，不再依追蹤器與序列幀率換算，`load_retention()` 只回傳逐類別的 `track_buffer`。Re-ID 閘門已是每幀計算，不變。
- **文件**：新增 ADR 0009（取代 ADR 0008）、修訂 ADR 0004 加入例外；更新 CLAUDE.md、CONTEXT.md「保留期限」、`class_aware_*_flow.md`、`HANDOFF.md`、`scripts/reid_data/README.md` 中關於 `track_buffer` 單位的說明。
- **不做**：不改 `__init__`、`remove_duplicate_stracks()`、`init_track()`；不重新調參（數值只做等價換算）；不重建 `fragment_merges.csv` 與 `v3`。

## Capabilities

### New Capabilities

- `tracker-retention`：兩個追蹤器的 `track_buffer` 統一為保留期限幀數，以及載入設定時對 `track_buffer` 的檢查。

### Modified Capabilities

- `project-structure`：「Tracker override 方法本體不變」加入兩個 `update()` 保留期限敘述的明列例外。
- `reid-dataset`：「遮擋碎片合併只產生合併紀錄」的候選窗改為 `track_buffer`（幀數）加 `margin_frames`，不再依追蹤器與序列幀率換算。

## Impact

- **程式碼**：`src/safetrajai/tracking/class_aware_botsort.py`、`class_aware_bytetrack.py`（各一行）、`src/safetrajai/tracking/config.py`（檢查、移除 `retention_frames()`）、`src/safetrajai/reid/fragment_merge.py`、`scripts/reid_data/extract_fragment_features.py`、`scripts/reid_data/build_fragment_merges.py`；測試 `tests/test_smoke.py`、`tests/test_reid_fragment_merge.py`、`tests/test_retention_frames.py`（移除）。
- **設定**：`config/*.yaml` 中所有追蹤情境、`tests/golden/configs/*.yaml`。grid search 的 `PARAM_GRIDS` 沒有 `track_buffer`，不需換算。
- **行為**：`frame_rate` 固定的 config 與 C000025 上的 `botsort_reid_timm.yaml` 逐幀相同；`reid_crops.yaml` 在非 10fps 的 4 支來源序列（C000011、C000012、C000043、C000056）上保留期限改變，只影響之後重新蒐集的裁切。
- **進行中的變更**：`align-reid-training-domain` 的 `reid-fusion` delta 有「依追蹤器換算保留期限」的情境，套用本變更後需改寫（不在本變更內修改）。
- **文件**：見 What Changes；`CONTEXT.md` 目前有使用者未 commit 的修改，實作時只改「保留期限」相關段落。
