# Design

## Context

動機見 proposal.md。現況：

- BoTSORT `update()`（`class_aware_botsort.py:382`）：`cls_max_time_lost = int(cls_buffer_sec * self.frame_rate)`；同方法內 `:365-373` 留有舊算式的註解、`:375`／`:381` 的註解寫「單位是秒」。ByteTrack `update()`（`class_aware_bytetrack.py:261`）：`cls_max_time_lost = int(self.frame_rate / 30.0 * cls_buffer)`。兩者在 `track_buffer` 缺少時都退回 `self.max_time_lost`（ultralytics `__init__`：`int(frame_rate / 30 × args.track_buffer)`，`args.track_buffer` 來自 `DEFAULT_TRACKER_CONFIG` 的 120）。
- `tests/test_smoke.py::test_tracker_override_bodies_unchanged` 以 `ast.dump` 比對 `BASE_COMMIT`（`9c6414b`）的方法本體，要求完全相同。
- 追蹤器的 `frame_rate` 來自 config 的 `tracking_params.frame_rate`：大多數情境（含全部 golden config）是固定 `30`；`botsort_reid_timm.yaml` 與 `reid_crops.yaml` 是 `auto`（讀影片幀率四捨五入，C000025 為 4.876 → 5）。
- 沒有逐類別 `track_buffer` 的情境：`bytetrack.yaml`、`botsort_reid.yaml`、`grid_search_bytetrack.yaml`，以及 golden 的 `bytetrack.yaml`、`botsort_reid.yaml`；它們實際用 fallback，`frame_rate: 30` 下為 120 幀。
- `retention_frames()` 只被 `fragment_merge.find_candidates()` 使用；碎片合併的 `track_config` 是 `reid_crops.yaml`。grid search 的 `PARAM_GRIDS` 不含 `track_buffer`。

## Goals / Non-Goals

**Goals:**
- `track_buffer` 在兩個追蹤器都是保留期限幀數，凍結方法只開兩個敘述的例外，並由 AST 檢查精確守住。
- 既有情境換算後逐幀行為不變（golden 差異 0）。

**Non-Goals:**
- 不重新調整保留期限的數值；不改 Kalman 的 `dt_scale` 或其他與幀率相關的行為。
- 不重建 `fragment_merges.csv`、`v3` 或既有裁切。

## Decisions

### D1. 凍結方法的例外：只換兩個 Assign 敘述

- BoTSORT：`cls_max_time_lost = int(cls_buffer_sec * self.frame_rate)` → `cls_max_time_lost = int(cls_buffer_sec)`。ByteTrack：`cls_max_time_lost = int(self.frame_rate / 30.0 * cls_buffer)` → `cls_max_time_lost = int(cls_buffer)`。
- 變數名稱 `cls_buffer_sec` 不改（改名會動到同方法的多個敘述，超出例外範圍）；它在 ADR 0009 與 flow 文件中註明「歷史名稱，現為幀數」。
- 方法內的註解不在 AST 中，但仍屬「方法本體一字不改」的範圍；**不改**任何註解，包括寫著「單位是秒」的那兩行（使用者的決定是只改那一行）。過時的註解由 ADR 0009 與 flow 文件說明。
- 替代方案：改 `__init__` 讓 fallback 也統一（使用者否決，見 D3）；在 `create_tracker()` 換算回原單位（ADR 0008 已否決，且使用者選擇直接改算式）。

### D2. AST 檢查：先對 BASE 版做精確替換再比對

- 在 `tests/test_smoke.py` 新增一張明列的例外表：`(類別, 方法) → (舊敘述, 新敘述)`，以 `ast.unparse` 的字串表示。比對前，對 `BASE_COMMIT` 版的方法本體以 `ast.NodeTransformer` 把 unparse 後等於「舊敘述」的 `Assign` 節點換成「新敘述」解析出的節點；每個例外必須恰好替換 1 次，否則測試失敗（防止例外表過期或誤配到其他敘述）。
- 替換後要求 `ast.dump` 完全相同，所以例外以外的任何改動仍會失敗。註解不影響 AST，`BASE_COMMIT` 中以 `#` 註解掉的舊算式不會被誤配。
- 替代方案：更新 `BASE_COMMIT` 為新的 commit（會讓「與重構前相同」失去基準，且無法區分例外與其他改動）；以文字替換（舊算式也出現在註解裡，會誤配）。

### D3. fallback 不可達：載入設定時強制每類別有正整數 `track_buffer`

- 在 `load_track_config()` 驗證（與既有的融合參數驗證同一階段、更新全域參數之前）：對 `class_names` 的每個類別，取 `class_tracking_params[類別]`，沒有該類別時取 `default_tracking_params`，其中的 `track_buffer` 必須是 `int`（排除 `bool`）且 ≥ 1。YAML 的 `90.0` 也視為不合格，強迫設定檔寫成整數幀數。
- 錯誤訊息指出類別與 `track_buffer`；驗證失敗時全域參數保持不變。
- grid search 只 patch `PARAM_GRIDS` 列出的鍵，不會移除 `track_buffer`，fallback 在 grid search 中同樣不可達。

### D4. config 換算規則

依各 config 的 `frame_rate` 換算成修改前的保留期限幀數（值皆為整數，無取整誤差）：

| config | 追蹤器 | frame_rate | 修改前 | 修改後 |
|---|---|---|---|---|
| `botsort_noreid`、`botsort_save_reid_crops`、`botsort_save_reid_crops_video`、`collision_detect`、`grid_search_botsort`（及 golden 同名） | BoTSORT | 30 | 3.0／2.0 秒 | 90／60 |
| `bytetrack`、`grid_search_bytetrack`（及 golden `bytetrack`） | ByteTrack | 30 | 無（fallback 120） | 每類別 120 |
| `botsort_reid`（及 golden 同名） | BoTSORT | 30 | 無（fallback 120） | 每類別 120 |
| `botsort_reid_timm` | BoTSORT | auto（C000025 → 5） | 3.0／2.0 秒 | 15／10 |
| `reid_crops` | BoTSORT | auto（來源 9／10／14／15） | 3.0／2.0 秒 | 30／20（以 10fps 換算） |

- `reid_crops.yaml` 無法對所有來源序列等價：39 支序列中 35 支是 10fps，C000043（9fps）、C000011／C000056（14fps）、C000012（15fps）的保留期限會改變。只影響之後重新蒐集的裁切；golden 的裁切情境是 `botsort_save_reid_crops`（`frame_rate: 30`），不受影響。
- 換算後在每個 config 的 `track_buffer` 旁註明單位（幀）與換算來源，並移除寫著「秒」的註解。

### D5. 衍生量

- 移除 `retention_frames()` 與 `tests/test_retention_frames.py`。
- `fragment_merge`：`load_retention()` 改為只回傳 `{類別: int(track_buffer)}`；`find_candidates()` 移除 `tracker` 參數，候選窗為 `track_buffers[cls] + margin_frames`。候選窗不再需要序列幀率，`sequence_fps()` 與 `Track.fps` 一併移除；外插仍以幀差計算。兩支 `scripts/reid_data/` 腳本跟著調整。
- 碎片合併現有輸入（`reid_crops.yaml` 10fps 換算）在 10fps 序列上的候選窗與修改前（`int(3 × 10) + 10 = 40`）相同；非 10fps 序列不同，但 `fragment_merges.csv` 與 `v3` 依先前決定不重建，不做資料比對。
- Re-ID 閘門（`reid_center_gate_px_per_frame`）已是每幀計算，不變。

### D6. 文件

- 新增 `docs/adr/0009`：`track_buffer` 統一為保留期限幀數、凍結方法的兩個例外敘述、`cls_buffer_sec` 為歷史名稱、方法內過時的註解不改；標明取代 ADR 0008。ADR 0008 開頭加「已由 ADR 0009 取代」。ADR 0004 加入例外並連到 0009。
- CLAUDE.md（`track_buffer` 落差那一段、碎片合併時間條件）、CONTEXT.md（「保留期限」定義與 217 行的換算說明；只改這兩處，保留使用者未 commit 的其他修改）、`class_aware_botsort_flow.md`、`class_aware_bytetrack_flow.md`、`HANDOFF.md`、`scripts/reid_data/README.md`、`config/fragment_merge.yaml` 的註解。

## Risks / Trade-offs

- **[風險] 外部的 config 或筆記仍用舊單位**（例如以秒寫的 BoTSORT 設定被直接沿用）→ D3 的整數檢查會擋下 `3.0` 這類值，但擋不住寫成整數 `3` 的舊秒數；在 ADR 0009 與 CLAUDE.md 明確寫出單位，config 註解標「幀」。
- **[取捨] 同一個幀數在不同幀率下代表不同的遮擋時間** → 這是選擇幀數的直接代價（ADR 0008 當初的理由）；5fps 評估與 10fps 來源若要相同的秒數，需在各自的 config 寫不同的幀數。
- **[風險] AST 例外表誤配** → D2 要求每個例外恰好替換 1 次。
- **[風險] 進行中的 `align-reid-training-domain` 依賴「依追蹤器換算」** → 在 proposal 的 Impact 註記；該變更實作前需改寫其 `reid-fusion` delta（屬於該變更的範圍）。

## Migration Plan

1. D2 的 AST 例外表與測試（先失敗）→ D1 兩行修改 → 測試通過。
2. D3 檢查與測試 → D4 換算所有 config（含 golden）→ `pytest`。
3. D5 衍生量與測試 → `pytest`。
4. D6 文件。
5. 經使用者確認後跑 golden（約 16 分鐘），與 `reid1` 比對差異須為 0。

**還原方式**：git revert；資料未變動。
