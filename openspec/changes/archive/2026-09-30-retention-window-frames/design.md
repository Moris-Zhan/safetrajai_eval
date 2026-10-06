# Design

## Context

動機見 proposal.md 與 ADR 0008，需求見兩份 delta spec。現況：

- `_gated_appearance()`：`lost_s = (self.frame_id − track.frame_id) / self.frame_rate`，半徑＝`gate_px + reid_center_gate_px_per_s × lost_s`；`config.py` 的 `_FUSION_PARAM_RULES` 驗證 `reid_center_gate_px_per_s ≥ 0`。沒有任何情境設定使用這兩個鍵。
- `fragment_merge.py`：`CropSample` 只有 `time_s`、`cx`、`cy`、`size`；`find_candidates()` 以 `track_buffers[cls] + time_margin_s`（秒）為候選窗，`_extrapolate()` 以每秒速度 × 秒數外插。`load_track_buffers()` 只讀 `track_buffer` 數值，不看追蹤器種類。呼叫端：`extract_fragment_features.py`、`build_fragment_merges.py`、`tests/test_reid_fragment_merge.py`。
- 裁切索引每列有 `frame` 與 `time_s`；來源影片 10fps，裁切約每 5 幀一張。
- `fragment_merge.yaml` 的 `track_config` 指向 `config/reid_crops.yaml`（BoTSORT，四輪車 `track_buffer` 3、person／scooter 2）。

## Goals / Non-Goals

**Goals:**
- 依賴保留期限的衍生量都以保留期限幀數計算，而且與追蹤器實作的換算方式相同。
- 在現有資料上，碎片合併的候選清單與修改前相同。

**Non-Goals:**
- 不改凍結方法、不改情境 config 的 `track_buffer`。
- 不重建 `fragment_merges.csv` 或資料集版本（除非比對發現差異，且使用者決定重建）。

## Decisions

### D1. 保留期限換算放在一個純函式

- 新增 `retention_frames(track_buffer, frame_rate, tracker) -> int`（放在 `safetrajai/tracking/config.py`，因為它描述的是追蹤設定的語意）：`botsort` → `int(track_buffer × frame_rate)`，`bytetrack` → `int(frame_rate / 30 × track_buffer)`，其他值報錯。`frame_rate` 必須是整數（與追蹤器 `frame_rate: auto` 的取整一致）。
- 單元測試直接對照兩個追蹤器 `update()` 裡的算式（以數值案例），確保一致。

### D2. Re-ID 閘門改為每幀

- `_gated_appearance()`：`lost_frames = self.frame_id − track.frame_id`（只對 Lost），半徑＝`gate_px + reid_center_gate_px_per_frame × lost_frames`。
- `_FUSION_PARAM_RULES` 改為 `reid_center_gate_px_per_frame ≥ 0`；`_validate_fusion_params()` 遇到 `reid_center_gate_px_per_s` 時丟出 `ValueError`，訊息指出新名稱。
- 既有的單元測試（「閘門隨遺失秒數放寬」）改為每幀版本，並新增舊名稱被拒絕的測試。

### D3. 碎片合併改為幀數

- `CropSample` 新增 `frame`；`Track` 新增 `fps`（整數）。序列幀率：同一序列所有軌跡中相鄰裁切的 `Δframe / Δtime_s`（Δtime_s > 0）取中位數後四捨五入；在 `collect_tracks()` 依序列計算一次。
- `load_track_buffers()` 改為 `load_retention(track_config) -> (tracker, {cls: track_buffer})`，讀取 `tracking_params.tracker`；候選窗在 `find_candidates()` 內以 D1 的函式依序列幀率換算：`window = retention_frames(...) + margin_frames`。
- 條件：`0 < gap_frames ≤ window`；外插：每幀速度＝`(last.cx − prev.cx) / (last.frame − prev.frame)`，位置＝`last + 速度 × gap_frames`。
- `Candidate.gap_s` 改為 `gap_frames`；`config/fragment_merge.yaml` 的 `time_margin_s: 1.0` 改為 `margin_frames: 10`，`load_merge_config()` 對應修改（舊鍵出現時報錯）。
- `bisect` 的排序鍵改為起始幀；同一序列的幀號單調，排序結果與以時間排序相同。

### D4. 驗收：候選清單比對

- 寫一支一次性的比對腳本（放在 scratchpad，不進 repo）：以 `git show` 取出修改前的 `fragment_merge.py` 作為舊模組，對同一份索引與兩份排除清單（與產生 `v3` 時相同的輸入）分別執行舊的 `find_candidates`（秒）與新的（幀），比較 `(class, sequence, identity_a, identity_b)` 集合，並列出 `distance_ratio` 的最大差值。
- 結果相同 → 端點裁切、特徵與合併結果都不會變，`fragment_merges.csv` 與 `v3` 不需重建；結果不同 → 列出差異的配對與原因（時間戳不規則的序列、幀率推算），回報使用者決定，不自動重建。
- 讀取的是索引 CSV 與排除清單（metadata），不讀任何影像。

## Risks / Trade-offs

- **[風險] 某些序列的時間戳不規則，幀率推算與實際不同** → 以中位數推算；比對若有差異會列出對應的序列。
- **[風險] 外插的數值差異讓臨界的配對進出候選** → 比對列出 `distance_ratio` 差值與進出的配對。
- **[取捨] 參數改名是 BREAKING** → 目前沒有 config 使用；舊名稱會明確報錯，不會被靜默忽略。

## Migration Plan

1. D1、D2 與測試；D3 與測試；`pytest`。
2. D4 比對；依結果回報。
3. 文件。

**還原方式**：git revert；資料未變動。
