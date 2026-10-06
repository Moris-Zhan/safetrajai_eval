# Proposal

## Why

ADR 0008 決定：追蹤器設定值 `track_buffer` 維持原單位（BoTSORT 為秒、ByteTrack 為 30fps 基準的幀數，換算寫在凍結的 `update()`），但**依賴保留期限的衍生量一律以保留期限幀數計算**（術語見 `CONTEXT.md`「保留期限」）。目前還有兩個已完成的功能以秒計算：

- Re-ID 閘門的放寬量 `reid_center_gate_px_per_s` × 遺失秒數（change `reid-center-gate-fusion`）。
- 碎片合併的候選窗「`track_buffer` 秒 + 1 秒」與以每秒速度外插（change `merge-occlusion-fragments`，產生 `v3` 的 `fragment_merges.csv`）。

這兩處的「秒」都隱含 BoTSORT 的解讀；若改用 ByteTrack 的設定檔，候選窗會差 30 倍。

## What Changes

- **Re-ID 閘門**：`reid_center_gate_px_per_s` 改名為 `reid_center_gate_px_per_frame`，閘門半徑＝`reid_center_gate_px` ＋ 每幀像素 × 遺失幀數。**BREAKING**（參數改名；目前沒有任何情境設定使用它）。
- **碎片合併**：
  - 候選窗改為「0 < B 的第一張與 A 的最後一張的幀差 ≤ 該類別保留期限幀數 ＋ `margin_frames`」。保留期限依 `track_config` 的追蹤器與序列幀率換算（BoTSORT：`int(track_buffer × 幀率)`；ByteTrack：`int(幀率 / 30 × track_buffer)`）；`config/fragment_merge.yaml` 的 `time_margin_s: 1.0` 改為 `margin_frames: 10`（來源影片 10fps，相當於原本的 1 秒）。
  - 外插改用每幀速度（以最後兩張的幀差計算）乘以幀差。
  - 序列幀率由裁切索引推得（相鄰裁切的 Δ幀 / Δ時間 取中位數後四捨五入，與追蹤器 `frame_rate: auto` 的取整方式一致），不再讀影片。
- **驗收條件：結果不變**。以現有索引與排除清單，分別用修改前（秒）與修改後（幀）的候選規則產生候選清單並比較；相同時 `fragment_merges.csv` 與 `v3` 不需要重建。不同時列出差異並回報，不自動重建。
- **文件**：CLAUDE.md 對 `track_buffer` 的說明改為「設定值以秒（BoTSORT）或 30fps 幀數（ByteTrack）表示，衍生量一律以保留期限幀數計算（ADR 0008）」。
- **不做**：不改追蹤器的凍結方法與情境 config 的 `track_buffer`；不改裁切蒐集的 0.5 秒抽樣與靜止標記的時間條件（與保留期限無關）。

## Capabilities

### New Capabilities

（無）

### Modified Capabilities

- `reid-fusion`：「中心距離閘門讓外觀距離重新關聯遺失中的軌跡」的放寬參數改為每幀像素 × 遺失幀數。
- `reid-dataset`：「遮擋碎片合併只產生合併紀錄」的時間條件改為保留期限幀數 ＋ 餘裕幀數，外插改用每幀速度。

## Impact

- **程式碼**：`src/safetrajai/tracking/class_aware_botsort.py`（`_gated_appearance()`，不在凍結範圍）、`src/safetrajai/tracking/config.py`（參數驗證改名）、`src/safetrajai/reid/fragment_merge.py`、`scripts/reid_data/extract_fragment_features.py`、`scripts/reid_data/build_fragment_merges.py`；`tests/test_reid_fusion.py`、`tests/test_reid_fragment_merge.py`。
- **設定**：`config/fragment_merge.yaml`（`time_margin_s` → `margin_frames`）。情境設定不變，golden 不受影響。
- **資料**：預期不變；以候選清單比對確認。
- **文件**：CLAUDE.md、`src/safetrajai/tracking/class_aware_botsort_flow.md`、`scripts/reid_data/README.md` 中提到秒數的段落。
