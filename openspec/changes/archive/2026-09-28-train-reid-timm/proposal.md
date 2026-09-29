# Proposal

## Why

`botsort_reid.yaml` 的 Re-ID 目前全部用偵測器 backbone（`yolo11x*.pt`）暫代，舊的 Re-ID 權重已確認無效並隔離，所以 ReID 預設關閉。`rebuild-reid-dataset` 與 `auto-reid-exclusions` 已交付可重現的 Re-ID 資料集版本（`v2`：22,164 個身分、156,141 張；`merge-occlusion-fragments` 完成後會有 `v3`），現在可以在本地訓練一個 6 類共用的 Re-ID 模型，並驗證它能否改善 C000025 的追蹤。

## What Changes

- **訓練**：以 timm `vit_small_patch14_dinov2` 為骨幹，在本地 GPU 上以資料集版本的 manifest 訓練 6 類共用的 Re-ID 模型。
  - 輸入：裁切 letterbox 到 224×224（人與車輛共用同一輸入尺寸）。
  - 損失：身分分類（label smoothing）＋ batch-hard 三元組；以「每批 P 個身分 × K 張」取樣。
  - 資料：`v2` 與 `v3` 各訓練一次（`v3` 不存在時只用 `v2`），以調參序列上的追蹤結果選一個進入驗收；訓練紀錄寫明使用的版本。
- **權重格式**：存成 `state_dict`＋設定（骨幹名稱、特徵維度、輸入尺寸、資料集版本），不再用完整模型 pickle；放在 `models/class_reid_weights/<run>/`（路徑必須含 `class_reid_weights`，凍結的追蹤器 `__init__` 以此選用 `Custom_ReID`）。
- **推論端**：`Custom_ReID` 支援新格式（letterbox 224、正規化、L2 正規化特徵），同時保留舊的完整模型 pickle 載入（ADR 0001）；追蹤器的 override 方法不改（ADR 0004）。
- **評估**：
  - val：以資料集版本的 val 計算各類別的 mAP 與 Rank-1，只作參考（val 身分都在同一序列內，分數偏樂觀）。
  - 驗收基準：目前最好的追蹤設定（`config/reid_crops.yaml` 的 R1 參數、`gmc_method: sparseOptFlow`、`frame_rate: auto`）關閉 ReID。
  - 驗收切分（術語見 `CONTEXT.md`）：C000025 的 145630、150634 兩個時段（4 段）為調參序列，152639 時段（2 段）為保留序列。
  - 逐類別啟用：在調參序列上逐類別開啟 `use_reid`；IDSW 不增加、IDP、AssA、CLEAR TP 都不下降的類別才啟用。
  - 保留序列只回報成績；全部啟用後保留序列的整體 IDSW 增加或 IDP 下降時，逐一關閉類別直到不變差。
- **情境設定**：新增 `config/botsort_reid_timm.yaml`（以驗收基準為底），記錄驗收通過的逐類別設定；既有的 `botsort_reid.yaml` 與 golden 情境不變。
- **不做**：
  - 不改追蹤器的 override 方法與 `CLASS_TRACKING_PARAMS` 的讀取順序。
  - 不做跨鏡頭的身分對應，也不上傳任何資料或權重到外部服務。
  - 不重新產生資料集版本。

## Capabilities

### New Capabilities

- `reid-model`：Re-ID 模型的訓練輸入、權重格式、推論端載入與特徵、逐類別啟用的驗收規則。

### Modified Capabilities

（無）

## Impact

- **程式碼**：
  - 新增訓練模組（manifest 資料集、PK 取樣、模型、損失）與訓練、評估腳本（`scripts/reid_train/`）。
  - `safetrajai/reid/encoders.py`：`Custom_ReID` 支援新格式。
- **依賴**：timm（已在 conda env `track_eval`）；DINOv2 預訓練權重第一次使用時從 HF Hub 下載（只下載，不上傳）。
- **資料與模型**：`models/class_reid_weights/<run>/`（權重與訓練紀錄）；訓練與驗收輸出在 `runs/`。
- **設定**：新增 `config/botsort_reid_timm.yaml`。
- **文件**：CLAUDE.md 的 Re-ID 章節、`scripts/reid_data/README.md`（訓練讀取 manifest 的方式）。
