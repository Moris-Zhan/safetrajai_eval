# Proposal

## Why

change `reid-center-gate-fusion` 的校準顯示，`dinov2s_v3` 在 C000025 上分不開同一條 GT 軌跡（間隔 ≥ 1 秒）與它的干擾物（car：正樣本 P95 0.41、干擾物 P5 0.21），擴大外觀的作用範圍反而讓串錯增加。檢查訓練資料後發現三個與評估鏡頭不一致的地方：

- **解析度**：C000025 的物體很小（最短邊中位數 car 16px、scooter 13px、person 12px，86～100% 小於 32px），訓練裁切卻規定最短邊 ≥ 32px（中位數 car 55px、scooter 40px）。
- **裁切方式**：推論端 `Custom_ReID` 沿用 `save_one_box`（框放大 1.02 倍再加 10px 邊），訓練裁切是緊框；一台 13px 的機車推論時有一半以上是背景。
- **沒看過干擾物**：PK 取樣從全部身分隨機抽，88% 的裁切來自 C000054，同一批的負樣本多半不在同一畫面，模型沒學到區分干擾物（同時出現、鄰近的同類道路使用者）。

不重新蒐集資料，先在訓練與推論端對齊這三點，並在追蹤驗收前以調參序列上的外觀分離度確認模型真的變好。

## What Changes

- **解析度增強**：訓練時以固定機率把裁切縮小到評估鏡頭常見的尺寸（最短邊隨機取在設定範圍內），再 letterbox 回輸入尺寸，模擬小物體的低解析度。
- **推論端改用緊框**：`Custom_ReID` 的新格式路徑改以 `xywh2xyxy` 轉換後的框直接裁切（不放大、不加邊），與訓練裁切一致。舊的完整模型 pickle 路徑與 YOLO 的 `ReID` 不變。**BREAKING**（只影響新格式權重的特徵值；目前沒有任何情境設定啟用它）。
- **干擾物取樣**：PK 取樣中，一部分身分改從種子身分的干擾物（同序列、同類別、同時出現，而且框中心在依鏡頭解析度換算的半徑內；座標讀自裁切索引）中選，讓同一批包含干擾物；不選時間上相鄰但不重疊的身分，避免把同一物體的碎片當成負樣本推開。比例與最少共同幀數寫入訓練紀錄。最佳 epoch 改以縮小版 val 的 mAP 選。
- **外觀分離度關卡**：候選先以 `fusion_distance_stats.py` 在調參序列上量測外觀分離度（新增 AUC 與 TPR@FPR5%），與參考（`dinov2s_v3` 以修改前的裁切量測）比較；car、scooter 需有最小改善量（不低於換種子重訓的 2 倍雜訊）才進入追蹤驗收；另依框的尺寸分組回報，作為之後是否限制極小框使用外觀的依據。
- **訓練與驗收**：四個候選——`dinov2s_v3` 以緊框量測（不需訓練），以及在 `v3` 上訓練的三個模型（只加解析度增強、只加干擾物取樣、兩者都加）；通過關卡的最佳候選沿用 `reid-model` 的逐類別規則與 `reid-center-gate-fusion` 的參數網格做追蹤驗收；結果更新 `config/botsort_reid_timm.yaml`。
- **不做**：不重新蒐集裁切、不改最小尺寸門檻、不改鏡頭比例、不產生新的資料集版本；不改追蹤器的融合方式與凍結方法。

## Capabilities

### New Capabilities

（無）

### Modified Capabilities

- `reid-fusion`：所有間隔改以幀數表示；校準的正樣本另外以「最小間隔幀數～保留期限幀數」（直接取設定檔的逐類別 `track_buffer`，見 ADR 0009）的間隔回報並用於門檻與關卡，負樣本稱為干擾物，並依框的尺寸分組回報。
- `reid-model`：推論端的裁切改為緊框；訓練的干擾物取樣只取同時出現的身分、以縮小版 val 選 epoch，訓練紀錄寫明相關設定；新增「候選先通過外觀分離度關卡才進入追蹤驗收」。

## Impact

- **程式碼**：`src/safetrajai/reid/encoders.py`（`Custom_ReID` 新格式裁切）、`src/safetrajai/reid/training/`（`data.py` 解析度增強與縮小版 val、`sampler.py` 干擾物取樣、`train.py` 設定與紀錄）、`scripts/reid_train/train_reid.py` 參數、`src/safetrajai/reid/fusion_stats.py`／`scripts/reid_train/fusion_distance_stats.py`（AUC、TPR@FPR5%、關卡判定、`--legacy-crop` 量測參考，`--config` 預設改為 `botsort_reid_timm.yaml`）；對應單元測試。
- **模型**：`models/class_reid_weights/<run>/` 新增三個權重；`dinov2s_v3` 保留作為參考與緊框候選。
- **設定**：`config/botsort_reid_timm.yaml` 依驗收結果更新；其他情境設定與 golden 不變（golden 只用 YOLO `ReID`）。
- **文件**：CLAUDE.md 的 Re-ID 章節、`src/safetrajai/reid/HANDOFF.md`；`CONTEXT.md` 已新增干擾物、外觀分離度，並在裁切的定義寫明緊框。
