# Proposal

## Why

change `align-reid-training-domain` 的外觀分離度關卡判定沒有候選通過：`dinov2s_v3_lowres`、`dinov2s_v3_both` 的 car、scooter 都遠超過最小改善量（both：car AUC +0.031、TPR@FPR5% +0.145），只因 small_truck AUC 下降約 0.03 而被擋下。檢查後發現，這個下降缺乏統計基礎：

- 關卡的「其他類別有 30 對以上干擾物配對才檢查」計算的是**幀配對**。small_truck 的 43 對只來自 **3 組不同的 GT 物體配對**，下降集中在其中一組（#5 與 #27）的 11 幀；person 是 476 對、20 組物體配對；truck 是 1 組物體配對。
- 其中 1 組物體配對含 GT id = -1 的框（調參序列 `150634_15s_1` 有 4 個框沒有身分：scooter 與 small_truck 各 2 個），量測工具把它當成一條正常軌跡。
- val（來源鏡頭，只作旁證）上 small_truck 的 mAP 變化（both 0.751 對 `dinov2s_v3` 0.760）小於只換種子的差距（0.012），沒有退步的證據。

關卡應以獨立證據判斷其他類別是否退步。修正後重新判定既有的候選，通過者才進入追蹤驗收。

## What Changes

- **量測跳過沒有身分的 GT 框**：`fusion_stats` 量測時排除 GT id < 0 的框（正樣本、干擾物、退回配對都不用），輸出記錄每個類別跳過的框數。GT 檔不修改。
- **以獨立物體配對計算證據**：每個類別回報干擾物來自的不同 GT 物體配對數。關卡的「其他類別 AUC 下降不超過 0.02」只套用在至少 10 組物體配對的類別（取代 30 對幀配對）；car、scooter 少於 10 組時，關卡判定為無法通過。
- **不變**：校準融合門檻時「干擾物少於 30 對改用任意配對」的退回規則、最小改善量與雜訊底線的計算、追蹤驗收的逐類別規則。
- **重新判定與驗收**：以新程式重新量測參考（`dinov2s_v3` 舊裁切）與 5 個候選（`dinov2s_v3` 緊框、seed1、lowres、distractor、both），重算雜訊底線與關卡；有候選通過時，以 car、scooter 平均 AUC 最高者依 `align-reid-training-domain` 的追蹤驗收方式驗收，並依結果更新 `config/botsort_reid_timm.yaml`；沒有通過時記錄並維持 config。
- **規則在看到結果後修改的風險**：N=10 取在 4～20 之間對目前結果都相同（small_truck 2～3 組、person 20 組），不是為了讓特定候選通過而挑選；追蹤驗收仍逐類別判定，保留序列仍獨立否決整體。

## Capabilities

### New Capabilities

（無）

### Modified Capabilities

- `reid-fusion`：門檻校準排除 GT id < 0 的框，並回報每個類別的干擾物物體配對數。
- `reid-model`：外觀分離度關卡的「其他類別」改以至少 10 組 GT 物體配對為列入條件，car、scooter 物體配對不足時無法通過。

## Impact

- **程式碼**：`src/safetrajai/reid/fusion_stats.py`（跳過 id < 0、物體配對數、關卡條件）、`scripts/reid_train/fusion_distance_stats.py`（輸出跳過數）；`tests/test_reid_fusion_stats.py`。
- **量測與驗收輸出**：`runs/reid_domain/`（重新量測）、追蹤驗收輸出；不需要重新訓練。
- **設定**：`config/botsort_reid_timm.yaml` 依追蹤驗收結果更新，或維持 6 類關閉；其他情境設定與 golden 不變。
- **文件**：CLAUDE.md 的 Re-ID 章節（關卡規則與結果）。
