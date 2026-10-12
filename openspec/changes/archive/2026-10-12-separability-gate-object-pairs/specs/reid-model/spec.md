# Spec Delta

## MODIFIED Requirements

### Requirement: 外觀分離度關卡判定

雜訊底線 SHALL 為與 `dinov2s_v3` 訓練設定完全相同、只換亂數種子的模型與 `dinov2s_v3` 以緊框量測的差距（各類別 AUC 與 TPR@FPR5% 的絕對差）。car 與 scooter 的最小改善量 SHALL 取「AUC 0.01、TPR@FPR5% 0.02」與「該類別 2 倍雜訊底線」中較大者。

候選通過的條件：car 與 scooter 的 AUC 與 TPR@FPR5% 都比參考高出至少最小改善量，而且其他干擾物來自至少 10 組 GT 物體配對的類別 AUC 下降不超過 0.02（物體配對數見 `reid-fusion` 的量測條文；不足 10 組的類別只回報，不列入判定）。car 或 scooter 的干擾物物體配對不足 10 組時，候選 MUST NOT 通過。多個候選通過時，只有 car 與 scooter 平均 AUC 最高者進入追蹤驗收；沒有候選通過時 MUST NOT 進行追蹤驗收，`config/botsort_reid_timm.yaml` 維持不變。val 的 mAP MUST NOT 作為關卡依據。

#### Scenario: 分離度沒有改善的模型不驗收
- **WHEN** 候選模型的 scooter AUC 低於參考
- **THEN** 這個模型不進入追蹤驗收

#### Scenario: 改善太小的模型不驗收
- **WHEN** 候選模型的 car AUC 比參考高 0.004、scooter 高 0.02
- **THEN** 這個模型不通過關卡

#### Scenario: 雜訊底線較大時以雜訊底線為準
- **WHEN** 換種子重訓的模型與 `dinov2s_v3` 的 car AUC 相差 0.008，候選的 car AUC 比參考高 0.012
- **THEN** car 的最小改善量是 0.016，這個候選不通過關卡

#### Scenario: 只有尾端沒改善的模型不驗收
- **WHEN** 候選模型的 car、scooter AUC 都比參考高 0.02，但 car 的 TPR@FPR5% 只高 0.005
- **THEN** 這個模型不通過關卡

#### Scenario: 物體配對不足的類別不列入下降檢查
- **WHEN** 候選的 car、scooter 都超過最小改善量，small_truck 的 AUC 下降 0.03，但 small_truck 的干擾物只來自 2 組 GT 物體配對
- **THEN** small_truck 不列入判定，這個候選通過關卡，輸出中仍回報 small_truck 的下降

#### Scenario: 物體配對足夠的類別下降過多時不通過
- **WHEN** 候選的 car、scooter 都超過最小改善量，person 的干擾物來自 20 組 GT 物體配對，AUC 下降 0.03
- **THEN** 這個候選不通過關卡

#### Scenario: car 或 scooter 證據不足時不通過
- **WHEN** 候選的 car、scooter 都超過最小改善量，但 scooter 的干擾物只來自 6 組 GT 物體配對
- **THEN** 這個候選不通過關卡，原因註明 scooter 的物體配對不足

#### Scenario: 只靠緊框就通過
- **WHEN** `dinov2s_v3` 以緊框量測時通過關卡，而且在通過者中 car 與 scooter 的平均 AUC 最高
- **THEN** 以 `dinov2s_v3`（緊框）進入追蹤驗收

#### Scenario: 多個候選通過時只驗收最好的
- **WHEN** 兩個候選模型都通過關卡，car 與 scooter 的平均 AUC 分別是 0.80 與 0.83
- **THEN** 只有平均 AUC 0.83 的模型進入追蹤驗收

#### Scenario: 沒有候選通過
- **WHEN** 所有候選都沒有通過關卡
- **THEN** 不進行追蹤驗收，`config/botsort_reid_timm.yaml` 維持不變
