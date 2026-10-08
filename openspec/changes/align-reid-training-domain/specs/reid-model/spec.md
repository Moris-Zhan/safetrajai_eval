# Spec Delta

## MODIFIED Requirements

### Requirement: 訓練只讀取資料集版本

Re-ID 模型的訓練與 val 評估 SHALL 只讀取一個資料集版本的 `manifest.csv`（以 `crop_path` 相對於裁切根目錄讀取影像，依 `split` 區分 train 與 val），MUST NOT 直接掃描裁切目錄。訓練紀錄 SHALL 寫明使用的資料集版本、骨幹、輸入尺寸、超參數與亂數種子，以及解析度增強（是否啟用、機率、最短邊範圍）、干擾物取樣（是否啟用、比例、最少共同幀數、配對幀差、半徑）與選擇最佳 epoch 的方式。要使用哪些裁切只依 manifest 決定；干擾物取樣所需的序列、幀與框座標 SHALL 由裁切索引中與 manifest `crop_path` 對應的列取得（只讀 metadata，不讀影像），鏡頭的畫面寬度由該鏡頭的 ROI 檔取得。解析度增強的範圍 MUST NOT 依評估資料集的統計逐類別設定。

干擾物取樣只能選種子身分的**干擾物**：同一序列、同一類別，兩者的裁切幀範圍（不擴張）至少重疊設定的共同幀數，而且至少有一對幀差不超過設定幀數的裁切，其框中心距離在半徑以內。半徑 SHALL 為該類別兩倍 `norm_factor` 乘以「來源鏡頭畫面寬度 / 評估鏡頭畫面寬度（640）」，與外觀分離度關卡的干擾物條件在評估鏡頭上的尺度一致。不足時以隨機身分補滿，MUST NOT 退回選擇不符合條件的身分（幀範圍不重疊的身分可能是同一個道路使用者的碎片）。

最佳 epoch SHALL 以縮小版 val 的 mAP 選擇：val 裁切以固定種子縮小到與解析度增強相同的最短邊範圍後評估；原始 val 的 mAP 仍記錄在訓練紀錄中。

#### Scenario: 被排除的裁切不會被訓練
- **WHEN** 某張裁切在裁切目錄中存在，但不在所用資料集版本的 manifest 中
- **THEN** 訓練與 val 評估都不會讀取這張裁切

#### Scenario: 訓練紀錄可追溯
- **WHEN** 訓練完成
- **THEN** 權重目錄中的訓練紀錄列出資料集版本名稱、骨幹名稱、輸入尺寸、亂數種子、解析度增強與干擾物取樣的設定，以及每個 epoch 的原始 val 與縮小版 val 的 mAP

#### Scenario: 解析度增強產生小物體的外觀
- **WHEN** 解析度增強啟用、最短邊範圍 8～40px，一張最短邊 60px 的裁切被選中增強
- **THEN** 這張裁切先縮小到最短邊落在 8～40px 之間，再 letterbox 到輸入尺寸

#### Scenario: 同時出現的干擾物進入同一批
- **WHEN** 干擾物取樣啟用、比例 0.5，種子身分有 10 個以上符合干擾物條件的身分
- **THEN** 以它為種子的一批中，至少一半的身分是它的干擾物

#### Scenario: 同時出現但距離太遠的身分不算干擾物
- **WHEN** C000054（寬 1280）的 car 的 `norm_factor` 為 80，身分 B 與種子同時出現，但同時段的框中心距離始終大於 320px
- **THEN** B 不會以干擾物的身分被選入同一批

#### Scenario: 碎片不會被選為干擾物
- **WHEN** 同一序列、同一類別的身分 A 的裁切在第 100～130 幀、身分 B 在第 140～170 幀
- **THEN** 以 A 為種子時，B 不會以干擾物的身分被選入同一批

#### Scenario: 以縮小版 val 選 epoch
- **WHEN** 第 20 個 epoch 的原始 val mAP 最高，但第 25 個 epoch 的縮小版 val mAP 最高
- **THEN** 權重採用第 25 個 epoch

### Requirement: 推論端對每個偵測框輸出 L2 正規化特徵

`Custom_ReID` SHALL 同時支援新格式權重與既有的完整模型 pickle。使用新格式時，每個偵測框 SHALL 以 `xywh2xyxy` 轉換後的框直接裁切（不放大、不加邊，與訓練裁切相同的緊框），letterbox 到權重設定的輸入尺寸並正規化，輸出一個 L2 範數為 1 的特徵；輸入 N 個框時輸出 N 個特徵。舊格式的完整模型 pickle 與 YOLO 的 `ReID` 的裁切方式 MUST NOT 改變。追蹤器的 override 方法 MUST NOT 修改。

#### Scenario: 每個框一個特徵
- **WHEN** 以新格式權重對 3 個偵測框抽特徵
- **THEN** 回傳 3 個長度等於特徵維度、L2 範數為 1 的特徵

#### Scenario: 緊框裁切
- **WHEN** 以新格式權重對一個中心 (100, 80)、寬 40、高 60 的框抽特徵
- **THEN** 裁切的範圍是 x 80～120、y 50～110，不含框外的像素

#### Scenario: 舊格式仍可載入
- **WHEN** 以既有的完整模型 pickle 建立 `Custom_ReID`
- **THEN** 載入與抽特徵的行為與修改前相同

## ADDED Requirements

### Requirement: 候選模型先通過外觀分離度關卡

新的 Re-ID 模型（或新的推論裁切方式）進入追蹤驗收前，SHALL 在調參序列上量測外觀分離度：正樣本為同一條 GT 軌跡、間隔 ≥ 1 秒的兩個框，負樣本為它的干擾物（同幀、同類別、中心距離在兩倍 `norm_factor` 內的不同 GT 軌跡）；每個類別計算 AUC（正樣本距離小於干擾物距離的機率）與尾端比例 TPR@FPR5%（正樣本距離小於干擾物距離 P5 的比例）。

正樣本的間隔 SHALL 為最小間隔幀數到該類別保留期限幀數之間（保留期限的取法見 `reid-fusion` 的校準條文）。

參考為 `dinov2s_v3` 以修改前的推論裁切（框放大 1.02 倍、加 10px 邊）量測的結果。候選 SHALL 以修改後的推論裁切（緊框）量測；`dinov2s_v3` 以緊框量測本身也是一個候選。

雜訊底線：以與 `dinov2s_v3` 完全相同的訓練設定、只換亂數種子再訓練一個模型，兩者以緊框量測的差距（各類別 AUC 與 TPR@FPR5% 的絕對差）即為雜訊底線。car 與 scooter 的最小改善量 SHALL 取「AUC 0.01、TPR@FPR5% 0.02」與「該類別 2 倍雜訊底線」中較大者。

候選通過關卡的條件：car 與 scooter 的 AUC 與 TPR@FPR5% 都比參考高出至少最小改善量，而且其他有 30 對以上干擾物配對的類別 AUC 下降不超過 0.02。多個候選通過時，只有 car 與 scooter 平均 AUC 最高者進入追蹤驗收。沒有候選通過時 MUST NOT 進行追蹤驗收，`config/botsort_reid_timm.yaml` 維持不變。保留序列 MUST NOT 參與關卡；val 的 mAP MUST NOT 作為關卡依據。

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

#### Scenario: 只靠緊框就通過
- **WHEN** `dinov2s_v3` 以緊框量測時通過關卡，而且在通過者中 car 與 scooter 的平均 AUC 最高
- **THEN** 以 `dinov2s_v3`（緊框）進入追蹤驗收

#### Scenario: 多個候選通過時只驗收最好的
- **WHEN** 兩個候選模型都通過關卡，car 與 scooter 的平均 AUC 分別是 0.80 與 0.83
- **THEN** 只有平均 AUC 0.83 的模型進入追蹤驗收

#### Scenario: 沒有候選通過
- **WHEN** 所有候選都沒有通過關卡
- **THEN** 不進行追蹤驗收，`config/botsort_reid_timm.yaml` 維持不變
