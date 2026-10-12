# reid-model Specification

## Purpose

定義 Re-ID 模型如何由資料集版本訓練、權重以什麼格式保存、追蹤器如何載入並抽取特徵，以及哪些類別可以在追蹤時啟用 Re-ID。

## Requirements

### Requirement: 訓練只讀取資料集版本

Re-ID 模型的訓練與 val 評估 SHALL 只讀取一個資料集版本的 `manifest.csv`（以 `crop_path` 相對於裁切根目錄讀取影像，依 `split` 區分 train 與 val），MUST NOT 直接掃描裁切目錄。訓練紀錄 SHALL 寫明使用的資料集版本、骨幹、輸入尺寸、超參數與亂數種子，以及解析度增強（是否啟用、機率、最短邊範圍）、干擾物取樣（是否啟用、比例、最少共同幀數、配對幀差、半徑）與選擇最佳 epoch 的方式。要使用哪些裁切只依 manifest 決定；干擾物取樣所需的序列、幀與框座標 SHALL 由裁切索引中與 manifest `crop_path` 對應的列取得（只讀 metadata，不讀影像），鏡頭的畫面寬度由該鏡頭的 ROI 檔取得。解析度增強的範圍 MUST NOT 依評估資料集的統計逐類別設定。

干擾物取樣只能選種子身分的**干擾物**：同一序列、同一類別，兩者的裁切幀範圍（不擴張）至少重疊設定的共同幀數，而且至少有一對幀差不超過設定幀數的裁切，其框中心距離在半徑以內。半徑 SHALL 為該類別兩倍 `norm_factor` 乘以「來源鏡頭畫面寬度 / 評估鏡頭畫面寬度（640）」，與外觀分離度關卡的干擾物條件在評估鏡頭上的尺度一致。干擾物 SHALL 先從本 epoch 尚未用過的身分中選，不足時可重複選入已用過的干擾物（同一批內身分不重複）；種子與補位身分在同一 epoch 不重複，每個 epoch 的批數不變。仍不足時以隨機身分補滿，MUST NOT 退回選擇不符合條件的身分（幀範圍不重疊的身分可能是同一個道路使用者的碎片）。

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

#### Scenario: 干擾物用完後仍可重複選入
- **WHEN** 干擾物取樣啟用、比例 0.5，種子身分的干擾物在本 epoch 都已被其他批選入過
- **THEN** 這些干擾物仍可再次被選入以它為種子的批，而這一批內每個身分只出現一次

#### Scenario: 同時出現但距離太遠的身分不算干擾物
- **WHEN** C000054（寬 1280）的 car 的 `norm_factor` 為 80，身分 B 與種子同時出現，但同時段的框中心距離始終大於 320px
- **THEN** B 不會以干擾物的身分被選入同一批

#### Scenario: 碎片不會被選為干擾物
- **WHEN** 同一序列、同一類別的身分 A 的裁切在第 100～130 幀、身分 B 在第 140～170 幀
- **THEN** 以 A 為種子時，B 不會以干擾物的身分被選入同一批

#### Scenario: 以縮小版 val 選 epoch
- **WHEN** 第 20 個 epoch 的原始 val mAP 最高，但第 25 個 epoch 的縮小版 val mAP 最高
- **THEN** 權重採用第 25 個 epoch

### Requirement: 權重以 state_dict 與設定保存

新訓練的權重 SHALL 存成包含 `state_dict` 與設定（骨幹名稱、特徵維度、輸入尺寸、資料集版本）的檔案，MUST NOT 依賴完整模型 pickle 才能載入；權重目錄的路徑 SHALL 包含 `class_reid_weights`，讓追蹤器選用 `Custom_ReID`。

#### Scenario: 不依賴 pickle 類別路徑
- **WHEN** 以新格式的權重建立 `Custom_ReID`
- **THEN** 載入成功，且不需要 `utils.model` 中的任何類別

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

### Requirement: 只在追蹤不變差的類別啟用 Re-ID

某個類別在情境設定中啟用 `use_reid` 前，SHALL 在評估資料集的調參序列（C000025 的 145630、150634 兩個時段）上，與關閉 Re-ID 的驗收基準（R1 追蹤參數、`gmc_method: sparseOptFlow`、`frame_rate: auto`，其他參數完全相同）比較。該類別的 IDSW 不增加，且 IDP、AssA 與 CLEAR TP 都不下降，是啟用的必要條件；另外 SHALL 有身分層級的改善（IDSW 減少，或 IDP 上升），才可在 `config/botsort_reid_timm.yaml` 中啟用。四項指標都與基準相同、或只有 AssA／CLEAR TP 上升的類別，以及調參序列上沒有該類別 GT、無法判定的類別，SHALL 維持 `use_reid: false`。沒有任何類別符合時，`config/botsort_reid_timm.yaml` 的全部類別 SHALL 為 `use_reid: false`。val 的 mAP 與 Rank-1 MUST NOT 作為啟用依據。

保留序列（C000025 的 152639 時段）MUST NOT 參與逐類別的決定，只用來回報成績；但所有啟用的類別一起開啟後，若保留序列的整體 IDSW 增加或 IDP 下降，SHALL 逐一關閉對整體貢獻最小的類別，直到保留序列的整體 IDSW 不增加且 IDP 不下降。

#### Scenario: 追蹤變差的類別不啟用
- **WHEN** 在調參序列上開啟 scooter 的 Re-ID 後，scooter 的 IDSW 比基準多
- **THEN** `config/botsort_reid_timm.yaml` 中 scooter 的 `use_reid` 為 false

#### Scenario: 關聯品質下降的類別不啟用
- **WHEN** 在調參序列上開啟 car 的 Re-ID 後，car 的 IDSW 與 IDP 不變，但 AssA 下降
- **THEN** car 的 `use_reid` 為 false

#### Scenario: 追蹤不變差的類別啟用
- **WHEN** 在調參序列上開啟 person 的 Re-ID 後，person 的 IDSW 減少，IDP、AssA、TP 都不下降
- **THEN** `config/botsort_reid_timm.yaml` 中 person 的 `use_reid` 為 true，且 `model` 指向新權重

#### Scenario: 沒有改善的類別不啟用
- **WHEN** 在調參序列上開啟 truck 的 Re-ID 後，truck 的 IDSW、IDP、AssA、TP 都與基準相同
- **THEN** truck 的 `use_reid` 為 false

#### Scenario: 只有 AssA 上升的類別不啟用
- **WHEN** 在調參序列上開啟 car 的 Re-ID 後，car 的 IDSW 與 IDP 不變，AssA 上升 0.006
- **THEN** car 的 `use_reid` 為 false

#### Scenario: 無 GT 的類別不啟用
- **WHEN** 調參序列上沒有 bus 的 GT
- **THEN** bus 的 `use_reid` 為 false

#### Scenario: 沒有類別符合時全部關閉
- **WHEN** 所有類別在調參序列上都沒有 IDSW 減少或 IDP 上升
- **THEN** `config/botsort_reid_timm.yaml` 的 6 個類別都是 `use_reid: false`

#### Scenario: 保留序列整體變差時逐一關閉
- **WHEN** person 與 truck 都在調參序列上啟用，但兩者一起開啟後保留序列的整體 IDSW 比基準多
- **THEN** 關閉貢獻較小的類別，直到保留序列的整體 IDSW 不增加且 IDP 不下降

#### Scenario: 保留序列不參與逐類別決定
- **WHEN** 某類別在調參序列上啟用，在保留序列上該類別單獨的 IDSW 較多，但整體 IDSW 不增加、IDP 不下降
- **THEN** 該類別仍然啟用

### Requirement: 候選模型量測外觀分離度

新的 Re-ID 模型（或新的推論裁切方式）進入追蹤驗收前，SHALL 在調參序列上量測外觀分離度：正樣本為同一條 GT 軌跡、間隔在最小間隔幀數到該類別保留期限幀數之間的兩個框（保留期限的取法見 `reid-fusion` 的校準條文），負樣本為它的干擾物（同幀、同類別、中心距離在兩倍 `norm_factor` 內的不同 GT 軌跡）；每個類別計算 AUC（正樣本距離小於干擾物距離的機率）與尾端比例 TPR@FPR5%（正樣本距離小於干擾物距離 P5 的比例）。

參考 SHALL 為 `dinov2s_v3` 以修改前的推論裁切（框放大 1.02 倍、加 10px 邊）量測的結果；候選 SHALL 以修改後的推論裁切（緊框）量測，`dinov2s_v3` 以緊框量測本身也是一個候選。保留序列 MUST NOT 參與量測。

#### Scenario: 參考以舊裁切量測
- **WHEN** 量測參考 `dinov2s_v3`
- **THEN** 裁切為框放大 1.02 倍、加 10px 邊，結果標示為舊裁切

#### Scenario: 候選以緊框量測
- **WHEN** 量測任何候選模型，包括 `dinov2s_v3` 本身
- **THEN** 裁切為不放大、不加邊的緊框

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
