# reid-model Specification

## Purpose

定義 Re-ID 模型如何由資料集版本訓練、權重以什麼格式保存、追蹤器如何載入並抽取特徵，以及哪些類別可以在追蹤時啟用 Re-ID。

## Requirements

### Requirement: 訓練只讀取資料集版本

Re-ID 模型的訓練與 val 評估 SHALL 只讀取一個資料集版本的 `manifest.csv`（以 `crop_path` 相對於裁切根目錄讀取影像，依 `split` 區分 train 與 val），MUST NOT 直接掃描裁切目錄。訓練紀錄 SHALL 寫明使用的資料集版本、骨幹、輸入尺寸、超參數與亂數種子。

#### Scenario: 被排除的裁切不會被訓練
- **WHEN** 某張裁切在裁切目錄中存在，但不在所用資料集版本的 manifest 中
- **THEN** 訓練與 val 評估都不會讀取這張裁切

#### Scenario: 訓練紀錄可追溯
- **WHEN** 訓練完成
- **THEN** 權重目錄中的訓練紀錄列出資料集版本名稱、骨幹名稱、輸入尺寸與亂數種子

### Requirement: 權重以 state_dict 與設定保存

新訓練的權重 SHALL 存成包含 `state_dict` 與設定（骨幹名稱、特徵維度、輸入尺寸、資料集版本）的檔案，MUST NOT 依賴完整模型 pickle 才能載入；權重目錄的路徑 SHALL 包含 `class_reid_weights`，讓追蹤器選用 `Custom_ReID`。

#### Scenario: 不依賴 pickle 類別路徑
- **WHEN** 以新格式的權重建立 `Custom_ReID`
- **THEN** 載入成功，且不需要 `utils.model` 中的任何類別

### Requirement: 推論端對每個偵測框輸出 L2 正規化特徵

`Custom_ReID` SHALL 同時支援新格式權重與既有的完整模型 pickle。使用新格式時，每個偵測框 SHALL 以與既有 `ReID` 相同的框座標轉換裁切，letterbox 到權重設定的輸入尺寸並正規化，輸出一個 L2 範數為 1 的特徵；輸入 N 個框時輸出 N 個特徵。追蹤器的 override 方法 MUST NOT 修改。

#### Scenario: 每個框一個特徵
- **WHEN** 以新格式權重對 3 個偵測框抽特徵
- **THEN** 回傳 3 個長度等於特徵維度、L2 範數為 1 的特徵

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
