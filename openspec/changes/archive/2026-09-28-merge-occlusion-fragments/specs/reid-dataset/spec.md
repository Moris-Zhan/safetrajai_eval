# Spec Delta

## ADDED Requirements

### Requirement: 遮擋碎片合併只產生合併紀錄

系統 SHALL 在同一序列、同一類別內找出遮擋造成的碎片並寫成合併清單（欄位至少包含 `class, identity, merged_identity`），MUST NOT 搬移、改名或刪除任何裁切，也 MUST NOT 修改索引與排除清單。兩段軌跡 A、B 只有在同時符合下列條件時才可合併：
- A 的最後一張與 B 的第一張的時間差大於 0，且不超過該類別 `track_buffer` 加 1 秒。
- B 第一張的框中心點，距離 A 以最後兩張推算的速度外插到 B 出現時刻的位置，不超過兩者框尺寸較大者的 2 倍。
- A 最後幾張與 B 最前幾張的外觀特徵平均的 cosine 相似度不低於合併門檻。

配對 SHALL 為一對一：每段軌跡最多接一個前段與一個後段，衝突時保留相似度最高的配對；串起來的軌跡屬於同一個合併後的身分，合併後的身分沿用時間最早的那段軌跡。被排除清單整段排除的身分 MUST NOT 參與配對。合併門檻 SHALL 以人工確認的候選配對校準：以固定種子抽出的 80 組候選中，門檻以上的配對準確率必須達到 95%，否則 MUST NOT 以合併清單產生資料集版本。相同的索引、特徵與門檻 SHALL 產生逐位元組相同的合併清單。

#### Scenario: 遮擋後的碎片被合併
- **WHEN** 同一序列的 car 軌跡 A 在 t=10.0 秒結束，car 軌跡 B 在 t=11.0 秒出現在 A 外插位置 1 倍框尺寸內，兩者外觀相似度高於門檻
- **THEN** 合併清單把 B 對應到 A 的身分

#### Scenario: 外觀不同的鄰車不合併
- **WHEN** 軌跡 B 的時間與位置符合條件，但與 A 的外觀相似度低於門檻
- **THEN** 合併清單不包含 A 與 B 的配對

#### Scenario: 不跨類別與序列合併
- **WHEN** 軌跡 A 是 car、B 是 small_truck，或 A、B 屬於不同序列
- **THEN** 兩者不會被合併

#### Scenario: 一對一配對
- **WHEN** 軌跡 B、C 都符合接在 A 之後的條件，B 與 A 的相似度較高
- **THEN** 合併清單只把 B 接在 A 之後，C 不接到 A

#### Scenario: 串接多段
- **WHEN** A 接 B、B 接 C 都符合條件
- **THEN** A、B、C 都對應到同一個合併後的身分（A 的身分）

#### Scenario: 合併不改動檔案與其他紀錄
- **WHEN** 產生合併清單
- **THEN** 裁切根目錄、索引、自動與人工排除清單都與執行前相同

## MODIFIED Requirements

### Requirement: 以版本化 manifest 定義訓練資料與切分

訓練資料 SHALL 由 `dataset/reid/splits/<version>/manifest.csv` 定義，欄位為 `crop_path, identity, class, camera, date, split`，`split` 為 `train` 或 `val`。manifest 產生時 SHALL 依序套用：
1. 移除自動排除清單與人工排除清單中的身分與影像（兩份清單都套用，任一份列出即移除）。
2. 有指定合併清單時，把合併清單中的原始身分改為其合併後的身分（同一序列、同一類別），一個合併後的身分包含所有對應原始身分剩下的裁切。
3. 捨棄剩餘少於 4 張的身分。
4. 每個身分最多保留 30 張，依 `time_s` 均勻抽樣。
5. 切分：鏡頭 C000054 日期為 20250623、20250625、20250627 者為 `train`，20250629 者為 `val`；其他來源鏡頭全部為 `train`。

同一個身分 MUST 只出現在一個 split。manifest 只引用既有裁切，MUST NOT 複製影像。相同的輸入（索引、兩份排除清單、合併清單、規則）SHALL 產生逐位元組相同的 manifest。

#### Scenario: 身分張數規則
- **WHEN** 某身分套用排除清單後剩 3 張，另一身分剩 80 張
- **THEN** 前者不出現在 manifest，後者在 manifest 中恰好有 30 列

#### Scenario: 兩份排除清單都套用
- **WHEN** 自動排除清單排除身分 A，人工排除清單排除身分 B 的一張裁切
- **THEN** manifest 中沒有身分 A 的任何列，也沒有身分 B 的那一張裁切

#### Scenario: 合併後的碎片共同計算張數
- **WHEN** 同一序列、同一類別的身分 A 有 3 張、身分 B 有 2 張，合併清單把 B 合併到 A
- **THEN** manifest 中身分 A 有 5 列，沒有身分 B

#### Scenario: 排除先於合併
- **WHEN** 身分 B 被自動排除清單整段排除，而合併清單把 B 合併到 A
- **THEN** manifest 中身分 A 只包含 A 自己的裁切

#### Scenario: 依日期切分
- **WHEN** 產生 manifest
- **THEN** 所有 `date=20250629` 的 C000054 身分都在 `val`，其他日期的 C000054 身分與其他來源鏡頭的身分都在 `train`

#### Scenario: 評估鏡頭永遠不出現
- **WHEN** 產生 manifest
- **THEN** manifest 中沒有任何 `camera` 屬於 `role: eval` 資料集的列

#### Scenario: 可重現
- **WHEN** 以相同輸入連續產生兩次 manifest
- **THEN** 兩份 manifest 內容完全相同

#### Scenario: 不複製影像
- **WHEN** 產生 manifest
- **THEN** 裁切根目錄與 splits 目錄內都沒有新增任何影像檔
