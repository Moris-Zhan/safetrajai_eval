# Spec Delta

## MODIFIED Requirements

### Requirement: 遮擋碎片合併只產生合併紀錄

系統 SHALL 在同一序列、同一類別內找出遮擋造成的碎片並寫成合併清單（欄位至少包含 `class, identity, merged_identity`），MUST NOT 搬移、改名或刪除任何裁切，也 MUST NOT 修改索引與排除清單。兩段軌跡 A、B 只有在同時符合下列條件時才可合併：
- B 的第一張與 A 的最後一張的幀差大於 0，且不超過該類別的保留期限幀數加上餘裕幀數（`margin_frames`）。保留期限幀數 SHALL 依合併設定所引用的追蹤設定檔的追蹤器與該序列的幀率換算：BoTSORT 為 `int(track_buffer × 幀率)`，ByteTrack 為 `int(幀率 / 30 × track_buffer)`；序列幀率由裁切索引中相鄰裁切的幀差與時間差推得（取中位數後四捨五入為整數）。
- B 第一張的框中心點，距離 A 以最後兩張推算的每幀速度外插該幀差後的位置，不超過兩者框尺寸較大者的 2 倍。
- A 最後幾張與 B 最前幾張的外觀特徵平均的 cosine 相似度不低於合併門檻。

配對 SHALL 為一對一：每段軌跡最多接一個前段與一個後段，衝突時保留相似度最高的配對；串起來的軌跡屬於同一個合併後的身分，合併後的身分沿用時間最早的那段軌跡。被排除清單整段排除的身分 MUST NOT 參與配對。合併門檻 SHALL 以人工確認的候選配對校準：以固定種子抽出的 80 組候選中，門檻以上的配對準確率必須達到 95%，否則 MUST NOT 以合併清單產生資料集版本。相同的索引、特徵與門檻 SHALL 產生逐位元組相同的合併清單。

#### Scenario: 遮擋後的碎片被合併
- **WHEN** 10fps 序列、BoTSORT 設定（car 的 `track_buffer` 為 3）、`margin_frames` 為 10，car 軌跡 A 在第 100 幀結束，car 軌跡 B 在第 110 幀出現在 A 外插位置 1 倍框尺寸內，兩者外觀相似度高於門檻
- **THEN** 候選窗為 40 幀，合併清單把 B 對應到 A 的身分

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

#### Scenario: 候選窗依追蹤器換算
- **WHEN** 合併設定引用 ByteTrack 的追蹤設定檔，car 的 `track_buffer` 為 90，序列幀率 10，`margin_frames` 為 10
- **THEN** car 的候選窗為 40 幀（保留期限 30 幀 ＋ 餘裕 10 幀）
