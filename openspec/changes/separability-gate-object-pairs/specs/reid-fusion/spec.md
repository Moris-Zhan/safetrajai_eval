# Spec Delta

## ADDED Requirements

### Requirement: 量測排除沒有身分的 GT 框並回報物體配對數

量測外觀距離分布與外觀分離度時，GT id 小於 0 的框（沒有身分的標註）MUST NOT 作為正樣本、干擾物或退回配對的任何一方；輸出 SHALL 記錄每個類別跳過的框數。GT 檔案 MUST NOT 被修改。

每個類別 SHALL 回報干擾物配對來自多少組不同的 GT 物體配對（同一序列中兩條不同 GT 軌跡的無序組合；同一組物體在多個幀的配對只算一組）。

#### Scenario: 沒有身分的框不參與量測
- **WHEN** 調參序列中有 2 個 small_truck 框的 GT id 為 -1，其中 1 個在同一幀、半徑內與另一個 small_truck 框相鄰
- **THEN** 這 2 個框不出現在任何正樣本、干擾物或退回配對中，輸出記錄 small_truck 跳過 2 個框

#### Scenario: 同一組物體的多幀配對只算一組
- **WHEN** small_truck 的干擾物共 41 對，來自軌跡 #24 與 #27（17 幀）以及 #5 與 #27（24 幀）
- **THEN** small_truck 的干擾物物體配對數為 2
