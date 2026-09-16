## Purpose

產生涵蓋 `crop_dataset/` 全部身分（而非隨機抽樣）的縮圖對照報告，讓人工能有系統地逐一審視每個身分是否歸類正確、是否有局部污染，並將 DSEC 來源身分獨立呈現以區隔「有自動分數」與「純人工審查」兩種情境。

## ADDED Requirements

### Requirement: 全量身分縮圖
系統 SHALL 為每個類別資料夾底下的每一個身分（而非抽樣子集）產生至少一張代表縮圖，並可依身分名稱追溯回原始資料夾路徑。

#### Scenario: 產生全量報告
- **WHEN** 對一個有 531 個身分的類別產生報告
- **THEN** 報告涵蓋全部 531 個身分的縮圖，沒有任何身分被跳過

### Requirement: 依可疑分數排序與分頁
系統 SHALL 在提供一致性檢查結果時，依身分的 agreement_rate 由低到高排序顯示；當單一類別的身分數超過設定的每頁上限時，SHALL 將報告分頁輸出。

#### Scenario: 高身分數類別分頁
- **WHEN** 某類別有 336 個身分且每頁上限設為 100
- **THEN** 報告產生至少 4 頁，每頁最多 100 個身分，且 agreement_rate 最低（最需複核）的身分出現在第一頁

### Requirement: DSEC 來源身分獨立區塊
系統 SHALL 將身分資料夾名稱前綴為 `dsecdet_` 的身分獨立呈現在各自的區塊或頁面，不與有一致性分數的身分混合排序，且不顯示不適用的分數欄位。

#### Scenario: bus 類別含 DSEC 身分
- **WHEN** 產生 bus 類別報告，其中同時含 CCTV 來源與 DSEC 來源身分
- **THEN** DSEC 來源身分出現在獨立區塊，該區塊內的身分不顯示 agreement_rate 排序欄位

### Requirement: 唯讀，不影響原始資料
系統 SHALL 只產生報告文件與縮圖檔案，不搬移或刪除 `crop_dataset/` 原始影像。

#### Scenario: 報告產生後資料不變
- **WHEN** 執行報告產生
- **THEN** `crop_dataset/` 底下原始檔案數量與內容不受影響
