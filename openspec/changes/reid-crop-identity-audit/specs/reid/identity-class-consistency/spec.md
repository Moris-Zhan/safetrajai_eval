## Purpose

對 `dataset/reid/crop_dataset/` 裡來自本專案 CCTV 資料的身分，重新以既有偵測器逐張核對「偵測結果類別」與「資料夾指定類別」是否一致，找出可能整個歸錯類別的身分或身分內局部污染的單張影像，只標記供人工複核，不自動修改任何原始資料。

## ADDED Requirements

### Requirement: 逐身分逐影像一致性檢查
系統 SHALL 對指定 crop root 底下每個類別資料夾的每個身分資料夾，逐張對裁切影像重新推論，取得該影像信心最高的偵測類別，並與資料夾所屬類別比較，記錄一致或不一致。「一致」的判定採車輛子類別群組放寬：`car`／`small_truck`／`truck` 三者互判為一致，因為偵測器對這三者外觀邊界的鑑別力弱——經小規模驗證證實，嚴格比對會讓車輛類別身分幾乎全數退化成同一分數、失去鑑別力（見 design.md Open Questions）；其餘類別之間、以及完全沒有偵測到任何框的情況，仍要求嚴格相等才算一致。

#### Scenario: 執行一致性檢查
- **WHEN** 對 `crop_dataset/` 執行一致性檢查
- **THEN** 輸出每張被掃描影像的比對結果（所屬類別、偵測類別、信心值、是否一致），且 `crop_dataset/` 底下原始檔案不變

#### Scenario: 車輛子類別群組放寬
- **WHEN** 一張 `small_truck` 資料夾底下的裁切被偵測器判為 `car`
- **THEN** 該張影像記錄為一致（`agree=True`），因為 `car` 與 `small_truck` 同屬車輛子類別群組

### Requirement: 排除 DSEC 來源身分
系統 SHALL 依身分資料夾名稱前綴（`dsecdet_`）自動辨識 DSEC 來源身分，跳過其一致性分數計算，不將其計入任何 agreement 統計。

#### Scenario: 掃描含 DSEC 身分的類別
- **WHEN** `bus/` 底下同時存在 CCTV 來源身分與 `dsecdet_` 前綴身分，對其執行一致性檢查
- **THEN** 只有 CCTV 來源身分出現在一致性標記結果中並計入統計；DSEC 來源身分被記錄為略過，不產生 agreement 分數

### Requirement: 身分層級聚合分數
系統 SHALL 為每個被檢查（非略過）的身分計算 agreement_rate（一致張數 / 該身分被掃描張數），並可依 agreement_rate 由低到高排序輸出。

#### Scenario: 身分聚合
- **WHEN** 一個身分有 10 張裁切，其中 3 張偵測類別與資料夾所屬類別不同
- **THEN** 該身分的 agreement_rate 為 0.7，並出現在排序後的清單中

### Requirement: 影像層級標記
系統 SHALL 標記單張影像其偵測類別與資料夾所屬類別不一致的紀錄，即使該身分整體 agreement_rate 很高，使局部污染（如追蹤器 ID switch 造成的個別錯誤裁切）可與整身分歸錯區分開來。

#### Scenario: 高一致率身分中的個別離群影像
- **WHEN** 一個身分 agreement_rate 為 0.95（20 張中 19 張一致）
- **THEN** 那 1 張不一致的影像仍出現在影像層級標記清單中，供人工複核是否為局部污染

### Requirement: 只標記不刪除
系統 SHALL NOT 自動搬移、刪除、或覆寫任何 `crop_dataset/` 原始影像檔案。

#### Scenario: 執行後檔案不變
- **WHEN** 對整個 `crop_dataset/` 執行一致性檢查
- **THEN** 執行前後 `crop_dataset/` 底下的檔案數量與內容皆不變
