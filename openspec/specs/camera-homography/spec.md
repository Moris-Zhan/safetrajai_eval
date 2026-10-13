# camera-homography Specification

## Purpose

定義以鏡頭為鍵的 Homography：如何找到鏡頭的轉換矩陣、如何取得鏡頭經緯度、如何以獨立驗證點與比例尺檢查量測誤差、如何換算公尺，以及如何取得衛星底圖，讓衝突分析能在有地理對位的鏡頭上使用公尺單位。

## Requirements

### Requirement: Homography 設定以鏡頭為鍵

每台鏡頭的 Homography 設定 SHALL 放在 `config/homography/<camera>.yaml`，`<camera>` 是序列名稱第一個空白前的鏡頭編號（與 ROI 相同）。設定 SHALL 能記錄：

- 鏡頭完整名稱、既有矩陣的 tag；
- 衛星圖來源參數（provider、zoom、scale、輸出尺寸）；
- 經緯度與其來源（`latlon_source`、`latlon_date`）、選用的查詢路名（`osm_roads`）；
- 比例尺檢查狀態（`scale_check`）；
- 選用的人工對應點（`cctv_points`／`map_points`，至少 4 組且數量相同）。

欄位缺漏或數量不符時，載入 MUST 以錯誤停止並指出檔案與欄位。

#### Scenario: 依序列取得鏡頭設定
- **WHEN** 處理序列 `C000025 中華三路 民生二路(東北角)_20251209145630_15s_1`
- **THEN** 讀取 `config/homography/C000025.yaml`

#### Scenario: 人工對應點數量不符
- **WHEN** 鏡頭設定的 `cctv_points` 有 4 組，`map_points` 有 3 組
- **THEN** 載入失敗，錯誤訊息包含檔案路徑與「數量不符」

### Requirement: 依鏡頭解析 Homography

流程 SHALL 依序決定鏡頭的 Homography：
1. 鏡頭設定指定的既有矩陣；
2. 鏡頭設定內的人工對應點所算出的矩陣；
3. 都沒有時不使用 Homography，下游改用 CCTV 原始像素座標，並在 log 中提示該鏡頭沒有 Homography。

指定的矩陣檔不存在時 MUST 以錯誤停止，不得靜默退回第 3 步。config 不得引用 `temp/` 之下的矩陣。

#### Scenario: 既有矩陣優先
- **WHEN** 鏡頭設定同時有矩陣 tag 與人工對應點
- **THEN** 使用矩陣 tag 指向的既有矩陣

#### Scenario: 指定的矩陣不存在
- **WHEN** 鏡頭設定的矩陣 tag 在矩陣目錄中找不到對應檔案
- **THEN** 流程以錯誤停止，錯誤訊息包含預期的檔案路徑

#### Scenario: 沒有 Homography
- **WHEN** 鏡頭沒有設定檔
- **THEN** 該鏡頭的序列使用原始像素座標，log 輸出一行提示

### Requirement: 既有 stage3 輸出不受鍵值改變影響

改為以鏡頭為鍵後，`collision_detect` 情境的 stage3 輸出（`ttc_events`、`pet_events`、log CSV、熱區圖）SHALL 與修改前的 golden 基準差異為 0。

#### Scenario: golden 比對
- **WHEN** 以 `collision_detect` 情境跑 stage1＋stage3，並與 golden 基準比對
- **THEN** `collision_output` 類別的差異為 0

### Requirement: 舊的序列鍵對應點檔不再使用

`config/homography_points/` 之下若存在以序列命名的對應點檔（README 以外的 `.yaml`），載入 MUST 以錯誤停止，並指出應搬到 `config/homography/<camera>.yaml` 的 `cctv_points`／`map_points`。

#### Scenario: 發現舊格式檔案
- **WHEN** `config/homography_points/` 有一個 `<seq>.yaml`
- **THEN** 載入失敗，錯誤訊息包含新的設定路徑

### Requirement: 經緯度自動取得

經緯度取得工具 SHALL 分兩步：
1. 查詢並寫出候選 CSV，欄位為鏡頭、查詢路名、來源、節點數、中心經緯度、分散距離、狀態；
2. 只有在明確下達套用指令時，才把狀態為 `ok` 的候選寫入鏡頭設定，並記錄 `latlon_source` 與查詢日期。

查詢方式與狀態規則：
- 先以 OpenStreetMap 查兩條路的共用節點，取平均作為路口中心；查不到時，再以 Google Geocoding 查「路名＋高雄市」作備援。
- 分散距離大於 50 m、查不到、名稱無法拆成兩條路時，狀態 MUST 為 `needs_review`，不得自動寫入。
- 鏡頭設定有 `osm_roads` 時，以它取代從名稱拆出的路名。
- 手動填寫的經緯度記為 `latlon_source: manual`。

#### Scenario: 自動取得成功
- **WHEN** 查詢 C000025（中華三路、民生二路），共用節點的分散距離不超過 50 m
- **THEN** 候選 CSV 的狀態為 `ok`；套用後鏡頭設定有 `latlon` 與 `latlon_source: osm_overpass`

#### Scenario: 名稱無法拆解
- **WHEN** 查詢名稱為「中正路交流道南下入口」的鏡頭
- **THEN** 候選狀態為 `needs_review`，套用時不寫入該鏡頭

#### Scenario: 未下達套用
- **WHEN** 只執行查詢、沒有下達套用指令
- **THEN** 任何鏡頭設定都不被修改

### Requirement: 外部服務金鑰

需要金鑰的工具 SHALL 從環境變數讀取金鑰；環境變數未設定時，從 repo 根目錄的 `.env` 讀取。`.env` MUST 被 git 忽略。金鑰值不得出現在任何輸出、log、metadata 或錯誤訊息中。工具 SHALL 依讀取到的變數判定金鑰種類：`GOOGLE_MAPS_DEMO_API_KEY` 為 `demo`，`GOOGLE_MAPS_API_KEY` 為 `billing`。

#### Scenario: 金鑰不外洩
- **WHEN** 工具以 `.env` 的金鑰執行，且請求失敗
- **THEN** 錯誤訊息與 log 只提到變數名稱與金鑰種類，不含金鑰值

#### Scenario: 沒有金鑰
- **WHEN** 環境變數與 `.env` 都沒有所需的金鑰
- **THEN** 工具以錯誤結束，不發出任何網路請求

### Requirement: 公尺換算

有 Homography 的鏡頭，其俯視平面每像素公尺數 SHALL 以 Web Mercator 地面解析度計算：`m_per_pixel = 156543.03392 × cos(緯度) / 2^zoom / scale`，zoom 與 scale 取自鏡頭設定的衛星圖參數。

以下任一情況，換算結果 MUST 為「不可用」，不得退回赤道值或任何預設值：
- 鏡頭沒有緯度；
- 沒有衛星圖參數；
- 俯視平面是人工自訂矩形；
- `scale_check` 不是 `passed`。

#### Scenario: 高雄鏡頭
- **WHEN** 鏡頭的 zoom 為 20、scale 為 2、緯度為 22.6，`scale_check` 為 `passed`
- **THEN** `m_per_pixel` 約為 0.0689（與 cos(22.6°) × 0.0746 的誤差小於 1e-4）

#### Scenario: 缺少緯度
- **WHEN** 鏡頭設定沒有緯度
- **THEN** `m_per_pixel` 為不可用，依賴公尺的下游指標不計算

#### Scenario: 比例尺未通過
- **WHEN** 鏡頭的 `scale_check` 為 `failed` 或 `unchecked`
- **THEN** `m_per_pixel` 為不可用

### Requirement: 比例尺檢查

鏡頭的 `scale_check` SHALL 只在以下兩項檢查都通過時才設為 `passed`；任一項不通過即設為 `failed`，該鏡頭不換算公尺：

1. 已知長度檢查：在衛星底圖上量測已知長度的地物（車道寬、停止線、斑馬線等），換算後的長度落在實際值 ±15% 以內。
2. 影像比對（有驗證用金鑰時才做）：以鏡頭經緯度與相同衛星圖參數下載一張驗證用底圖，與既有底圖做特徵匹配，估得的縮放比例落在 1±2% 以內。

沒有驗證用金鑰時，第 2 項記為略過，只依第 1 項判定。檢查結果與量測方式 SHALL 記錄在驗證紀錄中。

#### Scenario: 影像比對縮放超出範圍
- **WHEN** 已知長度檢查通過，但影像比對估得的縮放比例為 1.05
- **THEN** `scale_check` 為 `failed`

#### Scenario: 沒有驗證用金鑰
- **WHEN** 已知長度檢查通過，且沒有任何 Google Maps 金鑰
- **THEN** `scale_check` 為 `passed`，驗證紀錄註明影像比對略過

### Requirement: check points 以鏡頭為鍵

驗證用的 check points SHALL 放在 `config/homography_checkpoints/<camera>.yaml`（`cctv_points`／`sat_points` 一一對應）。check points 與算矩陣用的訓練點座標完全相同的點 MUST 被排除在驗證統計之外，並在報表中列出被排除的點數。

#### Scenario: 混入訓練點
- **WHEN** check points 中有 1 組與訓練點座標相同
- **THEN** 驗證統計只使用其餘的點，報表的排除點數為 1

### Requirement: 遷移既有 check points

遷移工具 SHALL 把既有的 tag 鍵 check points YAML，以及 `models/homography/checkpoints/` 之下的 `.npy` check points，轉成鏡頭鍵的 YAML，數值逐位元保留。原檔 MUST NOT 被修改或刪除。目標檔已存在且內容不同時，遷移 MUST 停止並列出衝突，不得覆寫。

#### Scenario: 轉入 C000297 的 .npy
- **WHEN** 對 `check_pts_cctv_C000297 …npy` 與 `check_pts_sat_C000297 …npy` 執行遷移
- **THEN** 產生 `config/homography_checkpoints/C000297.yaml`，點座標與 `.npy` 相同，原 `.npy` 仍在

#### Scenario: 目標已存在
- **WHEN** 目標 YAML 已存在且點座標不同
- **THEN** 遷移失敗，不修改任何檔案

### Requirement: 驗證報表內容

批次驗證 SHALL 對每台有矩陣的鏡頭輸出一列彙總，包含：
- check points 的重投影誤差統計（N、mean、std、median、RMSE、p95、max，單位像素）；
- 訓練點的重投影誤差統計，欄位名稱明確標示為擬合殘差；
- `m_per_pixel` 可用時，對應的公尺欄位；
- `scale_check` 狀態。

沒有 check points 的鏡頭，`validated` MUST 為 False 並註明原因。check points 少於 4 點時，`validated` MUST 為 False，原因為點數不足，但仍輸出統計。

#### Scenario: 有公尺換算的鏡頭
- **WHEN** 鏡頭有 8 個 check points 且 `m_per_pixel` 可用
- **THEN** 彙總列同時有 `check_dist_rmse_px` 與 `check_dist_rmse_m`，以及 `train_residual_rmse_px`

#### Scenario: 點數不足
- **WHEN** 鏡頭只有 2 個 check points
- **THEN** `validated` 為 False，原因為點數不足，統計欄位仍有數值

### Requirement: 衛星底圖下載

衛星底圖下載 SHALL 是獨立的手動指令，不得在 pipeline 的任何 stage 中執行，並 SHALL 以用途參數區分兩種下載：
- `verify`：驗證用底圖，存到驗證目錄，不取代既有底圖，也不供任何分析使用；接受 `demo` 與 `billing` 兩種金鑰。
- `basemap`：正式底圖；MUST 使用 `billing` 金鑰，使用 `demo` 金鑰時以錯誤結束。

兩種用途都輸出 PNG 與 metadata（provider、參數、中心經緯度、下載日期、用途、金鑰種類，不含金鑰值）。正式底圖已存在時 MUST 拒絕覆寫，除非明確要求替換；替換時舊檔依隔離慣例移到 `temp/` 並記錄。

#### Scenario: Demo 金鑰下載正式底圖
- **WHEN** 只設定 `GOOGLE_MAPS_DEMO_API_KEY`，以 `basemap` 用途下載
- **THEN** 指令以錯誤結束，不發出網路請求

#### Scenario: Demo 金鑰下載驗證用底圖
- **WHEN** 只設定 `GOOGLE_MAPS_DEMO_API_KEY`，以 `verify` 用途下載
- **THEN** 驗證目錄產生 PNG 與 metadata，metadata 的金鑰種類為 `demo`，既有底圖不變

#### Scenario: 正式底圖已存在
- **WHEN** 目標鏡頭的正式底圖已存在，且未要求替換
- **THEN** 指令以錯誤結束，既有檔案不變
