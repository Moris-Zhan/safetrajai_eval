# Proposal

## Why

接下來的 change `conflict-metrics` 要計算 CI、DRAC、PICUD，以及以公尺為門檻的 TTC／PET，這些都需要每台鏡頭的俯視平面能換算成公尺。目前做不到，原因有幾個：

- Homography 的資料散在三種鍵上：既有矩陣用完整鏡頭名稱加參數尾碼，人工對應點用序列名稱，check points 用 tag。這是 `CONTEXT.md` 已經記錄的落差。
- 驗證只算 check points 的像素誤差：既有的 `.npy` check points（C000297）從來沒有被讀取，訓練點誤差沒有被量測，也沒有公尺誤差。
- 沒有任何方式取得新鏡頭的衛星底圖。

參考實作 `Sub_Module/homongraphy_val` 涵蓋上述功能，但它依賴 GUI、寫死了 API key、公尺換算也漏乘 cos(緯度)，不能直接搬過來用。

## What Changes

- 新增以**鏡頭編號**為鍵的 Homography 設定 `config/homography/<camera>.yaml`，記錄矩陣 tag、衛星圖參數（provider、zoom、scale、size）、GPS 緯度與經度，以及選用的人工對應點。
- 解析 Homography 的入口改成依鏡頭查找：既有矩陣 → 鏡頭設定內的人工對應點 → 都沒有時回傳 None。對既有的 26 個路口，stage3 的輸出逐位元不變。
- **BREAKING**：`config/homography_points/<seq>.yaml` 不再讀取。目前這個目錄只有 README、沒有任何對應點檔，實際影響為零；載入時若發現這類檔案會報錯並提示搬遷位置。
- check points 改以鏡頭為鍵：`config/homography_checkpoints/<camera>.yaml`。既有的 C000025 YAML 與 C000297 的 `.npy` 以一次性遷移轉入，原檔保留不動。
- 新增鏡頭的公尺換算 `meters_per_pixel(camera)`：
  - 公式為 Web Mercator 地面解析度 × cos(緯度) ÷ scale。
  - 沒有緯度、或衛星圖參數不明時回傳 None，不退回赤道值。
- 批次驗證報表新增：
  - 訓練點（算 H 用的 4 點）重投影誤差，並明確標示為「擬合殘差，不是驗證」；
  - 有 `m_per_pixel` 時，誤差的公尺欄位。
- 新增經緯度自動取得 `scripts/geocode_cameras.py`：
  - 以 OpenStreetMap Overpass 查兩條路的共用節點，取平均作為路口中心；查不到時以 Google Geocoding 備援；
  - 先輸出候選 CSV，使用者確認後才以 `--apply` 寫入；
  - 實測 C000025 可以取得，26 台中有 9 台直接成功。
- 新增金鑰讀取：環境變數優先，其次是 repo 根目錄的 `.env`（已列入 `.gitignore`）。依變數名稱區分 `demo` 與 `billing` 金鑰，任何輸出都不含金鑰值。
- 新增比例尺檢查 `scripts/check_homography_scale.py`：已知長度的地物量測（±15%），加上以 Maps Demo Key 下載驗證用底圖、與既有底圖做影像比對（縮放比例 1±2%）。結果寫入鏡頭設定的 `scale_check`，沒有通過的鏡頭不換算公尺。
- 新增衛星底圖下載 CLI `scripts/download_satellite.py`：
  - 依鏡頭設定的經緯度呼叫 Google Static Maps；
  - 以 `--purpose verify|basemap` 區分用途，Demo Key 只能下載驗證用底圖；
  - 輸出 PNG 加 metadata sidecar，不覆寫既有底圖；
  - 不在 pipeline 中自動執行。
- 新增 ADR 0010：有地理對位的鏡頭可以使用公尺單位。同步更新 `CLAUDE.md`、`CONTEXT.md` 中「誤差單位是像素，不是公尺」的敘述。

## Capabilities

### New Capabilities
- `camera-homography`：以鏡頭為鍵的 Homography 設定與解析、check points 驗證報表、公尺換算、衛星底圖取得。

### Modified Capabilities
（無。目前 `openspec/specs/` 沒有 Homography 相關的 capability；`project-structure` 的 golden 等價要求不變，homography 驗證輸出的欄位變動依「刻意變更時重建基準」的既有規則處理，見 design。）

## Impact

- 程式：
  - `src/safetrajai/collision/homography/`：`points.py`、`validation.py`，新增 `camera_config.py`、`georef.py`、`satellite.py`；
  - `src/safetrajai/collision/__init__.py`：改呼叫新的解析入口；
  - `src/safetrajai/evaluation/config_loader.py`：新增 `homography_camera_dir`，移除 `homography_points_dir`；
  - `scripts/validate_homography.py`；
  - 新增 `src/safetrajai/secrets.py`；
  - 新增 `scripts/download_satellite.py`、`scripts/migrate_homography_config.py`、`scripts/geocode_cameras.py`、`scripts/check_homography_scale.py`。
- 外部服務（只在手動執行的工具中使用，不在 pipeline 中）：OpenStreetMap Overpass API（不需要金鑰）；Google Static Maps 與 Geocoding API（需要金鑰，讀取 `.env` 的 `GOOGLE_MAPS_API_KEY`／`GOOGLE_MAPS_DEMO_API_KEY`）。
- 設定：新增 `config/homography/`；改寫 `config/homography_checkpoints/`；`config/collision_detect.yaml` 的 `homography_points_dir` 改為 `homography_camera_dir`。
- 依賴：`requests` 已隨 ultralytics 安裝，要補登記到 `pyproject.toml`。
- 測試：
  - smoke 測試新增鏡頭設定載入檢查，以及「config 不得引用 `temp/`」對新目錄的檢查；
  - 新增 `tests/test_homography_georef.py`（m/px 公式、鏡頭解析、遷移）。
- golden：`collision_detect` 的 stage3 輸出不變；`homography_validation` 輸出新增欄位，要依 TOLERANCE.md 重建基準。
- 文件：`docs/adr/0010-*.md`、`CLAUDE.md`、`CONTEXT.md`、既有的 `config/homography_checkpoints/README.md`；鏡頭設定的欄位說明寫在各 YAML 的檔頭註解。
- 資料：參考實作所依賴的 GPS 對照 Excel 不在快照中，經緯度改以自動查詢取得、使用者確認；`.env` 已在規劃階段建立（權限 0600、列入 `.gitignore`）。
