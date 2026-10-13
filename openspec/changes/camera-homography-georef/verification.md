# Verification

## 已驗證（2026-10-13）

### 單元與 smoke 測試

- `pytest`：563 passed、11 skipped。新增的測試檔：`test_homography_georef.py`、`test_homography_migration.py`、`test_secrets.py`、`test_geocode_cameras.py`、`test_satellite_download.py`、`test_scale_match.py`、`test_homography_validation.py`，全部使用暫存目錄與 mock HTTP，不連網。
- smoke 新增：`.env` 被 git 忽略、追蹤的檔案中沒有 `AIza…` 金鑰字串、26 份 `config/homography/*.yaml` 都能載入，而且沒有引用 `temp/`。

### 遷移（任務 2.2）

- `scripts/migrate_homography_config.py --dry-run`：28 個 `create`，即 26 份鏡頭設定，加上 C000025（tag 鍵 YAML 原樣複製）與 C000297（`.npy` 轉入）兩份 check points；正式執行後寫入 28 個檔案。
- `ls config/homography | wc -l` 為 26；原始的 tag 鍵 YAML 與 `.npy` 沒有被修改。

### 解析等價

- 對 26 個既有 tag 建立 `<路口名>_20250101000000_15s_1` 形式的序列名，以新的 `resolve_camera_homography()` 解析：26/26 的 tag 與矩陣（`np.array_equal`）都和舊的前綴比對結果相同。

### 驗證報表結構（任務 7.2 的結構檢查，正式執行留待比例尺檢查之後）

以 `run_all(generate_videos=False)` 輸出到 `$CLAUDE_JOB_DIR/tmp/val72/`，沒有覆寫 `models/homography/validation_output/`：

| 鏡頭 | validated | reason | check_N | check_excluded | check RMSE (px) | 訓練點殘差 RMSE (px) |
|---|---|---|---|---|---|---|
| C000025 | False | too_few_checkpoints | 2 | 0 | 64.67 | 0.0 |
| C000297 | True | — | 10 | 0 | 396.54 | 0.0 |
| 其餘 24 台 | False | no_checkpoints | — | — | — | 0.0（4 點 DLT 恰好可解） |

- 所有 `_m` 欄位為空：目前沒有任何鏡頭的 `scale_check` 為 `passed`，符合 spec。
- **C000297 的 check RMSE 396.54 px 異常大**（若 m/px 約 0.069，相當於約 27 m）。可能原因是這組 `.npy` check points 不是在目前這張衛星底圖上點選的，或者點選有誤。數值已依 spec 逐位元轉入，沒有修改；需要使用者判斷是否保留這份 check points。

### 經緯度查詢（任務 4.2，2026-10-13，Google 備援使用 `GOOGLE_MAPS_DEMO_API_KEY`）

`python scripts/geocode_cameras.py` → `models/homography/geocode_candidates.csv`：`ok` 12、`needs_review` 12、`error` 2（C000056、C000222：Overpass HTTP 504，重試 3 次後仍失敗）。

有影片的 4 台鏡頭：

| 鏡頭 | 狀態 | 節點數 | 中心 (lat, lon) | 分散 |
|---|---|---|---|---|
| C000025 | ok | 16 | (22.625869, 120.297201) | 25 m |
| C000043 | ok | 2 | (22.640411, 120.314249) | 6 m |
| C000054 | ok | 1 | (22.668278, 120.316900) | 0 m |
| C000085 | needs_review | 3 | (22.624673, 120.341895) | 53 m（略超過 50 m） |

其他 `needs_review` 的原因：名稱無法拆成兩條路（C000011、C000012、C000047、C000294、C000297）、分散過大（037- 288 m、C000033 326 m、C000203 237 m）、Google 備援結果（C000094、C000132、C000221）。

### 套用經緯度（任務 4.3）

- 使用者決定：套用全部 `ok`（12 台），另外接受 C000085（分散 53 m）。
- 套用前把 `config/homography/` 複製到 `$CLAUDE_JOB_DIR/tmp/homography_before/`，`--apply --accept C000085` 之後以 `diff -r` 比對：13 份設定各只新增 `latlon`、`latlon_source`、`latlon_date` 三行，沒有其他差異。

### 已知長度檢查（任務 6.2，由 Claude 量測）

量測方式：在既有衛星底圖（1280×1280）上，以 10 px 格線放大 6 倍的裁切圖，目視判讀單一小客車車身前後端（不含陰影）的像素座標；參考長度取一般小客車 4.7 m（實際約 4.3～5.0 m）。以 `check_homography_scale.py --skip-image-match --dry-run` 換算（含 cos(緯度)）。

| 鏡頭 | 端點（底圖 px） | 長度 (px) | m/px | 換算 (m) | 相對誤差 | 判定 |
|---|---|---|---|---|---|---|
| C000025 | (850,592)–(917,562) | 73.4 | 0.068901 | 5.06 | +7.6% | 通過 |
| C000043 | (592,356)–(666,356) | 74.0 | 0.068893 | 5.10 | +8.5% | 通過 |
| C000054 | (110,438)–(180,452) | 71.4 | 0.068879 | 4.92 | +4.6% | 通過 |
| C000085 | (893,825)–(958,815) | 65.8 | 0.068901 | 4.53 | −3.6% | 通過 |

不確定度：衛星影像模糊加上車輛邊緣判讀，每個端點約 ±2 px，長度約 ±4 px（約 ±0.3 m、±6%）；另外車型不明，參考長度本身也有約 ±0.4 m 的變異。所以這項只能作為粗略的合理性檢查，比例尺的主要把關是影像比對（任務 6.4）。若照參考實作漏乘 cos(緯度)，所有換算值都會再高約 8.3%（C000043 會變成 +17.5%，超出 ±15%），這與 ADR 0010 的修正方向一致。C000054 原本先試的另一台白車與車道標線輪廓不清，改用左側道路上的深色小客車。

### 影像比對（任務 6.4，2026-10-13，未完成）

- 使用者同意後，以 `GOOGLE_MAPS_DEMO_API_KEY` 對 4 台鏡頭執行 `check_homography_scale.py`：Static Maps 全部回應 **HTTP 403「This API is not activated on your API project」**，也就是 Demo Key 所屬的 Cloud 專案沒有啟用 Maps Static API。
- 腳本在寫入前就中止：4 份鏡頭設定仍是 `scale_check: unchecked`，沒有下載任何影像。
- 已改善錯誤訊息：HTTP 錯誤會附上 Google 回應的前 200 字（金鑰值遮蔽），並補上測試。
- **使用者決定（2026-10-13）：不使用 Google Maps API**。改以 `--skip-image-match` 執行，`scale_check` 只依任務 6.2 的已知長度檢查判定：C000025、C000043、C000054、C000085 都寫入 `passed`（附 `scale_check_date`）。
  - 這偏離 design §8「有金鑰時加做影像比對」的原意（金鑰存在，但專案沒有啟用 API）。目前比例尺只經過不確定度約 ±6% 的粗略檢查。
  - 日後若啟用 Maps Static API，可以不加 `--skip-image-match` 重跑 `check_homography_scale.py`，補做影像比對。

### 經緯度補齊（2026-10-13）

- 使用者在 `manual-latlon.md` 手動查詢並核對了其餘 13 台鏡頭（含 037-、C000033、C000203），全部以 `latlon_source: manual` 寫入；`diff -r` 確認只新增 `latlon`、`latlon_source`、`latlon_date`。現在 26/26 台鏡頭都有經緯度。

### 驗證報表（任務 7.2）

`python scripts/validate_homography.py`（含比對影片），輸出 `models/homography/validation_output/summary.csv`，共 26 列：

| 鏡頭 | validated | reason | scale_check | m_per_pixel | check_N | check RMSE |
|---|---|---|---|---|---|---|
| C000025 | False | too_few_checkpoints | passed | 0.068900 | 2 | 64.67 px／4.46 m |
| C000043、C000054、C000085 | False | no_checkpoints | passed | 0.06889～0.06890 | — | — |
| C000297 | True | — | unchecked | — | 10 | 396.53 px（無公尺值） |
| 其餘 21 台 | False | no_checkpoints | unchecked | — | — | — |

### golden（任務 8.3，2026-10-13）

`tests/golden/run_all.sh georef1`（約 16 分鐘，全部步驟正常結束），`python tests/golden/compare.py reid1 georef1`：

| 類別 | 差異 | 說明 |
|---|---|---|
| mot_tracks、yolo_labels、trackeval、conflict_events、reid_crops、other | 0 | stage1／stage2／stage3 輸出不變（spec「既有 stage3 輸出不受鍵值改變影響」） |
| grid_search | `best_params_scooter.txt` 多一行 `'track_buffer': 120,`，數值差異 0 | 既有差異，已記錄在 `archive/2026-10-06-track-buffer-in-frames/verification.md`（golden 設定顯式補上 track_buffer），與本 change 無關 |
| homography | `summary.csv` 欄位改變；C000297 多出 7 個檔案（stats.xlsx 與 6 張熱力圖） | 預期差異：新的報表欄位、C000025 改為 too_few_checkpoints、C000297 的 `.npy` check points 轉入後開始驗證（design §9） |

### 隔離舊檔（任務 8.2）

- `config/homography_checkpoints/C000025 中華三路 民生二路(東北角)-2-20-1280-1280.yaml` 與新的 `C000025.yaml` 以 `cmp` 確認完全相同後，`mv` 到 `config/homography_checkpoints/temp/`，`DO_NOT_USE.md` 記錄原路徑、日期、原因與還原指令。
- C000297 的 check points 依使用者決定保留，並在檔頭標註「可疑」。

### 新基準（任務 8.4，2026-10-13）

- 第二次執行 `tests/golden/run_all.sh georef1b`，`compare.py georef1 georef1b` 的 8 個類別全部 0 差異（identical），容忍度維持 0。
- `georef1` 成為新的 golden 基準；`tests/golden/TOLERANCE.md` 新增「reid1 → georef1」一節，記錄差異來源，並註明之後 `latlon`／`scale_check`／check points 的改變屬於預期的 homography 差異；CLAUDE.md 的比對指令改為 `compare.py georef1 <run>`。

## 待完成

（無）
