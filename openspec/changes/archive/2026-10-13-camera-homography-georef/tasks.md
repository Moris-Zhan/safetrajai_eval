# Tasks

## 1. 鏡頭設定與解析

- [x] 1.1 先寫 `tests/test_homography_georef.py` 的設定載入測試：正常檔、缺 `camera`、`cctv_points` 與 `map_points` 數量不符、對應點少於 4 組、`scale_check` 不是 `passed`／`failed`／`unchecked`，都要依 spec 成功或報錯（訊息含檔案路徑與欄位）。確認失敗後新增 `src/safetrajai/collision/homography/camera_config.py`，測試通過
- [x] 1.2 先寫解析測試（暫存目錄內的假矩陣）：有 `matrix_tag` 時用既有矩陣；只有人工對應點時用算出的矩陣；沒有設定檔時回傳 None；`matrix_tag` 指向不存在的檔案時報錯，訊息含預期路徑。確認失敗後在 `points.py` 實作依鏡頭解析的入口，測試通過
- [x] 1.3 `config/homography_points/` 有 README 以外的 `.yaml` 時，載入報錯並指向新路徑（spec「發現舊格式檔案」）；以測試驗證
- [x] 1.4 `config_loader.py` 新增 `homography_camera_dir`（預設 `config/homography`）、移除 `homography_points_dir`；`config/collision_detect.yaml` 與 `tests/golden/configs/` 中對應的鍵同步改名。跑 `pytest` 確認全部 config 載入成功

## 2. 遷移

- [x] 2.1 先寫遷移測試（暫存目錄）：
  - 由 `H_<tag>.npy` 產生鏡頭設定，`satellite` 從 tag 尾碼解析為 zoom 20、scale 2、size [640, 640]，`scale_check` 為 `unchecked`；
  - tag 鍵 YAML 原樣複製，檔頭註解保留；
  - `.npy` 轉 YAML 後讀回與原陣列 `np.array_equal`；
  - 目標已存在且內容不同時失敗、不寫任何檔案；
  - 原檔不變。

  確認失敗後新增 `scripts/migrate_homography_config.py`（含 `--dry-run`），測試通過，`--help` 正常
- [x] 2.2 對 repo 執行遷移：先 `--dry-run` 列出要產生的檔案（應為 26 份鏡頭設定、C000025 與 C000297 兩份 check points），再正式執行；以 `ls` 與檔案數核對（不讀整個 `models/` 目錄樹的內容）

## 3. 金鑰讀取

- [x] 3.1 先寫測試（暫存的 `.env`、monkeypatch 環境變數）：
  - 環境變數優先於 `.env`；
  - `#` 註解與空值被忽略；
  - `GOOGLE_MAPS_DEMO_API_KEY` 判定為 `demo`、`GOOGLE_MAPS_API_KEY` 判定為 `billing`；
  - 物件 `repr` 與錯誤訊息不含金鑰值；
  - 都沒有金鑰時報錯。

  確認失敗後新增 `src/safetrajai/secrets.py`（只用標準函式庫），測試通過
- [x] 3.2 smoke 測試新增「`git check-ignore .env` 成立」與「repo 追蹤的檔案中沒有 `AIza` 開頭的字串」兩項檢查；`pytest` 通過

## 4. 經緯度自動取得

- [x] 4.1 先寫測試（mock HTTP，不連網）：
  - 名稱拆解：「C000025 中華三路 民生二路(東北角)」得到兩條路；「中正路交流道南下入口」為 `needs_review`；`osm_roads` 優先使用；
  - Overpass 回傳 16 個節點時取平均；分散距離大於 50 m 時為 `needs_review`；
  - 遇到 429 或非 JSON 回應時退避重試，超過次數記為 `error`；
  - Overpass 查不到且有金鑰時，以 Geocoding 備援，狀態為 `needs_review`；
  - 沒有 `--apply` 時不修改任何鏡頭設定；`--apply` 只寫入 `ok` 的列或 `--accept` 指定的列，並寫入 `latlon_source` 與日期。

  確認失敗後新增 `scripts/geocode_cameras.py`，測試通過，`--help` 正常
- [x] 4.2 **連網，向使用者確認後執行**：對 26 台鏡頭查詢，產生 `models/homography/geocode_candidates.csv`，並把各狀態的數量與四台有影片鏡頭（C000025、C000043、C000054、C000085）的結果寫入 `verification.md`
- [x] 4.3 **使用者確認**：把候選表交給使用者確認；C000085 等 `needs_review` 的鏡頭先嘗試以 `osm_roads` 重查，仍不行時由使用者手動填寫（`latlon_source: manual`）。使用者同意後，先把 `config/homography/` 複製到 `$CLAUDE_JOB_DIR/tmp/`（這些檔案尚未納入 git，`git diff` 看不到差異），再執行 `--apply`，以 `diff -r` 比對確認只有 `latlon`、`latlon_source`、`latlon_date` 三個欄位改變

## 5. 衛星底圖下載

- [x] 5.1 先寫測試（mock HTTP，不連網）：
  - 沒有任何金鑰時報錯，且沒有發出請求；
  - 只有 demo 金鑰時，`--purpose basemap` 報錯且沒有發出請求；`--purpose verify` 寫到驗證目錄，既有底圖不變；
  - 正式底圖已存在且沒有 `--replace` 時報錯，檔案不變；
  - metadata 有 purpose 與 key_type、不含金鑰值，日期為 ISO 8601；
  - 回應是單色圖時 sanity check 失敗；
  - `--replace` 時舊檔移到 `temp/`，`DO_NOT_USE.md` 有紀錄，鏡頭設定的 `matrix_tag` 改為 `needs_recalibration`。

  確認失敗後新增 `satellite.py` 與 `scripts/download_satellite.py`，測試通過，`--help` 正常（6.4 的比例尺檢查會用到）
- [x] 5.2 `pyproject.toml` 登記 `requests`；以 `pip install -e . --no-deps` 重裝後 `pytest` 通過
- [x] 5.3 文件：遷移腳本在每份 `<camera>.yaml` 寫入欄位說明的檔頭註解（風格依既有 config）；CLAUDE.md「常用指令」補上 geocoding、比例尺檢查、下載三個指令，以及 `.env` 的變數名稱、Demo Key 只能用於驗證、服務條款風險。不新增 README（使用者的全域規則）

## 6. 公尺換算與比例尺檢查

- [x] 6.1 先寫測試：zoom 20、scale 2、緯度 22.6、`scale_check: passed` 時，結果與 `156543.03392·cos(22.6°)/2^20/2` 的差小於 1e-12，約為 0.0689；缺 `latlon`、缺 `satellite`、只有人工對應點、`scale_check` 為 `failed` 或 `unchecked` 時回傳 None。確認失敗後新增 `georef.py::meters_per_pixel()`，測試通過
- [x] 6.2 已知長度檢查：在 C000025、C000043、C000054、C000085 的衛星底圖上，以 `pick_aid` 格線座標或放大裁切，各量測一個已知長度的地物（車道寬約 3.5 m、停止線或斑馬線長度），以 tag 推得的 m/px 換算，判定是否落在實際值 ±15% 以內；量測方式與結果寫入 `verification.md`
- [x] 6.3 先寫影像比對的測試（合成影像：已知縮放 1.00 與 1.05、已知平移）：估得的縮放誤差小於 0.5%；縮放 1.05 判定為不通過；平移量只記錄、不影響判定。確認失敗後實作比對函式（ORB 加上 `cv2.estimateAffinePartial2D` RANSAC），測試通過
- [x] 6.4 新增 `scripts/check_homography_scale.py`：讀取 6.2 的量測結果（以命令列參數輸入），有金鑰時下載 `verify` 底圖並比對，依 spec 寫入 `scale_check` 與檢查日期；`--help` 正常。**連網，向使用者確認後**對 4 台鏡頭執行，結果寫入 `verification.md`。任一台不通過時告知使用者，該鏡頭維持不可用
- [x] 6.5 新增 `docs/adr/0010-meters-for-georeferenced-cameras.md`：有衛星圖地理對位、有緯度，而且比例尺檢查通過的鏡頭可以使用公尺，其餘維持像素；公式與參考實作的 8% 落差也寫進去。同步修改 CLAUDE.md「衝突事件／Homography」一節與 CONTEXT.md 的「俯視平面」「單位」敘述；`grep "不是公尺"` 確認沒有矛盾的舊敘述

## 7. 驗證報表

- [x] 7.1 先寫測試（合成矩陣與點）：
  - 訓練點殘差使用 `train_residual_` 前綴；
  - 與訓練點重複的 check point 被排除，排除數正確；
  - check points 少於 4 點時 `validated=False`、`reason=too_few_checkpoints`，統計仍有數值；
  - 有 m/px 時輸出 `_m` 欄位，沒有時留空；
  - 彙總列有 `scale_check`。

  確認失敗後修改 `validation.py` 與 `scripts/validate_homography.py`（以鏡頭為鍵讀 check points），測試通過
- [x] 7.2 執行 `python scripts/validate_homography.py`，確認 `summary.csv` 有 26 列：C000025 為 `too_few_checkpoints`，C000297 有 check 統計（沒有影片，所以沒有 overlay），其餘為 `no_checkpoints`；把重點數值記到 `verification.md`
- [x] 7.3 更新 `config/homography_checkpoints/README.md`（改為鏡頭鍵、點數至少 4 的規則、排除訓練點的規則）

## 8. 驗收

- [x] 8.1 smoke 測試：新增「所有 `config/homography/*.yaml` 都能載入」，並把 `temp/` 引用檢查擴及 `config/homography/`；跑 `pytest` 全部通過
- [x] 8.2 依 design Migration Plan 第 5 步，把舊的 tag 鍵 check points YAML 移到 `config/homography_checkpoints/temp/`，寫好 `DO_NOT_USE.md`；`pytest` 通過
- [x] 8.3 （必須在 4.3、6.4 完成、四台鏡頭的 `latlon` 與 `scale_check` 都定案之後才執行，否則基準會記錄暫時狀態）向使用者確認後跑 `tests/golden/run_all.sh <run>`（約 16 分鐘），以 `python tests/golden/compare.py reid1 <run>` 比對：`collision_output` 與其他非 homography 類別差異為 0；`homography` 類別的差異只來自 design 第 9 節列出的來源。結果寫入 `verification.md`
- [x] 8.4 依 TOLERANCE.md 重建基準，新基準命名為 `georef1`（連跑兩次自比為 0）；在 TOLERANCE.md 記錄 `reid1` → `georef1` 的差異來源（只有 homography 類別），並註明之後鏡頭設定的 `latlon`／`scale_check` 變動會讓 `summary.csv` 的 `scale_check`／`m_per_pixel`／`_m` 欄位改變，屬於預期差異、需重建基準；並把 CLAUDE.md 中的基準名稱與 `compare.py reid1 <run>` 指令改為 `georef1`；跑 `openspec validate camera-homography-georef --strict` 確認通過
