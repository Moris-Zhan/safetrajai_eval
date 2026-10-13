# Design

## Context

動機見 proposal.md。目前的狀態：

- `src/safetrajai/collision/homography/` 已經從參考實作移植了大部分的無 GUI 功能：`compute_homography`、`transform_*`、`warp_and_overlay`、`validation.py`（check points 像素誤差、overlay、Excel）、`heatmap.py`、`compare_video.py`、`pick_aid.py`。`scripts/validate_homography.py` 批次輸出到 `models/homography/validation_output/`，結果也納入 golden 比對。
- 各種資料的鍵與位置：
  - 既有矩陣：`models/homography/matrices/H_<tag>.npy`，`tag` 是 `<鏡頭完整名稱>-<scale>-<zoom>-<w>-<h>`，共 26 台鏡頭，鏡頭編號（tag 第一個空白前的字串）互不重複。
  - 訓練點：同目錄的 `pts_cctv_<tag>.npy`、`pts_sate_<tag>.npy`。
  - 衛星底圖：`models/homography/satellite_images/<tag>.png`。
  - stage3 以 `find_tag_for_sequence()` 比對「序列名是否以 tag 去掉數字尾碼的部分開頭」來找矩陣。
- 參考實作的底圖來自 Google Static Maps（`size=640×640, scale=2, zoom=20`，輸出 1280×1280，見 `Sub_Module/homongraphy_val/utils_cctv.py::gen_cctv_satellite_image`）。兩個 `satellite_images/` 目錄中都沒有 metadata 檔（`*.meta.json`），也沒有參考實作 Esri 替換流程會留下的 `*.bak_map.png`／`*.sanity_fail.png`；26 張都是 1280×1280，所以 26 張底圖都只能從 tag 推得 zoom 與 scale。這只能排除「曾被 Esri 替換」的跡象，不能證明比例尺正確，因此 tasks 加入以已知長度地物做的比例尺檢查。
- 參考實作的 `simple_zoom_scale()` 是 `40075017 / (scale·256·2^zoom)`，漏乘 cos(緯度)，在高雄約高估 8%（見 `Sub_Module/trajectoryanalyst2/CONFLICT_METRICS.md` 2.2 節）。
- 既有 check points：
  - C000025 只有 2 點（tag 鍵 YAML），是目視估計，檔頭已註明信心度中等；
  - C000297 有 `.npy`，放在 `models/homography/checkpoints/`，目前沒有任何程式讀取它。
- 執行環境沒有 `DISPLAY`，參考實作的 matplotlib 點選視窗無法使用。

## Goals / Non-Goals

**Goals:**
- 鏡頭編號是 Homography 唯一的查找鍵，與 ROI 一致。
- 提供 `m_per_pixel(camera)`，作為後續 `conflict-metrics` 的單位基礎。
- 讓所有既有的驗證資料都進入報表，並把「擬合殘差」與「驗證誤差」分開呈現。

**Non-Goals:**
- 不提供互動式點選 GUI（維持 `pick_aid` 並排圖加上手填 YAML 的流程）。
- 不重新標定任何矩陣，也不判定既有矩陣是否合格。
- 不改變 stage3 既有 TTC／PET 的座標選擇（仍投影框中心）；接地點改用底邊中點，只套用在 `conflict-metrics` 的新指標。
- 不處理 Esri 圖磚、GPS 對照 Excel 的自動比對（參考實作 `match_video_to_intersection` 的路名分詞比對）。

## Decisions

### 1. 鏡頭設定檔 `config/homography/<camera>.yaml`

```yaml
camera: C000025
name: C000025 中華三路 民生二路(東北角)
matrix_tag: C000025 中華三路 民生二路(東北角)-2-20-1280-1280   # 選填
satellite:            # 選填；沒有時 m_per_pixel 不可用
  provider: google_static
  zoom: 20
  scale: 2
  size: [640, 640]    # 請求尺寸；輸出為 size × scale
latlon: [22.xxxx, 120.xxxx]   # 選填；使用者提供
cctv_points: []       # 選填：人工對應點（沒有 matrix_tag 時才會使用）
map_points: []
```

- 為什麼放在 `config/` 而不是 `models/`：設定需要納入版控，`models/` 被 gitignore；矩陣與底圖仍留在 `models/homography/`，設定檔只引用 tag。
- 為什麼不沿用前綴比對：鏡頭名稱中的視角描述在不同來源可能寫法不同，前綴比對會靜默找不到；以鏡頭編號查找與 ROI 一致，找不到時有明確提示。
- 替代方案「把設定併進 `config/roi/<camera>.yaml`」：被否決。ROI 與 Homography 的生命週期不同，合併後 ROI 的 `image_size` 檢查會牽動 Homography。

### 2. 遷移採用一次性腳本，不採執行期相容

`scripts/migrate_homography_config.py`：

- 由 26 個 `H_<tag>.npy` 產生 26 份鏡頭設定，`satellite` 參數從 tag 尾碼解析，`latlon` 留空；
- tag 鍵的 check points YAML 以**文字原樣複製**到 `<camera>.yaml`，保留檔頭的信心度註解；
- `.npy` check points 轉成 YAML，數值寫成 `repr(float)`，確保逐位元可還原；
- 目標已存在且內容不同時停止。原檔不刪除，遷移後由使用者確認，再依隔離慣例移到 `temp/`。

執行期的解析程式只認新格式，舊的 tag 鍵 YAML 若還留在 `config/homography_checkpoints/` 會被忽略並提示。

### 3. 公尺換算放在獨立的純函式

`georef.py::meters_per_pixel(camera_cfg) -> float | None`，公式見 spec。

- 用 `156543.03392 = 2π·6378137 / 256`（WGS84 赤道半徑），而不是參考實作的 `40075017 / 256`（赤道周長取整）。兩者差約 1e-5，對 golden 沒有影響（golden 目前不輸出公尺）。
- 人工對應點產生的矩陣（自訂矩形平面），一律回傳 None。

### 4. 驗證報表擴充

- 擬合殘差欄位一律加 `train_residual_` 前綴，避免和 `check_` 混淆。只有 4 點時 DLT 是恰好可解，殘差接近 0 是預期結果，報表註明「不代表準確」。
- check points 少於 4 點時 `validated=False`、`reason=too_few_checkpoints`。這會讓 C000025 從 True 變成 False，屬於刻意的語意改變。
- 公尺欄位使用 `_m` 後綴；`m_per_pixel` 不可用時欄位留空，不填 0。

### 5. 衛星底圖下載

- 只支援 `google_static`，與既有 26 張底圖的來源一致，m/px 公式也才一致。Esri World Imagery 的最大 zoom 是 19，解析度不同，若要支援要另外處理 m/px，這次不做。
- 金鑰依第 7 節讀取。參考實作把 key 寫死在 `utils_cctv.py` 第 53 行，這份不移植，並建議使用者撤銷該 key。
- `--purpose verify|basemap`：
  - `verify` 存到 `models/homography/verification/<camera>/`，接受 demo 與 billing 金鑰，只供第 8 節的影像比對使用；
  - `basemap` 存到 `models/homography/satellite_images/`，只接受 billing 金鑰。Maps Demo Key 的條款限定用於測試與原型開發，所以不拿來產生正式資料。
- 輸出 `<tag>.png` 與 `<tag>.png.meta.json`，metadata 欄位為 provider、zoom、scale、size、中心經緯度、下載日期（ISO 8601）、purpose、key_type。
- 下載後做基本 sanity check：尺寸符合，且不是單色或錯誤圖。參考實作 `satellite_sanity_check` 的用意相同，但這裡只保留尺寸與變異數檢查。
- 替換既有底圖要加 `--replace`（只適用 `basemap`）；舊檔移到 `models/homography/satellite_images/temp/` 並寫入 `DO_NOT_USE.md`。底圖換了之後，舊矩陣的對應點就失效了，所以 `--replace` 會把鏡頭設定的 `matrix_tag` 移除並改記在 `needs_recalibration` 欄位；重新標定前，該鏡頭沒有 Homography，`m_per_pixel` 也不可用。

### 6. 經緯度自動取得（`scripts/geocode_cameras.py`）

可行性已在規劃階段（2026-10-13）實測過，tasks 會重跑一次並寫入 `verification.md`：
- Nominatim 查不到「A 路 B 路」這種路口；
- Overpass 查高雄範圍內兩條路的共用節點可以查到：C000025 有 16 個節點，中心為 (22.625869, 120.297201)；
- 26 台中有 9 台直接取得，8 台被限流，4 台查不到共用節點，5 台名稱無法拆成兩條路。

做法：
- 名稱拆解：去掉鏡頭編號、括號內的視角描述，以及結尾的「東北角」「口」等字樣，以空白或頓號切開；結果必須剛好是兩條路，否則為 `needs_review`。鏡頭設定有 `osm_roads` 時直接使用。
- 查詢：限定高雄 bbox，路名以前綴比對（例如「中正路」也比對「中正一路」）；查詢間隔 ≥ 2 秒，遇到 429 或非 JSON 回應時指數退避，最多重試 3 次，並帶有可辨識的 User-Agent，遵守 Overpass 的使用政策。
- 備援：Overpass 查不到時，若有金鑰，以 Google Geocoding API 查「<路1>與<路2>交叉口, 高雄市」，`latlon_source: google_geocoding`，狀態一律為 `needs_review`。
- 分散距離大於 50 m 時為 `needs_review`。例如 C000033 的 331 m，可能是同名道路有多處交會。
- 候選 CSV 寫到 `models/homography/geocode_candidates.csv`；`--apply` 只寫入 `status=ok` 的列，或使用者以 `--accept <camera>` 指定的列。

m/px 只和緯度有關，就算偏差 300 m，m/px 的誤差也只有約 0.002%；中心精度主要影響第 5 節驗證用底圖與既有底圖的重疊程度，所以 50 m 的門檻是依影像比對的需要訂的。

### 7. 金鑰讀取

- `src/safetrajai/secrets.py` 用標準函式庫解析 repo 根目錄的 `.env`（`KEY=VALUE`、`#` 註解），不新增 python-dotenv 依賴。環境變數優先於 `.env`。
- 回傳金鑰值與種類（`demo`／`billing`）。物件的 `__repr__` 會遮蔽金鑰值，所有錯誤訊息只寫變數名稱。
- 下載時：`verify` 優先使用 demo 金鑰，沒有時才用 billing；`basemap` 只使用 billing。
- `.env` 已列入 `.gitignore`（本次規劃階段已建立，權限 0600）；smoke 測試新增一項檢查：`git check-ignore .env` 必須成立。

### 8. 比例尺檢查與 `scale_check`

- 已知長度檢查：在 `pick_aid` 並排圖或放大裁切上量兩個端點，以 tag 推得的 m/px 換算長度；結果、地物種類、參考值寫進 `verification.md`。
- 影像比對：下載 `verify` 用途的底圖，與既有底圖做 ORB 特徵匹配，再以 RANSAC 估計相似變換（`cv2.estimateAffinePartial2D`），取得縮放比例與平移量。縮放比例 1±2% 以內才算通過；平移量只記錄、不判定，因為 OSM 中心與當初 Excel 的 GPS 本來就可能不同。
- `scale_check` 的值：`passed`／`failed`／`unchecked`，由檢查腳本寫入鏡頭設定，並附上檢查日期。遷移產生的設定預設為 `unchecked`，所以檢查完成前，所有鏡頭的 m/px 都不可用。

### 9. golden

- `collision_detect`：stage3 解析出的矩陣相同，輸出逐位元不變，作為驗收條件。
- `homography_validation`：新增欄位、C000025 的 `validated` 改變、C000297 多出統計。依 `tests/golden/TOLERANCE.md` 重建基準：新基準連跑兩次自比為 0，並在 TOLERANCE.md 記錄差異來源。其餘情境要與舊基準等價。

## Risks / Trade-offs

- [自動取得的路口中心有誤（同名道路、OSM 路名不同）] → 分散距離門檻、`needs_review` 狀態、使用者確認後才套用；四台有影片的鏡頭中，C000085 預期要以 `osm_roads` 或手動補上。
- [Overpass 或 Google 服務暫時無法使用] → 查詢工具只在規劃的任務中手動執行，失敗不影響 pipeline；候選 CSV 可以重跑。
- [`.env` 中的金鑰外洩] → 列入 `.gitignore`、權限 0600，程式不印出金鑰值；參考實作中已明文外洩的 key 建議撤銷後重新產生。
- [Google Static Maps 的服務條款可能限制離線保存與衍生使用] → 下載是使用者手動執行，CLAUDE.md 的指令說明註明條款風險；不把底圖納入 git。
- [底圖中心座標與 GPS 的誤差] → 不影響 m/px（只和緯度與 zoom 有關），但會影響重新標定時的對應點品質。
- [平面近似] → Homography 假設路面是平面，坡道與高架下的誤差不在這次處理範圍內，報表只呈現量測結果。
- [C000025 只有 2 個目視估計的 check points] → 新規則會把它標成未驗證；要正式驗證，需要使用者補點。

## Migration Plan

1. 執行遷移腳本，產生 `config/homography/*.yaml` 與鏡頭鍵的 check points。
2. 執行經緯度查詢，使用者確認候選後套用（至少 4 台有影片的鏡頭）；接著做比例尺檢查，寫入 `scale_check`。
3. 切換 stage3 的解析入口，跑 `collision_detect` 的 golden，確認 stage3 輸出差異為 0。
4. 跑 `validate_homography.py`，檢查新欄位，依 TOLERANCE.md 重建 homography 基準。
5. 把舊的 tag 鍵 check points YAML 隔離到 `config/homography_checkpoints/temp/`（smoke 測試確保 config 不引用它）。

回復方式：還原程式即可；舊檔在 `temp/` 中，`DO_NOT_USE.md` 有還原指令。
