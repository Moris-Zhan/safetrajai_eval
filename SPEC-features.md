# 規格：Sub_Module 功能整合（Features Spec）

> 狀態：草稿，待使用者確認。
> 本規格在重構（`SPEC.md`）的 T9 完成後才開始執行（Q36）；兩份規格的範圍互不重疊，重構期間 `Sub_Module/` 完全不動。
> 套件路徑以重構後為準：`utils/modules/safetrajai/`；config 路徑以重構後為準（`config/<功能>/`）。

## 1. 目標（Objective）

把 `Sub_Module/` 中「主流程沒有」的功能，整合成主流程的選用輸出或新的工具腳本：

| 編號 | 功能 | 來源 | 形式 |
|---|---|---|---|
| F1 | 網頁式互動選點器 | `homongraphy_val/`（matplotlib 選點） | 新腳本 `scripts/picker/make_picker.py` |
| F2 | 計數線流量統計 | `trajectoryanalyst2` Demo 模式 | stage3 選用區塊 `counting:` |
| F3 | 衛星圖鳥瞰影片 | `trajectoryanalyst2` Demo 模式 | `conflict_analysis.bev_video` 選用輸出 |
| F4 | 盲區分析 | `trajectoryanalyst2` 的 `process_blind_spot` | `conflict_analysis.blind_area` 選用輸出 |

「不破壞既有行為」的定義：所有新功能預設關閉；沒有對應區塊或 config 檔時，既有輸出逐位元不變；既有 `pytest` 與 `--help` 輸出（扣除新增的腳本）不變。

## 2. 已盤點但不整合的項目（附理由）

| 項目 | 決定 | 理由 |
|---|---|---|
| 已移植的衝突指標、Homography 批次驗證、同幀同 ID 修復 | 不再整合 | 主流程已有（`collision/conflict/`、`homography/`、`evaluation/mot_converter.py`） |
| 文獻版 PICUD（Q40） | 不做 | ADR 0011 已決定以 DRAC 超限取代；需要時另開 spec |
| SSAM／SUMO 本體（Q41-a） | 不整合 | SSAM 只能在 Windows 執行、SUMO 未編譯；需要驗證指標時用純 Python 合成軌跡測試（已知答案情境），不依賴 SUMO |
| `MoRe - Final Version/` 資料集（Q42-a） | 不整合 | 無授權與出處；確認後另開 spec |
| Esri 衛星圖磚下載、Excel 路名比對索引（Q43） | 不移植 | 後者已被 `geocode.py` 取代；前者需先確認 Esri 使用條款 |
| `NASDownloader`、寫死的 API key 與 NAS 帳號 | 絕不移植 | 機密 |
| `trajectory_trackeval/` | 不整合 | 單檔舊原型，已被主流程取代；`Yolo/yolo_pt2onnx/` 只有 `yolo11x.pt`，沒有轉換程式 |
| `Sub_Module/` 目錄本身 | 不動 | 被 gitignore、約 6.5 GB；`Sub_Module.md` 已過時，留待文件盤點 |

## 3. 指令（Commands）

```bash
conda activate track_eval
pytest

# F1（規劃）
python scripts/picker/make_picker.py --camera C000054 --mode homography
python scripts/picker/make_picker.py --camera C000054 --mode checkpoints
python scripts/picker/make_picker.py --camera C000054 --mode counting-lines
python scripts/picker/make_picker.py --camera C000054 --mode camera-position

# F2–F4：由情境 config 的區塊開啟，沿用既有指令
python track_eval.py --config config/collision/conflict_analysis.yaml
```

## 4. 專案結構（Project Structure）

新增與修改的位置（重構後路徑）：

```
scripts/picker/make_picker.py                    F1
config/counting_lines/<camera>.yaml              F2（以鏡頭為鍵，與 roi/、homography/ 並列）
utils/modules/safetrajai/collision/counting/     F2
utils/modules/safetrajai/collision/conflict/bev.py         F3
utils/modules/safetrajai/collision/conflict/blind_area.py  F4
utils/modules/safetrajai/collision/conflict/config.py      新增 object_width_m、blind_area、bev_video 欄位
（不新增 ADR：ADR 已移出 repo，Q60-a；新單位／新輸出／限制記在本檔、輸出的 limitations 與 Markdown 教學）
Markdown/                                        每個功能補對應教學（放進既有的模組資料夾）
```

## 5. 程式風格與測試策略

- 沿用 `SPEC.md` 第 5 節的風格；新模組沿用 `collision/conflict/` 的拆檔方式（config／計算／輸出分開）。
- 優先使用既有依賴（`numpy`、`cv2`、`scipy`）；若需要 `shapely`（盲區多邊形運算）等新依賴，須先經使用者同意（Ask first）。
- 先寫失敗測試再實作（TDD）：
  - F2：用合成軌跡與已知的線，驗證穿越方向、去重、類別分組、`image_size` 不符時報錯。
  - F4：用合成矩形與已知視點，驗證陰影多邊形與面積比例。
  - F1：JS 無法由 pytest 驗證，所以把「點 → YAML」的格式化與驗證放在 Python 端（產生的 YAML 必須能被 `load_camera_config`／`load_counting_lines` 讀回，且內容與輸入點一致）；頁面僅檢查產生的 HTML 含必要標記與內嵌影像。
  - F3：不測影片像素；測試只驗證「開關關閉時不產生檔案」、「開啟時產生非空 mp4」與逐幀繪製函式的輸入輸出。
- 不跑 golden（新功能預設關閉，既有情境輸出不變）；若 `conflict_analysis` 既有輸出出現任何差異，先停下來問使用者。

## 6. 功能規格

### F1 網頁式互動選點器

- 腳本 `scripts/picker/make_picker.py --camera <c> --mode homography|checkpoints|counting-lines|camera-position`，輸出單檔 HTML 加可複製／下載的 YAML 片段；沿用 `make_roi_picker.py` 的做法（首幀 base64 內嵌、輸出放被 git 忽略的 `runs/`，影像不進 git、不上傳）。`make_roi_picker.py` 內容不改，只在重構 T2c 搬到 `scripts/picker/`。
- `homography` 模式：左 CCTV 首幀、右衛星圖並排，先點 4 組訓練點，可退回；輸出 `cctv_points`／`map_points`，貼進 `config/homography/<camera>.yaml`。
- `checkpoints` 模式：再點 8 組驗證點；輸出 `cctv_points`／`sat_points`（沿用既有 key 名稱，與訓練點的 `map_points` 不一致屬既有介面，不改名），貼進 `config/homography_checkpoints/<camera>.yaml`。
- `counting-lines` 模式：在 CCTV 首幀每兩點畫一條線，可命名、可設定方向標籤；輸出 `config/counting_lines/<camera>.yaml` 的內容。
- `camera-position` 模式：在衛星圖上點一個視點；輸出 `camera_view_point: [x, y]`（衛星圖像素），貼進 `config/homography/<camera>.yaml`。
- 不輸出 `.npy` 矩陣；矩陣由 `homography_from_config()` 載入時算出。若該鏡頭已有 `matrix_tag`，頁面提示「這組點只有在移除 `matrix_tag` 後才會生效」，不自動修改任何 config。
- 衛星圖來源為 `models/homography/satellite_images/`；沒有衛星圖的鏡頭先報錯並提示用 `download_satellite.py`。
- 新增腳本後 `README.md` 必須提到檔名（覆蓋測試）。

### F2 計數線流量統計

- 座標空間：CCTV 像素座標。投影變換保持線段相交關係，所以結果與衛星圖座標下相同，且不依賴 Homography，所有鏡頭可用。
- 設定檔 `config/counting_lines/<camera>.yaml`（以鏡頭為鍵，格式比照 `config/roi/`）：

  ```yaml
  camera: C000025
  image_size: [640, 480]
  lines:
    - name: north_gate
      points: [[120, 300], [480, 310]]   # 畫線順序 p1 → p2
      forward_label: southbound          # 沿 p1→p2 看（影像座標，y 向下），從左側穿越到右側
      backward_label: northbound         # 反向；兩個標籤皆可省略，預設 "forward"／"backward"
  ```

  `image_size` 與影片尺寸不符時報錯（與 ROI 相同）。
- 情境 config 新增頂層區塊（沒有這個區塊視為關閉，預設 `enabled: false`）：

  ```yaml
  counting:
    enabled: false
  ```

- 判定點固定為框底邊中點（接地點），不開放參數。方向依畫線順序，不再像原版依 x 座標隱含決定。同一條軌跡對同一條線同一方向只計一次（避免軌跡在線上抖動而重複計數）。軌跡鍵為 `(class, track_id)`。
- 沒有對應 `config/counting_lines/<camera>.yaml` 的序列，只寫 `summary.json` 並記錄 `skipped_reason: no_counting_lines`。
- 輸出：`collision_output/<seq>/counting/counts.csv`（逐線、逐方向、逐類別的累計與逐幀時間序列）與 `summary.json`；`limitations` 註明 ID 切換造成的高估或低估，以及 stage1 軌跡不依 ROI 過濾（ADR 0005）。
- 在 `run_stage3()` 之後呼叫，與既有輸出互不影響。

### F3 鳥瞰影片

- 由 `conflict_analysis.bev_video: true` 開啟，預設關閉；輸出 `collision_output/<seq>/conflict/bev.mp4`。
- 只對 `meters_per_pixel()` 可用且有衛星圖的鏡頭產生（目前 C000025、C000043、C000054、C000085），其餘跳過並在 `summary.json` 記錄原因。
- 疊加項目：軌跡尾跡與 ID、速度（km/h）、衝突事件的配對連線、計數線與累計數（F2 開啟時）、盲區（F4 開啟時）。
- 速度只顯示在影片，不另外輸出 `speeds.csv`；需要統計時另開 spec。
- 影片編碼沿用既有 `VideoWriterWithFfmpeg`（fps 寫死為 5 屬既有行為，不修正）。

### F4 盲區分析

- 掛在 `conflict_analysis.blind_area` 底下（需要 m/px，條件與衝突指標相同）；預設關閉。
- 輸入：新增 `config/homography/<camera>.yaml` 的 `camera_view_point: [x, y]`（衛星圖像素，用 F1 取得）。沒有視點或 m/px 不可用的鏡頭只寫 `summary.json` 並記錄 `skipped_reason`。
- 遮蔽物：車體矩形，中心為框底邊中點投影到地圖座標，長度取 `object_length_m`，寬度取新增的 `object_width_m`（預設 car 1.8、small_truck 2.0、bus 2.5、truck 2.5），朝向用平滑後的速度方向；速度太小時沿用前一個有效朝向，沒有就跳過該物件。
- 遮蔽類別：`blind_area.occluder_classes`，預設 `[car, small_truck, bus, truck]`，不含 `person`、`scooter`。
- 盲區：從視點看過去，矩形後方的陰影區域，與 ROI 在地圖座標下的多邊形相交。輸出「盲區面積／ROI 面積」。
- 輸出：`collision_output/<seq>/conflict/blind_area.csv`（逐幀比例）；`limitations` 註明視為 2D、不考慮物件高度，所以會高估盲區。
- 與衝突事件的整合（事件註記盲區比例）不在本規格內，等逐幀輸出完成並檢視數據後再決定。

## 7. 執行順序

F1 → F2 → F3 → F4。F1 最先，因為 F2、F4 需要它取得資料；F3 排在 F4 前，F4 完成後再補上盲區疊圖。每個功能：先在本檔補決策，再 TDD 實作，一個功能一個 commit 系列（測試、實作、文件分開）。

## 8. 邊界（Boundaries）

### 一定要做（Always）
- 所有新功能預設關閉；沒有對應區塊或 config 檔時既有輸出逐位元不變。
- 新增單位與限制寫進本檔、`Markdown/` 教學與輸出的 `limitations`；未經實測校正的數值只能當技術訊號（承襲既有 ADR 0010、0011 的原則，原文在 `ignore/docs/adr/`，不在 repo 內）。
- 新增腳本時同步更新 `README.md` 與 `Markdown/` 對應教學。
- 一個任務一個 commit（英文 conventional commits），commit 前跑 `pytest` 並出示 diff 摘要。

### 先問再做（Ask first）
- 新增任何依賴（含 `shapely`）。
- 修改 `config` 區塊名稱以外的既有欄位語意。
- 跑 golden 比對。

### 絕對不做（Never）
- 修改 `Sub_Module/` 內任何檔案，或把其中的大型內容（`sumo/`、`SSAM/`、`MoRe`、`trajectory_trackeval/`）加入 git。
- 移植或複製任何 API key、NAS 帳號、密碼；程式不得印出金鑰值。
- 修改 tracker override 方法、`utils/modules/_vendor/trackeval/`、`utils/model.py`。
- 對 `dataset/`、`models/`、`runs/` 做全量讀取。
- 把 CCTV 影像或衛星圖上傳到外部服務；選點器的輸出只放被 git 忽略的 `runs/`。

## 9. 成功標準（Success Criteria）

1. 全部新功能預設關閉時，既有 `pytest` 通過，既有情境輸出逐位元不變。
2. F1：每個模式產生的 YAML 都能被對應的載入函式讀回，內容與輸入點一致；`README.md` 提到新腳本。
3. F2：合成軌跡測試涵蓋方向、去重、類別分組、`image_size` 不符、無設定檔的 `skipped_reason`。
4. F3：開關關閉不產生檔案；開啟時對 4 個可用鏡頭產生非空 `bev.mp4`，其餘鏡頭只記錄跳過原因。
5. F4：合成矩形與已知視點的陰影面積比例與手算值一致（容許誤差寫進測試）；`limitations` 含 2D 假設。
6. 本檔與 `Markdown/` 教學涵蓋計數線座標空間、去重規則、盲區幾何假設。
7. 「絕對不做」清單零違反。

## 10. 已決定事項與待確認

| 編號 | 決定 |
|---|---|
| Q36 | 新功能另開本規格，T9 後才做 |
| Q37 | 計數線做成 stage3 選用區塊（座標與格式見 F2） |
| Q38-a | 盲區要整合，用途為逐幀比例輸出與影片疊圖，事件註記延後 |
| Q39-a | 鳥瞰影片要做，選用輸出 |
| Q40 | 不做文獻版 PICUD |
| Q41-a | 不整合 SSAM／SUMO，用純 Python 合成軌跡測試 |
| Q42-a | 不整合 MoRe |
| Q43 | 互動式選點要移植並改成網頁；Esri 底圖與 Excel 比對不做 |
| Q44 | 盲區輸出 (i) 逐幀比例＋(iii) 影片疊圖；相機視點用網頁取得 |
| Q45 | 計數線用 CCTV 像素座標，獨立 `config/counting_lines/<camera>.yaml` |
| Q46 | 判定點為框底邊中點；方向依畫線順序；同軌跡同線同方向只計一次 |
| Q47 | 新增 `scripts/picker/make_picker.py`，ROI 選點器內容不改 |
| Q48 | 鳥瞰影片疊加項目與輸出如 F3；速度只顯示在影片 |
| Q49 | 本檔為載體（不新增 ADR，Q60-a）；順序 F1 → F2 → F3 → F4 |
| Q50 | 選點器輸出格式如 F1，不輸出 `.npy` |
| Q51 | `counting:` 為獨立頂層區塊；盲區與影片掛在 `conflict_analysis:` 底下 |
| Q52 | 盲區幾何假設如 F4 |

待實作時確認（不阻擋規格）：
1. 盲區的多邊形運算要不要用 `shapely`（新依賴，須先問）；否則以 `cv2`／`numpy` 實作。
2. 計數 `counts.csv` 的欄位順序與時間序列的取樣（逐幀或逐秒）。
3. 鳥瞰影片的解析度與軌跡尾跡長度的預設值。
4. 選點器頁面上的退回／清除操作的細節（沿用原版 `z` 退回）。
