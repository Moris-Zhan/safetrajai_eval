# 以鏡頭為鍵的設定

> 涵蓋 `config/datasets.yaml`、`config/roi/`、`config/homography/`、`config/homography_checkpoints/`。這些設定屬於**鏡頭**，不屬於某個情境檔，所以多個情境共用同一份。

**鏡頭編號**是序列名稱第一個空白前的字串（例如序列 `C000054 民族一路 天祥二路_20250627071742` 的鏡頭是 `C000054`）；序列名稱裡沒有空白時會報錯。`roi/`、`homography/`、`homography_checkpoints/` 的檔名都是 `<鏡頭編號>.yaml`。

## 1. 新增一支鏡頭要補哪些檔

| 步驟 | 要做的事 | 沒做會怎樣 |
|---|---|---|
| 1 | 在 `datasets.yaml` 對應資料集的 `cameras` 列出鏡頭，並把影片放到該資料集的版面位置 | 序列不會被探索到 |
| 2 | 建立 `config/roi/<camera>.yaml` | 評估與 stage3 以整張影像為 ROI（會印出提示）；**蒐集 Re-ID 裁切時直接報錯** |
| 3 | 需要 stage3 的 Homography 時，建立 `config/homography/<camera>.yaml` | 該鏡頭的序列退回原始 CCTV 像素座標 |
| 4 | 要驗證 Homography，建立 `config/homography_checkpoints/<camera>.yaml` | 驗證報表標示為 `no_checkpoints` |

## 2. `datasets.yaml`：dataset id 登記表

情境檔的 `paths.dataset: <id>` 會查這張表。

```yaml
datasets:
  cctv_all8:
    root: "./dataset/eval/cctv_all8/"
    role: eval
    layout: annotated
    cameras: [C000025]
    run_det_eval_dir: annotations_all8
    run_track_eval_dir: mot_annotations_all8
```

| 鍵 | 說明 |
|---|---|
| `root` | 資料集根目錄 |
| `role` | `eval`：附人工 GT，只用於評估；`source`：沒有人工 GT，用來產生訓練資料。**`eval` 與 `source` 的鏡頭不得重疊**（`tests/test_smoke.py` 檢查），評估鏡頭也不能進任何訓練資料 |
| `layout` | `annotated`：`<root>/<run_det_eval_dir>/annotation_<seq>/<seq>.mp4`；`raw`：`<root>/<camera>/<seq>.mp4`，只探索 `cameras` 清單內的鏡頭 |
| `cameras` | 這個資料集登記的鏡頭；情境檔的 `paths.cameras` 只能取它的子集 |
| `run_det_eval_dir`、`run_track_eval_dir` | `annotated` 版面的來源子目錄名稱，也是 `OUTPUT_ROOT` 下輸出子目錄的名稱 |

目前有兩個資料集：`cctv_all8`（評估，鏡頭 `C000025`）與 `reid_sources`（來源，9 支鏡頭）。`C000026` 與評估鏡頭位於同一路口，影片雖放在 `dataset/raw/videos/C000026/`，但刻意不列入 `reid_sources`，避免 Re-ID 在 `C000025` 的驗收偏樂觀。

## 3. ROI：`config/roi/<camera>.yaml`

```yaml
camera: C000054
image_size: [1280, 720]
polygon:
  - [595, 145]
  - [1209, 193]
  - [1208, 580]
  - [168, 336]
```

- `image_size` 是 `[寬, 高]`，必須與影片的實際尺寸相同，**不符就報錯，不會自動縮放**。
- `polygon` 是依序列出的頂點（像素座標）。
- stage1 的軌跡**不依 ROI 過濾**；限縮範圍由下游自行套用：stage3 與 Re-ID 裁切以框中心點判斷，stage2 以忽略區域處理。
- 情境檔頂層的 `det_corner` 已不再支援，寫了會報錯。
- 產生 ROI：用離線點選頁，在瀏覽器點出頂點後把 YAML 放回這個資料夾：

  ```bash
  python scripts/picker/make_roi_picker.py --camera C000054
  ```

  完整步驟見 [ROI 教學](../roi/roi-guide.md)。

## 4. Homography：`config/homography/<camera>.yaml`

```yaml
camera: C000054
name: C000054 民族一路 天祥二路
matrix_tag: C000054 民族一路 天祥二路-2-20-1280-1280
satellite: {provider: google_static, scale: 2, size: [640, 640], zoom: 20}
scale_check: passed
latlon: [22.6682777, 120.3168995]
latlon_source: osm_overpass
latlon_date: '2026-10-13'
scale_check_date: '2026-10-13'
```

| 鍵 | 說明 |
|---|---|
| `camera` | 必須等於檔名（不含副檔名） |
| `name` | 選填，鏡頭的名稱 |
| `matrix_tag` | 指向 `models/homography/matrices/H_<matrix_tag>.npy` 的既有矩陣；檔案不存在時報錯 |
| `satellite` | 衛星底圖的請求參數：`provider`（目前只有 `google_static`）、`zoom`（0 到 22）、`scale`（1、2 或 4）、`size`（兩個正整數）。輸出尺寸是 `size × scale` |
| `latlon`、`latlon_source`、`latlon_date` | 路口中心經緯度，來源是 `osm_overpass`、`google_geocoding` 或 `manual` |
| `osm_roads` | 選填，覆寫經緯度查詢用的兩條路名 |
| `scale_check`、`scale_check_date` | 比例尺檢查結果：`passed`、`failed` 或 `unchecked` |
| `needs_recalibration` | 選填。替換衛星底圖後，舊的 `matrix_tag` 會移到這裡，表示矩陣要重新校正 |
| `cctv_points`、`map_points` | 選填的人工對應點（CCTV 與地圖座標的 `[[x, y], ...]`）。必須同時提供、數量相等、至少 4 組 |

規則：

- 每個頂層值都寫在**同一行**，工具才能逐欄更新並保留檔案裡的註解。
- 有不認得的鍵會報錯。
- **矩陣的查找順序**：`matrix_tag` 指向的矩陣 → 設定內的 `cctv_points`／`map_points` → 都沒有時，直接用原始 CCTV 像素座標。
- **只有同時滿足下列條件時才換算成公尺**：有 `matrix_tag`、有 `satellite`、有 `latlon`、`scale_check: passed`，且沒有 `needs_recalibration`。其餘一律維持像素，不會退回預設比例。換算公式：`156543.03392 × cos(緯度) ÷ 2^zoom ÷ scale`。

寫入這些欄位的工具（都是手動執行，不在主流程）：

| 工具 | 寫入的欄位 | 備註 |
|---|---|---|
| `scripts/homography/migrate_homography_config.py [--dry-run]` | 從舊的 tag 鍵資料產生整份設定 | 只寫新檔，衝突時不寫 |
| `scripts/homography/geocode_cameras.py [--cameras <cam...>]`，加 `--apply` 寫入 | `latlon` 與來源、日期 | 會連網；先產生候選表 `models/homography/geocode_candidates.csv` |
| `scripts/homography/check_homography_scale.py --camera <cam> --points X1,Y1 X2,Y2 --reference-m 3.5 --feature 車道寬` | `scale_check` | 衛星圖上兩點換算後的長度必須落在已知長度的 ±15% 內；加 `--dry-run` 只印結果 |
| `scripts/homography/download_satellite.py --camera <cam> --purpose verify\|basemap [--replace]` | 加 `--replace` 時把 `matrix_tag` 改記為 `needs_recalibration` | 會連網；金鑰從環境變數或專案根目錄的 `.env` 讀取，不會印出；Demo Key 只能用於 `verify` |

> 外部服務的條款可能限制離線保存與衍生使用，底圖不納入 git。

## 5. 驗證用的 check points：`config/homography_checkpoints/<camera>.yaml`

```yaml
cctv_points:
  - [x1, y1]
  - [x2, y2]
sat_points:
  - [x1, y1]
  - [x2, y2]
```

- 注意這裡的衛星圖座標鍵是 **`sat_points`**，與 `homography/` 的 `map_points` 名稱不同，兩邊都是既有介面，不要互換。
- `cctv_points` 與 `sat_points` 必須一一對應、順序相同、數量相等。
- check points 必須是**額外挑選、與算矩陣用的訓練點不同的點**，才是有意義的驗證。與訓練點的 CCTV 座標相同的點會被排除；排除後少於 4 點時，報表的 `validated` 為 `False`、`reason` 為 `too_few_checkpoints`。建議 8 點以上。
- 執行驗證：

  ```bash
  python scripts/homography/validate_homography.py
  ```

  輸出到 `models/homography/validation_output/`，其中 `summary.csv` 每支有設定的鏡頭一列。沒有 check points 的鏡頭會標示 `no_checkpoints`，不代表矩陣有問題，只是缺驗證資料。

## 6. 常見任務

| 我想要… | 做法 |
|---|---|
| 讓一支新鏡頭能蒐集 Re-ID 裁切 | 補 `config/roi/<camera>.yaml`，並把鏡頭列入 `reid_sources` 的 `cameras` |
| 讓 stage3 的事件用公尺 | 補 `config/homography/<camera>.yaml` 的矩陣、經緯度，並通過 `scale_check` |
| 確認某支鏡頭的設定能載入 | `python -c "from safetrajai.collision.homography.camera_config import load_camera_config; print(load_camera_config('config/homography/C000054.yaml'))"` |
| 換新的衛星底圖 | `download_satellite.py --purpose basemap --replace`，之後重新校正並重做比例尺檢查 |
