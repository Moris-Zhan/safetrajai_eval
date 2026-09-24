# Proposal

## Why

舊的 Re-ID 資料是舊追蹤器產生的虛擬身分，已確認無效並隔離（change `reorganize-dataset-and-eval-gt`）。後續要用 timm 在本地訓練 6 類共用的 Re-ID 模型（change `train-reid-timm`），在那之前需要一份可重現、可追溯、不混入評估鏡頭的 Re-ID 資料集。

現行的裁切蒐集不適合直接拿來重建：
- 每條軌跡、每一幀都存一張，10fps 的影片相鄰幀幾乎相同。
- 裁切沒有任何 metadata（幀號、時間、框、分數、鏡頭、日期），無法事後篩選，也無法依日期切分。
- 切分腳本會把影像複製成兩份。
- 來源鏡頭都沒有 ROI。

## What Changes

- **來源鏡頭的 ROI**：提供離線 HTML 點選頁（嵌入各鏡頭首幀，不需要伺服器有圖形介面、不上傳任何東西），由使用者畫出來源鏡頭的 ROI，輸出到 `config/roi/<camera>.yaml`。來源鏡頭為 C000043、C000054、C000085，以及 2026-09-24 新增的 C000011、C000012、C000033、C000048、C000056、C000256；同日新增的 C000026 與評估鏡頭 C000025 位於同一路口，影片歸位但不列入 `reid_sources`。
- **選定產生裁切的工具模型與追蹤參數**：
  - 工具模型：在評估資料集上以人工 GT 比較 `yolo11x_dennis.pt` 與 `yolo11x.pt`（後者以 1280 解析度），差距不大時選 dennis。detection 評估的推論解析度改為讀取情境設定，不再固定 640。
  - 追蹤參數：比較現行參數與新的「純度優先」參數（寧可斷開、不要串錯），以 C000025 的 IDSW 與 IDP 決定。結果寫成專用的情境設定。
- **裁切蒐集改寫**：
  - 每條軌跡依影片時間每 0.5 秒最多取 1 張。
  - 最短邊小於 32px 的裁切不存。
  - 每張裁切寫一列 metadata 索引：身分、類別、鏡頭、序列、日期、幀號、時間、框、分數。
  - 只從來源資料集蒐集；評估鏡頭一律拒絕；每支來源鏡頭都必須有 ROI 檔，否則在追蹤前就報錯。
- **污染偵測**：以 DINOv2 特徵、每張裁切只讀一次，同時做類別中心點比對與軌跡內外觀離群值偵測；原本以 dennis 特徵的類別一致性檢查改用同一份 DINOv2 特徵。只產生標記，不搬移或刪除任何裁切。
- **人工審查**：審查頁改讀新的索引與標記，涵蓋所有被標記的身分與 bus 的全部身分。審查結果寫成排除清單。
- **BREAKING：`quarantine_flagged.py` 改名為 `apply_review_exclusions.py`，改為寫入排除清單**，不再搬移檔案（排除與隔離的差異見 CONTEXT.md、ADR 0006）。
- **manifest 與切分**：
  - `build_reid_split.py` 改寫成產生版本化的 `manifest.csv`，欄位為 `crop_path, identity, class, camera, date, split`，不複製影像。**BREAKING**：移除 `--mode category|flat` 兩種複製式輸出。
  - 規則：
    - 抽樣後少於 4 張的身分捨棄。
    - 每個身分最多 30 張（依時間均勻抽樣）。
    - 套用排除清單。
    - C000054 的 0623、0625、0627 為 train，0629 為 val；其他來源鏡頭全部為 train。
    - 評估鏡頭永遠不會出現。
- **pilot 與 I/O 預算**：先對單支 C000054 影片跑完整流程，估算裁切總數與寫入量，並輸出 1 分鐘的疊框影片讓使用者目視抽查工具模型；經使用者同意後才對全部 39 支來源影片執行（見 CLAUDE.md「禁止大量 I/O」）。
- **不做**：
  - Re-ID 模型訓練、`Custom_ReID` 支援 timm、追蹤器接入與驗收：屬於 `train-reid-timm`。
  - 跨軌跡或跨影片的身分合併：碎片化以切分規避，bus 需要時再另外處理。遮擋造成的碎片（C000012、C000048、C000056）已確認無法只靠追蹤參數接回，離線碎片合併另開後續 change。
  - detection 標註或訓練。

## Capabilities

### New Capabilities

- `reid-dataset`：Re-ID 訓練資料的產生與整理契約，涵蓋以下內容：
  - 只從來源資料集、以鏡頭 ROI 蒐集裁切。
  - 抽樣間隔與最小尺寸。
  - 每張裁切的 metadata 索引。
  - 污染標記與人工審查排除，只標記，不改動檔案。
  - 版本化 manifest 的欄位、身分張數規則與依日期／鏡頭的切分。
  - 評估鏡頭永遠排除，影像不複製。

### Modified Capabilities

（無。資料分層與評估鏡頭排除規則已由 `dataset-layout` 規範，本 change 在其約束下運作，不修改既有 requirement。）

## Impact

- **程式碼**：
  - `safetrajai/pipeline/reid_crops.py`：抽樣、最小尺寸、metadata 索引、拒絕評估鏡頭。
  - `safetrajai/pipeline/sequence_tracker.py`：傳入影片時間。
  - `safetrajai/evaluation/pipeline.py`：detection 評估的推論解析度改讀設定。
  - `scripts/reid_data/` 的審查、排除、切分三支腳本改寫；`check_class_consistency.py` 併入新的 DINOv2 污染偵測腳本。
  - 新增：DINOv2 污染偵測腳本、ROI 點選頁產生腳本、manifest 相關模組。
- **設定**：
  - `config/roi/` 新增 9 支來源鏡頭的 ROI（C000043、C000054、C000085、C000011、C000012、C000033、C000048、C000056、C000256）。
  - `config/datasets.yaml` 的 `reid_sources` 改為上述 9 支鏡頭。
  - 新增裁切專用的情境設定（純度優先參數，或沿用現行參數，依比較結果而定）。
  - `botsort_save_reid_crops_video.yaml` 的輸出位置改為新的裁切目錄。
- **資料**：
  - `dataset/reid/crops/`：裁切與索引。
  - `dataset/reid/splits/<version>/`：manifest 與排除清單。
  - `dataset/reports/reid/`：標記與審查頁。
  - 寫入量取決於 pilot 估算，全量執行前需要使用者確認。
- **依賴**：DINOv2 權重第一次使用時從 HF Hub 下載（只下載，不上傳）；timm 已在 conda env `track_eval` 中。
- **文件**：`scripts/reid_data/README.md`、CLAUDE.md 的 Re-ID 章節、`CONTEXT.md`（「身分」與 manifest 相關術語，視需要更新）。
