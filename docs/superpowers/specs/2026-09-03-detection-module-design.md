# module/detection/ — 設計文件

- 日期:2026-09-03
- 狀態:待審查
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第四個子專案**。

## 1. 背景與結論

`module/detection/detector.py`(39 行)只有一個 `Detector` class:`__init__(model_path)` 載入 YOLO 模型,`detect(frame, iou, imgsz)` 執行推論並依 `module/tracker/config.py::get_conf_thresh_for_class()`/`get_min_conf_thresh()` 做逐類別信心過濾。職責單一、介面精簡,**結構本身不需要調整**。

在確認這個結論的過程中,發現一個跨模組重複:`module/grid_search/`(三支腳本現行的 `_run_tracking_for_video()`)沒有重用 `Detector`,而是自行呼叫 `model.predict()` 並手刻一模一樣的「取 `min_conf` → 推論 → 逐偵測套用類別門檻過濾」邏輯,與 `Detector.detect()` 內容逐字重複。這是本輪唯一要處理的項目。

## 2. 目標 / 非目標

**目標**
- 確認 `Detector` 對外介面(`Detector(model_path).detect(frame, iou, imgsz) -> boxes`)已經足夠通用,維持不變。
- 消除 `module/grid_search/` 與 `Detector.detect()` 之間的偵測+過濾邏輯重複,兩邊改共用同一份實作。

**非目標**
- 不調整 `Detector` 的介面或內部邏輯。
- 不處理 YOLO 訓練腳本(不在本專案範圍,見 `markdown/NEED.md`)。
- 不處理 Re-ID 訓練資料的產生/過濾規劃(已記錄於 `markdown/NEED.md`,留待 `module/reid/` 專屬 brainstorm)。

## 3. 新設計

`module/grid_search/core.py`(`2026-09-03-grid-search-dedup-design.md` 定義的新檔,已同步修訂該文件的 `_run_tracking_for_video()` 段落):

- `score_params()`:每個 trial 開始時建立 `detector = Detector(model_path)`,取代現行 `model = YOLO(model_path)`。建立頻率不變(每 trial 一次,trial 內的影片迴圈重用同一個 `detector`),不影響效能特性。
- `_run_tracking_for_video()`:參數從 `model: YOLO` 改成 `detector: Detector`;內部原本手刻的 `min_conf = get_min_conf_thresh()` / `results = model.predict(...)` / `keep = [...]` 整段,改成一行 `boxes = detector.detect(frame, iou=iou, imgsz=imgsz)`。`get_conf_thresh_for_class`/`get_min_conf_thresh` 的 import 與呼叫從 `_run_tracking_for_video()` 移除,收斂回 `Detector.detect()` 內部(單一實作)。

## 4. 影響檔案清單

| 檔案 | 改動 |
|---|---|
| `module/detection/detector.py` | 不變 |
| `module/grid_search/core.py`(尚未實作,已在 `2026-09-03-grid-search-dedup-design.md` 定義) | `score_params()`/`_run_tracking_for_video()` 改用 `Detector`(已寫入該 spec 的修訂註記) |
| `2026-09-03-grid-search-dedup-design.md` | 已修訂:`_run_tracking_for_video()` 簽章與說明更新為呼叫 `Detector` |

不動 `track_eval.py`(已經在用 `Detector`)、`module/tracker`、`module/reid`、`module/collision`。

## 5. 錯誤處理與邊界情況

- `Detector.detect()` 內部的信心過濾邏輯(`get_conf_thresh_for_class`/`get_min_conf_thresh`)依賴 `load_track_config()` 已先被呼叫(`CLASS_TRACKING_PARAMS` 非空)。grid_search 的 `main()` 本來就已經在解析引數後呼叫 `load_track_config()`,順序不變,不新增風險。
- 若未來 `Detector.detect()` 的過濾邏輯有變動,`track_eval.py` 與 `module/grid_search/` 會同時受影響(這是預期行為,消除重複的目的就是讓兩邊行為保證一致)。

## 6. 驗證計畫

1. `track_eval.py`(已使用 `Detector`)驗證不受影響——本輪未改動 `Detector` 本身。
2. `module/grid_search/*.py`(待前一輪 spec 實作完成後)用 `--max-videos 1` 跑一組 trial,確認改用 `Detector` 前後,同一組參數的偵測框數量與 `HOTA.AssA_mean` 數值一致。

## 7. 風險總結

- **低風險**:改動範圍小(單一函式內部邏輯替換為呼叫既有 class),且 `Detector.detect()` 的行為本身完全不變,只是呼叫方從「自己重寫一份」改成「呼叫既有實作」。
- 本輪需與 `module/grid_search/` 子專案的實作**一起驗證**(這個修改實際上是那個檔案的一部分,只是設計決策記錄在這裡)。
