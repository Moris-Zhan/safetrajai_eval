## Why

`dataset/reid/crop_dataset/` 現有的 Tier 2 品質過濾（`filter_crop_dataset.py`，來自 `reid-training-data-pipeline`）只做「身分內」的像素統計相對比較（模糊度、色彩直方圖），無法偵測兩類實際存在的問題：(1) 整個身分本身就歸錯類別（如 `bus/` 底下的資料夾實際內容是卡車或看板）；(2) 身分內因追蹤器 ID switch 混入其他物體的局部污染。人工檢視已發現這兩類問題存在，且目前沒有任何工具能有系統地找出來、也沒有刪除/隔離機制——只能逐資料夾手動翻。此外現有 `quality_flags.csv`／`quality_report.md` 是在 DSEC bus 資料三輪併入前產生的，已經過期（報告顯示 bus 5 身分，實際現為 156 身分）。

## What Changes

- 新增 `dataset/reid/check_class_consistency.py`：對每個身分逐張 crop 重新呼叫 `module/detection/detector.py::Detector.detect()`，比對偵測結果的最高信心類別是否等於資料夾指定類別，輸出身分層級（`agreement_rate`）與影像層級（單張不一致）兩種標記，只標記不刪除。**排除 DSEC 來源身分**（資料夾名稱前綴 `dsecdet_`）——`reid-training-data-pipeline` 的 pilot 已證實本專案偵測器在 DSEC 視角對 `bus` 系統性誤判，用同一顆模型驗證同視角資料沒有意義。
- 擴充/新增報告產生器：從現有「每類別隨機抽 6 個身分」改成「每類別所有身分都出縮圖」的全量 contact sheet，依 `check_class_consistency.py` 的可疑分數排序、依身分數分頁；DSEC 來源身分獨立成一個不帶自動分數的區塊，供純目視審查。
- 新增 `dataset/reid/quarantine_flagged.py`：讀取人工確認過的移除清單（CSV），把確認的身分或單張影像從 `crop_dataset/` 搬到 `dataset/reid/crop_dataset_quarantine/`（保留 `{class}/{identity}/` 結構），預設 `--dry-run`，`--apply` 才真的搬移，並寫入操作 manifest 供回溯/還原。
- 重新執行現有 `filter_crop_dataset.py` + `generate_quality_report.py`，涵蓋 DSEC 併入後的全量資料（純執行既有工具，非新增行為，不在此變更的 spec 範圍內，於 tasks 中作為前置驗證步驟）。
- 更新 `dataset/reid/README.md` 補上新工具的用途與目錄關係。

## Capabilities

### New Capabilities
- `reid/identity-class-consistency`：對現有 CCTV 來源身分重跑偵測器，逐身分/逐影像標記「偵測結果類別」與「資料夾指定類別」不一致的項目，只標記不刪除，明確排除 DSEC 來源身分。
- `reid/crop-review-report`：全量、可排序的 contact sheet 報告生成，讓人工能逐一審視每個身分（而非隨機抽樣），DSEC 來源身分獨立區塊呈現。
- `reid/crop-quarantine`：依人工確認清單把問題身分/影像從 `crop_dataset/` 搬到隔離目錄的可逆移除機制，預設 dry-run，操作留有 manifest。

### Modified Capabilities
（無——`reid-training-data-pipeline` 的既有 Tier 1/Tier 2 行為不變，本次為新增能力。）

## Impact

- 新增檔案：`dataset/reid/check_class_consistency.py`、`dataset/reid/quarantine_flagged.py`；`dataset/reid/generate_quality_report.py` 擴充或新增姊妹腳本（全量 gallery 模式）。
- 新增輸出：`dataset/reid/class_consistency_flags.csv`（或等效命名）、擴充後的 `dataset/reid/quality_report.md`（或新檔案）、`dataset/reid/crop_dataset_quarantine/`（新目錄，含 manifest）。
- 依賴既有 `module/detection/detector.py::Detector`、`module/tracker/config.py`（類別信心門檻）、既有 `CLASS_NAMES` 對照表；不改動這些模組本身。
- 不影響 `track_eval.py`（`ReidCropCollector` 不變）、`module/reid/training/`（訓練介面不變）。
- 執行面：`check_class_consistency.py` 需對全量 ~38000+ 張 crop 逐張跑一次 YOLO 推論，屬一次性/低頻批次作業，非常態 pipeline 步驟。
- `crop_dataset/` 原始檔案在確認前不受影響；`--apply` 執行後會實際搬移檔案（非刪除），影響下游 `build_reid_split.py`／訓練資料量統計，需搬移後重新產生 README 的「目前資料量」表。
