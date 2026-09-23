# Proposal

## Why

接下來要重建 Re-ID 訓練資料，並以 HF／timm 在本地訓練 Re-ID 模型，但 `dataset/` 與 `models/` 目前混著無效資料與有疑慮的權重，而且評估 GT 本身有問題。`dataset/reid/` 是舊追蹤器產生的虛擬身分，已確認不可用。`models/yolo-reid/*` 與 `models/class_reid_weights/*` 都是用這批資料訓練的。更嚴重的是，stage2 使用的 `annotations_all8/labels` 是被「限縮範圍」改過的版本：7 欄、帶 conf，逐框比對只有 7976／10401 框對得上，框的 IoU 中位數 0.83，另外多出 1339 框。它和 `mot_annotations_all8` 的 MOT GT 對不起來，所以 detection 評估其實不是在和人工 GT 比。在開始 Re-ID 資料重建（後續 change）之前，要先把資料的權威來源、有疑慮資產的隔離方式，以及評估時的範圍限縮機制定清楚。

## What Changes

- **隔離有疑慮的資產**：有疑慮的權重與資料一律用 `mv` 移到所在目錄底下的 `temp/`，不刪除。`temp/` 內附 `DO_NOT_USE.md`，說明原因、日期與原路徑。對象：
  - `models/yolo-reid/*`、`models/class_reid_weights/{person,scooter}`
  - `dataset/reid/` 全部內容，包含 crop 資料集、兩種切分、稽核報表與兩支散落的 `.py`
  - `all8` 的限縮版與重複版：`annotations_all8`、`annotations_all8(compasate)`、`mot_annotations_all8`、`mot_annotations_all8(Ver2)`
- **確立評估 GT 權威來源**：`annotations_all8(orign)`（5 欄 YOLO，10401 框）與 `mot_annotations_all8(orign)` 是最原始的人工 GT。**原始 GT 檔案內容不得修改**；任何範圍限縮只能發生在評估階段。
- **評估時的 ROI 忽略區域（ignore region）**：
  - 新增以鏡頭為鍵的 ROI 設定（`config/roi/<camera>.yaml`），C000025 的 ROI 由現行 `det_corner` 移植而來。
  - 在送進 TrackEval 之前，由本專案的前處理完成 ignore 匹配：ROI 外的 GT 標為 ignore，匹配到 ignore GT 的預測不計入 FP／TP，接著移除 ignore GT。結果寫到 stage2 輸出目錄，不回寫來源。
  - 取代現行只過濾預測、不過濾 GT 的不對稱行為。vendored TrackEval 不修改。
  - detection 評估改用同一份原始 GT。能否套用 ignore 語意由 design 決定；若暫時做不到，輸出必須明確標示「未套用 ROI」。
- **`dataset/` 分層**：原始影片、人工標註、Re-ID 衍生資料與稽核報表分開存放。`config/datasets.yaml`、`dataset_layout.py` 與各 config 的路徑同步更新。根目錄的 `bus_video/` 與 `dataset/` 根目錄散落的 mp4 一併納入原始影片區。
- **權重定位**：
  - `models/yolo/yolo11x.pt` 是 detection fine-tune 起點（base model），同時也是工具模型。
  - `models/yolo/yolo11x_dennis.pt` 是工具模型，用來產生 Re-ID 虛擬身分與日後的 detection 預標，config 內的引用維持不變。
  - `yolo11n.pt` 是 COCO 原版，與 `yolo26m-cls.pt` 一樣保留不動。
- **`botsort_reid.yaml` 暫代權重**：small_truck／scooter 原本引用 `yolo-reid/yolo11x-reid-ultra.pt`，被隔離後需要暫代設定，由 design 決定。
- **BREAKING：淘汰 YOLO-as-encoder 訓練端**：移除 `scripts/train_reid.py`、`src/safetrajai/reid/yolo_reid_trainer.py`、`src/safetrajai/reid/training/`。推論端 `encoders.ReID` 保留，因為凍結的 `ClassAwareBOTSORT.__init__` 會分派到它。`encoders.Custom_ReID` 與 `utils/model.py` shim 保留，留給後續 HF 模型接入。
- **重建 golden 基準**：換 GT 與 ROI 語意會刻意改變 stage2 指標。golden 基準要以新行為重跑，並記錄新舊差異是刻意變更。
- **不做**：
  - 重新產生 Re-ID 裁切、HF 訓練腳本、`Custom_ReID` 支援 timm：屬於後續 change `rebuild-reid-dataset`。
  - detection 標註或訓練。
  - 修改 `third_party/trackeval/`。
  - 修改 tracker override 方法本體。
  - 調整追蹤參數。

## Capabilities

### New Capabilities

- `dataset-layout`：資料目錄分層契約（原始影片／人工標註／衍生資料／報表），原始 GT 不可修改；有疑慮資產的 `temp/` + `DO_NOT_USE.md` 隔離慣例；評估鏡頭（C000025）的資料不得出現在任何訓練資料來源中。
- `evaluation-roi-filtering`：以鏡頭為鍵的 ROI 定義，以及評估前的 ignore-region 匹配語意。GT 與預測對稱處理，結果只寫到輸出目錄，TrackEval 本身不修改。

### Modified Capabilities

- `project-structure`：
  - 「完整模型 pickle 的 Re-ID 權重維持可載入」改為：權重可以被隔離到 `temp/`，但 `utils.model` shim 仍須保留，確保被隔離的權重仍能用舊方式載入。
  - 「重構不改變任何 stage 的輸出」補上例外：刻意的評估語意變更要重建 golden 基準並記錄差異。

## Impact

- **資料／權重**：`dataset/`、`models/`、`bus_video/` 大量搬移。這些目錄都被 gitignore，git 追蹤不到，靠 `mv` 與 `DO_NOT_USE.md` 保留可還原性。
- **程式碼**：
  - `safetrajai/evaluation/`：`dataset_layout.py`、`mot_format.py`／`mot_converter.py`、`trackeval_runner.py`、`dataset_prep.py`，並新增 ignore 前處理。
  - `safetrajai/pipeline/`：`det_corner` 改為以鏡頭為鍵。
  - `config/*.yaml`、`config/datasets.yaml`，新增 `config/roi/`。
- **移除**：`scripts/train_reid.py`、`safetrajai/reid/yolo_reid_trainer.py`、`safetrajai/reid/training/`。`tests/test_smoke.py` 的 import 與 `--help` 清單要同步更新。
- **測試**：
  - smoke 測試。
  - golden 基準重建：完整跑一次約 11 分鐘，並且會覆寫 `runs/refactor_golden/` 的新 run。
  - `tests/golden/configs/*` 同步更新路徑。
- **文件**：
  - CLAUDE.md（資料路徑、Re-ID 章節、已知限制）。
  - `src/safetrajai/reid/HANDOFF.md`、`scripts/reid_data/README.md`。
  - `reid_crops.py` docstring 中已失效的 `openspec/changes/reid-training-data-pipeline/` 引用。
  - `CONTEXT.md` 的術語：ROI、忽略區域、權威 GT。
