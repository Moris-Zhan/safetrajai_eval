# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概觀

CCTV 交通監控的物件偵測＋多物件追蹤（MOT）評估流程：YOLO11 偵測 → class-aware BoTSORT / ByteTrack 追蹤 → TrackEval 計算 HOTA / CLEAR / Identity 等指標，輸出 Excel 報表。追蹤類別固定 6 類：`person`、`car`、`scooter`、`small_truck`、`bus`、`truck`。

無 `requirements.txt` / `pyproject.toml`、無測試、無 lint/format 設定。專案已於 2026-07 導入 git 版本控制；實驗版本仍靠檔名區分（如 `track_config_BotSort_0413.yaml`）。主要依賴：`ultralytics`（YOLO + 內建 `BYTETracker`/`BOTSORT` 基底類別）、`torch`/`torchvision`、`opencv-python`、`numpy`、`scipy`、`openpyxl`、`pyyaml`。`trackeval/`（根目錄）是 vendored 的 TrackEval 套件並以 `import trackeval` 使用；舊有的重複副本 `trackers/trackeval/` 已歸檔到 `archive/trackers_trackeval_duplicate/`。

專案已依 Detection / Tracker / Re-ID 三大模組重組（2026-07），詳見下方「架構」。

## 專案文件（`markdown/`）
- `markdown/Project.md`：專案總覽、範圍、核心流程圖、已知限制。
- `markdown/TEST.md`：手動驗證流程（無自動化測試，改動後照此手動驗證）。
- `markdown/VERIFY.md`：已驗證過的實驗結論（ReID 開關、GMC 方法比較、FPS 比較等）與資料來源，避免重複測試。
- `markdown/NEED.md`：已知待辦與落差清單。

開始改動前先掃過 `markdown/VERIFY.md`，避免重工已驗證過的結論；改完後照 `markdown/TEST.md` 的流程驗證。

## 常用指令

沒有 CLI 套件包裝，都是直接執行根目錄腳本：

```bash
# 完整流程：class-aware tracking（第一階段）+ TrackEval 評估（第二階段）
python track_eval.py --config config.yaml --track-config tracker/track_config_ByteTrack_0411.yaml

# 針對單一類別做追蹤參數 grid search（以 TrackEval 的 HOTA.AssA_mean 為 scoring）
python tracker/grid_search/grid_search_bytetrack.py --target-class scooter --config config.yaml
python tracker/grid_search/grid_search_botsort.py --target-class scooter --tracker botsort
python reid/grid_search_botsort_reid.py --target-class person   # 含 ReID 開關的版本
# 常用旗標：--dry-run（只列組合數不執行）、--resume（跳過 CSV 已有結果）、
#          --max-videos（限制評估用影片數）、--params-subset（只搜尋指定參數）

# 疊圖視覺化 GT / 追蹤結果到影片
python visualize_gt_on_video.py --trackeval_root <path> --video_path <path> --output_video <path>
python visualize_gt_on_class_video.py --trackeval_root <path> --video_path <path> --output_video <path>
```

跑單一序列或除錯時，改 `config.yaml` 的 `paths.SOURCE_VIDEO_DIR`、`eval_flags`（`run_det_eval` / `run_track_eval`）縮小範圍，比另外加 CLI flag 更快。

## 架構

專案依 **Detection / Tracker / Re-ID** 三大模組重組（2026-07），另有一個橫跨兩者的共用 `evaluation/` 模組（DET 評估與 TrackEval 評估的共用邏輯，不屬於任一模型模組，故獨立）。詳細架構圖見 `markdown/Project.md`。

```
detection/    — YOLO 推論封裝（detector.py）。訓練腳本目前不在本專案範圍，見 markdown/NEED.md。
tracker/      — class-aware BoTSORT / ByteTrack（原 trackers/）+ grid_search/ 超參數搜尋腳本
reid/         — Re-ID 特徵抽取（encoders.py）、YOLO-ReID 訓練工具（yolo_reid_trainer.py）、
                grid_search_botsort_reid.py；training/ 下為孤兒訓練工具（train_utils.py）
evaluation/   — DET/TrackEval 共用 pipeline（config_loader、dataset_prep、mot_converter、
                trackeval_runner、excel_export、mot_format、pipeline）
utils/        — 通用小工具（visualization.py）+ model.py（見下方「⚠️ pickle 相容性」）
track_eval.py — CLI 入口，TrackingEvalPipeline 類別（stage1 tracking → stage2 TrackEval）
trackeval/    — vendored TrackEval 第三方套件本身（不動）
reid_models/  — vendored fast-reid 風格套件，**固定留在根目錄**（見下方「⚠️ pickle 相容性」）
archive/      — 歸檔區：past_conf/、past_py_conf/、trackers_trackeval_duplicate/、past_track_eval_orign.py
```

### ⚠️ pickle 相容性：`reid_models/` 與 `utils/model.py` 不可搬移或改名
`models/reid_model_weights/**/*.pth`、`models/class_reid_weights/*/best_full_model.pt` 是用 `torch.save()` 存整個模型物件（非 state_dict），`SOTA_ReID` / `Custom_ReID`（`reid/encoders.py`）用 `torch.load(..., weights_only=False)` 載入時，pickle 會用**存檔當下的完整模組路徑字串**（`reid_models.modeling....`、`utils.model.xxx`）去 `import` 對應類別定義。因此：
- `reid_models/` 必須留在專案根目錄，維持 `import reid_models` 可用（曾嘗試搬進 `reid/vendored/`，會讓所有 SOTA_ReID 權重載入失敗，已還原）。
- `utils/model.py` 必須留在 `utils/model.py`（`ReIDNet` 等類別被 `models/class_reid_weights/{person,scooter}/best_full_model.pt` 引用，經 `tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支載入，雖然目前預設 config 沒有直接 import 這個檔案，但不能移動）。
- 之後若要調整這兩者的位置，必須先用舊路徑載入權重、`torch.save` 成 state_dict 或用新路徑重新 pickle，而非直接搬檔案。

### 兩層設定檔（順序很重要）
- **`config.yaml`**（根目錄）：路徑、TrackEval 資料集參數、要評估的類別、偵測區域多邊形 `det_corner`、log 開關。由 `evaluation/config_loader.py::load_config()` 讀取。
- **`tracker/track_config*.yaml`**：class-aware tracker 的逐類別參數表（`match_thresh`、`track_buffer`、`use_reid` 等）。由 `tracker/config.py::load_track_config()` 讀取，**必須在建立 tracker 之前呼叫**，否則 `CLASS_TRACKING_PARAMS` 是空的（`track_eval.py` 的 CLI 入口已處理好順序，寫新腳本時要照做）。`tracker/` 目錄下多個 `track_config_*.yaml` 對應不同實驗（BotSort 有/無 ReID、ByteTrack 等）。

### `tracker/` — class-aware 追蹤器（核心邏輯，改動前務必先讀 `tracker/HANDOFF.md`）
繼承並改寫 Ultralytics 的 `BYTETracker` / `BOTSORT`。原版所有匹配閾值全域統一，且不同類別的偵測混在一起做匈牙利匹配，可能跨類別誤配（如 person 配到 scooter）。`class_aware_botsort.py` / `class_aware_bytetrack.py` override `update()`，改為**逐類別分組**做三階段匹配（各類別各自的 `match_thresh` / `second_match_thresh` / `unconfirmed_match_thresh`），track buffer、new-track 閾值、ReID 開關（僅 BoTSORT）也都依類別查表覆蓋，查表函式是 `tracker/config.py::get_params_for_class()`。

ReID 有兩種來源（`reid/encoders.py`）：`ReID`（用 YOLO 模型本身當 encoder，權重在 `models/yolo-reid/`）與 `SOTA_ReID`（載入根目錄 `reid_models/` 這個 vendored fast-reid 風格套件的權重，`models/reid_model_weights/`、`models/class_reid_weights/`）。目前預設關閉 ReID —— 實驗結論是對 person 有幫助但會損害 scooter 且增加 IDSW（見 `tracker/__init__.py` 註解）。`tracker/` 依賴 `reid/`（BoTSORT 匹配時用 Re-ID 外觀特徵），`reid/` 不依賴 `tracker/`。

對外入口只有一個工廠函式：`create_tracker(tracker_type='botsort'|'bytetrack', frame_rate=..., **overrides)`。

### `track_eval.py` — 兩階段主流程（OOP：`TrackingEvalPipeline`）
- `Detector`（`detection/detector.py`）：YOLO 推論 + 逐類別信心過濾。
- `VideoSequenceTracker`（`track_eval.py`）：單一影片的 class-aware tracking 迴圈（每支影片各自建立一個 Detector + tracker 實例，與舊版函式每次呼叫都重建的行為一致）。
- `TrackingEvalPipeline`（`track_eval.py`）：
  1. `run_stage1()`：對 `SOURCE_VIDEO_DIR` 下每個序列跑 YOLO 偵測 + class-aware tracking，輸出 MOT 格式追蹤結果與（可選）標註影片，寫到 `OUTPUT_ROOT/<run_det_eval_dir>/annotation_<seq>/`。
  2. `run_stage2()`：把第一階段輸出轉成 TrackEval 期望的目錄結構（`OUTPUT_ROOT/<run_track_eval_dir>/`），呼叫 `trackeval` 逐類別計算 HOTA/CLEAR/Identity，合併結果輸出 Excel（`evaluation/excel_export.py`）到 `OUTPUT_ROOT/trackeval_output/`。

腳本頂部 docstring 有完整輸入/輸出目錄結構圖，改動輸出路徑邏輯前先讀那段。

### `evaluation/` — DET/TrackEval 共用 pipeline
- `config_loader.py`：讀 `config.yaml`。
- `dataset_prep.py`：序列探索、DET yaml 產生、ultralytics tracking 標籤產生。
- `mot_converter.py`：Ultralytics 追蹤標籤 → MOT 7 欄格式，處理重複幀 / 重複 ID 等髒資料問題。
- `trackeval_runner.py`：組出 TrackEval 要求的 GT/tracker 目錄結構（`seqinfo.ini` 等）並呼叫評估。
- `excel_export.py`：多類別 TrackEval 結果合併輸出 Excel。
- `mot_format.py`：`save_mot_format` / `make_det_eval_dataset`（原 `utils/track_utils.py` 拆出）。
- `pipeline.py`：`run_eval_pipeline`（原 `utils/eval_utils.py`），被 `track_eval.py` 的 `run_stage2()` 呼叫。

### `tracker/grid_search/`、`reid/grid_search_botsort_reid.py` — 超參數搜尋
每種 tracker 一支腳本（botsort / botsort_reid / bytetrack），自行實作 `ParameterGrid` + 逐組評估（不依賴 sklearn Estimator 介面）。流程：動態 patch `tracker.config.CLASS_TRACKING_PARAMS` → 呼叫與 `track_eval.py` 相同的 tracking → `save_mot_format` → `run_trackeval_mot` 流程 → 從結果取出 `HOTA.AssA_mean` 當 score → 還原參數 → 寫入 CSV（可用 `--resume` 續跑）。ReID 版本放在 `reid/` 而非 `tracker/`，因為它調的是 Re-ID 相關參數。

### 其他注意事項
- `reid/training/train_utils.py`：ReID 模型訓練相關工具函式，**目前沒有任何腳本 import 它**（孤兒程式碼，疑似舊訓練流程殘留），改動或刪除前先確認是否仍在別處使用。
- `archive/`：舊版本程式碼保留供對照，不是目前流程的一部分。
- `dataset/`、`runs/`、`wandb/`、`models/`、`reid_models/**/__pycache__`：權重與歷史實驗輸出，體積大但不需要通篇閱讀即可理解架構。

> ⚠️ **下方「專案架構規範」章節與本 repo 實際結構不同**：它描述的是 `scripts/train.py`／`src/`／`configs/experiment/` 這類目標範本目錄結構，但目前程式碼實際上是根目錄 `track_eval.py` ＋ `detection/`／`tracker/`／`reid/`／`evaluation/` 四個模組的組合（見上方「架構」）。新增程式碼請沿用現有慣例，除非使用者明確要求依範本重構整個專案。

## 關於我
- 我是工程師，負責此專案的開發與維護。
- 使用繁體中文（台灣用語）溝通。
- 熟悉技術術語，不需要額外解釋。

## 回答方式
- 直接給結論，不要鋪陳。
- 需要決策時，提供 2–3 個方案並列出各自的取捨（效能、複雜度、維護成本等），不要只給單一方案。
- 不確定的事情直接說不確定，不要猜測或編造答案。
- 涉及原理或設計判斷時，簡述理由，但不要過度展開。

## 執行規則
- 動到程式碼之前，先說明要修改哪些檔案、做什麼改動，等待確認後再執行。
- 批次操作（重新命名、刪除、覆寫、大範圍重構）一律先確認。
- 遵循極簡原則：只改動解決問題所必要的部分，不做無關的重構或「順手優化」。
- 修改前先理解現有程式碼的邏輯與慣例，維持風格一致。
- 產出的檔案放到 `output/` 資料夾（或專案既定的輸出路徑）。
- 涉及刪除資料、覆寫檔案、或執行破壞性指令（如 `rm`、`git push --force`、資料庫遷移）時，一律先確認。
- 完成任務後，說明做了什麼改動、如何驗證（測試結果、執行輸出等），不要只回報「完成」。

## 程式碼風格
- 命名清楚、具描述性，避免縮寫或模糊變數名。
- 保持函式與模組單一職責，避免過度耦合。
- 若專案有既定的 linter / formatter 設定，一律遵循，不自行更改規則。

## 技術棧專屬規則（Python / PyTorch / CV）

### 環境與依賴
- Python 版本、CUDA/cuDNN 版本、PyTorch 版本三者要互相對齊，改動任一項前先確認相容性。
- 新增或升級套件時，同步更新 `requirements.txt` / `pyproject.toml`，並註明版本鎖定的原因（尤其是 `xformers`、`mistral_common` 這類容易衝突的套件）。
- Docker image 變動（base image、CUDA 版本）視為高風險操作，需先確認再執行。

### 模型與訓練 / 推論
- 涉及模型權重、checkpoint 路徑的改動，先確認是否會覆寫既有結果。
- 訓練或推論腳本的超參數改動，要列出改了什麼、預期影響是什麼，再執行。
- Grid Search / 大範圍參數掃描前，先確認搜尋空間大小與預估執行時間。
- TensorRT / ONNX 轉換：轉換前後的輸出要有基本一致性檢查（如 shape、數值誤差範圍），不要假設轉換一定成功。

### 電腦視覺任務（偵測 / 追蹤 / Re-ID）
- 座標系統（如 tlwh、xyxy、center point）轉換時要特別小心，明確標示目前用的格式。
- 評估指標（HOTA、AssA、mAP 等）變動時，附上評估用的資料集版本與參數設定，避免結果不可重現。
- 追蹤演算法（ByteTrack、BotSort）的 matching 邏輯改動，需先說明會影響哪些追蹤行為。

### 效能與除錯
- 效能相關改動（batch size、TensorRT 精度模式 fp16/int8）要附上改動前後的基準數據，不要只憑直覺調整。
- GPU 記憶體不足或 CUDA 錯誤時，先定位是模型、資料、還是環境問題，不要一開始就大改程式碼。
- debug flag（如 `CUDA_LAUNCH_BLOCKING=1`）僅用於除錯階段，確認問題後要提醒移除，避免影響效能。

### 測試與驗證
- 資料前處理 / 後處理邏輯的改動，附上小範例驗證輸入輸出是否符合預期。
- 涉及 API 端點（如 FastAPI）的改動，先確認是否有未授權或破壞性操作的風險（如可直接刪除/覆寫資料的端點）。

## 專案架構規範

### 目錄結構
```
project/
├── datasets/
│   ├── raw/                    # 原始未處理資料，唯讀
│   ├── processed/              # 前處理後可直接餵給訓練的資料
│   ├── annotations/            # 標註檔（COCO/YOLO格式）
│   └── splits/                 # train/val/test 切分清單（txt或json，不搬動實體檔案）
│
├── models/
│   ├── configs/                # 模型結構、超參數設定檔（yaml）
│   ├── checkpoints/            # 訓練中/完成的權重，依 experiment 命名子資料夾
│   │   └── {exp_name}/
│   │       ├── best.pt
│   │       ├── last.pt
│   │       └── metrics.json
│   ├── exported/                # 轉換後的部署格式（onnx, engine/TensorRT）
│   └── registry.yaml            # 模型版本索引：哪個 checkpoint 對應哪個資料集版本、指標
│
├── output/
│   ├── inference/               # 推論結果（圖片、影片、json）
│   ├── eval/                    # 評估報告（HOTA/mAP等指標、混淆矩陣）
│   └── logs/                    # 訓練/推論/評估的執行紀錄
│
├── web_ui/
│   ├── backend/                 # FastAPI，串接 models/exported 做線上推論
│   │   ├── main.py
│   │   ├── routers/
│   │   └── services/            # 封裝模型載入、前後處理邏輯
│   ├── frontend/                 # 若有前端頁面（React/簡單HTML）
│   └── Dockerfile
│
├── src/
│   ├── data/                    # dataset class、augmentation、collate_fn
│   ├── models/                  # 模型定義、backbone、tracker（ByteTrack/BotSort）
│   ├── engine/                  # train/eval/infer 共用邏輯（loop、metric計算）
│   └── utils/                   # 座標轉換（tlwh/xyxy）、logging、config loader
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   └── export.py                # 權重轉 ONNX/TensorRT
│
├── configs/
│   └── experiment/{exp_name}.yaml   # 單一實驗設定入口（資料路徑、模型、超參數）
│
├── tests/
├── requirements.txt
└── CLAUDE.md
```

**設計原則**
- `datasets/raw` 唯讀，任何前處理輸出到 `processed`，避免污染原始資料。
- `models/checkpoints/{exp_name}/` 每個實驗獨立資料夾，避免權重互相覆蓋。
- `output/` 依任務類型分層（inference / eval / logs），方便清理與追蹤。
- `configs/experiment/` 是單一入口：一個 yaml 決定資料集版本、模型設定、超參數，train/eval/infer 都讀同一份，避免參數散落各處對不齊。

### 主腳本架構

共同原則：`train.py`、`eval.py`、`infer.py` 只負責流程控制（讀 config → 呼叫 `src/engine` → 存結果），核心邏輯放在 `src/engine/`，避免腳本肥大難測試。三者都吃同一份 `configs/experiment/{exp_name}.yaml`，用 `--exp` 指定，不允許用一堆零散 CLI flag 覆寫關鍵參數。

**`scripts/train.py`**
- 輸入：`--exp {exp_name} [--resume]`
- 流程：load_config → build_dataset → build_model → engine.train_loop（每個 epoch 存 checkpoint，best.pt 依 val metric 更新）
- 輸出：`models/checkpoints/{exp_name}/{best,last}.pt`、`output/logs/train_{exp_name}_{timestamp}.log`、`metrics.json`

**`scripts/eval.py`**
- 輸入：`--exp {exp_name} --checkpoint {best|last|path} --split {val|test}`
- 流程：load_config → build_dataset(split) → load_model+checkpoint → engine.eval_loop（計算 mAP/HOTA/AssA 等指標）
- 輸出：`output/eval/{exp_name}_{split}_{timestamp}/`（metrics.json、per_class_report.csv、confusion_matrix.png）
- 注意：必須記錄用的是哪個 checkpoint + 哪個資料集版本，寫進報告，避免結果不可回溯。

**`scripts/infer.py`**
- 輸入：`--exp {exp_name} --checkpoint {best|path} --source {圖片/影片/資料夾} [--save-format {json,video,image}]`
- 流程：load_model（exported 優先，沒有則用 .pt）→ 前處理 → 推論 → 後處理（NMS/座標還原/tracker）
- 輸出：`output/inference/{exp_name}_{timestamp}/`（results.json、visualized/ 可選）

**`scripts/export.py`**
- 輸入：`--exp {exp_name} --checkpoint {best|path} --format {onnx|tensorrt} [--precision fp16|int8]`
- 流程：pt → onnx →（可選）TensorRT engine，轉換前後跑一次 sanity check（同輸入比對輸出誤差），寫進 log
- 輸出：`models/exported/{exp_name}/model.{onnx,engine}`

## 語氣
- 中性、清楚、可以直接套用。
- 不要用「不是 A 而是 B」這種對比句型。
- 不要勵志收尾。
- 不確定或有風險的地方要明確標示，不要淡化。
