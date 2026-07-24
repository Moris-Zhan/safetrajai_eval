# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概觀

CCTV 交通監控的物件偵測＋多物件追蹤（MOT）評估流程：YOLO11 偵測 → class-aware BoTSORT / ByteTrack 追蹤 → TrackEval 計算 HOTA / CLEAR / Identity 等指標，輸出 Excel 報表。追蹤類別固定 6 類：`person`、`car`、`scooter`、`small_truck`、`bus`、`truck`。

無 `requirements.txt` / `pyproject.toml`、無測試、無 lint/format 設定。專案已於 2026-07 導入 git 版本控制；實驗版本靠 `config/*.yaml` 的情境檔名區分（如 `config/botsort_reid.yaml`）。主要依賴：`ultralytics`（YOLO + 內建 `BYTETracker`/`BOTSORT` 基底類別）、`torch`/`torchvision`、`opencv-python`、`numpy`、`scipy`、`openpyxl`、`pyyaml`。`trackeval/`（根目錄）是 vendored 的 TrackEval 套件並以 `import trackeval` 使用；舊有的重複副本 `trackers/trackeval/` 已歸檔到 `archive/trackers_trackeval_duplicate/`。

專案已依 Detection / Tracker / Re-ID / grid_search 四個模組重組並統一放進 `module/`（2026-07），詳見下方「架構」。

## 專案文件（`markdown/`）
- `markdown/Project.md`：專案總覽、範圍、核心流程圖、已知限制。
- `markdown/USAGE.md`：四大模組（Detection/Tracker/Re-ID/Evaluation）+ `TrackingEvalPipeline`、grid search、`visualize_gt_on_*.py`、`yolo_reid_trainer` 的實際使用範例。
- `markdown/TEST.md`：手動驗證流程（無自動化測試，改動後照此手動驗證）。
- `markdown/VERIFY.md`：已驗證過的實驗結論（ReID 開關、GMC 方法比較、FPS 比較等）與資料來源，避免重複測試。
- `markdown/NEED.md`：已知待辦與落差清單。

開始改動前先掃過 `markdown/VERIFY.md`，避免重工已驗證過的結論；改完後照 `markdown/TEST.md` 的流程驗證。

## 常用指令

沒有 CLI 套件包裝，都是直接執行根目錄腳本：

```bash
# 完整流程：class-aware tracking（第一階段）+ TrackEval 評估（第二階段）
# 不指定 --config 時預設 config/bytetrack.yaml
python track_eval.py
python track_eval.py --config config/bytetrack.yaml          # 情境 1：ByteTrack + 評估
python track_eval.py --config config/botsort_noreid.yaml     # 情境 2：BoTSORT（ReID 關閉）+ 評估
python track_eval.py --config config/botsort_save_reid_crops.yaml  # 情境 3：BoTSORT + 蒐集 Re-ID 訓練用裁切（只跑 stage1）
python track_eval.py --config config/botsort_save_reid_crops_video.yaml  # 情境 3 變體：來源改用 video/ 底下不同路口影片
python track_eval.py --config config/botsort_reid.yaml       # 情境 4：BoTSORT + ReID 開啟 + 評估

# 針對單一類別做追蹤參數 grid search（以 TrackEval 的 HOTA.AssA_mean 為 scoring）
python module/grid_search/grid_search_bytetrack.py --target-class scooter          # 情境 5，預設 config/grid_search_bytetrack.yaml
python module/grid_search/grid_search_botsort.py --target-class scooter --tracker botsort   # 情境 6，預設 config/grid_search_botsort.yaml
python module/grid_search/grid_search_botsort_reid.py --target-class person   # 含 ReID 開關的版本，預設 config/botsort_reid.yaml
# 常用旗標：--dry-run（只列組合數不執行）、--resume（跳過 CSV 已有結果）、
#          --max-videos（限制評估用影片數）、--params-subset（只搜尋指定參數）

# 疊圖視覺化 GT / 追蹤結果到影片
python visualize/visualize_gt_on_video.py --trackeval_root <path> --video_path <path> --output_video <path>
python visualize/visualize_gt_on_class_video.py --trackeval_root <path> --video_path <path> --output_video <path>
```

六支情境設定檔都在 `config/`（見下方「設定檔」）。跑單一序列或除錯時，改對應 `config/*.yaml` 的 `paths.SOURCE_VIDEO_DIR`、`eval_flags`（`run_det_eval` / `run_track_eval`）縮小範圍，比另外加 CLI flag 更快。

## 架構

專案依 **Detection / Tracker / Re-ID / grid_search** 四個模組重組，2026-07-23 再次搬移統一放進 `module/` 底下、彼此同一層。另有一個橫跨 Detection eval 與 Tracker eval 的共用 `evaluation/` 模組（不屬於這四者之一，故不在 `module/` 底下，留在根目錄）。詳細架構圖見 `markdown/Project.md`。

```
module/detection/ — YOLO 推論封裝（detector.py）。訓練腳本目前不在本專案範圍，見 markdown/NEED.md。
module/tracker/   — class-aware BoTSORT / ByteTrack（原 trackers/ → tracker/ → module/tracker/）
module/reid/      — Re-ID 特徵抽取（encoders.py：`ReID`、`Custom_ReID`）、YOLO-ReID 訓練工具（yolo_reid_trainer.py，
                     模型架構與訓練腳本說明見 module/reid/HANDOFF.md）
evaluation/       — DET/TrackEval 共用 pipeline（config_loader、dataset_prep、mot_converter、
                     trackeval_runner、excel_export、mot_format、pipeline）
utils/            — 通用小工具（visualization.py）+ model.py（見下方「⚠️ pickle 相容性」）
track_eval.py     — CLI 入口，TrackingEvalPipeline 類別（stage1 tracking → stage2 TrackEval）
trackeval/        — vendored TrackEval 第三方套件本身（不動）
archive/          — 歸檔區：past_conf/、past_py_conf/、trackers_trackeval_duplicate/、past_track_eval_orign.py
```

> `module/grid_search/` 曾是第四個並列模組（`botsort`、`bytetrack`、`botsort_reid` 三支腳本），2026-07-24 併回 `module/tracker/` 未再列於此樹狀圖，實際檔案位置仍是 `module/grid_search/*.py`——若之後有調整以此處樹狀圖與實際目錄為準。

### ⚠️ pickle 相容性：`utils/model.py` 不可搬移或改名
`models/class_reid_weights/*/best_full_model.pt` 是用 `torch.save()` 存整個模型物件（非 state_dict），`Custom_ReID`（`module/reid/encoders.py`）用 `torch.load(..., weights_only=False)` 載入時，pickle 會用**存檔當下的完整模組路徑字串**（`utils.model.xxx`）去 `import` 對應類別定義。因此：
- `utils/model.py` 必須留在 `utils/model.py`（`ReIDNet` 等類別被這兩個權重檔引用，經 `module/tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支載入，不能移動）。
- 之後若要調整位置，必須先用舊路徑載入權重、`torch.save` 成 state_dict 或用新路徑重新 pickle，而非直接搬檔案。

> **2026-07-24 更新**：`reid_models/`（vendored fast-reid 風格套件）、`SOTA_ReID`（`module/reid/encoders.py`）、`models/reid_model_weights/`（FastReID/TransReID 權重）已全數移除。這是先前用來討論同一個 pickle 相容性限制的第三個範例，現在已不適用；上面只留 `utils/model.py`/`Custom_ReID` 這組還存在的限制。移除原因與過程見 `NEED.md`。

### 設定檔（2026-07-24 統一放進 `config/`，固定 6 個情境檔 + 1 個變體）
`module/tracker/track_config_*.yaml` 與根目錄 `config.yaml` 已全數移除／併入，改成 `config/` 底下固定 6 個「情境設定檔」，每個都是完整、獨立可用的 config（一般設定 + 內建 `class_tracking:` 章節），不再需要另外指定 `--track-config`：

| 檔案 | 情境 | tracker | ReID | stage1 | stage2 |
|---|---|---|---|---|---|
| `config/bytetrack.yaml` | 1 | bytetrack | 關閉 | ✅ | ✅ |
| `config/botsort_noreid.yaml` | 2 | botsort | 關閉 | ✅ | ✅ |
| `config/botsort_save_reid_crops.yaml` | 3 | botsort | 關閉 | ✅（`save_reid_crops: true`） | ❌ |
| `config/botsort_reid.yaml` | 4 | botsort | 開啟 | ✅ | ✅ |
| `config/grid_search_bytetrack.yaml` | 5（`module/grid_search/grid_search_bytetrack.py` 用） | bytetrack | 關閉 | — | — |
| `config/grid_search_botsort.yaml` | 6（`module/grid_search/grid_search_botsort.py` 用） | botsort | 關閉 | — | — |

`track_eval.py`／3 支 `module/grid_search/*.py` 的 `--config` 預設值各自對應上表；`grid_search_botsort_reid.py` 沒有專屬情境檔，預設用 `config/botsort_reid.yaml`（該腳本的 `PARAM_GRIDS` 每次試驗都會覆寫 ReID 相關欄位，起始值不重要）。舊的 `module/tracker/track_config.yaml`（base 版）與 `track_config_new.yaml`（另一個 ReID 變體，`track_buffer`/調參值跟現有 6 個情境檔不完全一樣）已歸檔到 `archive/track_config_variants/`，未直接刪除。

**第 7 個變體檔**：`config/botsort_save_reid_crops_video.yaml`（2026-07-24 新增）是情境 3 的變體——來源改成 `video/` 底下不同路口的原始影片（各取前 60 秒放進 `dataset/video_seqs/annotations_video/annotation_<seq>/<seq>.mp4`，符合 stage1 掃描結構），`det_corner` 關閉（不同路口鏡頭，原本的多邊形不適用）。用來補足既有 6 支序列裡 `bus`/`person`/`truck`/`small_truck` 身分數過少的問題。不計入固定的 6 個情境檔，因為只是資料來源不同、其餘設定完全比照情境 3。

**由 `evaluation/config_loader.py::load_config()` 讀取一般設定**（路徑、TrackEval 資料集參數、要評估的類別、偵測區域多邊形 `det_corner`、log 開關、`stages.run_stage1`/`run_stage2` 兩階段開關、`save params.save_reid_crops`/`reid_crop_dir`）。**由 `module/tracker/config.py::load_track_config()` 讀取 `class_tracking:` 章節**（class-aware tracker 的逐類別參數表：`match_thresh`、`track_buffer`、`use_reid` 等），**必須在建立 tracker 之前呼叫**，否則 `CLASS_TRACKING_PARAMS` 是空的（`track_eval.py` 的 CLI 入口已處理好順序，寫新腳本時要照做）。

**CLI 用法**：`--track-config` 預設 `None`——不指定時，`load_track_config(args.track_config or args.config)` 直接讀 `--config` 檔案本身的 `class_tracking:` 章節（6 個情境檔都內建了這個章節）。`module/tracker/config.py::load_track_config()` 內部用 `data.get("class_tracking", data)` 判斷：有 `class_tracking` 鍵就當作合併格式，沒有就當作扁平格式（`archive/track_config_variants/` 底下的舊檔仍是扁平結構，理論上還能用 `--track-config` 指到那裡，但不建議——內容已經跟現在 6 個情境檔不同步）。

**stage1/stage2 開關**：`config/*.yaml` 的 `stages.run_stage1`/`run_stage2`（預設都 `true`）控制 `TrackingEvalPipeline.run()` 要不要跑該階段。情境 3（`botsort_save_reid_crops.yaml`）把 `run_stage2` 設 `false`，只跑 tracking + 裁切蒐集，不跑 TrackEval 評估。

### `module/tracker/` — class-aware 追蹤器（核心邏輯，改動前務必先讀 `module/tracker/HANDOFF.md`）
繼承並改寫 Ultralytics 的 `BYTETracker` / `BOTSORT`。原版所有匹配閾值全域統一，且不同類別的偵測混在一起做匈牙利匹配，可能跨類別誤配（如 person 配到 scooter）。`class_aware_botsort.py` / `class_aware_bytetrack.py` override `update()`，改為**逐類別分組**做三階段匹配（各類別各自的 `match_thresh` / `second_match_thresh` / `unconfirmed_match_thresh`），track buffer、new-track 閾值、ReID 開關（僅 BoTSORT）也都依類別查表覆蓋，查表函式是 `module/tracker/config.py::get_params_for_class()`。

`HANDOFF.md` 講的是「為什麼這樣改、原版 vs 改版對照」；逐步的 `update()` 執行流程（每個階段做什麼、`get_dists_class_aware()` 怎麼算距離矩陣）分別寫在 `class_aware_botsort_flow.md` / `class_aware_bytetrack_flow.md`，兩者也整理了差異對照表。⚠️ 這兩份文件記錄了一個既有落差：`track_buffer` 在 BoTSORT 版被當作「秒」，在 ByteTrack 版被當作「30fps 基準幀數」，同一個設定值換算出的實際存活時間差 30 倍，詳見 `NEED.md`。

ReID 有兩種來源（`module/reid/encoders.py`）：`ReID`（用 YOLO 模型本身當 encoder，權重在 `models/yolo-reid/`，模型架構與訓練腳本說明見 `module/reid/HANDOFF.md`）與 `Custom_ReID`（自己訓練的模型，權重在 `models/class_reid_weights/`，訓練程式碼不在本專案範圍內）。目前預設關閉 ReID —— 實驗結論是對 person 有幫助但會損害 scooter 且增加 IDSW（見 `module/tracker/__init__.py` 註解）。`module/tracker/` 依賴 `module/reid/`（BoTSORT 匹配時用 Re-ID 外觀特徵），`module/reid/` 不依賴 `module/tracker/`。

對外入口只有一個工廠函式：`create_tracker(tracker_type='botsort'|'bytetrack', frame_rate=..., **overrides)`。

### `track_eval.py` — 兩階段主流程（OOP：`TrackingEvalPipeline`）
- `Detector`（`module/detection/detector.py`）：YOLO 推論 + 逐類別信心過濾。
- `VideoSequenceTracker`（`track_eval.py`）：單一影片的 class-aware tracking 迴圈（每支影片各自建立一個 Detector + tracker 實例，與舊版函式每次呼叫都重建的行為一致）。
- `TrackingEvalPipeline`（`track_eval.py`）：
  1. `run_stage1()`：對 `SOURCE_VIDEO_DIR` 下每個序列跑 YOLO 偵測 + class-aware tracking，輸出 MOT 格式追蹤結果與（可選）標註影片，寫到 `OUTPUT_ROOT/<run_det_eval_dir>/annotation_<seq>/`。
  2. `run_stage2()`：把第一階段輸出轉成 TrackEval 期望的目錄結構（`OUTPUT_ROOT/<run_track_eval_dir>/`），呼叫 `trackeval` 逐類別計算 HOTA/CLEAR/Identity，合併結果輸出 Excel（`evaluation/excel_export.py`）到 `OUTPUT_ROOT/trackeval_output/`。

腳本頂部 docstring 有完整輸入/輸出目錄結構圖，改動輸出路徑邏輯前先讀那段。

### `evaluation/` — DET/TrackEval 共用 pipeline
- `config_loader.py`：讀 `config/*.yaml`（一般設定章節）。
- `dataset_prep.py`：序列探索、DET yaml 產生、ultralytics tracking 標籤產生。
- `mot_converter.py`：Ultralytics 追蹤標籤 → MOT 7 欄格式，處理重複幀 / 重複 ID 等髒資料問題。
- `trackeval_runner.py`：組出 TrackEval 要求的 GT/tracker 目錄結構（`seqinfo.ini` 等）並呼叫評估。
- `excel_export.py`：多類別 TrackEval 結果合併輸出 Excel。
- `mot_format.py`：`save_mot_format` / `make_det_eval_dataset`（原 `utils/track_utils.py` 拆出）。
- `pipeline.py`：`run_eval_pipeline`（原 `utils/eval_utils.py`），被 `track_eval.py` 的 `run_stage2()` 呼叫。

### `module/grid_search/` — 超參數搜尋
每種 tracker 一支腳本（botsort / botsort_reid / bytetrack），自行實作 `ParameterGrid` + 逐組評估（不依賴 sklearn Estimator 介面）。流程：動態 patch `module.tracker.config.CLASS_TRACKING_PARAMS` → 呼叫與 `track_eval.py` 相同的 tracking → `save_mot_format` → `run_trackeval_mot` 流程 → 從結果取出 `HOTA.AssA_mean` 當 score → 還原參數 → 寫入 CSV（可用 `--resume` 續跑）。`grid_search_botsort_reid.py` 雖然調的是 Re-ID 開關/參數，但本質仍是 tracker 超參數搜尋，因此放在 `module/grid_search/` 而非 `module/reid/`。

### 其他注意事項
- 原 `module/reid/training/train_utils.py`（孤兒訓練迴圈）已於 2026-07-24 移除，ReID 訓練資料（`dataset/reid/crop_dataset/`）與訓練方式正在重新設計，見 `NEED.md`。
- `archive/`：舊版本程式碼保留供對照，不是目前流程的一部分。
- `dataset/`、`runs/`、`wandb/`、`models/`、`reid_models/**/__pycache__`：權重與歷史實驗輸出，體積大但不需要通篇閱讀即可理解架構。

> ⚠️ **下方「專案架構規範」章節與本 repo 實際結構不同**：它描述的是 `scripts/train.py`／`src/`／`configs/experiment/` 這類目標範本目錄結構，但目前程式碼實際上是根目錄 `track_eval.py` ＋ `module/detection/`／`module/tracker/`／`module/reid/`／`module/grid_search/`／`evaluation/` 的組合（見上方「架構」）。新增程式碼請沿用現有慣例，除非使用者明確要求依範本重構整個專案。

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
