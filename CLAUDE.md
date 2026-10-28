# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

術語見 `CONTEXT.md`（鏡頭、序列、軌跡、身分、衝突事件等），關鍵決策見 `docs/adr/`。

## 專案概觀

離線 CCTV 交通監控評估流程：YOLO11 偵測 → class-aware BoTSORT／ByteTrack 追蹤 → TrackEval（HOTA／CLEAR／Identity）→ Excel 報表，另有 stage3 的衝突事件分析（TTC／PET＋Homography）。固定 6 類，類別 ID 是跨模組契約：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`，唯一定義在 `utils/modules/safetrajai/classes.py`（TrackEval／MOT GT 用的 1-based ID 由此推導為 `id + 1`）。

- 可安裝套件 `safetrajai`（原始碼放在 `utils/modules/`，`pyproject.toml`）。`requirement.txt` 是舊的 pip 安裝腳本，包含許多與本專案無關的套件，**不是依賴的權威來源**；依賴以 `pyproject.toml` 為準（核心為 `ultralytics==8.3.141`、`torch`、`opencv-python`、`numpy`、`scipy`、`openpyxl`、`pyyaml`、`scikit-learn` 等）。
- `utils/modules/_vendor/trackeval/` 是修改過的 vendored TrackEval（6 類別對照），透過同一份 pyproject 安裝，以 `import trackeval` 使用；視為唯讀，不要修改或格式化（見 `docs/adr/0002`）。
- `dataset/`、`models/`、`runs/`、`Sub_Module/`、`ignore/` 被 gitignore（權重、影片、實驗輸出）。
- 規格驅動的變更放在 `openspec/`（`openspec/changes/<name>/` 有 proposal、design、specs、tasks），用 `/opsx:*` 指令操作。

## 安裝

只能使用 conda env `track_eval`（不要用 uv 或另建 venv）。安裝一律加 `--no-deps`，避免 pip 把 CUDA 版 torch 換成 PyPI 版本：

```bash
conda activate track_eval
pip install -e . --no-deps
pip install pytest          # dev 依賴
```

未安裝時執行 `python track_eval.py` 會印出上面的安裝指令並以 exit 1 結束。

## 常用指令

所有指令都從專案根目錄執行（資料與模型路徑如 `models/`、`dataset/`、`config/` 相對於 repo 根目錄）：

```bash
# 主流程：stage1 tracking → stage2 TrackEval → stage3 衝突事件（由 config 的 stages.* 開關控制）
python track_eval.py --config config/tracking/bytetrack.yaml               # 不帶 --config 時預設就是這個檔；等同 python scripts/pipeline/run_pipeline.py
python track_eval.py --config config/tracking/botsort_noreid.yaml
python track_eval.py --config config/tracking/botsort_reid.yaml
python track_eval.py --config config/reid/reid_crops.yaml              # 只跑 stage1，從 reid_sources 蒐集 Re-ID 裁切到 dataset/reid/crops/
python track_eval.py --config config/collision/collision_detect.yaml        # stage1 + stage3（TTC/PET）
python track_eval.py --config config/collision/conflict_analysis.yaml       # stage1 + stage3（TTC/PET）＋衝突指標（只限有 m/px 的鏡頭）
python scripts/pipeline/run_evaluation.py --config <config>                # 只對既有 stage1 輸出跑 stage2

# 單一類別的追蹤參數 grid search（score = HOTA.AssA_mean，屬於關聯指標，不等於 HOTA）
python scripts/grid_search/grid_search_bytetrack.py --target-class scooter
python scripts/grid_search/grid_search_botsort.py --target-class scooter --tracker botsort
python scripts/grid_search/grid_search_botsort_reid.py --target-class person
#   --dry-run（只算組合數）  --max-videos 1（最小驗證）  --params-subset  --resume  --result-csv

# Homography 驗證（輸出到 models/homography/validation_output/）
python scripts/homography/validate_homography.py

# Homography 鏡頭設定（config/homography/<camera>.yaml）：遷移 → 經緯度 → 比例尺檢查；後兩者會連網
python scripts/homography/migrate_homography_config.py [--dry-run]           # tag 鍵資料 → 鏡頭鍵設定（只寫新檔，衝突時不寫）
python scripts/homography/geocode_cameras.py [--cameras <cam...>]            # 候選表 models/homography/geocode_candidates.csv
python scripts/homography/geocode_cameras.py --apply [--accept <cam...>]     # 只寫入 status=ok 的列或使用者確認過的鏡頭
python scripts/homography/check_homography_scale.py --camera <cam> --points X1,Y1 X2,Y2 --reference-m 3.5 --feature 車道寬  # 寫入 scale_check
python scripts/homography/download_satellite.py --camera <cam> --purpose verify|basemap [--replace]  # Demo Key 只能 verify

# Re-ID 資料工具：蒐集 → 標記 → 自動排除 → 抽查 → 排除 → 碎片合併 → manifest
python scripts/reid_data/detect_contamination.py                  # DINOv2 標記，輸出 dataset/reports/reid/
python scripts/reid_data/apply_auto_exclusions.py [--apply]       # 由標記整份重算 dataset/reid/auto_exclusions.csv
python scripts/reid_data/generate_identity_review.py --audit-only # 自動排除後的抽查頁＋audit_sample.csv（勾選＝排除）
python scripts/reid_data/summarize_audit.py --confirm-csv <csv...> # 每支鏡頭的不合格率（> 10% 要處理）
python scripts/reid_data/apply_review_exclusions.py --confirm-csv <csv> [--apply]  # 合併進 dataset/reid/exclusions.csv
python scripts/reid_data/extract_fragment_features.py            # 只讀碎片候選端點裁切的 DINOv2 特徵
python scripts/reid_data/build_fragment_merges.py [--sample 80 --seed 0 | --calibrate <csv> | --apply]  # 碎片合併清單
python scripts/reid_data/build_reid_split.py --version <v> [--exclude-cameras <cam...>] [--merges dataset/reid/fragment_merges.csv]  # 兩份排除清單都套用

# Re-ID 模型訓練：manifest → 本機快取（只讀一次掛載磁碟）→ 訓練 → val 評估
python scripts/reid_train/pack_manifest.py --version <v> [--reuse-version <v0>]   # ~/.cache/safetrajai/reid_packs/<v>/；--reuse-version 只補讀缺少的裁切
python scripts/reid_train/train_reid.py --version <v> --output models/class_reid_weights/<run>  # 背景執行，30 epoch 約 25～45 分鐘（單獨一個任務）
#   [--downscale-prob 0.5]（解析度增強，最短邊 --downscale-min/--downscale-max 預設 8～40px）
#   [--distractor-fraction 0.5]（干擾物取樣，讀 crops/_index/ 與 config/roi/）  [--select-by reduced_val|val]（預設縮小版 val）
python scripts/reid_train/eval_reid.py --weights models/class_reid_weights/<run>/model.pt --version <v>  # val mAP／Rank-1，只作參考
python scripts/reid_train/fusion_distance_stats.py --out <json>   # 調參序列 GT 的外觀距離分布、AUC、TPR@FPR5%、融合門檻候選（拒絕保留序列）
#   [--legacy-crop]（量測參考用的舊裁切）  [--reference <json> --noise-reference <seed_json> <base_json>]（外觀分離度關卡判定）
python scripts/reid_train/acceptance_grid.py configs --name <run> [--subsets tune holdout] --set CLASS.KEY=VALUE ...  # 追蹤驗收設定（以 botsort_reid_timm.yaml 為底）
#   執行：python track_eval.py --config runs/reid_domain/accept/configs/<run>_tune.yaml > runs/reid_domain/accept/<run>/tune/run.log 2>&1
python scripts/reid_train/acceptance_grid.py summarize --baseline baseline --runs <run...> --out <prefix>  # 逐類別彙總與 spec 判定（CSV＋Markdown）

# 在序列影片上疊 GT／追蹤結果
python scripts/visualization/visualize_gt_on_video.py --trackeval_root <path> --video_path <path> --output_video <path>
```

### 測試與回歸驗證

- `pytest`：smoke 測試（`tests/test_smoke.py`），涵蓋全部模組 import、全部 config 載入、類別 ID 契約、tracker override 方法的 AST 不變性、`sys.path` 掃描、`scripts/**/*.py --help`。幾秒內跑完，任何結構調整後都要跑。
- 黃金比對（golden output）：`tests/golden/run_all.sh <run>` 以 6 個情境＋grid search＋homography 驗證跑一次完整流程（平行執行，約 16 分鐘，輸出在 `runs/refactor_golden/<run>/`），再用 `python tests/golden/compare.py conflict1 <run>` 與基準比對（基準於 2026-10-14 由 `georef1` 改為 `conflict1`：新增 `conflict_analysis` 情境，多出 `conflict` 類別；鏡頭設定的 `latlon`／`scale_check`／check points 改變時，homography 類別與 `conflict` 類別的差異屬於預期，需重建基準。原因見 `tests/golden/TOLERANCE.md`）。基準自比的容忍度為 0（見 `tests/golden/TOLERANCE.md`），任何非零差異都代表行為改變。
- **禁止大量 I/O**：`dataset/`、`models/`、`runs/` 位於掛載磁碟，檔案數以百萬計（例如 Re-ID 裁切）。禁止對整個目錄樹逐檔讀取內容，例如全量 `sha256sum`／`md5sum`、全量 `du`、`find … -exec cat`、大範圍 `rsync --checksum`，這類操作容易造成磁碟損壞。需要核對搬移或複製結果時，只比對路徑清單與檔案數（`find -type f | wc -l`、只列路徑不讀內容），同一磁碟內的 `mv` 不需要核對內容；真的需要比對內容時，只針對少數指定檔案（例如單支影片或權重），並先向使用者確認範圍。長時間的背景讀寫任務要先告知使用者。
- **隔離慣例**：有疑慮、不可使用的權重或資料一律用 `mv` 移到所在目錄下的 `temp/`（不刪除），並在 `temp/DO_NOT_USE.md` 逐項記錄原路徑、日期、原因與還原指令；config 不得引用 `temp/` 內的路徑（smoke 測試檢查）。評估資料集的人工 GT 內容不可修改。
- 改動演算法時，驗證方式仍是實際跑受影響的 stage 再比對輸出。想縮小範圍時，直接在 config 的 `paths:` 加 `SOURCE_VIDEO_DIR` 指定單一序列，或調整 `eval_flags.run_det_eval`／`run_track_eval`。全量序列、grid search 和重跑 stage1 都很耗時，也可能覆寫既有輸出，執行前要先確認範圍。

## 架構

```
track_eval.py                相容入口 wrapper → safetrajai.cli.run_pipeline.main()
scripts/                     CLI，依功能分資料夾：pipeline/（run_pipeline、run_evaluation）、grid_search/、homography/（validate_homography 等）、
                             visualization/（visualize_gt_*）、picker/（ROI 選點器）、reid_data/（Re-ID 資料切分與稽核）、reid_train/
utils/modules/safetrajai/
  paths.py                   唯一的 PROJECT_ROOT／CONFIG_DIR（需要定位 repo 內檔案時一律用它）
  classes.py                 類別 ID 契約（CLASS_NAMES、MOT_CLASS_ID）
  cli/run_pipeline.py        argparse + main()：先 load_track_config() 再 load_config() → TrackingEvalPipeline
  pipeline/                  tracking_pipeline（TrackingEvalPipeline，run_stage1/2/3）、sequence_tracker（單序列迴圈）、
                             reid_crops、video_io
  detection/                 Detector：單幀 YOLO 推論 + 逐類別信心門檻（class_conf_thresh）過濾
  tracking/                  class-aware ByteTrack / BoTSORT（繼承 ultralytics trackers），唯一入口 create_tracker()
  reid/                      推論端：ReID（YOLO 當 encoder）與 Custom_ReID（models/class_reid_weights/）兩條路徑；
                             training/（manifest 資料集與本機快取、PK 取樣、骨幹＋BNNeck、CE＋三元組、val 評估）
  roi.py                     以鏡頭為鍵的 ROI 解析（config/roi/<camera>.yaml），stage1 繪製／裁切、stage2、stage3 共用
  grid_search/core.py        grid search 共用邏輯，scripts/grid_search/grid_search_*.py 只保留 PARAM_GRIDS + argparse
  collision/                 stage3：MOT 軌跡 → TTC/PET 衝突事件、熱區圖；homography/ 子套件負責轉換／點選／驗證、
                             以鏡頭為鍵的設定（camera_config）、公尺換算（georef）、經緯度查詢（geocode）、
                             底圖下載（satellite）、比例尺比對（scale_match）；conflict/ 子套件是公尺單位的衝突指標分析
  secrets.py                 外部服務金鑰（環境變數優先，其次 repo 根目錄 .env；不輸出金鑰值）
  evaluation/                stage2 共用：config_loader、dataset_layout（路徑組裝純函式）、mot_converter、trackeval_runner、
                             ignore_region（TrackEval 前的忽略區域前處理）、dataset_prep（人工 GT 的 detection 評估資料集）、excel_export
  visualization/drawing.py   ROI 與追蹤框繪製
utils/model.py               pickle 相容 shim（見下方 Re-ID）
utils/modules/_vendor/trackeval/       vendored TrackEval
config/                      情境設定檔（每檔自包含，依功能分 tracking/、grid_search/、reid/、collision/）+ datasets.yaml（dataset id 對照表）+ roi/（以鏡頭為鍵的 ROI）+ homography/、homography_checkpoints/（以鏡頭為鍵）
tests/                       smoke 測試與 golden 比對工具
```

### 設定檔

- 每個情境 `config/<功能>/*.yaml` 都是完整、自包含的情境設定：一般設定（`paths`、`models`、`stages`、`eval_flags`、`tracking_params`…）加上 `class_tracking:` 章節。這些區塊名稱是穩定介面，不能改名。新實驗請另外建檔，不要覆寫既有的基準情境。
- 一般設定由 `safetrajai/evaluation/config_loader.py::load_config()` 讀取；`paths.dataset: <id>` 會去查 `config/datasets.yaml`，取得 `root`／`run_det_eval_dir`／`run_track_eval_dir`，也可以逐欄位顯式覆寫。每個資料集還有 `role`（`eval`：附人工 GT、只用於評估；`source`：用來產生訓練資料）、`layout`（`annotated` 是 `annotation_<seq>/` 結構，`raw` 是 `dataset/raw/videos/<camera>/` 平放影片）與 `cameras`；eval 與 source 的鏡頭不得重疊，評估鏡頭（目前是 C000025）不能進任何訓練資料，smoke 測試會檢查。
- ROI（關注區域）屬於鏡頭，定義在 `config/roi/<camera>.yaml`（`camera`、`image_size`、`polygon`），依序列名稱第一個空白前的鏡頭編號查找；沒有檔案時以整張影像為 ROI，`image_size` 與影片不符時報錯。情境 config 不再支援頂層 `det_corner`，出現會直接報錯。
- `class_tracking:` 由 `safetrajai/tracking/config.py::load_track_config()` 讀取，寫入模組層級的 `CLASS_TRACKING_PARAMS`。**這個函式必須在 `create_tracker()`／`Detector` 建立之前呼叫**；順序錯了不會報錯：模組在 import 時已載入預設的 `config/tracking/bytetrack.yaml`，所以只會悄悄用到這份預設參數（或先前載入的設定），而不是指定的情境。BoTSORT 在建立時就固定類別集合（`CLASS_KEYS`），之後再載入設定不會更新已建立的 tracker，要使用新設定必須重新建立。`--track-config` 不指定時，會直接讀 `--config` 檔本身的 `class_tracking:`。
- `stages.run_stage1/2/3` 控制要跑哪幾個階段；`eval_flags.run_det_eval/run_track_eval` 只控制 stage2 裡面的子工作，兩者語意不要混用。

### 路徑慣例（`safetrajai/evaluation/dataset_layout.py`）

SOURCE 端（序列影片、人工 GT）沿用舊命名 `annotation_<seq>/`、`mot_annotation_<seq>/`；OUTPUT 端（本專案產生的 stage1/stage2 目錄）直接用 `<seq>/`，不加前綴。所有路徑都透過 `dataset_layout.py` 的函式組出來（`safetrajai/pipeline/`、`safetrajai/grid_search/core.py`、`safetrajai/collision/` 共用），不要自己拼字串。`safetrajai/pipeline/tracking_pipeline.py` 開頭的 docstring 有完整的輸入／輸出目錄結構圖。repo 內檔案（`config/` 等）一律透過 `safetrajai.paths` 定位，不要用 `Path(__file__).parents[N]`，也不要修改 `sys.path`。

### 軌跡格式契約（stage1 → stage2／stage3）

- 穩定的下游輸入是 `trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt`：每行 7 欄、無 header，`frame,track_id,left,top,w,h,score`，像素座標、以左上角為原點，`frame` 從 1 開始。檔案內沒有 class 欄，類別看資料夾名稱決定。
- 6 個類別都必須產生檔案，沒有結果時建立 0-byte 空檔。stage1 軌跡**不依 ROI 過濾**，保留全部追蹤結果；限縮範圍由下游自行套用（見 `docs/adr/0005`）：stage3 與 Re-ID 裁切以框中心點硬過濾，stage2 以忽略區域處理（UA-DETRAC 式）——ROI 外的人工 GT 標為 ignore，預測與全部 GT 做 IoU ≥ 0.5 的一對一匹配；匹配到 ROI 內 GT 的預測保留，其餘預測中匹配到 ignore GT 或中心點在 ROI 外者移除，再移除 ignore GT，所以 FP 只來自 ROI 內（代價與風險見 `specs/evaluation-roi-filtering`）。這一步在 TrackEval 之前、只改 stage2 輸出目錄中的副本（原始 tracker 副本備份在 `trackers_raw/`），不修改 vendored TrackEval。
- stage2 的 detection 評估以評估資料集的人工 GT 為基準（在 `OUTPUT_ROOT/det_eval/<seq>/` 組出資料集），不再使用 stage1 的追蹤標籤；目前不套用 ROI，報表會標示「ROI：未套用（全畫面）」。
- `track_id` 只在單一序列、單一類別內唯一（每段序列都新建 tracker，而且是逐類別配號），跨序列要用「序列名＋track_id」當鍵。
- GT 有 class 欄，但只給 TrackEval 比對用；stage3 和其他下游分析一律讀 tracker 輸出，不讀 GT。

### Tracker（改動前先讀 `utils/modules/safetrajai/tracking/HANDOFF.md` 與同目錄的 `class_aware_*_flow.md`）

- 核心設計：偵測結果**依類別分組後才做三階段匈牙利匹配**，每個類別有自己的 `match_thresh`／`second_match_thresh`／`unconfirmed_match_thresh`／`new_track_thresh`／`track_buffer`／`use_reid`，由 `get_params_for_class()` 查表取得。不能讓不同類別共用同一個 matching matrix。
- **不可更動（使用者明確指示，見 `docs/adr/0004`）**：`ClassAwareBOTSORT` 與 `ClassAwareBYTETracker` 中 override 自 ultralytics 的方法，包括 `__init__`、`update()`、`remove_duplicate_stracks()`、`init_track()`（僅 BoTSORT），除 ADR 0009 明列的兩個 `update()` 保留期限敘述外，方法本體一字不能改，也不能搬到其他類別或抽成共用函式。整個檔案可以搬位置，但只能改檔案層級的 import；smoke 測試會以 AST 對照 `tests/golden/BASE_COMMIT` 檢查。這條優先於任何去重複或修 bug 的目標。本專案自己新增的 `get_dists_class_aware()`、`_center_distance()`，以及 `matching.py`、`config.py`、`__init__.py` 不受這條限制。
- `track_buffer` 在 BoTSORT 與 ByteTrack 都是正整數保留期限幀數，不依幀率換算（ADR 0009，取代 ADR 0008）。載入設定時每個類別都必須有有效值（沒有該類別設定時取預設參數），浮點值與布林值會報錯。碎片候選窗直接用 `track_buffer + margin_frames`；Re-ID 閘門維持每幀計算。BoTSORT 的 `cls_buffer_sec` 是歷史名稱，方法內舊註解仍依凍結規則保留。
- GMC 只在 BoTSORT 有作用；ReID 只透過 BoTSORT 的逐類別 `use_reid` 啟用，預設關閉。
  - 啟用條件（spec `reid-model`）：在調參序列上與驗收基準比較，不變差，而且 IDSW 減少或 IDP 上升；只有 AssA／TP 上升、沒有 GT 的類別不啟用；沒有類別符合時全部關閉。目前 `botsort_reid_timm.yaml` 的 6 類都關閉。
  - 本地訓練的 Re-ID 模型（change `train-reid-timm`）在原本的融合下幾乎沒有作用：外觀距離只作用在 IoU 已重疊的配對上。
  - change `reid-center-gate-fusion` 新增選用的逐類別參數（未設定時距離矩陣逐位元不變）：`reid_center_gate_px`／`reid_center_gate_px_per_frame`（只對遺失中的軌跡，以中心距離閘門擴大外觀的作用範圍；放寬量是每幀像素 × 遺失幀數，舊名稱 `reid_center_gate_px_per_s` 會報錯）與 `appearance_veto_sim`（外觀否決）。在 C000025 上閘門讓 car、scooter 的 IDSW 明顯增加，否決沒有效果。瓶頸在模型：C000025 上同一條 GT 軌跡相隔 ≥ 1 秒的外觀距離，和鄰近的不同物體高度重疊（`fusion_distance_stats.py` 可量測）；下一步應改善模型或資料，而不是再調融合。
  - change `align-reid-training-domain` 依此在訓練與推論端對齊評估鏡頭（解析度增強、干擾物取樣、推論改用緊框），並在追蹤驗收前加上外觀分離度關卡（見下方 Re-ID）。四個候選都沒有通過關卡，沒有做追蹤驗收，config 不變。
  - 外觀門檻的方向：`appearance_thresh` 與 `appearance_veto_sim` 都是相似度方向（容許的外觀距離上限為 1 − 值，值越大越嚴）；新增外觀門檻時沿用這個方向。
  - 啟用 ReID 會讓每幀追蹤時間增加數倍（凍結的 `init_track` 對每個框各呼叫一次 encoder）。
- 驗證方式：先跑單一序列或目標類別，比較同一基準下的 HOTA、AssA、IDSW，再用疊圖影片檢查 ID switch 和跨類別誤配。

### Re-ID

- **`Custom_ReID` 支援兩種權重格式**（`models/class_reid_weights/` 之下的路徑才會走 `Custom_ReID`）：
  - 新格式 `safetrajai-reid-v1`（`scripts/reid_train/train_reid.py` 產生的 `model.pt`）：state_dict＋設定（`arch`、`embed_dim`、`input_size`、`mean`、`std`、`manifest_version`），以 `weights_only=True` 讀取、不依賴 pickle 類別路徑；每個框以 `xywh2xyxy` 後的框**緊框**裁切（`save_one_box` 的 `gain=1.0, pad=0`，與訓練裁切一致；change `align-reid-training-domain`），letterbox 到 `input_size`、正規化、批次推論，回傳每個框一個 L2 正規化特徵。`Custom_ReID(..., legacy_crop=True)` 恢復舊的 `gain=1.02, pad=10`，只供 `fusion_distance_stats.py --legacy-crop` 量測參考，情境設定無法開啟。同目錄的 `train_record.json` 記錄資料集版本、超參數、種子、解析度增強與干擾物取樣設定，以及每個 epoch 的原始 val 與縮小版 val 的 mAP。
  - 目前的模型：`models/class_reid_weights/dinov2s_v3/model.pt`（DINOv2 ViT-S/14，`v3`，val mAP 0.783），是 `models/class_reid_weights/` 下唯一保留的模型；`dinov2s_v2`（`v2` 對照組）與下列實驗權重已於 2026-10-12 隔離到 `models/class_reid_weights/temp/`（HF `Moris555/safetrajai-source` 同步，原佈局保留在 tag `reid-v3-2026-10-08`）。情境設定 `config/tracking/botsort_reid_timm.yaml`（R1 追蹤參數；目前 6 類 ReID 都關閉，等同驗收基準，檔頭記錄重新驗收的方式）。
  - 已隔離的實驗權重（change `align-reid-training-domain`；改以物體配對判定後 `_lowres`、`_both` 通過外觀分離度關卡，但 `_both` 的追蹤驗收沒有任何類別通過，沒有情境設定使用）：`dinov2s_v3_lowres`（解析度增強）、`dinov2s_v3_distractor`（干擾物取樣）、`dinov2s_v3_both`（兩者）、`dinov2s_v3_seed1`（只換種子、`--select-by val`，用來估雜訊底線）。
- **外觀分離度關卡**（spec `reid-model`）：新模型或新裁切方式在追蹤驗收前，先以 `fusion_distance_stats.py` 在調參序列上量測。正樣本是同一條 GT 軌跡、間隔 5 幀～該類別 `track_buffer` 幀；負樣本是干擾物（同幀、同類別、中心距離 ≤ 2×`norm_factor`）；GT id < 0（沒有身分）的框不參與量測。參考是 `dinov2s_v3` 舊裁切；car 與 scooter 的 AUC、TPR@FPR5% 都要高出最小改善量（AUC 0.01、TPR 0.02 與 2 倍雜訊底線取大者）。證據以**不同 GT 物體配對**計算（同一組物體的多幀只算一組）：car、scooter 少於 10 組時不能通過；其他類別至少 10 組才檢查 AUC 下降 ≤ 0.02，不足者只列在 `unchecked_classes`（change `separability-gate-object-pairs`，取代原本的「≥ 30 對幀配對」）。val mAP 不能當關卡依據。
  - 結果：lowres／both 讓 car、scooter 大幅改善（both：car AUC +0.031、TPR +0.145），但 small_truck AUC 下降約 0.03（只有 43 對干擾物，下降集中在最短邊 < 16px 的 13 對），依規則不通過；distractor 與緊框 `dinov2s_v3` 的 car 未達最小改善量（TPR@FPR5% 反而下降）。結果不代表追蹤 IDSW 會改善（沒有做追蹤驗收）。後來發現 small_truck 的 43 對只來自 3 組物體配對（其中一組含 GT id -1），change `separability-gate-object-pairs` 因此改以物體配對計算證據並重新判定：`_lowres`、`_both` 通過，`_both` 進入追蹤驗收；40 組網格（閘門、否決、原本融合 × person／car／scooter／small_truck／truck）**沒有任何類別通過**——原本融合與否決完全沒有作用，閘門讓 car（IDSW 2 → 最多 18）、scooter（4 → 28～40）明顯變差。外觀分離度改善沒有轉換成追蹤改善；`botsort_reid_timm.yaml` 維持 6 類關閉。
  - 訓練新參數預設關閉，但 `--select-by` 預設是縮小版 val；要重現 `dinov2s_v3` 的設定時加 `--select-by val`。干擾物取樣比例受干擾物數限制（中位數 5），P=32 的理論上限約 0.196，實測約 0.18。
  - 舊格式：完整模型 pickle，行為不變（只處理第一個框）。
- **`utils/model.py` 是 pickle 相容 shim，不可刪除或改名（見 `docs/adr/0001`）**：舊格式的 `best_full_model.pt` 是用 `torch.save` 存下的完整模型 pickle，`Custom_ReID` 用 `weights_only=False` 載入時，會依 `utils.model.<Class>` 找類別。實作在 `safetrajai/reid/model.py`，shim 只 re-export；pyproject 把 `utils` 登記為頂層 package，所以任意工作目錄下都能載入。真的要移除 shim，先用舊路徑載入、轉存成 state_dict 並確認可用。
- 不要再引入已經移除的 SOTA_ReID、FastReID、TransReID。
- 舊的 Re-ID 資料與權重（`dataset/reid/temp/`、`models/yolo-reid/temp/`、`models/class_reid_weights/temp/`）已確認無效並隔離；不要引用或拿來訓練。
- Re-ID 資料已由 change `rebuild-reid-dataset` 從 `dataset/raw/videos/` 重建（流程、目錄與資料量見 `scripts/reid_data/README.md`）：
  - 裁切在 `dataset/reid/crops/{class}/{seq}_track{id}/{frame:06d}.jpg`，每支序列一份 metadata 索引 `crops/_index/<seq>.csv`；只從 `role: source` 的鏡頭蒐集，評估鏡頭一律拒絕，每支鏡頭都必須有 ROI 檔。蒐集設定是 `config/reid/reid_crops.yaml`。
  - **訓練一律讀 `dataset/reid/splits/<version>/manifest.csv`**（`crop_path` 相對於 `crops/`），不要直接掃 `crops/`；排除清單、每身分 4～30 張、依鏡頭與日期切分（`config/reid/reid_split.yaml`）只在 manifest 中套用。訓練與 val 評估只從本機快取 `~/.cache/safetrajai/reid_packs/<version>/` 讀取影像（`pack_manifest.py` 依 manifest 從掛載磁碟各讀一次），不要讓每個 epoch 直接讀 `crops/`。
  - 標記、排除、抽查都只寫紀錄，不移動裁切（排除與隔離的差異見 `docs/adr/0006`）。品質把關以**自動排除為主、人工只抽查**（`docs/adr/0007`）：`auto_exclusions.csv` 由標記整份重算、`exclusions.csv` 只累加人工結果，manifest 兩份都套用；不要改回全量人工審查，而是加大抽查樣本或新增規則。
  - `v2`（22,164 個身分、156,141 張）與 `v3` 都可訓練；兩者訓練出的模型在追蹤上表現相同，平手依規則採用 `v3`。`v1` 是審查前的基準，只供比較。
  - 遮擋碎片由 change `merge-occlusion-fragments` 離線合併：只寫 `dataset/reid/fragment_merges.csv`（不動裁切），manifest 以 `--merges` 在排除之後套用。
    - 同一序列、同一類別內，以時間、外插位置加上 DINOv2 外觀門檻（`config/reid/fragment_merge.yaml`，以人工確認校準到 ≥ 95%）一對一配對。
    - 時間條件以幀數計算（ADR 0009）：幀差 ≤ `track_config` 的逐類別 `track_buffer` ＋ `margin_frames`，不依追蹤器或序列幀率換算，外插用每幀速度。
    - 只靠時間與位置不可靠（C000054 車流密集）。
    - `v3`＝`v2`＋合併（22,223 個身分、156,186 張）。
    - `v3` 的 `fragment_merges.csv` 是用舊的秒數規則產生的，決定不重建。改用幀數規則後少 2 筆，都在非 10fps 序列，見該 change 的 `verification.md`；下次重建時才套用新規則。

### Grid search

每個 trial 的順序固定：深拷貝原始參數 → patch `CLASS_TRACKING_PARAMS` → tracking → MOT 輸出 → TrackEval → 取出 `HOTA.AssA_mean` → 寫入 CSV → 還原參數。無論成功、跳過或拋出例外，都必須還原參數。`--resume` 只能跳過完全相同、而且已經完成的組合。

### 衝突事件／Homography（stage3，程式碼沿用 `collision` 命名）

- 程式碼與設定檔中的 `collision`（`safetrajai/collision/`、`config/collision/collision_detect.yaml`）指的是**衝突事件**（TTC／PET 低於門檻的接近事件），不是實際碰撞（見 `docs/adr/0003`）。
- 輸入只讀 stage1 的 7 欄 MOT 軌跡。事件請讀 `ttc_events.{json,csv}`、`pet_events.{json,csv}`，不要依賴舊 log 的 `colision_occur_sec`（PET 的這個欄位可能記的是 frame，而不是秒）。
- Homography 屬於鏡頭，設定在 `config/homography/<camera>.yaml`（鏡頭編號為鍵，與 ROI 相同；由 `scripts/homography/migrate_homography_config.py` 從 26 個既有矩陣產生）。查找順序：設定的 `matrix_tag` 指向的既有矩陣（`models/homography/matrices/`，檔案不存在時報錯）→ 設定內的人工對應點 `cctv_points`／`map_points` → 都沒有時直接用 CCTV 原始像素座標。舊的 `config/homography_points/<seq>.yaml` 不再讀取，存在時報錯。套用矩陣時，TTC 的 bounds 要用衛星圖尺寸；速度和方向要從轉換後的位置重新差分；輸出的事件點則要用反矩陣投影回 CCTV 像素座標。
- 單位（ADR 0010）：只有指定既有衛星矩陣、有經緯度、而且 `scale_check: passed` 的鏡頭，`georef.meters_per_pixel()` 才回傳公尺換算（`156543.03392·cos(緯度)/2^zoom/scale`）；其餘一律維持像素，不得退回預設值。check points 驗證以鏡頭為鍵（`config/homography_checkpoints/<camera>.yaml`，少於 4 點為點數不足）。未經實測校正的事件數只能當作技術訊號，不能當成正式的安全結論。
- 衝突指標分析（change `conflict-metrics`，ADR 0011）：`conflict_analysis.enabled` 為 true 時，`run_stage3()` 在既有 TTC／PET 之後另外呼叫 `safetrajai.collision.conflict.run_conflict_analysis_for_sequence()`，既有輸出逐位元不變；沒有這個區塊時視為關閉。
  - 只在 `meters_per_pixel()` 可用的鏡頭計算（目前 C000025、C000043、C000054、C000085），其餘序列只寫 `summary.json` 並記錄 `skipped_reason`（`no_homography`／`no_meters_per_pixel`），不退回像素。位置取框的底邊中點，軌跡鍵為 `(class, track_id)`。
  - 輸出在 `collision_output/<seq>/conflict/`：`pair_frames.csv`、`conflict_events.{csv,json}`、`pet.csv`、`chains.csv`、`summary.json`、`conflict_heatmap.png`；點座標有 `_m`／`_map_px`／`_cctv_px` 三組欄位。
  - 指標定義以 `Sub_Module/trajectoryanalyst2` 為準（TTC 碰撞距離為中心距離 2 m、三種衝突類型、DRAC 超限取代 PICUD、CI 取大小、PET 只算同時出現過的配對），唯一例外是 DRAC 接近速度的符號錯誤已修正；限制寫在 `summary.json` 的 `limitations`。
  - fps 低於 `min_fps_for_acceleration`（預設 10）時不計算加速度類指標（MTTC、DR、MaxD、DRAC 超限、CI、CI_m）；C000025 約 5 fps，所以只有速度類指標。C000025 上的事件多在畫面遠端，那裡投影誤差最大；指標只當技術訊號。
- 外部服務與金鑰：經緯度查詢（`scripts/homography/geocode_cameras.py`，OSM Overpass，查不到時以 Google Geocoding 備援）、比例尺檢查（`scripts/homography/check_homography_scale.py`）與底圖下載（`scripts/homography/download_satellite.py`）都是手動執行，不在 pipeline 中。金鑰從環境變數或 repo 根目錄的 `.env`（git 忽略）讀取：`GOOGLE_MAPS_API_KEY`（billing）、`GOOGLE_MAPS_DEMO_API_KEY`（Maps Demo Key，依條款只能用於測試與原型，只允許 `--purpose verify`，不能下載正式底圖）。程式不得印出金鑰值。Google Static Maps 條款可能限制離線保存與衍生使用，底圖不納入 git。
