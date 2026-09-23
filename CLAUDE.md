# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概觀

離線 CCTV 交通監控評估流程：YOLO11 偵測 → class-aware BoTSORT／ByteTrack 追蹤 → TrackEval（HOTA／CLEAR／Identity）→ Excel 報表，另有第三階段的軌跡碰撞風險分析（TTC／PET＋Homography）。固定 6 類，類別 ID 是跨模組契約：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`。

- 沒有 `pyproject.toml`、測試（test）、lint 設定。`requirement.txt` 是 pip 安裝腳本，包含許多與本專案無關的套件；實際核心依賴是 `ultralytics==8.3.141`、`torch`、`opencv-python`、`numpy`、`scipy`、`openpyxl`、`pyyaml`。
- `trackeval/`（根目錄）是 vendored 的第三方套件，以 `import trackeval` 使用，不要修改或格式化。
- `dataset/`、`models/`、`runs/`、`video/`、`Sub_Module/`、`ignore/` 被 gitignore（權重、影片、實驗輸出），`dataset/reid/` 底下的幾支腳本與 README 例外，有進版控。
- 規格驅動的變更放在 `openspec/`（`openspec/changes/<name>/` 有 proposal、design、specs、tasks），用 `/opsx:*` 指令操作。

## 常用指令

所有腳本都從專案根目錄執行，沒有額外的 CLI 包裝：

```bash
# 主流程：stage1 tracking → stage2 TrackEval → stage3 碰撞偵測（由 config 的 stages.* 開關控制）
python track_eval.py --config config/bytetrack.yaml               # 不帶 --config 時預設就是這個檔
python track_eval.py --config config/botsort_noreid.yaml
python track_eval.py --config config/botsort_reid.yaml
python track_eval.py --config config/botsort_save_reid_crops.yaml # 只跑 stage1，並蒐集 Re-ID 裁切
python track_eval.py --config config/collision_detect.yaml        # stage1 + stage3（TTC/PET）

# 單一類別的追蹤參數 grid search（score = HOTA.AssA_mean，屬於關聯指標，不等於 HOTA）
python module/grid_search/grid_search_bytetrack.py --target-class scooter
python module/grid_search/grid_search_botsort.py --target-class scooter --tracker botsort
python module/grid_search/grid_search_botsort_reid.py --target-class person
#   --dry-run（只算組合數）  --max-videos 1（最小驗證）  --params-subset  --resume  --result-csv

# Homography 驗證
python -m module.collision.run_homography_validation

# Re-ID 訓練（YOLO-as-encoder + projector + batch-hard triplet，PK sampling）
python module/reid/training/train.py --class-name person
python dataset/reid/build_reid_split.py --mode category|flat

# 在影片上疊 GT／追蹤結果
python visualize/visualize_gt_on_video.py --trackeval_root <path> --video_path <path> --output_video <path>
```

沒有自動化測試，驗證方式是實際跑受影響的 stage，然後比對輸出。想縮小範圍時，直接在 config 的 `paths:` 加 `SOURCE_VIDEO_DIR` 指定單一序列，或調整 `eval_flags.run_det_eval`／`run_track_eval`，會比另外新增 CLI flag 來得快。全量影片、grid search 和重跑 stage1 都很耗時，也可能覆寫既有輸出，執行前要先確認範圍。

## 架構

```
track_eval.py      CLI 入口；TrackingEvalPipeline（run_stage1/2/3）+ VideoSequenceTracker（單影片迴圈）
module/detection/  Detector：單幀 YOLO 推論 + 逐類別信心門檻（class_conf_thresh）過濾
module/tracker/    class-aware ByteTrack / BoTSORT（繼承 ultralytics trackers），唯一入口 create_tracker()
module/reid/       ReID（YOLO 當 encoder）與 Custom_ReID（models/class_reid_weights/）兩條路徑；training/ 為新訓練流程
module/grid_search/ core.py 收斂共用邏輯，三支腳本只保留 PARAM_GRIDS + argparse
module/collision/  stage3：MOT 軌跡 → TTC/PET 事件、Homography 轉換／驗證、熱區圖
evaluation/        stage2 共用：config_loader、dataset_layout（路徑組裝純函式）、mot_converter、trackeval_runner、excel_export
config/            情境設定檔（每檔自包含）+ datasets.yaml（dataset id 對照表）+ homography_points/checkpoints
```

### 設定檔

- 每個 `config/*.yaml` 都是完整、自包含的實驗情境：一般設定（`paths`、`models`、`stages`、`eval_flags`、`tracking_params`…）加上 `class_tracking:` 章節。這些區塊名稱是穩定介面，不能改名。新實驗請另外建檔，不要覆寫既有的基準情境。
- 一般設定由 `evaluation/config_loader.py::load_config()` 讀取；`paths.dataset: <id>` 會去查 `config/datasets.yaml`，取得 `root`／`run_det_eval_dir`／`run_track_eval_dir`，也可以逐欄位顯式覆寫。
- `class_tracking:` 由 `module/tracker/config.py::load_track_config()` 讀取，寫入模組層級的 `CLASS_TRACKING_PARAMS`。**這個函式必須在 `create_tracker()`／`Detector` 建立之前呼叫**；順序錯了不會報錯，只會悄悄用到空的或舊的參數。`--track-config` 不指定時，會直接讀 `--config` 檔本身的 `class_tracking:`。
- `stages.run_stage1/2/3` 控制要跑哪幾個階段；`eval_flags.run_det_eval/run_track_eval` 只控制 stage2 裡面的子工作，兩者語意不要混用。

### 路徑慣例（`evaluation/dataset_layout.py`）

SOURCE 端（影片、人工 GT）沿用舊命名 `annotation_<seq>/`、`mot_annotation_<seq>/`；OUTPUT 端（本專案產生的 stage1/stage2 目錄）直接用 `<seq>/`，不加前綴。所有路徑都透過 `dataset_layout.py` 的函式組出來（`track_eval.py`、`grid_search/core.py`、`module/collision/` 共用），不要自己拼字串。`track_eval.py` 開頭的 docstring 有完整的輸入／輸出目錄結構圖。

### 軌跡格式契約（stage1 → stage2／stage3）

- 穩定的下游輸入是 `trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt`：每行 7 欄、無 header，`frame,track_id,left,top,w,h,score`，像素座標、以左上角為原點，`frame` 從 1 開始。檔案內沒有 class 欄，類別看資料夾名稱決定。
- 6 個類別都必須產生檔案，沒有結果時建立 0-byte 空檔。`det_corner` 多邊形會在 MOT 輸出階段依 bbox 中心點過濾掉區域外的偵測。
- `track_id` 只在單一影片、單一類別內唯一（每支影片都新建 tracker，而且是逐類別配號），跨影片要用「序列名＋track_id」當鍵。
- GT 有 class 欄，但只給 TrackEval 比對用；stage3 和其他下游分析一律讀 tracker 輸出，不讀 GT。
- 已知限制：`save_mot_format()` 用 `C000025_*` 做 glob，換成非預設資料集時可能靜默輸出全空的軌跡。修好之前，不要把空檔解讀成沒有偵測到東西。

### Tracker（改動前先讀 `module/tracker/HANDOFF.md` 與 `class_aware_*_flow.md`）

- 核心設計：偵測結果**依類別分組後才做三階段匈牙利匹配**，每個類別有自己的 `match_thresh`／`second_match_thresh`／`unconfirmed_match_thresh`／`new_track_thresh`／`track_buffer`／`use_reid`，由 `get_params_for_class()` 查表取得。不能讓不同類別共用同一個 matching matrix。
- **不可更動（使用者明確指示）**：`ClassAwareBOTSORT` 與 `ClassAwareBYTETracker` 中 override 自 ultralytics 的方法，包括 `__init__`、`update()`、`remove_duplicate_stracks()`、`init_track()`（僅 BoTSORT），程式碼一字不能改，也不能搬移或抽成共用函式。這條優先於任何去重複或修 bug 的目標。本專案自己新增的 `get_dists_class_aware()`、`_center_distance()`，以及 `matching.py`、`config.py`、`__init__.py` 不受這條限制。
- 已知但不會修正的落差：`track_buffer` 在 BoTSORT 版被當成「秒」（會乘上 frame rate），在 ByteTrack 版被當成 30fps 基準的幀數，同一個數值換算成實際存活時間，兩者差 30 倍。
- GMC 只在 BoTSORT 有作用；ReID 只透過 BoTSORT 的逐類別 `use_reid` 啟用。實驗結論是 ReID 對 person 有幫助，但會讓 scooter 變差、IDSW 增加，所以預設關閉。
- 驗證方式：先跑單一序列或目標類別，比較同一基準下的 HOTA、AssA、IDSW，再用疊圖影片檢查 ID switch 和跨類別誤配。

### Re-ID

- **`utils/model.py` 不可搬移或改名**：`models/class_reid_weights/*/best_full_model.pt` 是用 `torch.save` 存下的完整模型 pickle，`Custom_ReID` 用 `weights_only=False` 載入時，會依 `utils.model` 這個模組路徑找類別。真的要遷移的話，先用舊路徑載入，再轉存成 state_dict。
- 不要再引入已經移除的 SOTA_ReID、FastReID、TransReID。
- Crop 資料集結構是 `dataset/reid/crop_dataset/{class}/{video}_track{id}/*.jpg`（由 `save_reid_crops` 產生），train／val 要依身分（identity）互斥切分；`--mode category` 和 `--mode flat` 是兩種不同的資料集，不能混用。資料稽核相關腳本（`check_class_consistency.py`、`quarantine_flagged.py` 等）的規格見 `openspec/changes/reid-crop-identity-audit/`。

### Grid search

每個 trial 的順序固定：深拷貝原始參數 → patch `CLASS_TRACKING_PARAMS` → tracking → MOT 輸出 → TrackEval → 取出 `HOTA.AssA_mean` → 寫入 CSV → 還原參數。無論成功、跳過或拋出例外，都必須還原參數。`--resume` 只能跳過完全相同、而且已經完成的組合。

### Collision／Homography（stage3）

- 輸入只讀 stage1 的 7 欄 MOT 軌跡。事件請讀 `ttc_events.{json,csv}`、`pet_events.{json,csv}`，不要依賴舊 log 的 `colision_occur_sec`（PET 的這個欄位可能記的是 frame，而不是秒）。
- Homography 的查找順序：既有矩陣（`models/homography/`）→ `config/homography_points/<seq>.yaml` 人工點選 → 都沒有時直接用 CCTV 原始像素座標。套用矩陣時，TTC 的 bounds 要用衛星圖尺寸；速度和方向要從轉換後的位置重新差分；輸出的事件點則要用反矩陣投影回 CCTV 像素座標。
- 誤差單位是像素，不是公尺。未經實測校正的事件數只能當作技術訊號，不能當成正式的安全結論。
