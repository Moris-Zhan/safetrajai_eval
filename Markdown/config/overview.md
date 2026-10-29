# Config 總覽

> 對象：要跑情境、改參數或新增實驗檔的人。各資料夾的細節見同目錄的其他教學；輸出目錄與各階段的行為見 [主流程](../pipeline/main-flow.md) 與 [各階段說明](../pipeline/stages.md)。

## 1. 目錄對照

| 位置 | 內容 | 教學 |
|---|---|---|
| `config/tracking/` | 追蹤＋評估情境：`bytetrack.yaml`、`botsort_noreid.yaml`、`botsort_reid.yaml`、`botsort_reid_timm.yaml` | [tracking-scenarios.md](tracking-scenarios.md) |
| `config/grid_search/` | grid search 的起始設定：`grid_search_bytetrack.yaml`、`grid_search_botsort.yaml` | [tracking-scenarios.md](tracking-scenarios.md)、[Grid Search](../grid_search/grid-search.md) |
| `config/reid/` | Re-ID 資料蒐集情境（`reid_crops.yaml`、`botsort_save_reid_crops.yaml`、`botsort_save_reid_crops_video.yaml`）與工具設定（`reid_split.yaml`、`fragment_merge.yaml`） | [reid-configs.md](reid-configs.md) |
| `config/collision/` | stage3 情境：`collision_detect.yaml`、`conflict_analysis.yaml` | [collision-configs.md](collision-configs.md) |
| `config/datasets.yaml` | dataset id 登記表 | [per-camera-configs.md](per-camera-configs.md) |
| `config/roi/`、`config/homography/`、`config/homography_checkpoints/` | 以鏡頭為鍵的設定，檔名是鏡頭編號 | [per-camera-configs.md](per-camera-configs.md) |

## 2. 三種 config

- **情境檔**（`tracking/`、`collision/`，以及 `reid/` 裡的 `reid_crops.yaml` 與 `botsort_save_reid_crops*.yaml`）：交給 `track_eval.py --config <檔>`，每個檔都完整、自包含，同一套區塊（見第 3 節）。
- **工具設定**（`reid/reid_split.yaml`、`reid/fragment_merge.yaml`）：不是情境檔，由 `scripts/reid_data/` 的腳本以 `--rules`、`--config` 讀取。
- **以鏡頭為鍵的設定**（`roi/`、`homography/`、`homography_checkpoints/`）：檔名就是鏡頭編號，也就是序列名稱第一個空白前的字串（例如 `C000054`）。

## 3. 情境檔的區塊

下表的區塊名稱是穩定介面，不能改名；其中 `save params` 的名稱含一個空白，照原樣書寫。沒有寫的鍵會使用載入器的預設值。

| 區塊 | 鍵 | 說明 |
|---|---|---|
| `paths` | `dataset`、`SOURCE_VIDEO_DIR`、`cameras`、`OUTPUT_ROOT`、`TRACKEVAL_OUT_DIR`、`COLLISION_OUT_DIR`、`run_det_eval_dir`、`run_track_eval_dir` | 見第 4 節 |
| `models` | `det_model_path`、`reid_model_path` | 偵測模型路徑；`reid_model_path` 預設 `./Yolo/yolo11x-reid.pt`，現行情境檔多數不寫 |
| `trackeval` | `benchmark_name`、`split_name`、`tracker_name` | TrackEval 的資料集命名，預設 `CCTV`、`val`、`YOLO11x` |
| `stages` | `run_stage1`、`run_stage2`、`run_stage3` | 要跑哪幾個階段；預設 stage1、stage2 開，**stage3 關** |
| `eval_flags` | `run_det_eval`、`run_track_eval` | 只控制 stage2 內的子工作（detection 評估、追蹤評估），與 `stages` 的語意不同，不要混用；預設 `false`、`true` |
| `classes_to_run` | `person: 0` … `truck: 5` | 類別名稱對 YOLO class id（0-based） |
| `logging` | `verbose`、`debug` | `verbose: false` 關掉 GT 處理過程的詳細 log；`debug: true` 開啟 TrackEval tracker 檔案逐項檢查 log |
| `tracking_params` | `tracker`、`conf`、`iou`、`imgsz`、`frame_rate`、`gmc_method` | `tracker` 是 `bytetrack` 或 `botsort`；`frame_rate` 可寫數字，或寫 `auto` 讀影片實際幀率（四捨五入）；`gmc_method` 只在 BoTSORT 有作用 |
| `save params` | `show`、`save`、`save_txt`、`show_unmatched`、`show_removed`、`save_reid_crops`、`reid_crop_dir` | 影片與標籤的儲存控制；`save_reid_crops: true` 才會蒐集 Re-ID 裁切 |
| `class_tracking` | `class_names`、`class_tracking_params`、`default_tracking_params`、`class_conf_thresh`、`default_conf_thresh` | 逐類別追蹤參數，見 [tracking-scenarios.md](tracking-scenarios.md) |
| `collision` | `homography_camera_dir`、`max_ttc_frames`、… | stage3 的 TTC／PET 參數，見 [collision-configs.md](collision-configs.md) |
| `conflict_analysis` | `enabled`、… | stage3 的衝突指標分析，沒有這個區塊視為關閉，見 [collision-configs.md](collision-configs.md) |

## 4. `paths` 怎麼解析

- `dataset: <id>` 會去 `config/datasets.yaml` 查表，取得 `root`（資料集根目錄）、`run_det_eval_dir`、`run_track_eval_dir` 當起始值；`paths` 裡明確寫的 `SOURCE_VIDEO_DIR`、`run_det_eval_dir`、`run_track_eval_dir` 會逐欄位覆寫。id 不在登記表時報錯，並列出可用的 id。
- `cameras: [<camera>, ...]` 只能是該資料集登記鏡頭的**子集**，寫了登記以外的鏡頭會報錯；用它可以只跑某幾支鏡頭。
- 輸出位置由 `OUTPUT_ROOT` 決定：TrackEval 的輸出在 `OUTPUT_ROOT/<TRACKEVAL_OUT_DIR>/`，stage3 的輸出在 `OUTPUT_ROOT/<COLLISION_OUT_DIR>/`（預設 `trackeval_output`、`collision_output`）。

## 5. 載入順序

`class_tracking:` 由 `load_track_config()` 讀進模組層級的參數表，**必須在建立 tracker 與 Detector 之前呼叫**。順序錯了不會報錯，只會悄悄使用預設載入的 `config/tracking/bytetrack.yaml`（或先前載入的設定），而不是你指定的情境。

- 用 `track_eval.py` 或 `scripts/pipeline/run_pipeline.py` 時，順序由程式處理，不需要操心。
- `--track-config` 不指定時，會直接讀 `--config` 檔本身的 `class_tracking:`。
- BoTSORT 在建立時就固定類別集合，之後再載入設定不會更新已建立的 tracker；要換設定必須重新建立。只有在你自己寫程式呼叫這些函式時才會遇到。

## 6. 會報錯的寫法

這些寫法已不再支援，載入時直接報錯，不會悄悄忽略：

| 寫法 | 改用 |
|---|---|
| 情境檔頂層的 `det_corner` | `config/roi/<camera>.yaml`（見 [per-camera-configs.md](per-camera-configs.md)） |
| `collision.homography_points_dir` | `collision.homography_camera_dir` |
| `config/homography_points/<seq>.yaml`（舊的以序列為鍵的對應點） | `config/homography/<camera>.yaml` |
| `reid_center_gate_px_per_s` | `reid_center_gate_px_per_frame` |
| `track_buffer` 為浮點數、布林值或小於 1 | 正整數幀數 |

## 7. 常見任務

| 我想要… | 做法 |
|---|---|
| 跑基準評估 | `python track_eval.py --config config/tracking/bytetrack.yaml` |
| 做新實驗 | 複製一份基準檔到同資料夾並改名，只改要試的鍵；**不要覆寫基準情境**。新增的 yaml 還要在 `README.md` 提到檔名，否則 `tests/test_readme_coverage.py` 會失敗 |
| 只重算 stage2 | `python scripts/pipeline/run_evaluation.py --config <config>`，或在 config 設 `stages.run_stage1: false` |
| 只跑某幾支鏡頭 | `paths.cameras: [<camera>, ...]` |
| 縮小評估範圍 | 調整 `eval_flags.run_det_eval`、`eval_flags.run_track_eval`；或讓 `paths.SOURCE_VIDEO_DIR` 指向只放該序列的資料夾 |
| 確認一份 config 能載入 | `python -c "from safetrajai.evaluation.config_loader import load_config; load_config('<config>')"` |
| 確認 `class_tracking` 合法 | `python -c "from safetrajai.tracking.config import load_track_config; print(load_track_config('<config>'))"` |
