# 主流程：`track_eval.py`

```bash
python track_eval.py --config <config>          # 等同 python scripts/pipeline/run_pipeline.py --config <config>
python track_eval.py --config <config> --track-config <tracker_yaml>
```

| 參數 | 說明 |
|---|---|
| `--config` | 情境設定檔；不帶時預設為 `config/tracking/bytetrack.yaml` |
| `--track-config` | 另外指定 class-aware tracker 參數 YAML；不帶時直接讀 `--config` 檔內的 `class_tracking:` |

執行順序：`load_track_config()` → `load_config()` → `TrackingEvalPipeline`，依 config 的 `stages.run_stage1/2/3` 執行各階段。

## 1. 內建情境

| config | stage | 用途 |
|---|---|---|
| `config/tracking/bytetrack.yaml` | 1 + 2 | ByteTrack 完整流程（預設） |
| `config/tracking/botsort_noreid.yaml` | 1 + 2 | BoTSORT，不開 ReID |
| `config/tracking/botsort_reid.yaml` | 1 + 2 | BoTSORT + 逐類別 ReID（small_truck／scooter 權重是暫代的 `yolo11x.pt`） |
| `config/reid/botsort_save_reid_crops.yaml` | 1 | 追蹤並蒐集 Re-ID 裁切（`paths.dataset: cctv_all8`，見下方注意事項） |
| `config/reid/botsort_save_reid_crops_video.yaml` | 1 | 同上，改對 `reid_sources` 蒐集 |
| `config/collision/collision_detect.yaml` | 1 + 3 | 追蹤後做衝突事件分析（TTC／PET） |
| `config/grid_search/grid_search_*.yaml` | — | grid search 腳本使用的基準設定 |

> ⚠️ `rebuild-reid-dataset` 的 2.6 已加上裁切前檢查：開啟 `save_reid_crops` 時，**評估鏡頭一律拒絕**，而且每支鏡頭都必須有 `config/roi/<camera>.yaml`。因此：
> - `botsort_save_reid_crops.yaml`（`cctv_all8`，C000025）目前會直接報錯。
> - 正式的裁切設定是 `config/reid/reid_crops.yaml`（`reid_sources`、`frame_rate: auto`，輸出到 `dataset/reid/crops`）。
> - `botsort_save_reid_crops_video.yaml` 保留作為 `frame_rate: 30` 的對照情境，裁切輸出改到 `runs/reid_crops_video/crops`。

## 2. 情境設定檔結構

每個情境 `config/<功能>/*.yaml` 都是**完整、自包含**的設定。下列區塊名稱是穩定介面，不能改名。新實驗請另外建檔，不要覆寫既有的基準情境。

```yaml
paths:
  dataset: cctv_all8               # 對照 config/datasets.yaml
  # cameras: [C000043]             # 選用：只跑登記鏡頭中的一部分
  # SOURCE_VIDEO_DIR: ...          # 選用：顯式覆寫來源目錄
  TRACKEVAL_OUT_DIR: trackeval_output
  COLLISION_OUT_DIR: collision_output   # stage3 使用
  OUTPUT_ROOT: ./runs/track_eval
models:
  det_model_path: ./models/yolo/yolo11x_dennis.pt
trackeval: { benchmark_name: CCTV, split_name: val, tracker_name: YOLO11x }
stages:        { run_stage1: true, run_stage2: true, run_stage3: false }
eval_flags:    { run_det_eval: false, run_track_eval: true }   # 只控制 stage2 內的子工作
classes_to_run: { person: 0, car: 1, scooter: 2, small_truck: 3, bus: 4, truck: 5 }
tracking_params:
  tracker: bytetrack               # bytetrack | botsort
  conf: 0.15
  iou: 0.50
  imgsz: 640
  frame_rate: 30                   # 數值，或 auto（讀影片實際幀率並四捨五入）
  gmc_method: ecc                  # 只對 BoTSORT 有作用
save params:
  save: true                       # 輸出標註影片 Demo.mp4
  save_txt: true
  save_reid_crops: false
  reid_crop_dir: ./dataset/reid/crops
class_tracking:                    # 由 load_track_config() 讀取
  class_names: { 0: person, 1: car, ... }
  class_tracking_params:
    car: { match_thresh: 0.7, second_match_thresh: 0.6, unconfirmed_match_thresh: 0.9,
           new_track_thresh: 0.25, track_buffer: 3.0, use_reid: false, ... }
```

重點：

- `stages.*` 決定要跑哪幾個階段；`eval_flags.*` 只控制 stage2 內部要做 detection 評估還是 tracking 評估，兩者意義不同。
- 頂層 `det_corner` 已停用，出現就報錯；ROI 請改寫在 `config/roi/<camera>.yaml`。
- `track_buffer` 在 BoTSORT 版被當成「秒」（會乘上 frame rate），在 ByteTrack 版則是以 30fps 為基準的幀數。這是已知落差，不會修正。
- ReID 預設關閉：實驗結論是它對 person 有幫助，但會讓 scooter 變差、IDSW 增加。

## 3. 輸出目錄

```
OUTPUT_ROOT/                                   （預設 ./runs/track_eval）
├── annotations_all8/<seq>/                    stage1
│   ├── images/<seq>/*.jpg、labels/<seq>/*.txt
│   ├── trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt   ← 下游的穩定輸入
│   └── Demo.mp4（save: true 時）
├── mot_annotations_all8/<seq>/                stage2 用的 tracker 副本（原始副本備份在 trackers_raw/）
├── det_eval/<seq>/                            stage2 detection 評估資料集（人工 GT 的 hardlink）
├── trackeval_output/<seq>/                    stage2 報表（det_metrics_*.xlsx、trackeval_metrics_*_MULTI_new.xlsx）
└── collision_output/<seq>/                    stage3（ttc_events.*、pet_events.*、熱區圖）
```

輸出端的目錄直接用 `<seq>/`；來源端沿用舊命名 `annotation_<seq>/`、`mot_annotation_<seq>/`。程式內的路徑一律用 `safetrajai/evaluation/dataset_layout.py` 的函式組出來，不要自己拼字串。

## 4. 軌跡格式契約（stage1 → stage2／stage3）

- 檔案：`trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt`
- 每行 7 欄、沒有 header：`frame,track_id,left,top,w,h,score`（像素座標、以左上角為原點，`frame` 從 1 開始）。
- 檔案內沒有 class 欄，類別由資料夾名稱決定；6 個類別都一定會有檔案，沒有結果時是 0-byte 空檔。
- **不依 ROI 過濾**，保留全部追蹤結果。
- `track_id` 只在「單一序列 × 單一類別」內唯一；跨序列要用「序列名＋track_id」當鍵。
