# 指令索引

所有指令都在 repo 根目錄、conda env `track_eval` 中執行（另有說明的除外）。完整參數見 [`CLAUDE.md`](../CLAUDE.md) 與各腳本的 `--help`。

`tests/test_readme_coverage.py` 要求 `scripts/` 的每支腳本與 `config/` 的每份設定都出現在 `README.md` 或本檔。新增腳本或設定時，請同步補在這裡。

## 主流程

| 指令 | 用途 |
|---|---|
| `python track_eval.py --config config/tracking/bytetrack.yaml` | stage1 追蹤＋stage2 TrackEval；`stages.run_stage1/2/3` 控制階段。同樣用法：`botsort_noreid.yaml`、`botsort_reid.yaml`、`botsort_reid_timm.yaml`（Re-ID 驗收基準，6 類關閉） |
| `python scripts/pipeline/run_pipeline.py --config <config>` | 與 `track_eval.py` 等價的入口 |
| `python scripts/pipeline/run_evaluation.py --config <config>` | 只對既有 stage1 輸出重跑 stage2 |
| `python track_eval.py --config config/collision/collision_detect.yaml` | stage1＋stage3（TTC／PET） |
| `python track_eval.py --config config/collision/conflict_analysis.yaml` | 另外計算公尺單位的衝突指標；只在 `scale_check: passed` 的鏡頭計算，其餘只寫 `summary.json` 並記錄略過原因 |

輸出在 `runs/track_eval/`。要縮小範圍，在 config 的 `paths:` 加 `SOURCE_VIDEO_DIR` 指定單一序列。資料集對照表是 `config/datasets.yaml`。

## 調參（grid search）

針對**單一類別**掃描追蹤參數；全量掃描會跑完整 stage1＋stage2，執行前先確認範圍，並先用 `--dry-run` 看組合數。

| 腳本 | 預設設定檔 |
|---|---|
| `scripts/grid_search/grid_search_bytetrack.py` | `config/grid_search/grid_search_bytetrack.yaml` |
| `scripts/grid_search/grid_search_botsort.py` | `config/grid_search/grid_search_botsort.yaml` |
| `scripts/grid_search/grid_search_botsort_reid.py` | `config/tracking/botsort_reid.yaml`，另需 Re-ID 權重 |

```bash
python scripts/grid_search/grid_search_bytetrack.py --target-class scooter --dry-run
python scripts/grid_search/grid_search_botsort.py --target-class scooter --tracker botsort --max-videos 1
```

## 衝突事件與 Homography（stage3）

程式碼中的 `collision` 指**衝突事件**（TTC／PET 低於門檻的接近事件），不是實際碰撞。選用輸出都在情境 config 開啟，預設關閉：

| 選項 | 輸出 | 說明 |
|---|---|---|
| `counting: {enabled: true}` | `collision_output/<seq>/counting/` | 計數線流量；先用選點器畫線存成 `config/counting_lines/<camera>.yaml`。[`counting-guide.md`](counting/counting-guide.md) |
| `conflict_analysis.bev_video` | `conflict/bev.mp4` | 衛星圖上的軌跡、事件連線與計數線。[`bev-video-guide.md`](bev/bev-video-guide.md) |
| `conflict_analysis.blind_area` | `conflict/blind_area.csv` | 大型車輛造成的 ROI 盲區比例，需要 `camera_view_point`，視為 2D、會高估。[`blind-area-guide.md`](blind_area/blind-area-guide.md) |

Homography 鏡頭設定（`config/homography/<camera>.yaml`）依下列順序操作，後三者會連網：

| 步驟 | 腳本 | 用途 |
|---|---|---|
| 驗證 | `scripts/homography/validate_homography.py` | 批次驗證既有矩陣，**會覆寫** `models/homography/validation_output/`；沒有參數 |
| 驗收 | `scripts/homography/acceptance_check.py` | 讀驗證輸出的 `summary.csv`，套用誤差門檻判定每支鏡頭是否通過；門檻 `ACCEPTANCE_RMSE_M` 目前刻意留 `None`（待補實測數據），此時回報 `threshold_tbd`，不會誤報通過或失敗 |
| 1 遷移 | `scripts/homography/migrate_homography_config.py` | 舊的 tag 鍵資料 → 鏡頭鍵設定，先用 `--dry-run` |
| 2 經緯度 | `scripts/homography/geocode_cameras.py` | OSM Overpass 查詢，Google 備援；先出候選表再 `--apply` |
| 3 比例尺 | `scripts/homography/check_homography_scale.py` | 以已知長度地物核對比例尺，寫入 `scale_check` |
| 底圖 | `scripts/homography/download_satellite.py` | 下載衛星圖，`--purpose verify\|basemap` |

金鑰從環境變數或 repo 根目錄的 `.env`（git 忽略）讀取（`GOOGLE_MAPS_API_KEY`、`GOOGLE_MAPS_DEMO_API_KEY`；Demo Key 只能用於 `verify`），程式不會印出金鑰值，文件與紀錄也不要貼金鑰。這些腳本都是手動執行，不在 pipeline 中。沒有 `scale_check: passed` 的鏡頭單位維持像素。

## 選點器與視覺化

| 腳本 | 用途 |
|---|---|
| `scripts/picker/make_roi_picker.py --camera <camera>` | 產生 ROI 離線點選頁，輸出到 `runs/roi_picker/`。[`roi-guide.md`](roi/roi-guide.md) |
| `scripts/picker/make_picker.py --camera <camera> --mode <mode>` | 離線點選頁，`--mode` 為 `homography`、`checkpoints`、`counting-lines`、`camera-position`；頁面產生 YAML，工具不修改任何 config |
| `scripts/picker/serve_picker.py [--host] [--port 8765]` | 把選點器部署成網頁服務，需要存取權杖，**純 HTTP，只在信任網路、SSH 通道或 `kubectl port-forward` 使用**。[`picker-guide.md`](picker/picker-guide.md) |
| `scripts/visualization/visualize_gt_on_video.py` | 把人工 GT 疊在序列影片上 |
| `scripts/visualization/visualize_gt_on_class_video.py` | 把追蹤結果疊在序列影片上，檢查 ID switch 與跨類別誤配 |

## Re-ID 資料與訓練

資料流程是 **蒐集 → 標記 → 自動排除 → 抽查 → 人工排除 → 碎片合併 → manifest → 訓練 → 驗收**。各腳本的前提與用法見 [`reid/reid-data-tools.md`](reid/reid-data-tools.md) 與 [`scripts/reid_data/README.md`](../scripts/reid_data/README.md)；寫入類腳本不帶 `--apply` 時只列印計畫。

| 階段 | 腳本／設定 |
|---|---|
| 蒐集 | `config/reid/reid_crops.yaml`（只跑 stage1）、`config/reid/botsort_save_reid_crops.yaml`、`config/reid/botsort_save_reid_crops_video.yaml` |
| 標記與排除 | `scripts/reid_data/detect_contamination.py`、`apply_auto_exclusions.py`、`generate_identity_review.py`、`summarize_audit.py`、`apply_review_exclusions.py` |
| 碎片合併 | `scripts/reid_data/extract_fragment_features.py`、`build_fragment_merges.py`、`config/reid/fragment_merge.yaml` |
| manifest | `scripts/reid_data/build_reid_split.py`、`config/reid/reid_split.yaml` |
| 訓練與驗收 | `scripts/reid_train/pack_manifest.py`、`train_reid.py`、`eval_reid.py`、`fusion_distance_stats.py`、`acceptance_grid.py` |

新模型要上線，須在調參序列上與驗收基準比較：結果不變差，而且 IDSW 減少或 IDP 上升；只有 val mAP 上升不算通過。**從 HF 還原後不要執行 `pack_manifest.py`**：打包檔已還原，個別裁切 JPEG 不在 HF。

## 偵測資料與 CVAT 覆核

這條線用目前在用的偵測權重（`models/yolo/yolo11x_dennis.pt`）產生偽標籤，匯入自架的 CVAT，再以腳本輔助人工覆核，用來精煉偵測資料集。背景與決議見 [`training_eval_audit.md`](training_eval_audit.md)，CVAT 的啟停與覆核流程見 [`detection/cvat-operations.md`](detection/cvat-operations.md)，標註規則見 [`detection/labeling-guide.md`](detection/labeling-guide.md)。

### 取樣與偽標籤

| 腳本 | 用途 |
|---|---|
| `scripts/detection_data/sample_and_pseudolabel.py` | Pilot 階段：對 `detection_source` 挑出的 10 個檔案依固定秒數間隔取樣，用 `yolo11x_dennis.pt` 產生偽標籤。只做取樣與偽標籤，不做人工覆核與資料集切分 |
| `scripts/detection_data/track_pilot_videos.py` | 對同一份 10 支影片跑完整逐幀 class-aware 追蹤（沿用 `botsort_reid_timm.yaml`），輸出到 `runs/cvat_pilot_v1_tracking/`，供把 track ID 一起匯入 CVAT |
| `cvat/nuclio/yolo11_detector/` | CVAT 自動標註用的 Nuclio 函式（`cvat/nuclio/deploy.sh` 部署，nuctl 在 `cvat/bin/nuctl`；`function-gpu.yaml`）。`main.py` 與 `model_handler.py` 是函式本體，不是人會執行的 CLI |

### 調教評估與 fair_val（`track_eval` 環境，見 [`detection/finetune-plan.md`](detection/finetune-plan.md)）

| 腳本 | 用途 |
|---|---|
| `scripts/detection_data/eval_finetune.py` | 在 export 的 val split 上比較多組權重，逐鏡頭與合併報告（評估規則見 `dataset-build-rules.md` 第 8 節） |
| `scripts/detection_data/count_detection_errors.py` | 在 val split 上以固定信心門檻逐鏡頭、逐類別計算誤報與漏框；`--dump-fp` 輸出誤報清單給 `serve_fp_review.py` |
| `scripts/detection_data/compare_label_style.py` | `style`：訓練標籤與 cctv_all8 GT 的風格距離關卡；`thresholds`：偽標籤門檻檢查 |
| `scripts/detection_data/build_fair_val.py` | 從 C000012、C000056 驗證序列平均抽幀，產生基準模型底稿 `dataset/detection/fair_val/prefill.json`，建一次 |

### 與 CVAT 往來（需要 `cvat_sdk`）

`cvat_sdk` 裝在獨立 venv `cvat/deploy/cvat_client_venv`，不在 `track_eval`；這個 venv 沒有 cv2。連線資訊從環境變數或 `.env` 讀取（`CVAT_URL`、`CVAT_USERNAME`、`CVAT_PASSWORD`），不要把值寫進文件或回覆。

| 腳本 | 用途 |
|---|---|
| `cvat/review/fetch_task_annotations.py` | 把 CVAT task 的全部追蹤標註抓成 JSON（`runs/cvat_review/pilot_v1_annotations.json`），供覆核網頁使用 |
| `cvat/review/apply_corrections.py` | 把 `corrections.csv`（欄位 `track_id,action,new_label,reason`；`delete` 或 `relabel`）套用回 CVAT。套用前一定先備份目前標註，套用後重新抓一次驗證 |
| `cvat/review/create_fair_val_task.py` | 建立 fair_val task、上傳影像並匯入底稿（矩形 shape，每類別有 `low_conf` 屬性），task id 寫入 `fair_val/cvat_task.json` |
| `cvat/review/export_fair_val.py` | 把 fair_val task 目前的框匯出成 YOLO 標籤、`annotations_<UTC>.json` 快照與 `data.yaml`；可重複執行 |

### 覆核輔助（`track_eval` 環境）

這些腳本只讀 CVAT 匯出的標註，不修改 CVAT。

| 腳本 | 用途 |
|---|---|
| `cvat/review/dedup_clusters.py` | 把同一個實體物件被拆成多條破碎 track 的狀況合併成一群（同鏡頭、位置、尺寸、幾乎靜止、DINOv2 外觀高度相似），一群只需看一條代表 |
| `cvat/review/appearance_sample.py` | 無法去重合併的獨立 track，以 DINOv2 外觀分成 K 群各抽一條代表。群內其餘 track 不是同一物件，代表的判決**不能**安全套用到整群 |
| `cvat/review/reid_outlier_scan.py` | 以 DINOv2 外觀掃描某類別，找出與已確認正確的參考 track 差異很大的候選，只供排序優先序，不自動改標籤 |
| `cvat/review/static_clusters.py` | 找出同一鏡頭同一位置反覆出現的 track（目前只支援 scooter，強烈暗示靜態誤判，例如樹葉、路標）。**不適用 car**：停放的車是真實存在的 |
| `cvat/review/rank_suspicion.py` | 對 car／scooter 的大量未覆核 track 依可疑程度排序（群組框、過大框兩條規則），只決定覆核先後，不判定對錯 |
| `cvat/review/build_review_montage.py` | 為某類別尚未覆核的 track 產生覆核拼貼圖，`index.json` 對應拼貼格與 track_id |
| `cvat/review/build_audit_sample.py` | 抽出供使用者驗收的固定種子樣本（Claude 自行判決並已套用的修正，加上維持不變的外觀代表），寫成 `runs/cvat_review/audit_sample.json` |

### 覆核網頁服務

都需要存取權杖，頁面在請求時才產生，影像不離開這台主機，部署模式沿用 `serve_picker.py`。

| 腳本 | 預設埠 | 用途 |
|---|---|---|
| `cvat/review/serve_review.py` | 8766 | 逐 track 覆核指定類別（預設 small_truck、bus）的完整追蹤標註 |
| `cvat/review/serve_audit.py` | 8767 | 驗收 Claude 自己的判決；回答寫入 `runs/cvat_review/audit_responses.jsonl`，`/summary` 報告各樣本群的不一致率與 5% 門檻 |
| `cvat/review/serve_fp_review.py` | 8771 | 抽查 `count_detection_errors.py --dump-fp` 的誤報：真的錯，還是 GT 漏標 |

## 測試

| 層級 | 指令 | 耗時 |
|---|---|---|
| smoke 與單元 | `pytest` | 約 1 分鐘，不需要資料或權重 |
| golden 比對 | `tests/golden/run_all.sh <run>`，再 `python tests/golden/compare.py <基準> <run>` | 約 16 分鐘，需要完整資料與權重 |

測試分層與單元測試對照見 [`testing/testing.md`](testing/testing.md)。
