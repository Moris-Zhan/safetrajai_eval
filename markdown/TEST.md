# TEST.md

## 現況
本專案沒有自動化測試（無 `tests/`、無 pytest 設定）。所有驗證都是**手動跑完整或部分 pipeline，比對輸出**。修改 `module/detection/`、`module/tracker/`、`module/reid/`、`evaluation/`、`track_eval.py` 後，至少要跑過下方對應的手動測試項目再視為完成。

## 手動測試項目

### 1. 完整 pipeline 煙霧測試
確認第一階段（追蹤）與第二階段（評估）都能跑完、不報錯，且輸出目錄結構符合 `track_eval.py` 頂部 docstring 的描述。

```bash
python track_eval.py --config config/bytetrack.yaml
```

檢查點：
- `runs/track_eval/<run_det_eval_dir>/annotation_<seq>/trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt` 有產生且非空
- `runs/track_eval/trackeval_output/` 下有對應序列的 `det_metrics_*.xlsx` / `trackeval_metrics_*_MULTI_new.xlsx`
- 若 `config/bytetrack.yaml` 的 `save: true`，`Demo.mp4` 標註影片有正常產生（可開啟播放）

其餘 5 個情境（`config/botsort_noreid.yaml`、`botsort_save_reid_crops.yaml`、`botsort_reid.yaml`、`grid_search_bytetrack.yaml`、`grid_search_botsort.yaml`）改動後也應各跑一次對應腳本；`botsort_save_reid_crops.yaml` 額外檢查 `stages.run_stage2: false` 有沒有正確跳過 stage2、`dataset/reid/crop_dataset/{class}/` 有沒有正確產生裁切影像。

### 2. 縮小範圍快速驗證
改動 tracker 邏輯（`module/tracker/class_aware_botsort.py`、`class_aware_bytetrack.py`）時，先用少量序列跑，不要每次都跑全量資料集：
- 縮小 `config/*.yaml` 的 `paths.SOURCE_VIDEO_DIR` 或直接改 `eval_flags` 只開 `run_track_eval`
- 或用 grid search 腳本的 `--dry-run` 先確認參數組合數量，再用 `--max-videos 1` 只跑一支影片

```bash
python module/grid_search/grid_search_bytetrack.py --target-class scooter --dry-run
python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1
```

### 3. 追蹤參數改動後的迴歸比較
改 `config/*.yaml` 的 `class_tracking:` 章節或 `module/tracker/config.py` 的類別參數查表邏輯後，跑同一序列前後兩次，比較 Excel 報表中的 `HOTA` / `AssA` / `IDSW`，確認改動方向符合預期（例如：預期 scooter 的 AssA 上升，但 person 沒有明顯下降）。目前沒有腳本自動做這個 diff，需人工開 Excel 比對。

### 4. 視覺化抽查
數值指標異常時，用疊圖工具實際看畫面，確認是偵測漏檢、track ID 跳號，還是分類錯誤：

```bash
python visualize/visualize_gt_on_class_video.py --trackeval_root <該序列的 trackeval_output 路徑> --video_path <原始影片> --output_video <輸出路徑>
```

### 5. 設定檔載入順序
新增使用 `module.tracker.config` 的腳本時，務必確認 `load_track_config()` 有在建立 tracker（`create_tracker(...)`）之前呼叫，否則 `CLASS_TRACKING_PARAMS` 會是空字典，追蹤會用不到任何類別專屬參數但不會報錯（靜默失敗，最容易漏測）。檢查方式：程式印出的 `Loaded track config: <path>` 訊息是否有出現在建立 tracker 之前。

## 待補充
> 目前沒有「預期基準值」（baseline metrics）可供比對合格與否，建議之後在此補上：各序列/各類別的基準 HOTA/AssA 區間，作為迴歸測試的判斷標準。
