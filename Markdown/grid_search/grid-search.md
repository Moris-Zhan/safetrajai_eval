# 追蹤參數 Grid Search

```bash
python scripts/grid_search/grid_search_bytetrack.py   --target-class scooter
python scripts/grid_search/grid_search_botsort.py     --target-class scooter --tracker botsort
python scripts/grid_search/grid_search_botsort_reid.py --target-class person
```

| 參數 | 說明 |
|---|---|
| `--target-class` | 要調優的類別：`person`／`car`／`scooter`／`small_truck`／`bus`／`truck` |
| `--config`／`--track-config` | 基準設定（`--track-config` 不帶時讀 `--config` 的 `class_tracking:`） |
| `--tracker` | `botsort`／`bytetrack` |
| `--model` | YOLO 權重，預設取 config 的 `det_model_path` |
| `--video-glob`／`--max-videos` | 指定影片範圍；`--max-videos` 預設 2 |
| `--params-subset` | 只搜尋指定參數，例如 `match_thresh,new_track_thresh` |
| `--dry-run` | 只計算組合數，不實際執行 |
| `--resume`／`--result-csv` | 從既有 CSV 接續（只跳過完全相同且已完成的組合） |
| `--output-root`／`--verbose` | 暫存輸出位置／印出逐幀進度 |
| （僅 botsort）`--conf`、`--iou`、`--imgsz`、`--frame_rate`、`--gmc_method` | 覆寫推論與 GMC 參數 |

- 分數是 `HOTA.AssA_mean`，屬於關聯指標，**不等於 HOTA**。
- 每個 trial 的流程固定：深拷貝參數 → patch `CLASS_TRACKING_PARAMS` → tracking → MOT 輸出 → TrackEval → 取分數 → 寫 CSV → 還原參數。無論成功、跳過或拋出例外，都會還原參數。
- 建議先用 `--dry-run` 看組合數，再用 `--max-videos 1` 做最小驗證。
