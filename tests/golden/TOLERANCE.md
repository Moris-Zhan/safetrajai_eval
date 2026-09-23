# 黃金比對容忍度（nondeterminism baseline）

以未改動的程式碼（`BASE_COMMIT` = `9c6414b`）連跑兩次 `tests/golden/run_all.sh`（`before1`、`before2`），再以 `compare.py before1 before2` 量測自比差異。

- 環境：conda env `track_eval`（Python 3.10、torch 2.7.1+cu128、ultralytics 8.3.141）、NVIDIA GeForce RTX 5090
- 日期：2026-09-23

## 結論

**兩次執行完全一致，容忍度為 0。** 之後每個 Phase 的 `compare.py before1 after_pN` 都必須回報 `identical`；任何非零差異都視為行為改變。

## 比對結果

| 類別 | 檔案數 | 比對方式 | 差異檔案 | 最大絕對誤差 |
|---|---:|---|---:|---:|
| mot_tracks | 360 | stage1／stage2 的 7 欄 MOT 軌跡逐行逐值比對（5 情境 × 6 序列 × 6 類別 × 2 份） | 0 | 0 |
| yolo_labels | 1764 | stage1 逐幀 YOLO label 逐值比對 | 0 | 0 |
| trackeval | 126 | TrackEval `*_summary.txt`（108）與 stage2 Excel 報表（18）逐值比對 | 0 | 0 |
| grid_search | 2 | 結果 CSV（不含 `elapsed_sec`）與 best params 檔（不含 `# Generated:` 行） | 0 | 0 |
| conflict_events | 44 | `ttc_events`／`pet_events` 的 JSON 與 CSV 逐筆逐欄、`collision_{ttc,pet}_log.csv`；熱區圖只比路徑 | 0 | 0 |
| reid_crops | 3830 | 裁切檔相對路徑集合 | 0 | — |
| homography | 4 | `summary.csv`、`<tag>_stats.xlsx` 逐值；overlay／比對影片只比路徑 | 0 | 0 |
| other | 2214 | GT、seqinfo、`data.yaml` 等文字檔逐值；圖片／影片／PDF 只比路徑 | 0 | 0 |

## 比對規則備註

- 圖片、影片、PDF、`.npy`、`.pt` 只比對路徑是否存在；PDF 內嵌產生時間，逐位元比對會誤報。
- 字串中的 `runs/refactor_golden/<run>/` 會先正規化再比對（例如 `data.yaml` 的 `path:` 欄位會寫入 run 名稱）。
- `_configs/` 與 `logs/` 不列入比對。

## 各步驟耗時（before1／before2，秒）

| 步驟 | before1 | before2 |
|---|---:|---:|
| pipeline_bytetrack | 40 | 40 |
| pipeline_botsort_noreid | 412 | 411 |
| pipeline_botsort_reid | 635 | 628 |
| pipeline_collision_detect | 404 | 404 |
| pipeline_botsort_save_reid_crops | 410 | 409 |
| grid_search | 13 | 14 |
| homography_validation | 2 | 2 |
