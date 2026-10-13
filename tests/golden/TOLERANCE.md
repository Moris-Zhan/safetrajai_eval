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

## 基準切換：before1 → after1（2026-09-23，change `reorganize-dataset-and-eval-gt`）

評估 GT 改用人工 GT（`annotations_all8(orign)`／`mot_annotations_all8(orign)`）、ROI 改以鏡頭為鍵、stage1 軌跡不再依 ROI 過濾、stage2 改以忽略區域（UA-DETRAC 式）限縮範圍，這些是刻意的評估語意變更，golden 基準因此改為 `after1`。

- 自比：以最終程式碼連跑 `after1`、`after2`，`compare.py after1 after2` 全部為 0 差異，容忍度仍為 0。
- 與 `before1` 相同：`conflict_events`、`reid_crops`、`homography`、`yolo_labels`。
- 與 `before1` 不同（預期）：`mot_tracks`（stage1 不再過濾）、`trackeval`（評估語意）、`grid_search`（分數；最佳參數不變）、`other`（移除 stage1 `data.yaml`、新增 `trackers_raw/`）。
- 逐項說明與指標對照見 `openspec/changes/reorganize-dataset-and-eval-gt/verification.md`。`before1` 與 MOT17 式的 `after1_mot17`、`after2_mot17` 保留供對照。

（已被下一節取代：基準改為 `reid1`。）

## 基準切換：after1 → reid1（2026-09-24，change `rebuild-reid-dataset`）

Re-ID 裁切蒐集改為拒絕評估鏡頭（C000025），golden 的 `botsort_save_reid_crops` 情境因此改在來源資料集 `reid_sources` 的 C000043 上蒐集（`paths.cameras: [C000043]`），這是刻意的變更，golden 基準改為 `reid1`。

- 自比：以最終程式碼連跑 `reid1`、`reid2`，`compare.py reid1 reid2` 全部為 0 差異，容忍度仍為 0。
- 與 `after1` 相同：`bytetrack`、`botsort_noreid`、`botsort_reid`、`collision_detect` 四個情境，以及 grid search、homography；兩邊共有的檔案沒有任何差異。
- 與 `after1` 不同（預期，全部位於 `botsort_save_reid_crops/`）：
  - `mot_tracks`：C000025 的 6 段序列換成 C000043 的 1 段序列。這個情境 `save_txt: false`，兩邊的軌跡檔都是 0-byte。
  - `reid_crops`：3830 張 C000025 裁切換成 415 張 C000043 裁切（0.5 秒抽樣、最短邊 32px、ROI 過濾），另外新增 `_index/<seq>.csv`、`_index/<seq>.skipped.json`。
  - `other`：序列影片與文字檔換成 C000043。
- 耗時：`pipeline_botsort_save_reid_crops` 從約 410 秒增加到約 964 秒，整體約 16 分鐘。
- 逐項說明見 `openspec/changes/archive/2026-09-24-rebuild-reid-dataset/verification.md`。
- 2026-09-24 整理 `runs/` 時，`reid1` 以外的舊 run（`before1`、`after1`、`after2`、`after1_mot17`、`after2_mot17`、`reid2`）已移到 `runs/tmp/refactor_golden/`。`compare.py` 只讀 `runs/refactor_golden/`，需要與舊 run 比對時，先把它搬回該目錄。

（已被下一節取代：基準改為 `georef1`。）

## 基準切換：reid1 → georef1（2026-10-13，change `camera-homography-georef`）

Homography 改以鏡頭為鍵（`config/homography/<camera>.yaml`）、驗證報表新增欄位與點數規則，homography 驗證輸出因此刻意改變，golden 基準改為 `georef1`。

- 自比：以最終程式碼連跑 `georef1`、`georef1b`，`compare.py georef1 georef1b` 全部為 0 差異，容忍度仍為 0。
- 與 `reid1` 相同：mot_tracks、yolo_labels、trackeval、conflict_events（stage3 以新的鏡頭解析入口取得相同矩陣）、reid_crops、other 全部 0 差異。
- 與 `reid1` 不同（預期）：
  - `homography`：`summary.csv` 改為鏡頭鍵的新欄位（`camera`、`scale_check`、`m_per_pixel`、`check_*`、`train_residual_*`、`_m`）；C000025 只有 2 個 check points，改為 `too_few_checkpoints`；C000297 的 `.npy` check points 轉入後開始驗證，多出 `stats.xlsx` 與 6 張誤差熱力圖。
  - `grid_search`：`best_params_scooter.txt` 多一行 `'track_buffer': 120,`（數值差異 0）。這是 change `track-buffer-in-frames`（2026-10-06）就已存在、記錄在其 verification.md 的差異，與本 change 無關，這次一併納入基準。
- **之後的預期差異**：`summary.csv` 的 `scale_check`、`m_per_pixel` 與 `_m` 欄位直接取自鏡頭設定。任何鏡頭的 `latlon`、`scale_check` 或 check points 改變，homography 類別都會出現差異，屬於預期變更，要重建基準並在這裡記錄。
- 逐項說明見 `openspec/changes/camera-homography-georef/verification.md`。

之後的比對一律使用 `python tests/golden/compare.py georef1 <run>`。
