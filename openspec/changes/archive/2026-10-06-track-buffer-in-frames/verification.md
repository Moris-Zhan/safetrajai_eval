# Verification — track-buffer-in-frames

日期：2026-10-06。環境：既有 conda `track_eval`。

## 實作與測試

- 兩個追蹤器相對本次開始前快照各只改一行；`git diff --no-index` 確認所有其他敘述、註解與空白未變動。原有未提交的 Re-ID 改動保留。
- AST red：修改追蹤器前 2 failed、8 passed；green：修改後 10 passed。明列例外每項恰好替換一次，0／2 次拒絕；暫存原始碼對每個凍結方法加上未授權敘述，確認比對拒絕。
- 保留期限：BoTSORT／ByteTrack × 5／10fps × 15／30 幀；邊界仍保留、下一幀移除，邊界回來的偵測維持同一 ID。
- 設定檢查 red：14 failed（舊實作未拒絕無效值）；green：全部通過。缺少、0、負數、2.5、90.0、true 與字串都拒絕；覆蓋逐類別與預設參數，失敗時全部設定字典、預設信心值與來源路徑不變。
- 設定換算後的中途全套：370 passed、11 skipped、2 failed，兩項為尚未移除的碎片候選窗二次 fps 換算與舊回傳格式斷言。完成第 3 組任務後已解決。
- 碎片測試 red：17 failed、6 passed；實作後與設定測試合跑：69 passed、3 skipped。候選窗 30 + 10 = 40，覆蓋兩種追蹤器設定、10／14fps、無法推得 fps 的單裁切軌跡，40 幀接受、41 幀拒絕。
- 兩支碎片腳本 `--help` 成功；`src/`、`scripts/`、`tests/` 無 `retention_frames`、`sequence_fps`、`Track.fps` 引用。
- 最終完整 `python -m pytest -q`：**368 passed、11 skipped**（26.57s）。
- `openspec validate track-buffer-in-frames --strict`：通過。
- `git diff --check`：通過。ADR 0004／0008／0009 的相對引用已檢查。

## 設定等價比對

先保存本次開始時設定快照於 `/tmp/track-buffer-before/`，以 `/tmp/migrate_track_buffers.py` 換算；另以唯讀 `/tmp/verify_track_buffer_conversion.py` 重新計算並比對完整 YAML mapping。15 份設定除了預期的 track_buffer 整數值外沒有其他值改動。以下順序固定為 person、car、scooter、small_truck、bus、truck。

| Config | FPS used | Old → new (person, car, scooter, small_truck, bus, truck) | Equivalent |
|---|---|---|---|
| `config/botsort_noreid.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `config/botsort_reid.yaml` | 30 | fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120 | yes |
| `config/botsort_reid_timm.yaml` | 5 | 2.0 → 10; 3.0 → 15; 2.0 → 10; 3.0 → 15; 3.0 → 15; 3.0 → 15 | yes |
| `config/botsort_save_reid_crops.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `config/botsort_save_reid_crops_video.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `config/bytetrack.yaml` | 30 | fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120 | yes |
| `config/collision_detect.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `config/grid_search_botsort.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `config/grid_search_bytetrack.yaml` | 30 | fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120 | yes |
| `config/reid_crops.yaml` | 10 | 2.0 → 20; 3.0 → 30; 2.0 → 20; 3.0 → 30; 3.0 → 30; 3.0 → 30 | yes (10fps only) |
| `tests/golden/configs/botsort_noreid.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `tests/golden/configs/botsort_reid.yaml` | 30 | fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120 | yes |
| `tests/golden/configs/botsort_save_reid_crops.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |
| `tests/golden/configs/bytetrack.yaml` | 30 | fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120; fallback → 120 | yes |
| `tests/golden/configs/collision_detect.yaml` | 30 | 2.0 → 60; 3.0 → 90; 2.0 → 60; 3.0 → 90; 3.0 → 90; 3.0 → 90 | yes |

## reid_crops.yaml 已知差異

35／39 支來源序列為 10fps，保留期限等價。其餘依計畫採固定 30／20 幀（分別對應原 3.0／2.0）：

| 序列鏡頭 | FPS | 原保留期限（幀） | 新保留期限（幀） |
|---|---|---|---|
| C000043 | 9 | 27／18 | 30／20 |
| C000011、C000056 | 14 | 42／28 | 30／20 |
| C000012 | 15 | 45／30 | 30／20 |

碎片候選窗在上述幀數上加 10。只影響之後重跑的裁切蒐集與候選產生，既有裁切、`fragment_merges.csv`、v3 未重建。

## Golden 驗收

執行：`bash tests/golden/run_all.sh track_buffer_frames_20261006`（2026-10-06 12:00–12:16，`track_eval` 環境）。5 個情境、scooter grid search、homography 驗證的 log 都正常結束；log 中唯一的 error 字樣是 TrackEval 匯入 BURST 時缺 `tabulate` 的既有警告，基準 `reid1` 的 log 也有。

比對：`python tests/golden/compare.py reid1 track_buffer_frames_20261006`

| category | files A | files B | only A | only B | differing files | value mismatches | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| mot_tracks | 300 | 300 | 0 | 0 | 0 | 0 | 0 |
| yolo_labels | 1764 | 1764 | 0 | 0 | 0 | 0 | 0 |
| trackeval | 126 | 126 | 0 | 0 | 0 | 0 | 0 |
| grid_search | 2 | 2 | 0 | 0 | 1 | 0 | 0 |
| conflict_events | 44 | 44 | 0 | 0 | 0 | 0 | 0 |
| reid_crops | 417 | 417 | 0 | 0 | 0 | 0 | 0 |
| homography | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| other | 2265 | 2265 | 0 | 0 | 0 | 0 | 0 |

唯一的差異：`grid_search/best_params_scooter.txt` 行數 13 vs 14（compare 結果為 `differences found`）。`diff` 只有兩處：

- `# Generated:` 時間戳（compare 本來就忽略）。
- 新版多一行 `'track_buffer': 120,`。

原因：`grid_search/core.py` 寫出 patch 時，會把 config 的類別參數與最佳組合合併。golden 的 grid search 讀 `tests/golden/configs/bytetrack.yaml`，任務 2.3 依 design D4 在該檔為每個類別顯式補上 `track_buffer: 120`，原本這個值由 fallback 取得，數值同樣是 120。分數 `HOTA.AssA_mean = 0.6929` 與所有數值欄位都相同（value mismatches 0），追蹤行為沒有改變。

結論：追蹤、評估、衝突事件、裁切、homography 的輸出與基準逐位元相同。grid search patch 檔多出的一行是設定顯式化造成的報表差異，不是行為差異。使用者於 2026-10-06 決定接受這項差異，spec「golden 比對不變」已加註這個例外，5.1 勾選完成。
