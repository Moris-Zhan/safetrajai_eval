# Stage3 設定（`config/collision/`）

> 程式與設定中的 `collision` 指的是**衝突事件**（TTC／PET 低於門檻的接近事件），不是實際碰撞。區塊共通說明見 [overview.md](overview.md)，階段行為見 [各階段說明](../pipeline/stages.md)。

## 1. 兩個檔案

| 檔案 | 內容 |
|---|---|
| `collision_detect.yaml` | BoTSORT 追蹤（Re-ID 關閉）＋ stage3 的 TTC／PET 事件；`run_stage2` 關閉 |
| `conflict_analysis.yaml` | 與 `collision_detect.yaml` 完全相同，**只多一個 `conflict_analysis:` 區塊**，額外計算公尺單位的衝突指標 |

```bash
python track_eval.py --config config/collision/collision_detect.yaml
python track_eval.py --config config/collision/conflict_analysis.yaml
```

stage3 只讀 stage1 產生的 7 欄 MOT 軌跡，不讀人工 GT。與基準 `bytetrack.yaml` 相比，這兩個檔的差異是：`tracker: botsort`、`class_tracking` 沿用 `botsort_noreid.yaml` 的 BoTSORT 參數（Re-ID 關閉，見 [tracking-scenarios.md](tracking-scenarios.md)）、`save params.save: false`、`stages.run_stage2: false`、`stages.run_stage3: true`、`paths.COLLISION_OUT_DIR: collision_output`，以及新增 `collision:` 區塊。

## 2. `stages` 與輸出

| 鍵 | 值 | 說明 |
|---|---|---|
| `stages.run_stage1` | `true` | 追蹤，產生軌跡 |
| `stages.run_stage2` | `false` | 不做 TrackEval 評估 |
| `stages.run_stage3` | `true` | 預設是關閉的，要在 config 明確開啟 |
| `paths.COLLISION_OUT_DIR` | `collision_output` | 輸出在 `OUTPUT_ROOT/<COLLISION_OUT_DIR>/<seq>/` |

`collision_output/<seq>/` 內是：`ttc_events.{json,csv}`、`pet_events.{json,csv}`、`collision_ttc_log.csv`、`collision_pet_log.csv`、`ttc_collision_heatmap.png`、`pet_collision_heatmap.png`。要看事件請讀 `ttc_events`、`pet_events`；熱區圖疊在 stage1 存下的 `first_frame_with_box.jpg` 上，找不到這張圖時會略過熱區圖，沒有事件點時也不會產生圖檔。

## 3. `collision:` 區塊（TTC／PET）

| 鍵 | 預設 | 說明 |
|---|---|---|
| `homography_camera_dir` | `config/homography` | 以鏡頭為鍵的 Homography 設定資料夾。鏡頭沒有設定檔時，該鏡頭的序列退回原始 CCTV 像素座標 |
| `homography_matrices_dir` | `models/homography/matrices` | `matrix_tag` 指向的既有矩陣所在位置 |
| `homography_satellite_dir` | `models/homography/satellite_images` | 衛星底圖所在位置 |
| `max_ttc_frames` | `30` | TTC 模擬時間上限（幀數） |
| `iou_threshold` | `0.8` | 外推後邊界框 IoU 達此值才判定為碰撞 |
| `lookback_sec` | `3.0` | PET 允許比對的最大時間差（秒） |
| `overlap_threshold` | `30.0` | PET 的距離門檻。這個值是**像素尺度**，沒有套 Homography 的鏡頭才合理；套用轉換後的座標時量級不同，需要依實測結果調整，目前沒有用實際資料驗證過 |

`collision.homography_points_dir` 已不再支援，寫了會報錯，請用 `homography_camera_dir`。舊的 `config/homography_points/<seq>.yaml` 存在時同樣報錯。

## 4. `conflict_analysis:` 區塊（公尺單位的衝突指標）

沒有這個區塊視為關閉；寫了區塊但沒寫 `enabled`，也視為關閉。區塊內不認得的鍵會報錯。

**只有在能換算公尺的鏡頭才會計算**：鏡頭設定要有 `matrix_tag`（既有衛星矩陣）、經緯度，而且 `scale_check: passed`（見 [per-camera-configs.md](per-camera-configs.md)）。其他序列只寫 `conflict/summary.json` 並記錄略過原因（`no_homography` 或 `no_meters_per_pixel`），不會退回像素。位置取框的底邊中點，軌跡以 `(類別, track_id)` 區分。

| 分組 | 鍵 | 說明（單位） |
|---|---|---|
| 開關 | `enabled` | `true` 才執行 |
| 運動學 | `smooth_window_s` | 置中移動平均視窗（秒，須 > 0） |
| | `max_gap_frames` | 幀差大於此值時切段，不跨空檔差分（須 ≥ 0） |
| | `min_speed_mps` | 速率低於此值時航向未定義（公尺／秒，須 ≥ 0） |
| | `min_fps_for_acceleration` | 序列 fps 低於此值時不計算加速度類指標（MTTC、DR、MaxD、DRAC 超限、CI）。C000025 約 5 fps，所以只有速度類指標 |
| 配對 | `pair_radius_m` | 同幀中心距離超過此值的配對不計算（公尺，須 > 0） |
| | `class_pairs` | 要計算的類別配對；預設是 6 類兩兩組合（含同類），排除 person–person。要自訂時列出 `[類別, 類別]`，類別名稱必須是 6 類之一 |
| | `angle_types` | 速度夾角的類型：由小到大、連續，最後一段必須到 180 度，例如 `[30, Rear-end]`、`[85, Lane-change]`、`[180, Crossing]` |
| TTC／MTTC | `ttc_distance_m` | 中心距離小於此值視為命中（公尺，須 > 0） |
| | `ttc_distance_mode` | `fixed` 或 `class_length`（改用兩類別長度一半的和） |
| | `ttc_horizon_s`、`ttc_step_s` | 外推上限與步進（秒，須 > 0） |
| 追撞鏈 | `chain_angle_deg`、`chain_dist_m`、`chain_dist_weight`、`chain_eps` | Floyd-Warshall 連通的成本參數（須 ≥ 0）；最短路徑總成本不超過 `chain_eps` 視為同一追撞鏈 |
| PET | `pet_zone_m` | 衝突區半徑（公尺，須 > 0） |
| | `pet_max_gap_s` | `0` 表示只算同時出現過的配對；大於 0 時允許的時間間隔（秒） |
| 衝突事件 | `event_threshold_s` | TTC 或 MTTC 低於此值的幀才構成事件（秒，須 > 0） |
| | `event_gap_frames` | 幀差不超過此值時合併為同一事件（須 ≥ 0） |
| | `stat_thresholds_s` | 摘要中以事件數統計的門檻（秒） |
| 類別資料 | `object_length_m`、`object_mass_kg` | 6 類都必須給值且為正數；長度只用於 `ttc_distance_mode: class_length`，質量用於 CI_m |

輸出在 `collision_output/<seq>/conflict/`：`pair_frames.csv`、`conflict_events.{csv,json}`、`pet.csv`、`chains.csv`、`summary.json`、`conflict_heatmap.png`；點座標有 `_m`、`_map_px`、`_cctv_px` 三組欄位。`summary.json` 的 `limitations` 記錄這次分析的限制。

> ⚠️ 未經實測校正的事件數與指標只能當技術訊號，不能當成正式的安全結論。C000025 上的事件多在畫面遠端，那裡投影誤差最大。

## 5. 常見任務

| 我想要… | 做法 |
|---|---|
| 只重跑 stage3，沿用之前的 stage1 輸出 | 複製 `collision_detect.yaml`，設 `stages.run_stage1: false`、`stages.run_stage3: true` |
| 在既有情境上加衝突指標 | 加上 `conflict_analysis:` 區塊並設 `enabled: true`（或直接用 `conflict_analysis.yaml`） |
| 調 TTC 命中距離 | 改 `conflict_analysis.ttc_distance_m`；要依物體大小改用 `ttc_distance_mode: class_length` |
| 調事件的嚴格程度 | 改 `event_threshold_s`；摘要要統計的門檻改 `stat_thresholds_s` |
| 確認區塊合法 | `python -c "from safetrajai.evaluation.config_loader import load_config; print(load_config('config/collision/conflict_analysis.yaml').conflict_analysis)"` |
