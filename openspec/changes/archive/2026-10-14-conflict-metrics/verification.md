# Verification

## 已驗證（2026-10-14）

### 單元與 smoke 測試

- `tests/test_conflict_metrics.py`：71 passed（設定、軌跡、運動學、配對、TTC／MTTC、追撞後車指標、PET、追撞鏈、事件、序列入口與輸出檔）。
- 全部 `pytest`：653 passed、11 skipped。

### 真實資料執行（任務 5.3）

範圍與做法：
- 沒有重跑 stage1。golden 基準 `runs/refactor_golden/georef1/collision_detect/annotations_all8/` 已經有 C000025 六段序列的 stage1 輸出，這次把它複製到 `runs/conflict_metrics/verify1/annotations_all8/`（936 個檔案，與來源的檔案數相同），基準本身沒有修改。
- 情境設定是 `runs/conflict_metrics/verify1/_configs/conflict_analysis.yaml`，與 `config/conflict_analysis.yaml` 只差兩處：`OUTPUT_ROOT` 指向 `verify1`，以及 `run_stage1: false`。
- 原本規劃只跑一段序列，但第一段只有 1 個事件，不夠抽查 5 個，所以六段都跑。

鏡頭 C000025：`m_per_pixel` 為 0.068901，矩陣 tag 是 `C000025 中華三路 民生二路(東北角)-2-20-1280-1280`。

| 序列 | fps | 軌跡數 | 配對列（寫出） | 事件數（類型） | TTC 事件數 <1.0／1.5／2.0／2.5 s | PET 配對數 | PET <1.0／1.5／2.0／2.5 s | overlap_at_start 配對 |
|---|---|---|---|---|---|---|---|---|
| 145630_15s_1 | 4.876 | 34 | 2371（638） | 1（Rear-end 1） | 0／1／1／1 | 46 | 10／15／20／27 | 0 |
| 145630_15s_2 | 4.776 | 31 | 1378（853） | 1（Rear-end 1） | 0／0／0／1 | 30 | 9／11／13／17 | 0 |
| 150634_15s_1 | 4.955 | 37 | 2071（713） | 5（Rear-end 3、Undefined 2） | 0／2／3／5 | 33 | 5／11／15／21 | 2 |
| 150634_15s_2 | 5.000 | 34 | 1918（550） | 2（Rear-end 2） | 1／1／2／2 | 52 | 14／22／26／33 | 0 |
| 152639_15s_1 | 4.955 | 45 | 3132（1129） | 1（Lane-change 1） | 0／0／0／1 | 66 | 8／17／27／39 | 0 |
| 152639_15s_2 | 4.860 | 28 | 1324（492） | 1（Rear-end 1） | 1／1／1／1 | 25 | 2／5／8／12 | 1 |
| 合計 | | 209 | 12194（4375） | 11 | 2／5／7／11 | 252 | 48／81／109／149 | 3 |

- 六段都在 5 fps 左右，低於 `min_fps_for_acceleration=10`。所以每一段的 `acceleration_available` 都是 false，`mttc_event_counts` 為 null，`limitations` 也都有 `no_acceleration_low_fps`。事件的 MTTC、CI、CI_m、DRAC 超限全部未定義：`drac_ever_exceeded` 為空、`max_drac_excess` 為 nan。DRAC 本身只需要速度，照常計算。
- 執行時間：stage3 整體約 7.6 秒（六段序列，含既有 TTC／PET 與 Python 啟動），衝突指標分析本身不是瓶頸。

### 追撞鏈分布與 `chain_dist_m`

以相同的運動學與配對，只改 `chain_dist_m`，統計六段序列合計的追撞鏈大小（鏈大小：鏈數）：

| `chain_dist_m` | 鏈數 | 大小分布 | 最大鏈 |
|---|---|---|---|
| 2.0 | 148 | 1:121、2:19、3:6、4:2 | 4 |
| 5.0 | 116 | 1:78、2:17、3:14、4:4、5:3 | 5 |
| **10.0（目前預設）** | 75 | 1:40、2:11、3:9、4:6、5:2、6:3、8:1、10:1、13:2 | 13 |
| 15.0 | 44 | 1:18、2:6、3:3、4:3、5:5、6:1、8:2、10:1、11:1、12:1、13:1、18:1、23:1 | 23 |
| 20.0 | 27 | …、32:1、36:1 | 36 |

design 第 6 節說「換成公尺 2.0 時追撞鏈不可能形成」，這個說法不正確。邊成本是 `max(0, ΔS − chain_dist_m)`，只要不超過 `chain_eps=3` 就連通，所以實際條件是「最小 ΔS ≤ chain_dist_m + 3」：2.0 時是 ΔS ≤ 5 m，仍有少量追撞鏈。預設 10.0 等於 ΔS ≤ 13 m，最大的鏈有 13 條軌跡，大約是號誌前的一整列車隊。使用者確認保留 `chain_dist_m=10.0`（2026-10-14），design 第 6 節與 config 註解已同步修正。

### 事件抽查（圖片在 `runs/conflict_metrics/verify1/spot_check/`，使用者已確認，2026-10-14）

腳本是 `runs/conflict_metrics/verify1/spot_check.py`（`chain_sweep.py` 是上一節的追撞鏈統計）。每張圖取事件最小 TTC 的那一幀，畫出兩條軌跡的框、最近 10 幀的底邊中點路徑，以及 CCTV 像素的碰撞點（紅色十字），下方是放大 3 倍的局部。

| 事件 | 最小 TTC | 類型 | 初步觀察 |
|---|---|---|---|
| 145630_15s_2 `scooter:10-scooter:14@12` | 2.22 s | Rear-end | 同向的兩台機車 |
| 150634_15s_1 `scooter:4-scooter:14@9` | 1.24 s | Undefined | scooter:4 幾乎靜止（航向未定義），scooter:14 從後方接近，合理 |
| 150634_15s_1 `scooter:104-scooter:111@65` | 1.62 s | Rear-end | 6 幀連續命中 |
| 150634_15s_2 `car:40-truck:38@29` | 0.86 s | Rear-end | 畫面右上遠處，car 在 truck 前方並部分遮擋 truck；像是框的底邊中點位置太接近，不像真正的追撞衝突 |
| 152639_15s_1 `car:91-scooter:38@42` | 2.08 s | Lane-change | 右轉車與遠處機車，碰撞點落在兩者之間的路口出口，合理 |
| 152639_15s_2 `scooter:51-scooter:52@45` | 0.14 s | Rear-end | 兩台機車在遠處並排行駛；2 m 中心距離門檻對並排機車太寬，是已記錄的限制 |

共同觀察：11 個事件的碰撞點幾乎都落在畫面上方（CCTV y 約 90～250），也就是路口遠端，那裡每個像素對應的公尺數最大，Homography 誤差也最大。因此遠端的事件數偏多，可能是投影誤差造成的。這個現象和「2 m 中心距離對機車並排太寬」都已經寫在 summary 的 `limitations` 中，只能當作技術訊號。

### 既有 stage3 輸出不變（任務 6.1）

- 同一份 stage1 輸出，在分析開啟（`collision_output/`）與關閉（`collision_output_off/`，`enabled: false`）時各跑一次 stage3。排除 `conflict/` 後，兩邊的檔案清單相同（44 個檔案），`cmp` 逐位元比對有 0 個不同。
- 關閉時沒有產生任何 `conflict/` 目錄。
- 分析開啟時的既有輸出與 golden 基準 `georef1/collision_detect/collision_output` 也是 44 個檔案、0 個不同。

### 重現性（任務 6.2）

- 以同一份設定再跑一次（`collision_output_on2/`），整個 collision output（含 `conflict/`，共 86 個檔案）與第一次逐位元相同。

### golden 全量比對（任務 6.3）

- `tests/golden/run_all.sh conflict0`（既有 5 個情境，還沒加入新情境）：collision_detect 414 s、botsort_save_reid_crops 965 s，全部成功。
- `python tests/golden/compare.py georef1 conflict0`：mot_tracks 300、yolo_labels 1764、trackeval 126、grid_search 2、conflict_events 44、reid_crops 417、homography 11、other 2265 個檔案，沒有只存在於一邊的檔案，所有類別差異為 0，結果 **identical**；`conflict` 類別兩邊都是 0 個檔案。

### 與參考實作不同之處

- **DRAC 接近速度的符號**：參考實作 `six_pairs_index.py` 第 93 行是 `vrel = (v_front − v_rear)·ê`，其中 ê 從後車指向前車，結果是兩車分開時 DRAC 為正、接近時為 0，方向相反。本實作改用 `(v_rear − v_front)·ê`，接近時為正（`following.py`），單元測試「DRAC」驗證了數值。這是 ADR 0011「以參考實作的核心定義為主」的例外，屬於錯誤修正，不是定義變更。
- 最近前後車的分組鍵改用 `(frame, 軌跡)`（design 第 5 節）；門檻統計以事件數計算（spec「衝突事件彙總」）；ΔS 以公尺計算。這些都已經寫在 design 與 spec 中。

### golden 基準 `conflict1`（任務 6.4）

- 使用者確認抽查圖與 `chain_dist_m=10.0` 後（2026-10-14），`conflict_analysis` 加入 `tests/golden/run_all.sh` 的 `SCENARIOS`。`tests/golden/configs/conflict_analysis.yaml` 與 `config/conflict_analysis.yaml` 只差輸出路徑；`compare.py` 新增 `conflict` 類別。
- 連跑 `conflict1`、`conflict1b`：`pipeline_conflict_analysis` 約 426 秒，與其他情境平行執行，整體時間由 `botsort_save_reid_crops`（968 秒）決定。
- 第一次自比時，6 個 `pair_frames.csv` 回報有數值不同、但最大差異為 0；用 `cmp` 檢查，42 個 `conflict/` 檔案逐位元相同。原因是 `compare.py` 把兩個相同的 `inf` 算成不同（inf − inf 為 NaN），而 `pair_frames.csv` 的 TTC 欄位有 `inf`。先寫失敗測試（`test_golden_compare_treats_equal_infinities_as_identical`）再修正 `compare_scalars`。修正後 `compare.py conflict1 conflict1b` 全部為 0 差異，結果 **identical**。
- `compare.py georef1 conflict1`：沒有只存在於 `georef1` 的檔案，也沒有不同的檔案。只存在於 `conflict1` 的檔案全部在 `conflict_analysis/` 之下，包括 mot_tracks 72、yolo_labels 441、conflict_events 44、conflict 42、other 471 個。其餘類別（trackeval、grid_search、reid_crops、homography）兩邊相同。
- 同一次執行中，`conflict_analysis/collision_output/` 的既有 stage3 輸出（44 個檔案）與 `collision_detect/collision_output/` 逐位元相同。
- TOLERANCE.md 新增「基準切換：georef1 → conflict1」，CLAUDE.md 的基準名稱與情境數已同步更新。

### 文件（任務 6.5）

- CONTEXT.md 的衝突術語與 spec 一致；行進方向的定義改為「最多一個」（移動太慢時沒有）。
- CLAUDE.md「衝突事件／Homography」一節新增衝突指標分析的說明，常用指令新增 `config/conflict_analysis.yaml`。
- `config/conflict_analysis.yaml` 檔頭已寫明用法；`chain_dist_m` 的註解改為實際的連通條件。
- design 第 124 行已修正：「DRAC 超限」需要 MaxD，改列為加速度類指標。
