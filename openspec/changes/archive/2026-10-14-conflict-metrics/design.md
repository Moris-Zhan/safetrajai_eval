# Design

## Context

動機見 proposal.md，參考實作的逐項分析見 `Sub_Module/trajectoryanalyst2/CONFLICT_METRICS.md`（以下簡稱 CM）。現況與限制：

- stage3 入口是 `TrackingEvalPipeline.run_stage3()`，逐序列呼叫 `safetrajai.collision.run_collision_detection_for_sequence()`。輸入是 stage1 的 7 欄 MOT 檔，逐類別各一檔，`trajectory_adapter.load_sequence_tracks()` 會以 ROI 框中心點過濾。
- 前置 change `camera-homography-georef` 提供兩個介面：依鏡頭解析 Homography（回傳 H 或 None），以及 `meters_per_pixel(camera_cfg)`（回傳 float 或 None）。
- 抽查 golden 的一段 C000025 序列：94 條軌跡在六個類別檔之間沒有重複的 track_id。但 CONTEXT.md 定義 track_id 只在類別內唯一，所以新程式不依賴跨類別唯一。
- golden 比對會把任何多出的檔案算成差異（`compare.py` 的 `only_b`），所以既有 5 個情境不得多出輸出。
- `pyproject.toml` 沒有登記 pandas；可用的是 numpy、scipy、numba。

## Goals / Non-Goals

**Goals:**
- 依 ADR 0011，指標的核心定義以參考實作為準，只修正 CM 列出的錯誤（單位、最近前後車分組、區間缺口、`arccos` 範圍、第一幀補 0）；與文獻不同之處寫進 summary 的 `limitations`。
- 每個指標都是可以獨立測試的純函式，以合成軌跡的解析解驗證。
- 輸出可以重現（固定排序、不使用平行運算的非決定性順序）。

**Non-Goals:**
- 不修改、不取代現行 IoU 版的 TTC／PET。
- 不做路口幾何（實際進口道、車道線）；「方向」只依航向分成四個扇區。
- 不做視覺化影片（CM 的 Demo 模式），只輸出事件熱區圖。
- 不校正門檻的安全意義；預設值沿用參考實作，校準另外處理。

## Decisions

### 1. 模組切分

`src/safetrajai/collision/conflict/`：

| 模組 | 內容 |
|---|---|
| `config.py` | `ConflictConfig`（frozen dataclass），由 `conflict_analysis:` 載入並驗證範圍 |
| `tracks.py` | 讀 MOT、ROI 過濾、底邊中點投影、公尺換算，鍵為 `(class_name, track_id)` |
| `kinematics.py` | 平滑、分段、中心差分、航向 |
| `pairs.py` | 同幀配對（cKDTree 半徑查詢）與夾角、類型 |
| `ttc.py` | TTC、MTTC、碰撞點、前導與跟隨 |
| `following.py` | 最近前後車、ΔS、ΔV、DR、MaxD、DRAC、DRAC 超限、CI、CI_m |
| `chains.py` | Floyd-Warshall 追撞鏈、方向扇區、排序 |
| `pet.py` | 軌跡配對 PET |
| `events.py` | 逐幀合併成事件、門檻統計 |
| `output.py` | CSV／JSON 輸出、三種座標、熱區圖（重用 `collision.output.draw_collision_heatmap`） |
| `__init__.py` | `run_conflict_analysis_for_sequence(seq_output_dir, cfg, fps, bounds)` |

資料以 numpy 陣列與 frozen dataclass 傳遞，不使用 pandas。這樣避免依賴未登記的套件，也比參考實作逐列 merge 的方式容易驗證。

### 2. 呼叫位置

`run_stage3()` 在原本的 `run_collision_detection_for_sequence()` 之後，只有在 `cfg.conflict_analysis.enabled` 時才呼叫新入口，而且新入口自行讀取軌跡。理由：既有函式的程式碼不需要改動，既有輸出逐位元不變的條件最容易成立；代價是軌跡讀取兩次，每段序列只有幾 MB，可以接受。

### 3. 運動學

- 平滑：在每個分段內以置中移動平均處理。視窗以秒數設定（`smooth_window_s`，預設 0.6），換算為 `2·floor(s·fps/2)+1` 幀；4.9 fps 時為 3 幀，30 fps 時為 19 幀。平滑是物理時間的概念，所以不比照 ADR 0009 用幀數設定。分段長度不足視窗時，該段不平滑。
- 速度：`v_i = (p_{i+1} − p_{i−1}) / (t_{i+1} − t_{i−1})`；加速度對速度做相同處理。端點為 NaN。
- 分段：幀差大於 `max_gap_frames`（預設 3）時切開，不內插。
- 預設 `min_speed_mps = 0.5`。
- 低幀率：`min_fps_for_acceleration`（預設 10）。評估鏡頭 C000025 的影片約 4.9 fps（每段 15 秒約 72 幀），加速度要對位置做兩次差分，雜訊會被放大，結果大多不可信。所以 fps 低於門檻時，加速度一律為 NaN，MTTC、DR、MaxD、DRAC 超限、CI、CI_m 跟著是未定義；TTC、ΔS、ΔV、DRAC、PET 只需要速度，照常計算。程式與測試都完成，日後有高幀率影片時就能直接使用。

替代方案 Savitzky-Golay（`scipy.signal.savgol_filter`）可以同時給出平滑的導數，但對短分段與不等間距的處理較複雜。先採用移動平均，在 `summary.json` 記錄所用方法；之後若要更換，改動也只在這個模組內。

### 4. TTC／MTTC 的計算方式

沿用參考實作的「時間取樣後判斷距離」，以向量化一次處理一批配對（`ttc_horizon_s=5.0`、`ttc_step_s=0.02`）。等速情況其實可以解二次方程得到解析解，但 MTTC 是四次式，兩者用同一套取樣方式，結果比較一致，測試也簡單。

- 碰撞距離：預設沿用參考實作的中心距離 2 m（`ttc_distance_m`，ADR 0011）。限制是 2 m 對大型車偏寬鬆（車身早就重疊）、對行人偏嚴格，寫進 `limitations`。`ttc_distance_mode: class_length` 為選用模式，改用兩條軌跡類別長度一半的和（car–car 5.0 m、car–scooter 3.4 m），預設關閉。
- 只算正在接近的配對：相對位置與相對速度的內積小於 0。
- t=0 時已經小於碰撞距離的列標成 `overlap_at_start`，不計算 TTC，避免偵測框抖動或車隊停等產生大量 TTC=0 的假事件。配對時就已排除兩條軌跡都靜止的情況（spec「同幀配對」）。
- MTTC 沿用參考實作，只在 TTC 命中時計算（CM 第 4 節）；只靠加速度才會接近的情況不會出現在 MTTC 中，是 ADR 0011 接受的限制。
- 衝突類型沿用參考實作的三類（Rear-end、Lane-change、Crossing），只把整數區間改成連續區間；區間放在 config，需要時可以自行加入 Head-on。
- `class_pairs` 預設是 6 個類別的所有組合（含同類），但排除 person–person，因為行人之間的接近通常不是交通衝突。

### 5. 前導／跟隨與最近前後車

- 前導判定用一條統一規則（spec），涵蓋追撞與交叉；這等同 CM 5.1 節夾角法的符號判斷，但改用投影值，避免 atan2 在殘差很小時不穩定。
- 最近前後車以 `(frame, 軌跡)` 為分組鍵，修正 CM 5.2 節用欄位位置 1／2 分組的錯誤。

### 6. 追撞鏈參數

邊成本與連通規則沿用參考實作，但 ΔS 改用公尺。預設 `chain_angle_deg=30`、`chain_dist_m=10.0`、`chain_dist_weight=1.0`、`chain_eps=3.0`。

`chain_dist_m` 不沿用參考的 2.0（參考值原本是像素）。邊成本不超過 `chain_eps` 就連通，所以角度相符時的實際條件是「最小 ΔS ≤ chain_dist_m + chain_eps」：2.0 只容許 ΔS ≤ 5 m，大約一個車長，車隊幾乎無法串起來。採用 10.0（ΔS ≤ 13 m），在 C000025 六段序列上最大的追撞鏈有 13 條軌跡，大致是號誌前的一列車隊；各值的分布見 `verification.md`，已與使用者確認（tasks 5.3）。Floyd-Warshall 的節點數就是單一序列中參與追撞配對的軌跡數（約數十個），O(n³) 可以接受。

### 7. 追撞指標的定義選擇（ADR 0011）

- ΔS 與 DRAC 都用中心距離，沿用參考實作；文獻的 DRAC 用車頭到車尾的間距，中心距離會讓 DRAC 偏低，寫進 `limitations`。
- 類別長度與質量（`object_length_m`、`object_mass_kg`）放在 config，預設值沿用參考實作的 `OBJECT_SIZE`；長度只用於 TTC 的選用模式，質量用於 CI_m。
- 參考實作的 `PICUD_flag`／`PICUD_value`／`PICUD_pair`，實際計算的是「DRAC > MaxD」、`DRAC − MaxD` 與同一配對的超限比例。保留這個計算，但輸出欄位改名為 `drac_exceeds_maxd`、`drac_excess`、`drac_exceed_ratio`，不使用 PICUD 的名稱；參考實作中沒有用到的 `PICUD_a`、`PICUD_tr` 不移植。
- CI 與 CI_m 的速度與加速度取向量大小，沿用參考實作；因此無法區分加速與減速，寫進 `limitations`。

### 8. PET

- 候選配對：預設沿用參考實作，只納入至少同時出現過一幀的兩條軌跡（`pet_max_gap_s=0`）；`pet_max_gap_s > 0` 時，改為納入時間範圍相距不超過該值的配對（選用模式）。預設會漏掉「前一方離開畫面後另一方才進入」的情況，寫進 `limitations`。
- 最接近點：對其中一條路徑建 cKDTree，以另一條查詢。
- 進出時間：一樣用 cKDTree 半徑查詢。
- 每對軌跡只取一個衝突點，這和參考實作相同，是已知限制，記錄在 summary 的 `limitations`。

### 9. 事件

- `event_threshold_s` 預設 2.5，等於統計門檻的最大值；`event_gap_frames` 預設 2。
- 事件 ID 以 `(class1, id1, class2, id2, start_frame)` 決定，配對內兩條軌跡依 `(class_name, track_id)` 排序，確保輸出可以重現。

### 10. 輸出與座標

- 公尺座標是俯視平面像素乘以 m/px，原點與俯視平面相同。
- CCTV 像素由 H 的反矩陣投影回去。
- CSV 以固定欄位順序寫出，浮點數用 `repr` 寫，確保連跑結果逐位元相同。
- 熱區圖疊在 `first_frame_with_box.jpg` 上，使用事件在 CCTV 像素中的碰撞點。

### 11. 新情境與 golden

`config/conflict_analysis.yaml` 以 `collision_detect.yaml` 為底，加上 `conflict_analysis.enabled: true`。實作期間的驗收方式是連跑兩次相同，加上與分析關閉時的既有輸出逐位元相同（spec）。

等使用者目視確認抽查的事件之後，再把它加入 golden，成為第 6 個情境：
- `tests/golden/run_all.sh` 的 `SCENARIOS` 加入 `conflict_analysis`；
- `compare.py` 新增 `conflict` 類別；
- 以這次的輸出建立新基準 `conflict1`（連跑兩次自比為 0），在 TOLERANCE.md 記錄。

之後任何指標修改，都會被 golden 抓到。

## Risks / Trade-offs

- [CCTV 偵測框的抖動會放大成加速度] → 用平滑與中心差分處理，低幀率序列不計算加速度類指標；`summary.json` 記錄平滑參數與 fps；事件需要搭配疊圖目視確認後才能解讀。
- [現有評估影片都是約 5 fps] → 這次在真實資料上只能驗證速度類指標（TTC、ΔS、ΔV、DRAC、PET、追撞鏈）；DRAC 超限需要 MaxD，與 MTTC、CI 同屬加速度類；加速度類指標（MTTC、CI 等）只有合成資料的單元測試，要等有高幀率影片才能做實際驗證。
- [5 fps 時兩幀相隔 0.2 秒，TTC 步進 0.02 秒的精度高於資料本身的精度] → 不影響正確性；summary 記錄 fps，解讀時以幀間隔為實際解析度。
- [底邊中點對被遮擋的框不準] → 不另外處理，記錄為已知限制。
- [m/px 只修正緯度，不修正 Homography 本身的誤差] → `summary.json` 一併列出該鏡頭的 check points RMSE（取自前置 change 的報表，沒有時記為 null），讓使用者判斷可信度。
- [配對數隨軌跡數平方成長] → 以 `pair_radius_m` 的 cKDTree 查詢限制配對數；在 C000025 上記錄執行時間，寫入 verification.md。
- [門檻沿用參考值，沒有在地校準] → 結果只能當作技術訊號（CLAUDE.md 既有原則），summary 中註明。

## Migration Plan

純新增功能，預設關閉。回復方式是在 config 關閉，或移除新的子套件；既有輸出不受影響。
