# conflict-analysis Specification

## Purpose

定義 stage3 選用的衝突指標分析：在有地理對位的鏡頭上，以公尺為單位，從 stage1 軌跡計算逐幀配對的替代安全指標、追撞鏈、PET 與衝突事件，並規定輸出格式。指標只是技術訊號，不能當成正式的安全結論。

## Requirements

### Requirement: 分析的啟用與適用範圍

衝突指標分析 SHALL 只在情境 config 的 `conflict_analysis.enabled` 為 true 且 stage3 開啟時執行；沒有這個區塊時視為關閉。每個序列只有在鏡頭有 Homography、而且 `m_per_pixel` 可用時才計算；否則該序列只輸出摘要，記錄略過原因（`no_homography` 或 `no_meters_per_pixel`），不得退回像素單位計算。

#### Scenario: 預設關閉
- **WHEN** 以沒有 `conflict_analysis:` 區塊的情境跑 stage3
- **THEN** `collision_output/<seq>/` 下不產生 `conflict/` 目錄

#### Scenario: 鏡頭沒有緯度
- **WHEN** 分析開啟，但序列所屬鏡頭的 `m_per_pixel` 不可用
- **THEN** `conflict/summary.json` 的 `skipped_reason` 為 `no_meters_per_pixel`，不產生其他輸出檔

### Requirement: 不影響既有 stage3 輸出

不論分析是否開啟，既有 TTC／PET 的事件檔、log CSV 與熱區圖 SHALL 與不做分析時逐位元相同。

#### Scenario: 開啟分析時既有輸出不變
- **WHEN** 以同一份 stage1 輸出分別在分析開啟與關閉下跑 stage3
- **THEN** 兩次的 `ttc_events.*`、`pet_events.*`、`collision_*_log.csv`、熱區圖完全相同

### Requirement: 位置取點與軌跡識別

每個框的位置 SHALL 取底邊中點 `(left + w/2, top + h)`，經鏡頭 Homography 投影到俯視平面，再乘以 `m_per_pixel` 換算成公尺。是否位於 ROI 內 SHALL 以框中心點判斷（與現行 stage3 相同），ROI 外的框不參與分析。軌跡 MUST 以「類別＋track_id」識別，所有輸出都同時列出類別與 track_id。

#### Scenario: 底邊中點投影
- **WHEN** 一個框為 `left=100, top=200, w=40, h=20`
- **THEN** 投影的是 CCTV 像素 `(120, 220)`

#### Scenario: 不同類別相同 track_id
- **WHEN** car 與 scooter 各有一條 track_id 為 7 的軌跡
- **THEN** 兩者被視為不同軌跡，輸出中可以分辨

### Requirement: 速度與加速度

速度與加速度 SHALL 在公尺座標下計算，步驟如下：
1. 平滑：位置以置中移動平均處理，視窗以秒數設定（`smooth_window_s`），依序列 fps 換算成 `2·floor(smooth_window_s·fps/2) + 1` 幀（必為奇數，至少 1）。
2. 速度：以中心差分（依實際時間差）求得。
3. 加速度：對速度再做一次中心差分。

其他規則：
- 軌跡端點與無法做中心差分的幀，速度或加速度 MUST 為未定義，不得補 0。
- 幀差超過 `max_gap_frames` 時，軌跡在該處切成兩段分別計算。
- 速率低於 `min_speed_mps` 時，航向為未定義。
- 序列 fps 低於 `min_fps_for_acceleration` 時，加速度一律為未定義。

#### Scenario: 等速直線
- **WHEN** 一條軌跡以 10 m/s 等速直線前進 30 幀
- **THEN** 中間各幀速率為 10 m/s（誤差 < 1e-6），加速度為 0，第一幀與最後一幀的速度未定義

#### Scenario: 平滑視窗換算
- **WHEN** `smooth_window_s` 為 0.6，序列分別為 4.9 fps 與 30 fps
- **THEN** 視窗分別為 3 幀與 19 幀

#### Scenario: 低幀率序列
- **WHEN** 序列為 4.9 fps，`min_fps_for_acceleration` 為 10
- **THEN** 所有加速度為未定義，速度仍正常計算

#### Scenario: 軌跡中斷
- **WHEN** 一條軌跡在第 10 幀與第 30 幀之間沒有偵測，且 `max_gap_frames` 為 5
- **THEN** 速度不會跨過這個空檔做差分

### Requirement: 同幀配對

每一幀 SHALL 對所有速度已定義的軌跡兩兩配對，以下情況不計算：
- 類別組合不在 config 的 `class_pairs` 清單中（預設是 6 個類別的所有組合，但排除 person–person）；
- 兩條軌跡中心距離大於 `pair_radius_m`；
- 兩條軌跡的速率都低於 `min_speed_mps`。

#### Scenario: 超出配對半徑
- **WHEN** 兩條軌跡同幀相距 80 m，`pair_radius_m` 為 50
- **THEN** 這一幀不產生這個配對

#### Scenario: 預設排除行人對行人
- **WHEN** 同一幀有兩個 person 相距 3 m，config 使用預設的 `class_pairs`
- **THEN** 不產生這個配對；person 與 car 的配對照常產生

#### Scenario: 兩條軌跡都靜止
- **WHEN** 紅燈停等的兩台 car 速率都低於 `min_speed_mps`
- **THEN** 這一幀不產生這個配對

### Requirement: 衝突角度與類型

衝突角度 SHALL 是兩條軌跡速度向量的夾角（0～180 度），計算前把餘弦值限制在 [-1, 1]。類型依 config 的連續區間分類，預設沿用參考實作的三類（ADR 0011），只補上整數區間的缺口：[0, 30] Rear-end、(30, 85] Lane-change、(85, 180] Crossing；任一方航向未定義時，類型為 Undefined。

#### Scenario: 區間邊界
- **WHEN** 夾角為 30.5 度
- **THEN** 類型為 Lane-change

#### Scenario: 接近相向
- **WHEN** 夾角為 170 度
- **THEN** 類型為 Crossing

#### Scenario: 一方靜止
- **WHEN** 一方速率低於 `min_speed_mps`
- **THEN** 夾角未定義，類型為 Undefined

### Requirement: TTC 與 MTTC

配對的碰撞距離 SHALL 預設為固定值 `ttc_distance_m`（2.0 m，沿用參考實作，ADR 0011）；`ttc_distance_mode` 設為 `class_length` 時，改為兩條軌跡類別長度（`object_length_m`）一半的和。

TTC 與 MTTC 的定義：
- TTC：兩條軌跡以目前速度等速外推，中心距離首次小於碰撞距離的時間；
- MTTC：以目前速度與加速度等加速外推，判定方式相同；只在 TTC 有限時計算，TTC 為無限大時 MTTC 也記為無限大，加速度未定義時 MTTC 為未定義；
- 搜尋範圍是 0 到 `ttc_horizon_s`，步進為 `ttc_step_s`，範圍內沒有命中時記為無限大；
- 命中時記錄碰撞點，也就是兩個外推位置的中點。

只計算正在接近的配對（中心距離對時間的變化率小於 0）。t=0 時中心距離已經小於碰撞距離的配對，標成 `overlap_at_start`，TTC 與 MTTC 不計算。

#### Scenario: 正面接近
- **WHEN** 兩條軌跡相距 22 m，在同一直線上以各 10 m/s 相向而行，碰撞距離為預設的 2.0 m
- **THEN** TTC 為 1.0 秒（誤差不超過一個步進）

#### Scenario: 依類別調整碰撞距離（選用模式）
- **WHEN** `ttc_distance_mode` 為 `class_length`，配對分別是 car–car（各 5 m）與 car–scooter（5 m、1.8 m）
- **THEN** 碰撞距離分別是 5.0 m 與 3.4 m

#### Scenario: 一開始就重疊
- **WHEN** 兩條軌跡在 t=0 時中心距離 1.5 m（小於 2.0 m）
- **THEN** 這一列標成 `overlap_at_start`，TTC 與 MTTC 未定義

#### Scenario: 正在遠離
- **WHEN** 兩條軌跡的中心距離正在增加
- **THEN** TTC 與 MTTC 為無限大

#### Scenario: 只有加速度才會接近
- **WHEN** 兩條軌跡速度相同、不會等速接近，但後方軌跡有加速度，且會在範圍內追上
- **THEN** TTC 為無限大，MTTC 也為無限大（MTTC 以 TTC 命中為前提）

### Requirement: 前導與跟隨

TTC 有限的配對 SHALL 判定前導與跟隨：對每條軌跡計算「碰撞點減去自身在 TTC 時刻的外推位置」在自身航向上的投影，投影較小（較已越過碰撞點）的一方是前導，另一方是跟隨。航向未定義或類型為 Undefined 時不判定。

#### Scenario: 追撞
- **WHEN** 同向同車道，後車較快，TTC 有限
- **THEN** 前車為前導，後車為跟隨

### Requirement: 最近前車與後車

每一幀、每個航向已定義的軌跡 SHALL 找出最近前車與最近後車。候選是同幀、類型為 Rear-end 的配對軌跡；以兩者中心連線在該軌跡航向上的投影正負區分前後，取中心距離最小者。沒有候選時為空。

#### Scenario: 三車同向
- **WHEN** A、B、C 同向排成一列，B 在中間
- **THEN** B 的最近前車是前方那台，最近後車是後方那台；A 與 C 各只有一側有值

### Requirement: 追撞鏈與行進方向

軌跡 SHALL 依追撞配對合併成追撞鏈：兩條軌跡的邊成本為 `max(0, 最小夾角 − chain_angle_deg) + chain_dist_weight × max(0, 最小 ΔS − chain_dist_m)`，以 Floyd-Warshall 求最短路徑，總成本不超過 `chain_eps` 的軌跡屬於同一追撞鏈。每條軌跡依航向中位數分為 E [315, 45)、S [45, 135)、W [135, 225)、N [225, 315)（俯視平面北方朝上），並在同一（追撞鏈, 行進方向）內依位置投影排序。

#### Scenario: 一列車隊
- **WHEN** 三台車同向前後相隔 8 m 行駛，`chain_dist_m` 為 10
- **THEN** 三台車屬於同一追撞鏈、同一方向，排序依行進方向由後到前

### Requirement: 追撞後車指標

類型為 Rear-end、前後關係已判定的配對 SHALL 計算以下指標，單位都是公尺制：

- ΔS：中心距離；
- ΔV：速度差的大小；
- DR：後車沿自身航向的減速度，加速時為 0；加速度未定義時為未定義；
- MaxD：後車整條軌跡 DR 的最大值，沒有任何有定義的 DR 時為未定義；
- DRAC：接近速度大於 0 且 ΔS 大於 0 時，為接近速度平方除以兩倍 ΔS，否則未定義（沿用參考實作以中心距離計算，ADR 0011）。

#### Scenario: DRAC
- **WHEN** 後車 15 m/s、前車 10 m/s 同向，中心距離 ΔS 為 10 m
- **THEN** DRAC 為 1.25 m/s²

### Requirement: DRAC 超限

追撞配對 SHALL 計算 DRAC 超限（參考實作稱為 PICUD 的量，ADR 0011）：DRAC 大於後車 MaxD 時為超限，並記錄超限量 `DRAC − MaxD`，以及同一對軌跡所有列中超限的比例。DRAC 或 MaxD 未定義時，DRAC 超限也未定義。

#### Scenario: 超限
- **WHEN** DRAC 為 1.25 m/s²，後車 MaxD 為 0.8 m/s²
- **THEN** 判為超限，超限量為 0.45 m/s²

#### Scenario: 低幀率序列
- **WHEN** 序列 fps 低於 `min_fps_for_acceleration`
- **THEN** MaxD 未定義，DRAC 照常計算，DRAC 超限為未定義

### Requirement: Crash Index

類型為 Rear-end、MTTC 有限且大於 0 的配對 SHALL 計算：

```
CI   = ((V_F + a_F·MTTC)² − (V_L + a_L·MTTC)²) / (2·MTTC)
CI_m = (m_F·(V_F + a_F·MTTC)² − m_L·(V_L + a_L·MTTC)²) / (2·MTTC)
```

V 與 a 為速度與加速度向量的大小（沿用參考實作，恆為正，無法區分加速與減速，ADR 0011）；F 為後車、L 為前車；質量取自 config 的類別質量。加速度未定義（包括低幀率序列）時，CI 與 CI_m 為未定義。

#### Scenario: CI 數值
- **WHEN** V_F=15、V_L=10 m/s，兩車加速度皆為 0，MTTC 為 2.0 秒
- **THEN** CI 為 31.25

#### Scenario: 加速度只取大小
- **WHEN** MTTC 固定為 2.0 秒，其餘條件完全相同，只差在前車的加速度向量與航向同向或反向（大小都是 2 m/s²）
- **THEN** 兩者的 CI 相同

#### Scenario: 低幀率序列
- **WHEN** 序列 fps 低於 `min_fps_for_acceleration`
- **THEN** 所有配對的 CI、CI_m、MTTC、DR、MaxD 都是未定義，summary 註明原因

### Requirement: PET

PET SHALL 以軌跡配對計算。預設沿用參考實作，只納入至少同時出現過一幀的兩條軌跡（ADR 0011）；`pet_max_gap_s` 大於 0 時，改為納入時間範圍相距不超過該值的配對，不要求同時出現。

計算步驟：
1. 衝突點：兩條路徑空間上最接近的兩點的中點；最近距離大於 `pet_zone_m` 時，該配對沒有 PET。
2. 進出時間：每條軌跡距離衝突點在 `pet_zone_m` 以內的最早時間是進入、最晚時間是離開。
3. 兩段停留時間不重疊時，PET 為後到者的進入時間減去先到者的離開時間；重疊時為 0。

#### Scenario: 同時出現過的配對
- **WHEN** 甲在 3.0 秒離開衝突區，乙在 4.2 秒進入，兩者在 2.5～3.0 秒同時出現在畫面中
- **THEN** PET 為 1.2 秒，先到者為甲

#### Scenario: 沒有同時出現（預設）
- **WHEN** 甲在 3.0 秒離開畫面，乙在 4.2 秒才進入畫面，`pet_max_gap_s` 為預設的 0
- **THEN** 這一對軌跡不計算 PET

#### Scenario: 沒有同時出現（選用模式）
- **WHEN** 同上，但 `pet_max_gap_s` 為 5
- **THEN** PET 為 1.2 秒，先到者為甲

### Requirement: 衝突事件彙總

同一對軌跡 TTC 或 MTTC 低於 `event_threshold_s` 的幀，若幀差不超過 `event_gap_frames`，SHALL 合併成一個衝突事件。事件記錄：起訖幀、最小 TTC 與發生幀、最小 MTTC、該幀的類型與碰撞點、前導與跟隨，以及事件期間的最大 DRAC、是否曾 DRAC 超限與最大超限量、最大 CI 與 CI_m。摘要的門檻統計（預設 1.0、1.5、2.0、2.5 秒）以事件數計算，不以逐幀列數計算。標成 `overlap_at_start` 的列不構成事件，但摘要中另外列出這類配對的數量。

#### Scenario: 連續命中
- **WHEN** 同一對軌跡在連續 12 幀的 TTC 都低於門檻
- **THEN** 產生 1 個事件，摘要中該門檻的事件數為 1

#### Scenario: 一開始就重疊不算事件
- **WHEN** 某一對軌跡只有 `overlap_at_start` 的列
- **THEN** 不產生事件，摘要的 `overlap_at_start_pairs` 加 1

### Requirement: 輸出檔與座標

開啟分析且鏡頭可計算時，`collision_output/<seq>/conflict/` SHALL 包含以下檔案：
- `pair_frames.csv`：只含至少一項指標有定義的列；
- `conflict_events.csv` 與 `conflict_events.json`；
- `pet.csv`；
- `chains.csv`；
- `summary.json`：鏡頭、`m_per_pixel`、使用的參數、門檻統計，以及 `limitations`（ADR 0011 列出的指標限制：固定 2 m 碰撞距離、CI 不分加減速、中心距離、PET 要求同時出現，以及本序列是否因低幀率而沒有加速度類指標）；
- 事件熱區圖。

碰撞點與衝突點 SHALL 同時提供公尺、俯視平面像素與 CCTV 像素三種座標，欄位名稱標示座標系。相同輸入連跑兩次，輸出 MUST 完全相同。

#### Scenario: 座標欄位
- **WHEN** 檢查 `conflict_events.csv`
- **THEN** 碰撞點有 `_m`、`_map_px`、`_cctv_px` 三組欄位

#### Scenario: 重現性
- **WHEN** 以 `config/conflict_analysis.yaml` 對同一份 stage1 輸出跑兩次 stage3
- **THEN** 兩次的 `conflict/` 目錄內容完全相同
