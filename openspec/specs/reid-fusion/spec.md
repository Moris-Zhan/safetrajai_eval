# reid-fusion Specification

## Purpose

定義 BoTSORT 追蹤器在逐類別匹配時如何把外觀距離與運動距離融合：外觀距離在什麼範圍內可以降低配對成本、何時否決配對，以及沒有設定新參數時必須維持的相容行為。

## Requirements

### Requirement: 未設定新參數時融合行為不變

類別的 `class_tracking_params` 沒有設定 `reid_center_gate_px` 與 `appearance_veto_sim` 時，該類別的配對成本 SHALL 與本 change 之前逐位元相同：外觀距離（特徵 cosine 距離的一半，範圍 0～1）只作用在 IoU 距離 ≤ `1 − proximity_thresh` 的配對上，外觀距離 > `1 − appearance_thresh` 的配對不給外觀成本，最後取運動距離與外觀距離的較小者。`use_reid` 為 false 的類別 MUST 忽略新參數。既有情境設定與 golden 情境的輸出 MUST NOT 改變。

#### Scenario: 既有情境輸出不變
- **WHEN** 以本 change 之後的程式重跑 golden 情境
- **THEN** 輸出與基準的差異為 0

#### Scenario: 關閉 Re-ID 的類別忽略新參數
- **WHEN** 某類別 `use_reid: false`，但設定了 `reid_center_gate_px`
- **THEN** 該類別的配對成本只由運動距離決定，與未設定時相同

### Requirement: 中心距離閘門讓外觀距離重新關聯遺失中的軌跡

類別設定 `use_reid: true` 與 `reid_center_gate_px`（像素，> 0）時，對遺失中的軌跡（追蹤器狀態 Lost），配對只要「IoU 距離 ≤ `1 − proximity_thresh`」或「中心距離不超過閘門半徑」其中之一成立，SHALL 可取得外觀成本。中心距離 SHALL 取偵測框中心到「track 預測框中心」與「track 最後一次處於 Tracked 時的框中心」兩者的較小值。閘門半徑 SHALL 為 `reid_center_gate_px` 加上 `reid_center_gate_px_per_frame`（像素／幀，未設定時為 0）乘以 track 已遺失的幀數（目前幀號減去 track 最後一次更新的幀號）。外觀距離 > `1 − appearance_thresh` 的配對仍不給外觀成本；配對成本仍為運動距離與外觀距離的較小者。不是遺失中的軌跡（Tracked、未確認）MUST 維持只以 IoU proximity 判定外觀成本的規則。舊的參數名稱 `reid_center_gate_px_per_s` 出現在設定中時 SHALL 在載入追蹤設定時以錯誤停止，並在訊息中指出新的參數名稱。

#### Scenario: 遮擋後位移的遺失中的軌跡可由外觀重新關聯
- **WHEN** car 的一條遺失中的軌跡的預測框與新偵測框 IoU 為 0，中心距離 60 像素，閘門半徑 80 像素，兩者外觀距離 0.1，`appearance_thresh` 0.8
- **THEN** 這個配對的成本為 0.1（外觀距離），可在該類別的 `match_thresh` 內配對

#### Scenario: 停下的物體以最後位置重新關聯
- **WHEN** 一條遺失中的軌跡的預測框因外推而離偵測框 150 像素，但它最後一次 Tracked 時的中心離偵測框 30 像素，閘門半徑 80 像素，外觀距離在門檻內
- **THEN** 這個配對可取得外觀成本

#### Scenario: 閘門外的配對不取得外觀成本
- **WHEN** 一條遺失中的軌跡的兩個中心距離都超過閘門半徑，而且 IoU 距離大於 `1 − proximity_thresh`
- **THEN** 這個配對的成本等於運動距離

#### Scenario: IoU 重疊的遺失中的軌跡仍可取得外觀成本
- **WHEN** 一條遺失中的軌跡的中心距離超過閘門半徑，但 IoU 距離 ≤ `1 − proximity_thresh`，外觀距離在門檻內
- **THEN** 這個配對可取得外觀成本（與本 change 之前相同）

#### Scenario: 閘門隨遺失時間放寬
- **WHEN** `reid_center_gate_px` 為 80、`reid_center_gate_px_per_frame` 為 8，track 已遺失 7 幀，較近的中心距離為 120 像素
- **THEN** 閘門半徑為 136 像素，配對可取得外觀成本

#### Scenario: Tracked 的 track 不受閘門影響
- **WHEN** 一條 Tracked 的 track 與一個偵測 IoU 為 0、中心距離 40 像素、外觀距離 0.05，閘門半徑 80 像素
- **THEN** 這個配對的成本等於運動距離

#### Scenario: 外觀差異大的配對不取得外觀成本
- **WHEN** 遺失中的軌跡的配對在閘門內，但外觀距離大於 `1 − appearance_thresh`
- **THEN** 這個配對的成本等於運動距離

#### Scenario: 舊參數名稱被拒絕
- **WHEN** 某類別設定 `reid_center_gate_px_per_s: 40`
- **THEN** 載入追蹤設定時以錯誤停止，訊息指出應改用 `reid_center_gate_px_per_frame`

### Requirement: 外觀否決

類別設定 `use_reid: true` 與 `appearance_veto_sim`（0～1，相似度方向）時，外觀距離大於 `1 − appearance_veto_sim` 的配對 SHALL 視為不可配對（成本 1），不論 track 狀態與運動距離；未設定時 MUST NOT 否決任何配對。否決與中心距離閘門可以各自單獨開啟。

#### Scenario: 外觀差異大的鄰近配對被否決
- **WHEN** scooter 設定 `appearance_veto_sim` 0.6（外觀距離上限 0.4），一條 Tracked 的 track 與一個偵測 IoU 很高但外觀距離 0.5
- **THEN** 這個配對的成本為 1，不會被配對

#### Scenario: 未設定否決時不影響運動配對
- **WHEN** 同上，但未設定 `appearance_veto_sim`
- **THEN** 這個配對的成本與本 change 之前相同

### Requirement: 新參數在追蹤開始前驗證

`reid_center_gate_px` 設定時 MUST > 0，`reid_center_gate_px_per_frame` MUST ≥ 0，`appearance_veto_sim` MUST 介於 0 與 1 之間；不符合時 SHALL 在載入追蹤設定時以錯誤停止，錯誤訊息指出類別與參數名稱，不得開始追蹤。

#### Scenario: 無效的否決門檻
- **WHEN** 某類別設定 `appearance_veto_sim: 1.5`
- **THEN** 載入追蹤設定時以錯誤停止，訊息包含該類別名稱與 `appearance_veto_sim`

### Requirement: 門檻校準只使用調參序列

量測外觀距離分布以決定候選門檻時，SHALL 只讀評估資料集的調參序列（C000025 的 145630、150634 時段）與資料集版本的 val；MUST NOT 讀取保留序列，MUST NOT 產生任何裁切或訓練資料。

所有時間間隔 SHALL 以幀數表示。正樣本 SHALL 為同一條 GT 軌跡的兩個框，並以兩種間隔分別回報：「≥ 1 秒」（依影片幀率換算成幀數，與先前的量測可比較），以及「最小間隔幀數（預設 5 幀）≤ 間隔 ≤ 該類別的保留期限幀數」（追蹤器實際會嘗試重新關聯的範圍）；候選門檻與外觀分離度關卡使用後者。保留期限幀數 SHALL 直接取所用設定檔中該類別的 `track_buffer`（幀數，見 ADR 0009），不依追蹤器或影片幀率換算；校準 C000025 時 SHALL 使用與追蹤驗收相同的設定檔 `config/botsort_reid_timm.yaml`。

負樣本 SHALL 為干擾物：同一幀、同類別、中心距離在該類別兩倍 `norm_factor` 以內的不同 GT 軌跡的兩個框；某類別的干擾物配對少於 30 對時改用同類別任意不同 GT 軌跡的配對，並在輸出中標示。

每個類別另外 SHALL 依配對中較小的框的最短邊分成 < 16px、16～32px、≥ 32px 三組分別回報；分組結果只供參考，不影響候選門檻與關卡。

#### Scenario: 拒絕保留序列
- **WHEN** 以保留序列的路徑執行校準
- **THEN** 以錯誤停止，沒有輸出任何結果

#### Scenario: 硬負樣本不足時退回
- **WHEN** truck 在調參序列上符合干擾物條件的配對只有 12 對
- **THEN** truck 改用任意不同 GT 軌跡的配對，輸出中標示 truck 使用了退回規則

#### Scenario: 正樣本間隔以保留期限為上限
- **WHEN** scooter 的 `track_buffer` 為 10、影片幀率 5，同一條 GT 軌跡有間隔 8 幀與 25 幀的兩個配對
- **THEN** scooter 的保留期限為 10 幀；兩個配對都列入「≥ 1 秒」的回報，只有 8 幀的配對列入候選門檻與關卡使用的正樣本

#### Scenario: 保留期限不依追蹤器或幀率換算
- **WHEN** 設定檔為 ByteTrack、car 的 `track_buffer` 為 15、影片幀率 30
- **THEN** car 的保留期限為 15 幀

#### Scenario: 分尺寸回報
- **WHEN** 一個 car 的配對中，兩個框的最短邊分別是 12px 與 30px
- **THEN** 這個配對歸在 < 16px 那一組

### Requirement: 量測排除沒有身分的 GT 框並回報物體配對數

量測外觀距離分布與外觀分離度時，GT id 小於 0 的框（沒有身分的標註）MUST NOT 作為正樣本、干擾物或退回配對的任何一方；輸出 SHALL 記錄每個類別跳過的框數。GT 檔案 MUST NOT 被修改。

每個類別 SHALL 回報干擾物配對來自多少組不同的 GT 物體配對（同一序列中兩條不同 GT 軌跡的無序組合；同一組物體在多個幀的配對只算一組）。

#### Scenario: 沒有身分的框不參與量測
- **WHEN** 調參序列中有 2 個 small_truck 框的 GT id 為 -1，其中 1 個在同一幀、半徑內與另一個 small_truck 框相鄰
- **THEN** 這 2 個框不出現在任何正樣本、干擾物或退回配對中，輸出記錄 small_truck 跳過 2 個框

#### Scenario: 同一組物體的多幀配對只算一組
- **WHEN** small_truck 的干擾物共 41 對，來自軌跡 #24 與 #27（17 幀）以及 #5 與 #27（24 幀）
- **THEN** small_truck 的干擾物物體配對數為 2
