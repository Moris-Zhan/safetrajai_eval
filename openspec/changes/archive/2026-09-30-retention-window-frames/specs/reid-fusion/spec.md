# Spec Delta

## MODIFIED Requirements

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

### Requirement: 新參數在追蹤開始前驗證

`reid_center_gate_px` 設定時 MUST > 0，`reid_center_gate_px_per_frame` MUST ≥ 0，`appearance_veto_sim` MUST 介於 0 與 1 之間；不符合時 SHALL 在載入追蹤設定時以錯誤停止，錯誤訊息指出類別與參數名稱，不得開始追蹤。

#### Scenario: 無效的否決門檻
- **WHEN** 某類別設定 `appearance_veto_sim: 1.5`
- **THEN** 載入追蹤設定時以錯誤停止，訊息包含該類別名稱與 `appearance_veto_sim`
