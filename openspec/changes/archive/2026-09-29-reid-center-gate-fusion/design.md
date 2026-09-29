# Design

## Context

動機見 proposal.md，需求見 `specs/reid-fusion/spec.md`、`specs/reid-model/spec.md`。現況：

- `ClassAwareBOTSORT.update()`（凍結）在第一階段以 `strack_pool = tracked ∪ lost` 逐類別呼叫 `get_dists_class_aware()`，再以該類別的 `match_thresh` 做匈牙利匹配；未確認軌跡那一階段也呼叫同一個函式。第二階段（低分偵測）只用 IoU、只處理 Tracked，不在本 change 範圍。
- `get_dists_class_aware()`（可修改）目前的順序：IoU 距離 → `dists_mask = iou > 1 − proximity_thresh` → 可選的中心距離線性融合（`center_dist_weight`、`norm_factor`）→ 可選的 `fuse_score` → 若 `use_reid`：`emb = embedding_distance / 2`，`emb[emb > 1 − appearance_thresh] = 1`、`emb[dists_mask] = 1`，`dists = min(dists, emb)`。
- 遺失中的軌跡在匹配前已做 Kalman `multi_predict` 與 GMC，框是外推的預測位置；5fps 影片上 `dt_scale` 會把速度項放大 6 倍，遺失數秒後預測可能偏離很遠。
- **ultralytics 的 `STrack` 不保存最後觀測到的框**：`_tlwh` 只有建立時的框，`update()` 只更新 Kalman 狀態。track 類別與 `update()` 都不能改。
- `track.frame_id` 是最後一次更新的幀，`self.frame_id − track.frame_id` 是已遺失的幀數；tracker 的 `self.frame_rate` 是實際幀率（`frame_rate: auto`）。
- 使用 Re-ID 的類別，每個偵測都有 `curr_feat`，每條 track 都有 `smooth_feat`。
- `class_tracking_params` 的任意鍵會原樣進入 `CLASS_TRACKING_PARAMS`，新增參數只需要加驗證。
- 調參序列的 GT 每一幀都有標註（每段 71～75 幀、約 5fps、39～60 條 GT 軌跡），small_truck 與 truck 的框較少。
- `train-reid-timm` 的驗收工具（`runs/reid_accept/` 子集、config 產生器、彙總腳本）與基準數字可沿用。

## Goals / Non-Goals

**Goals:**
- 讓外觀距離能重新關聯遮擋後位置偏移或停下的遺失中的軌跡，以及可選地否決外觀明顯不同的配對。
- 未設定新參數時距離矩陣逐位元不變。

**Non-Goals:**
- 不改 `update()`、`init_track()` 等凍結方法，因此每個框各呼叫一次 encoder 的成本不變。
- 不改變 Tracked 與未確認 track 的外觀適用範圍（降低密集車流中相鄰同款車互搶的風險）。
- 不追求最佳門檻；只試少量有依據的組合，沒有類別通過就停止。

## Decisions

### D1. 融合的計算放在純函式，`get_dists_class_aware()` 只負責準備輸入

- 在 `matching.py` 新增純函式，輸入：運動距離矩陣、外觀距離矩陣、IoU proximity 遮罩、中心距離（像素）矩陣、每條 track 是否 Lost、每條 track 的閘門半徑、`appearance_thresh`、否決門檻（可為 None）；輸出融合後的成本。以合成矩陣做單元測試，不需要建立追蹤器。
- `get_dists_class_aware()`：類別 `use_reid` 為 false，或新參數都沒設定時，**原樣走現有程式碼**（不經過新函式），保證逐位元不變；否則準備輸入後呼叫新函式。
- 外觀成本的適用遮罩：非遺失中軌跡的列＝IoU proximity 遮罩（與現在相同）；遺失中軌跡的列＝IoU proximity 遮罩 **或** 中心距離 ≤ 閘門半徑（聯集，使用者決定）。
- 否決在融合之後套用到所有列：外觀距離 > `1 − appearance_veto_sim` 的配對成本設為 1。只設否決、沒設閘門時，適用遮罩與現在相同。
- **替代方案**：在 `update()` 後另加一個 lost-track 重新配對階段。**不採用的原因**：需要修改凍結的 `update()`（ADR 0004）。
- **替代方案**：加權融合 `λ·運動 + (1−λ)·外觀`。**不採用的原因**：使用者選擇保留 `min`，改動最小、影響只限於遺失中的軌跡的閘門內配對。

### D2. 中心距離：預測中心與最後位置取較近者

- 預測中心：track 目前（預測後）的 `tlwh` 中心，與 `_center_distance()` 的定義相同，但以像素計、不除以 `norm_factor`。
- 最後位置：`get_dists_class_aware()` 每次遇到 **Tracked** 狀態的 track，就把它當下的中心記在 track 物件的附加屬性上（`track._reid_last_center`）；track 變成 Lost 後使用這個值。這個值是「最後一次 Tracked 時的一步預測位置」，與真正的觀測位置只差一幀的預測誤差，遠小於閘門半徑。不修改任何類別或凍結方法；屬性隨 track 物件一起消失，不需要清理。只有啟用閘門的類別才記錄。
- 沒有記錄值（例如 track 在啟用閘門前就已經遺失）時，只用預測中心。
- 中心距離 = `min(‖det − 預測中心‖, ‖det − 最後位置‖)`。
- 閘門半徑 = `reid_center_gate_px + reid_center_gate_px_per_s × lost_s`，`lost_s = (self.frame_id − track.frame_id) / self.frame_rate`。
- **替代方案**：只用預測中心。**不採用的原因**：停下的車（等紅燈）在遮擋期間仍會被外推，預測位置偏離實際位置。
- **替代方案**：在 tracker 上維護 `{track_id: center}` 字典。**不採用的原因**：效果相同，但要處理 track 移除時的清理。

### D3. 參數驗證在 `load_track_config()`

- 驗證規則見 spec「新參數在追蹤開始前驗證」；錯誤以 `ValueError` 丟出，訊息含類別名稱與參數名稱。`load_track_config()` 在 tracker 建立前呼叫。
- 只驗證有出現的鍵；既有 config 沒有這些鍵，不受影響。

### D4. 候選門檻由調參序列 GT 的外觀距離分布決定

- 新增 `scripts/reid_train/fusion_distance_stats.py`（放在 repo，之後換模型或融合方式都要重新量）：
  - 輸入：Re-ID 權重、調參子集路徑（`dataset/eval/cctv_all8_subsets/tune/`）、資料集版本。路徑中含保留序列的時段（152639）或不是調參子集時以錯誤停止。
  - 讀取調參序列的影片與 MOT GT，逐幀裁切 GT 框（與推論端相同的 `save_one_box` 裁切與 letterbox），以 `Custom_ReID` 相同的前處理抽特徵；只在記憶體中處理，不寫出裁切。
  - 正樣本：同一條 GT 軌跡、間隔 ≥ 1 秒（依影片幀率換算）的兩個框。
  - 硬負樣本：同一幀、同類別、中心距離 ≤ 2×該類別 `norm_factor` 的不同 GT 軌跡；某類別少於 30 對時改用同類別任意不同 GT 軌跡的配對，輸出中標示。
  - 另以資料集版本 val 的特徵計算同樣的百分位數（val 的負樣本為同類別任意配對），只作對照。
  - 輸出各類別的百分位數表（JSON）。
- 候選：寬 `appearance_thresh` = 1 − P95(正樣本)；嚴 = 1 − P5(硬負樣本)，若比寬的更寬則取兩者較嚴者；`appearance_veto_sim` = 1 − P99、1 − P99.9(正樣本)。
- GT 的分布只用來挑候選值；啟用與否仍以追蹤指標判定（`reid-model` spec）。

### D5. 驗收網格（沿用 `train-reid-timm` D5，調參序列、`dinov2s_v3`）

- 類別：person、car、scooter、small_truck、truck；bus 不測（調參序列無 GT，維持關閉）。
- 閘門半徑以該類別的 `norm_factor` 為單位。
- **A 閘門**：`reid_center_gate_px ∈ {1×, 2×} norm_factor` × `appearance_thresh ∈ {寬, 嚴}`，`px_per_s = 0`：每類別 4 組，共 20 組。
- **B 放寬**：A 中該類別通過的最佳者加上 `reid_center_gate_px_per_s = norm_factor`：每類別最多 1 組（A 沒有通過的類別跳過）。
- **C 否決**：不設閘門，`appearance_veto_sim ∈ {1 − P99, 1 − P99.9}`（正樣本）：每類別 2 組，共 10 組。
- **D 合併**：同一類別的閘門與否決都通過時，試兩者一起開；再把所有啟用的類別一起開，做合併驗證。
- 判定：依 `reid-model` spec（不變差，且 IDSW 減少或 IDP 上升）。同一類別多組通過時選 IDSW 最少者，再依 IDP、AssA。每組記錄每幀追蹤時間。
- 保留序列：只跑基準與最終設定，依 spec 只否決整體。
- 沒有類別通過：`config/botsort_reid_timm.yaml` 全部 `use_reid: false`（等同驗收基準），檔頭註明原因與模型路徑，change 以負面結果收尾，不再做第二輪調參（使用者決定）。
- 基準與候選在同一批平行執行，確保追蹤時間可比較。

### D6. 相容性驗證

- 單元測試：修改前先以隨機的 tracks／detections（含遺失中的軌跡）固定 `get_dists_class_aware()` 的輸出，修改後在未設定新參數時比對逐元素相同。
- golden：`tests/golden/run_all.sh` 跑一次（約 16 分鐘，執行前告知使用者），以 `compare.py reid1 <run>` 確認差異為 0。
- `config/botsort_reid.yaml`、golden configs 不改；smoke 測試的 AST 檢查確認凍結方法未變。

## Risks / Trade-offs

- **[風險] 閘門內的不同物體以外觀配上（串錯增加）** → 只作用在遺失中的軌跡；嚴門檻由硬負樣本決定；IDSW／IDP 不變差才可啟用；另有否決可用。
- **[風險] 否決打斷外觀劇變（半遮擋、邊緣進出）的正確配對，造成碎片** → 否決門檻取正樣本的 P99／P99.9；IDSW 增加的類別不會啟用。
- **[風險] 最後位置是一步預測，不是真正的觀測** → 誤差為一幀的預測誤差，遠小於閘門半徑；單元測試驗證記錄與使用的時機。
- **[風險] 校準讀了評估鏡頭的調參序列** → 只量距離、不訓練、不寫裁切；保留序列由程式拒絕。
- **[風險] 調參序列小，改善可能只有 1～2 個 ID** → 改善條件要求身分層級的指標；verification 如實標示幅度；保留序列否決整體。
- **[取捨] 追蹤時間** → 與 `train-reid-timm` 相同，只記錄、不設上限。

## Migration Plan

1. D6 的回歸測試先寫；實作 D1～D3 與單元測試；跑 `pytest`。
2. D4 量測距離分布，定出候選門檻。
3. D5 驗收，更新 `config/botsort_reid_timm.yaml`。
4. D6 golden 比對；更新文件。

**還原方式**：新參數都是選用的，從 config 移除即回到原行為；程式以 git revert 還原。
