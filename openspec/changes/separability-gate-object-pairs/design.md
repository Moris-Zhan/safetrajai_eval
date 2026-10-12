# Design

## Context

動機見 proposal.md。現況：

- `fusion_stats.py`：`read_mot_gt` 讀入全部 GT 框（含 id < 0）；`hard_negative_pairs` 以「同序列、同幀、同類別、不同 gt_id、半徑內」產生幀配對；`class_stats` 在幀配對少於 30 對時改用任意配對並標 `any_fallback`；`separability_gate` 對 `negative_kind == "hard"` 的其他類別檢查 AUC 下降。
- 調參序列的干擾物來源（以 `botsort_reid_timm.yaml` 的半徑）：car 754 組物體配對、scooter 367（含 id -1 的框；排除後 356）、person 20、small_truck 3（含一組 id -1；排除後 2）、truck 1、bus 0。id < 0 的框只有 `150634_15s_1` 的 4 個（scooter、small_truck 各 2；探索時曾誤記為 small_truck 與 bus）。
- 量測輸出在 `runs/reid_domain/*.json`（`align-reid-training-domain` 產生）；權重 `dinov2s_v3`、`_seed1`、`_lowres`、`_distractor`、`_both` 都在 `models/class_reid_weights/`。
- 追蹤驗收的流程（`reid-center-gate-fusion` D5、`align-reid-training-domain` D6）沒有留下可重用的程式：當時的 config 產生器與彙總腳本只在暫存位置，`runs/reid_accept/configs/*.yaml` 只有產生出的設定檔。

## Goals / Non-Goals

**Goals:**
- 關卡以獨立的 GT 物體配對計算「其他類別」的證據，並排除沒有身分的 GT 框。
- 以修正後的量測重新判定既有候選；通過者完成追蹤驗收，且驗收流程可重現。

**Non-Goals:**
- 不重新訓練、不改資料、不改 GT 檔。
- 不改校準融合門檻的 30 對退回規則、最小改善量與雜訊底線的算法、追蹤驗收的逐類別規則與保留序列規則。
- 不改追蹤器的融合方式與凍結方法。

## Decisions

### D1. 跳過 id < 0 的框

- 在 `read_mot_gt` 之後、配對之前過濾：id < 0 的框不進入 boxes，因此不參與正樣本、干擾物或退回配對；特徵也不抽取。每個類別記錄跳過數，寫入輸出 JSON（`skipped_unlabeled_boxes`）。
- **替代方案**：只在干擾物中排除。**不採用的原因**：id -1 的多個框若被當成同一條軌跡，也會形成錯誤的正樣本。

### D2. 物體配對數與關卡條件

- `class_stats` 為每個類別回報 `negative_object_pairs`：干擾物（hard）幀配對對應的 `(序列, min(id), max(id))` 不重複數；類別使用退回規則時回報 0（退回配對不是干擾物，不能當成證據）。
- `separability_gate`：
  - 其他類別：參考與候選的 `negative_object_pairs` 都 ≥ `MIN_OBJECT_PAIRS`（10）才檢查 AUC 下降；不足者寫入回報（`unchecked_classes`，含物體配對數與 AUC 變化），不影響通過與否。
  - car、scooter：參考或候選的 `negative_object_pairs` < 10 時判定失敗，原因註明哪一類證據不足。
  - 取代原本的 `negative_kind == "hard"` 條件（hard 只代表幀配對 ≥ 30）。
- N=10 的理由：4～20 之間對目前結果都相同，取一般認為多於少數個案的最小樣本；寫成常數，不開放 CLI 參數，避免在看到結果後調整。
- **替代方案**：以物體配對為單位的群集 bootstrap 信賴區間。**不採用的原因**：2～3 組物體配對時區間極寬，結論與下限相同，卻多了亂數種子與實作複雜度（使用者決定）。

### D3. 重新量測與判定

- 以新程式重跑 6 份量測到 `runs/reid_domain/v2/`（不覆寫 `align-reid-training-domain` 的輸出，便於對照）：參考 `dinov2s_v3 --legacy-crop`，候選 `dinov2s_v3`、`_seed1`、`_lowres`、`_distractor`、`_both`（緊框）。
- 雜訊底線仍由 `_seed1` 與 `dinov2s_v3`（緊框）計算；每個候選以 `--reference`、`--noise-reference` 產生關卡判定，再以 `select_best` 選出最佳者。

### D4. 追蹤驗收工具（新增 `scripts/reid_train/acceptance_grid.py`）

- `configs`：以 `config/botsort_reid_timm.yaml` 為底，套用 `--set CLASS.KEY=VALUE`（例如 `car.use_reid=true`、`car.model=...`、`car.reid_center_gate_px=160`），為 `tune`／`holdout` 子集（`dataset/eval/cctv_all8_subsets/`）各產生一份設定，輸出到 `runs/reid_domain/accept/<name>/<subset>/`；載入前以 `load_track_config()` 驗證，無效參數直接報錯。
- `summarize`：讀取每組的 TrackEval 輸出，彙總方式與 `train-reid-timm` 相同（IDSW、CLEAR TP 各序列加總；IDP 以 IDTP／IDFP 加總後計算；HOTA、AssA 依 GT 框數加權平均；每幀追蹤時間），輸出逐類別與整體表格（CSV＋Markdown），並依 `reid-model` spec 標示每組每類別是否通過。
- 判定邏輯寫成純函式，以單元測試涵蓋 spec 的情境（變差、只有 AssA 上升、IDSW 減少、無 GT）。
- **替代方案**：沿用 `runs/reid_accept/configs/` 的 YAML 手動改。**不採用的原因**：40 組以上手改容易出錯，且無法重現。

### D5. 追蹤驗收網格（只對通過關卡的最佳候選）

- 先以該候選重跑 `fusion_distance_stats.py`，取新的候選門檻（寬／嚴 `appearance_thresh`、否決門檻）。
- 調參序列上與基準同批平行執行：A 閘門（`reid_center_gate_px ∈ {1×, 2×} norm_factor` × 寬／嚴，`reid_center_gate_px_per_frame = 0`）20 組；C 否決（`appearance_veto_sim ∈ {1 − P99, 1 − P99.9}`）10 組；原本融合方式（不設新參數）寬／嚴 10 組；共 40 組，類別為 person、car、scooter、small_truck、truck（bus 無 GT）。
- 依 `reid-model` spec 逐類別判定；同類別多組通過時選 IDSW 最少者，再依 IDP、AssA。有通過的類別再做合併驗證（所有啟用類別一起開）與保留序列（只否決整體），最後更新 `config/botsort_reid_timm.yaml`；沒有類別通過時維持 6 類關閉，只更新檔頭說明。
- 執行前回報預估時間給使用者（Re-ID 會讓每幀追蹤時間增加數倍）。

## Risks / Trade-offs

- **[風險] 規則在看到結果後修改** → N 取在對結果不敏感的範圍、寫成常數；追蹤驗收仍逐類別判定，保留序列獨立否決整體；small_truck 是否啟用仍由它自己的追蹤結果決定。
- **[風險] 排除 id < 0 改變了參考的數字** → 參考與所有候選都以新程式重新量測，不混用舊輸出；`runs/reid_domain/` 的舊輸出保留對照。
- **[風險] small_truck 的真實退步被忽略** → 回報中保留不列入判定的類別與其 AUC 變化；追蹤驗收時 small_truck 若變差，依 spec 不啟用。
- **[取捨] 新增驗收工具擴大了程式範圍** → 只做設定產生與彙總，不改 pipeline；以 `--help` smoke 與單元測試涵蓋。
- **[風險] 40 組驗收耗時** → 先以 1 組冒煙確認流程與每幀時間，估算後回報再全量執行。

## Migration Plan

1. D1、D2 與測試；`pytest`。
2. D3 重新量測與判定，結果寫入 `verification.md`。
3. 有候選通過：D4 工具與測試 → D5 驗收 → 更新 config；沒有通過：記錄並結束。
4. 更新文件。

還原：程式以 git revert；新的量測輸出在 `runs/reid_domain/v2/`、驗收輸出在 `runs/reid_domain/accept/`，可直接移除；config 以 git 還原。
