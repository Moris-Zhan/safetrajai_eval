# Verification

## 1. 量測與關卡規則

- `pytest`：405 passed、11 skipped，tracker override 方法的 AST 檢查通過。
- 新增與修改的測試（`tests/test_reid_fusion_stats.py`），修改前皆失敗：
  - 沒有身分的框不參與量測：id -1 的框被排除並逐類別計數，排除後不形成任何配對。
  - 同一組物體的多幀配對只算一組：兩段序列共 41 對幀配對（#24／#27 17 幀、#5／#27 24 幀）→ 2 組；使用退回規則時為 0。
  - 關卡：物體配對不足的類別（small_truck 2 組、AUC 下降 0.03）不列入判定並回報在 `unchecked_classes`；物體配對足夠的類別（person 20 組）下降 0.03 不通過；scooter 只有 6 組時不通過且原因註明物體配對不足；既有關卡情境改用物體配對欄位後仍通過。
- CLAUDE.md 的關卡規則改為以物體配對計算、排除 GT id < 0。

## 2. 重新量測與判定

### 2.1 重新量測

新程式重跑 6 份量測，輸出在 `runs/reid_domain/v2/`（`align-reid-training-domain` 的 `runs/reid_domain/*.json` 保留對照）。設定 `config/botsort_reid_timm.yaml`，調參序列 4 段。

- 跳過的沒有身分的框：scooter 2、small_truck 2（都在 `150634_15s_1`）。探索時把類別欄位誤讀為 small_truck 與 bus，已更正 proposal 與 design。
- 參考的干擾物：

| 類別 | 幀配對 | 物體配對 | 種類 |
|---|---|---|---|
| car | 14,841 | 754 | hard |
| scooter | 6,051 | 356（含 id -1 時 367） | hard |
| person | 476 | 20 | hard |
| small_truck | 41 | 2 | hard（≥ 30 幀配對） |
| truck | 退回 | 0 | any_fallback |
| bus | 退回（無配對） | 0 | any_fallback |

- 排除 id -1 只讓 scooter 與 small_truck 的數字小幅變化（例如參考 scooter TPR@FPR5% 0.6639 → 0.6637、small_truck AUC 0.9874 → 0.9872），car、person、truck 不變。

### 2.2 雜訊底線與關卡

雜訊底線（`seed1` 與 `dinov2s_v3`，都以緊框）：car AUC 0.0006、TPR 0.0102；scooter AUC 0.0020、TPR 0.0141。最小改善量：car AUC 0.01、TPR 0.0204；scooter AUC 0.01、TPR 0.0282。

| 候選（緊框） | car AUC／TPR | scooter AUC／TPR | person AUC | 舊判定 | 新判定 | 未列入判定（物體配對、AUC 變化） |
|---|---|---|---|---|---|---|
| 參考（舊裁切） | 0.9418／0.7412 | 0.9130／0.6637 | 0.9576 | — | — | — |
| `dinov2s_v3` | 0.9370／0.6769 | 0.9367／0.7138 | 0.9515 | 未通過 | **未通過**（car 變差） | small_truck 2 組 −0.002；truck 0 組 −0.008 |
| `dinov2s_v3_lowres` | 0.9710／0.8762 | 0.9444／0.7656 | 0.9445 | 未通過（small_truck） | **通過** | small_truck 2 組 −0.032；truck 0 組 −0.011 |
| `dinov2s_v3_distractor` | 0.9440／0.6969 | 0.9427／0.7472 | 0.9511 | 未通過 | **未通過**（car 改善不足） | small_truck 2 組 −0.009；truck 0 組 −0.003 |
| `dinov2s_v3_both` | 0.9725／0.8866 | 0.9473／0.7866 | 0.9464 | 未通過（small_truck） | **通過** | small_truck 2 組 −0.034；truck 0 組 −0.012 |

- person（20 組物體配對）是唯一列入下降檢查的其他類別：lowres 下降 0.013、both 下降 0.011，都在 0.02 以內。
- `select_best`：`dinov2s_v3_both`（car、scooter 平均 AUC 0.9599；lowres 0.9577）進入追蹤驗收。
- small_truck 的下降仍回報在 `unchecked_classes`；是否對 small_truck 啟用 Re-ID 由追蹤驗收逐類別決定。

## 3. 追蹤驗收工具

### 3.1 工具與測試

- 新增 `src/safetrajai/reid/acceptance.py`（`parse_summary`、`aggregate`、`judge_class`、`parse_track_time`、`apply_overrides`）與 `scripts/reid_train/acceptance_grid.py`（`configs`、`summarize`）。
- `tests/test_reid_acceptance.py` 14 個測試：summary 讀取、彙總（IDSW／TP 加總、IDP 由 IDTP／IDFP 加總、AssA／HOTA 依 GT 框數加權、沒有 GT 時不算比例）、逐類別判定（IDSW 增加、只有 AssA 上升、IDSW 減少或 IDP 上升且不變差、TP 或 AssA 下降、沒有 GT）、每幀時間依幀數加權、覆寫解析與錯誤格式；CLI 端到端（`--set` 套用到 tune／holdout、`appearance_veto_sim=1.5` 由 `load_track_config` 報錯、`summarize` 對合成輸出的判定與 CSV／Markdown）。
- `pytest`：422 passed、11 skipped（含 `scripts/**/*.py --help` smoke）。

### 3.2 冒煙驗收（調參序列）

設定由 `acceptance_grid.py configs` 產生（`runs/reid_domain/accept/configs/`）；候選為 `dinov2s_v3_both`、car 開啟 Re-ID、A 閘門 1×`norm_factor`（80px）、寬門檻 0.6619（取自 `runs/reid_domain/v2/dinov2s_v3_both_tight.json`）。彙總：`runs/reid_domain/accept/smoke_tune.{csv,md}`。

- **基準可重現**：6 類加總 IDSW 6、CLEAR TP 2502，與 `train-reid-timm`／`reid-center-gate-fusion` 的調參基準相同。
- `a_car_1w`：car IDSW 2 → 10、IDP 0.9963 → 0.9941、AssA 92.19 → 91.43、TP 1349 → 1347，不通過；其他類別與基準相同（未開啟 Re-ID）。
- 時間：基準每幀追蹤 0.0087 秒、整組 30 秒；開啟一類 Re-ID 每幀 0.0497 秒（約 5.7 倍）、整組 42 秒（兩組平行）。
