# BoTSORT + 本地訓練 Re-ID 模型（`botsort_reid_timm.yaml`）正式驗收報告

## 背景

`Markdown/grid_search/botsort_reid_all6_rerun/botsort_reid_all6_report.md`（BoTSORT grid search 全 6 類覆蓋補強）發現：以 `botsort_noreid.yaml` 結構為底測試 ReID 開關時，只有 scooter 符合 spec `reid-model` 的啟用條件。但該測試的基準參數化（`center_dist_weight`／`track_buffer`／`gmc_method: ecc`／`frame_rate: 30`）與 CLAUDE.md 標示為正式驗收基準的 `config/tracking/botsort_reid_timm.yaml`（R1 參數化：四輪車 `center_dist_weight` 加重、`track_buffer` 15/15/15/15/10/10、`gmc_method: sparseOptFlow`、`frame_rate: auto`）結構不同，不能直接套用結論。

`botsort_reid_timm.yaml` 檔頭明確要求：重新驗收時以 `scripts/reid_train/acceptance_grid.py` 產生逐類別設定並彙總，再依 spec `reid-model` 判定。本報告即依此流程，以該檔案自己的 R1 參數為底，正式重新驗收。

## 方法

- 工具：`scripts/reid_train/acceptance_grid.py`（`configs` 產生設定、`summarize` 彙總並逐類別判定）。
- 基準（baseline）：`botsort_reid_timm.yaml` 原樣（6 類 `use_reid: false`）。
- 候選（6 組，逐類別獨立測試，其餘類別維持關閉）：每組只把一個類別的 `use_reid` 改為 `true`、`model` 改為目前唯一保留的正式模型 `models/class_reid_weights/dinov2s_v3/model.pt`，R1 的其他參數（`center_dist_weight`、`track_buffer`、`match_thresh` 等）全部不動。
- 資料：`dataset/eval/cctv_all8_subsets/{tune,holdout}/`。這兩個子集合起來就是 `cctv_all8` 的全部 6 支序列（tune 4 支、holdout 2 支，均為 C000025），與本次 grid search 覆蓋度補強工作全程使用的全 6 序列資料範圍一致。
- 判定依 `utils/modules/safetrajai/reid/acceptance.py::judge_class()`：候選的 IDSW、IDP、AssA、CLEAR TP 都不能比基準差，而且 IDSW 要減少或 IDP 要上升；沒有 GT 的類別不判定為通過。
- 共執行 14 次 `track_eval.py`（baseline + 6 類別 × tune/holdout），全部 exit code 0。

## 結果

### tune 子集（4 序列）

| run | class | IDSW | TP | IDP | AssA | HOTA | GT | 判定 |
|---|---|---|---|---|---|---|---|---|
| baseline | person | 0 | 177 | 0.9779 | 87.1543 | 87.1021 | 178 | 基準 |
| baseline | car | 2 | 1349 | 0.9963 | 92.1873 | 88.8243 | 1355 | 基準 |
| baseline | scooter | 4 | 825 | 0.9669 | 80.7305 | 76.6762 | 865 | 基準 |
| baseline | small_truck | 0 | 70 | 1.0000 | 90.6591 | 88.7042 | 70 | 基準 |
| baseline | bus | 0 | 0 | — | — | — | 0 | 基準（tune 無 bus GT） |
| baseline | truck | 0 | 81 | 1.0000 | 93.8620 | 93.5376 | 81 | 基準 |
| reid_person | person | 0 | 177 | 0.9779 | 87.1543 | 87.1021 | 178 | **不通過**：無身分層級改善 |
| reid_car | car | 2 | 1349 | 0.9963 | 92.1928 | 88.8390 | 1355 | **不通過**：無身分層級改善（AssA 微升但 IDSW／IDP 不變） |
| reid_scooter | scooter | 4 | 825 | 0.9669 | 80.7305 | 76.6762 | 865 | **不通過**：無身分層級改善 |
| reid_small_truck | small_truck | 0 | 70 | 1.0000 | 90.6591 | 88.7042 | 70 | **不通過**：無身分層級改善 |
| reid_bus | bus | 0 | 0 | — | — | — | 0 | **不通過**：tune 無 GT，無法判定 |
| reid_truck | truck | 0 | 81 | 1.0000 | 93.8620 | 93.5376 | 81 | **不通過**：無身分層級改善 |

（每組候選同時也對其餘 5 個未開 ReID 的類別重跑，數值與基準完全一致，證實逐類別獨立、互不干擾；上表只列該組實際開啟 ReID 的那個類別。）

### holdout 子集（2 序列）

| run | class | IDSW | TP | IDP | AssA | HOTA | GT | 判定 |
|---|---|---|---|---|---|---|---|---|
| baseline | person | 0 | 55 | 1.0000 | 92.2248 | 92.1046 | 56 | 基準 |
| baseline | car | 1 | 724 | 0.9917 | 85.9553 | 83.7498 | 729 | 基準 |
| baseline | scooter | 2 | 436 | 0.9476 | 82.8534 | 76.4058 | 459 | 基準 |
| baseline | small_truck | 0 | 76 | 1.0000 | 96.8040 | 96.7160 | 76 | 基準 |
| baseline | bus | 0 | 21 | 1.0000 | 94.9700 | 94.9700 | 21 | 基準 |
| baseline | truck | 0 | 0 | — | — | — | 0 | 基準（holdout 無 truck GT） |
| reid_person | person | 0 | 55 | 1.0000 | 92.2248 | 92.1046 | 56 | **不通過**：無身分層級改善 |
| reid_car | car | 1 | 724 | 0.9917 | 85.9463 | 83.7510 | 729 | **不通過**：AssA 下降 |
| reid_scooter | scooter | 2 | 436 | 0.9476 | 82.9513 | 76.2399 | 459 | **不通過**：無身分層級改善 |
| reid_small_truck | small_truck | 0 | 76 | 1.0000 | 96.8040 | 96.7160 | 76 | **不通過**：無身分層級改善 |
| reid_bus | bus | 0 | 21 | 1.0000 | 94.9700 | 94.9700 | 21 | **不通過**：無身分層級改善 |
| reid_truck | truck | 0 | 0 | — | — | — | 0 | **不通過**：holdout 無 GT，無法判定 |

完整輸出（含每個候選對其餘類別的對照數值）見同目錄 `summary_tune.{csv,md}`、`summary_holdout.{csv,md}`。

## 結論

在 `botsort_reid_timm.yaml` 自己的 R1 參數化（四輪車 `center_dist_weight` 加重、`track_buffer` 15/15/15/15/10/10、`gmc_method: sparseOptFlow`、`frame_rate: auto`）下，**沒有任何類別**符合 spec `reid-model` 的啟用條件：

- person／scooter／small_truck／truck：所有身分層級指標（IDSW、IDP）與關聯指標（AssA、HOTA）、CLEAR TP 與基準逐位元相同，ReID 完全沒有造成任何變化。
- car：tune 子集 AssA 有極小幅上升（+0.0055），但 IDSW／IDP 不變，不滿足「IDSW 減少或 IDP 上升」；holdout 子集 AssA 反而下降，判定不通過。
- bus：tune 子集沒有 GT；truck：holdout 子集沒有 GT；依規則沒有 GT 的子集不能判定為通過。

這與 CLAUDE.md 已記錄的機制一致：本地訓練的 Re-ID 模型在原本的融合方式下，外觀距離只作用在 IoU 已重疊的配對上，而 R1 的 `track_buffer` 偏短（10～15 幀），軌跡幾乎不會進入「遺失中、需要靠外觀找回」的狀態，因此 ReID 幾乎沒有機會起作用。

**依 spec `reid-model`「沒有類別符合時全部關閉」，`config/tracking/botsort_reid_timm.yaml` 維持現狀（6 類 `use_reid: false`），不需改動。** 本報告是這份設定檔目前狀態的正式驗收依據（2026-11-10，`cctv_all8` 全 6 序列）。

## 與 BoTSORT grid search 報告的關係

`Markdown/grid_search/botsort_reid_all6_rerun/botsort_reid_all6_report.md` 中「僅 scooter 符合 ReID 啟用條件」的結論，是在 `botsort_noreid.yaml` 結構的基準上得出的，`botsort_reid_timm.yaml` 的 `track_buffer` 遠短於該基準（10～15 幀 vs. 60～90 幀），這很可能正是兩者結論不同（該報告 scooter IDP 0.8997→0.9480 明顯上升，本報告 scooter 完全沒有變化）的主要原因：`track_buffer` 愈短，軌跡愈少進入需要外觀比對的「遺失中」狀態。本報告的結論僅適用於 `botsort_reid_timm.yaml` 本身的 R1 參數化，不建議把其他基準下的 ReID 開關結論套用到這份設定檔，反之亦然。

## 備份

本目錄另含：
- `run_acceptance.sh`：本次驗收重跑腳本（14 次 `track_eval.py` 呼叫：baseline + 6 類別 × tune/holdout）。
- `configs/`：14 份由 `acceptance_grid.py configs` 產生的情境設定備份。
- `summary_tune.{csv,md}`、`summary_holdout.{csv,md}`：`acceptance_grid.py summarize` 的完整彙總輸出。
