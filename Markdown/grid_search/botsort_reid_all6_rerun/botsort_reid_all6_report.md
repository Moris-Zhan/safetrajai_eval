# BoTSORT＋ReID 6 段序列 grid search 與驗收報告

執行日期：2026-11-10。對應 `Markdown/training_eval_audit.md` 補強優先順序第 2 項，ByteTrack → BoTSORT（no ReID）之後的 BoTSORT＋ReID 階段，也是 CLAUDE.md `spec reid-model` 的重新驗收。

## 執行方式

- 資料集：`cctv_all8`（唯一鏡頭 C000025，全部 6 段序列）
- 搜尋參數子集：`match_thresh`、`second_match_thresh`、`unconfirmed_match_thresh`
- 搜尋網格：`scripts/grid_search/grid_search_botsort_reid.py` 的 car/bus/truck/small_truck 原為固定單值，本次已改成真實候選範圍（沿用 ByteTrack／BoTSORT no-ReID 同款範圍）；person/scooter 沿用腳本原有的範圍
- ReID 權重：`models/class_reid_weights/dinov2s_v3/model.pt`（目前唯一保留的模型，`appearance_thresh`/`proximity_thresh` 等融合參數沿用 `botsort_reid_timm.yaml` 原值）
- 基準 config（`gs_botsort_reid_all6.yaml`）：car／small_truck／truck 採用 BoTSORT（no ReID）階段已驗收的新參數，person／scooter／bus 維持原始基準參數（未採用該階段被拒絕/打平的候選值），6 類全部強制 `use_reid: true` 以便搜尋
- 腳本與設定備份：見本資料夾 `run_gs_botsort_reid_all6.sh`、`gs_botsort_reid_all6.yaml`、`gs_botsort_reid_all6_best.yaml`

## 兩種分數的說明（與前兩階段相同）

- **GS 分數**：grid search 回傳的「排除 0 分序列後取第 25 百分位數」（`core.py:523`），不是平均值。
- **平均準確率**：本報告額外計算的逐序列算術平均（排除無 GT 序列）。
- small_truck／bus／truck 的網格搜尋最佳分數恰好等於目前基準參數本身（打平）；person／scooter 的基準參數不在原始搜尋網格內，GS 分數(基準) 為 N/A。

## 各類別 grid search 分數彙總

| 類別 | GS分數(基準) | GS分數(最佳) | 平均準確率(基準) | 平均準確率(最佳) | 最佳參數（match/second/unconfirmed） | 結論 |
|---|---|---|---|---|---|---|
| person | N/A | 0.756847 | 0.8856 | 0.8856 | 0.8/0.2/0.6 | 打平／不敏感（逐序列分數完全一致），不採用 |
| car | 0.834227 | 0.835340 | 0.8941 | 0.8789 | 0.85/0.5/0.85 | ⚠️ GS 分數微幅「改善」，但真實平均準確率反而下降（-0.0152），**不採用** |
| scooter | N/A | 0.689818 | 0.7933 | 0.7771 | 0.9/0.4/0.7 | ⚠️ 平均準確率反而下降，**不採用新參數**（但見下方 ReID 開關的獨立結論） |
| small_truck | 0.869323 | 0.869323 | 0.9252 | 0.9252 | 0.7/0.45/0.85（與基準打平） | 打平，不用改 |
| bus | 0.903620 | 0.903620 | 0.9514 | 0.9514 | 0.6/0.45/0.85（與基準打平） | 打平（僅 1 段序列有 GT），不用改 |
| truck | 0.846728 | 0.846728 | 0.8882 | 0.8882 | 與基準完全相同 | 打平，不用改 |

**結論：grid search 在這 3 個門檻參數上找到的「最佳組合」，沒有一個類別能帶來真實平均準確率的改善**——car／scooter 的 GS 分數雖然看起來持平或微增，但獨立驗證後的平均準確率反而下降，與 ByteTrack／BoTSORT(no ReID) 兩階段看到的「GS 分數 ≠ 平均值」現象一致。因此本階段的門檻參數維持 `gs_botsort_reid_all6.yaml`（= 沿用 BoTSORT no-ReID 已驗收值）不變。

## ReID 開關驗收（CLAUDE.md `spec reid-model`）

驗收規則：「在調參序列上與驗收基準比較，不變差，而且 IDSW 減少或 IDP 上升；只有 AssA／TP 上升、沒有 GT 的類別不啟用；沒有類別符合時全部關閉。」

驗收基準＝目前已採用的 BoTSORT（no ReID）設定（`gs_botsort_noreid_all6_best.yaml`，即 car／small_truck／truck 用新參數、person／bus／scooter 維持原參數、6 類 `use_reid: false`）。

| 類別 | 基準 AssA | ReID(基準門檻) AssA | ΔAssA | 基準 IDSW(合計) | ReID IDSW(合計) | 基準 IDP(平均) | ReID IDP(平均) | 判定 |
|---|---|---|---|---|---|---|---|---|
| person | 0.8856 | 0.8856 | 0 | 0 | 0 | 0.9902 | 0.9902 | 無差異，不啟用 |
| car | 0.8942 | 0.8941 | -0.0001 | 3 | 3 | 0.9935 | 0.9935 | 無差異，不啟用 |
| scooter | 0.7815 | 0.7933 | **+0.0118** | 6 | 8 | 0.8997 | **0.9480** | AssA 升、IDP 升 → **符合啟用條件** |
| small_truck | 0.9252 | 0.9252 | 0 | 0 | 0 | 1.0000 | 1.0000 | 無差異，不啟用 |
| bus | 0.9514 | 0.9514 | 0 | 0 | 0 | 1.0000 | 1.0000 | 無差異，不啟用 |
| truck | 0.8882 | 0.8882 | 0 | 0 | 0 | 1.0000 | 1.0000 | 無差異，不啟用 |

注意事項：
- person／car／small_truck／bus／truck 這 5 類在目前融合機制下，ReID 完全沒有造成任何逐序列分數變化（逐位元相同），與 CLAUDE.md 既有記載的「本地訓練的 Re-ID 模型在原本的融合下幾乎沒有作用：外觀距離只作用在 IoU 已重疊的配對上」一致。沒有效果就不應該為了啟用 ReID 多付每幀數倍的推論時間與風險，因此這 5 類維持 `use_reid: false`。
- scooter 是本次唯一出現差異的類別：AssA 平均準確率上升（0.7815→0.7933）、IDP 平均上升（0.8997→0.9480），符合規則的「IDSW 減少或 IDP 上升」（IDP 上升已滿足 OR 條件）。IDSW 總數略增（6→8，6 段序列合計僅增加 2 次），在規則字面上不構成否決條件，但增加量不大、樣本量（1324 個 GT、合計 IDSW 個位數）仍偏小，建議列為初步結果，後續有新資料或新 Re-ID 模型時再重新確認。
- 這是啟用 scooter ReID 時使用**原始基準門檻參數**（0.9/0.4/0.8，非本次 grid search「最佳參數」0.9/0.4/0.7）的結果；套用 grid search 最佳門檻參數反而讓 scooter 的 ReID 效果變差（AssA 0.7933→0.7771、IDSW 8→11、IDP 0.9480→0.9096，見下方三方比較表），再次確認 grid search 找到的門檻並非更優。

### 三方比較（no-ReID 基準／ReID+基準門檻／ReID+grid-search最佳門檻）

| 類別 | AssA(no-ReID) | AssA(ReID基準) | AssA(ReID最佳) | IDSW(no-ReID) | IDSW(ReID基準) | IDSW(ReID最佳) | IDP(no-ReID) | IDP(ReID基準) | IDP(ReID最佳) |
|---|---|---|---|---|---|---|---|---|---|
| person | 0.8856 | 0.8856 | 0.8856 | 0 | 0 | 0 | 0.9902 | 0.9902 | 0.9902 |
| car | 0.8942 | 0.8941 | 0.8789 | 3 | 3 | 2 | 0.9935 | 0.9935 | 0.9793 |
| scooter | 0.7815 | 0.7933 | 0.7771 | 6 | 8 | 11 | 0.8997 | 0.9480 | 0.9096 |
| small_truck | 0.9252 | 0.9252 | 0.9252 | 0 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| bus | 0.9514 | 0.9514 | 0.9514 | 0 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| truck | 0.8882 | 0.8882 | 0.8882 | 0 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

## 逐序列 × 逐類別明細（AssA_mean，0 = 該序列此類別無可評估 GT）

### ReID baseline（固定基準參數，6 類 use_reid=true）

| 序列 | person | car | scooter | small_truck | bus | truck |
|---|---|---|---|---|---|---|
| C000025 中華三路 民生二路(東北角)_20251209145630_15s_1 | 0 | 0.9525 | 0.6868 | 0.9123 | 0 | 0 |
| C000025 中華三路 民生二路(東北角)_20251209145630_15s_2 | 0.9070 | 0.9011 | 0.7876 | 0.9157 | 0 | 0.8158 |
| C000025 中華三路 民生二路(東北角)_20251209150634_15s_1 | 0.8085 | 0.8562 | 0.8357 | 0.9055 | 0 | 0.8821 |
| C000025 中華三路 民生二路(東北角)_20251209150634_15s_2 | 0.8687 | 0.9389 | 0.8429 | 0 | 0 | 0.9667 |
| C000025 中華三路 民生二路(東北角)_20251209152639_15s_1 | 0.9115 | 0.8862 | 0.8318 | 0 | 0 | 0 |
| C000025 中華三路 民生二路(東北角)_20251209152639_15s_2 | 0.9323 | 0.8297 | 0.7750 | 0.9673 | 0.9514 | 0 |

### ReID 套用 grid search 最佳參數後

| 序列 | person | car | scooter | small_truck | bus | truck |
|---|---|---|---|---|---|---|
| C000025 中華三路 民生二路(東北角)_20251209145630_15s_1 | 0 | 0.9525 | 0.6653 | 0.9123 | 0 | 0 |
| C000025 中華三路 民生二路(東北角)_20251209145630_15s_2 | 0.9070 | 0.9011 | 0.7876 | 0.9157 | 0 | 0.8158 |
| C000025 中華三路 民生二路(東北角)_20251209150634_15s_1 | 0.8085 | 0.8559 | 0.7487 | 0.9055 | 0 | 0.8821 |
| C000025 中華三路 民生二路(東北角)_20251209150634_15s_2 | 0.8687 | 0.8678 | 0.8529 | 0 | 0 | 0.9667 |
| C000025 中華三路 民生二路(東北角)_20251209152639_15s_1 | 0.9115 | 0.8677 | 0.8335 | 0 | 0 | 0 |
| C000025 中華三路 民生二路(東北角)_20251209152639_15s_2 | 0.9323 | 0.8286 | 0.7747 | 0.9673 | 0.9514 | 0 |

### 逐序列 × 逐類別 GT 數量（ROI 內可評估）

| 序列 | person | car | scooter | small_truck | bus | truck |
|---|---|---|---|---|---|---|
| C000025 中華三路 民生二路(東北角)_20251209145630_15s_1 | 0 | 569 | 129 | 2 | 0 | 0 |
| C000025 中華三路 民生二路(東北角)_20251209145630_15s_2 | 68 | 238 | 178 | 31 | 0 | 5 |
| C000025 中華三路 民生二路(東北角)_20251209150634_15s_1 | 32 | 266 | 321 | 37 | 0 | 18 |
| C000025 中華三路 民生二路(東北角)_20251209150634_15s_2 | 78 | 282 | 237 | 0 | 0 | 58 |
| C000025 中華三路 民生二路(東北角)_20251209152639_15s_1 | 29 | 439 | 343 | 0 | 0 | 0 |
| C000025 中華三路 民生二路(東北角)_20251209152639_15s_2 | 27 | 290 | 116 | 76 | 21 | 0 |

## 資料覆蓋度限制

與前兩階段相同的 `cctv_all8` 限制：bus 僅 1 段序列有 ROI 內 GT，person 僅 5/6 段，small_truck 僅 4/6 段、truck 僅 3/6 段。bus／small_truck／truck／person 的統計強度弱於 car／scooter，解讀結論時要考量樣本量；scooter 的 ReID 啟用結論同樣建議視為初步結果看待。

## 最終建議

- **門檻參數（match/second/unconfirmed）**：維持 `gs_botsort_reid_all6.yaml` 既有值（＝沿用 BoTSORT no-ReID 已驗收結果）不變。本次 grid search 沒有找到任何能真正改善平均準確率的新組合。
- **ReID 開關**：僅 **scooter** 啟用 `use_reid: true`（搭配原始基準門檻 0.9/0.4/0.8，`model: models/class_reid_weights/dinov2s_v3/model.pt`），其餘 person／car／small_truck／bus／truck 維持 `use_reid: false`——這是本專案目前為止第一次有類別通過 ReID 驗收。
- 待辦：下次有新 Re-ID 訓練資料或模型時，應重新量測 scooter 的 IDSW 增加是否只是樣本噪音，再決定是否維持啟用。

## 後續

三個追蹤器（ByteTrack、BoTSORT no-ReID、BoTSORT+ReID）的全 6 類別 grid search 補強皆已完成；ByteTrack／BoTSORT(no ReID) 驗收通過的參數已寫回 `config/tracking/bytetrack.yaml`、`config/tracking/botsort_noreid.yaml`。

**`config/tracking/botsort_reid_timm.yaml` 已另行用 `scripts/reid_train/acceptance_grid.py` 以它自己的 R1 參數重新正式驗收完畢**，結論與本報告不同：本報告「scooter 符合 ReID 啟用條件」的結論驗證的是 `botsort_noreid.yaml` 結構的基準（`track_buffer` 60～90 幀），**不適用**於 `botsort_reid_timm.yaml` 的 R1 基準（`center_dist_weight` 四輪車加重至 0.6/0.4/0.4/0.4、`track_buffer` 僅 15/15/15/15/10/10、`gmc_method: sparseOptFlow`、`frame_rate: auto`）——`center_dist_weight` 改變距離矩陣組成、`track_buffer` 偏短讓軌跡更少進入需要外觀比對的「遺失中」狀態，R1 基準下 scooter 等 6 類全部沒有任何身分層級改善。`botsort_reid_timm.yaml` 依此維持 6 類 `use_reid: false` 不變。完整方法與數據見 `Markdown/grid_search/botsort_reid_timm_acceptance/botsort_reid_timm_acceptance_report.md`。
