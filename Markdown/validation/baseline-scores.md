# 評測資料集與追蹤基準分數

## 評測資料集

追蹤評估一律使用 `config/datasets.yaml` 的 `cctv_all8`（`role: eval`，附人工 GT，只用於評估，不能進任何訓練資料）：

| 項目 | 內容 |
|---|---|
| dataset id | `cctv_all8`（情境 config 以 `paths.dataset: cctv_all8` 引用） |
| 本機路徑 | `dataset/eval/cctv_all8/`：`annotations_all8/`（YOLO 格式＋序列影片）、`mot_annotations_all8/`（MOT GT） |
| 來源 | HF `Moris555/safetrajai-eval`，tag `gt-2026-09-23`（見 [`hf-restore.md`](../getting-started/hf-restore.md) 第 2 節） |
| 鏡頭 | C000025（中華三路 民生二路，東北角），2025-12-09 拍攝 |
| 序列 | 6 段 15 秒片段，約 5 fps（每段 71～75 幀），640×480 |
| 類別 | 6 類（person、car、scooter、small_truck、bus、truck） |

6 段序列依時段分成兩個子集（`subsets.yaml`，本機以 symlink 放在 `dataset/eval/cctv_all8_subsets/`，重建方式見 [`hf-restore.md`](../getting-started/hf-restore.md) 的 2.3）：

| 子集 | 時段 | 序列 | 幀數 | GT 列數 | 用途 |
|---|---|---|---:|---:|---|
| `tune` | 145630、150634 | `…_145630_15s_1`、`_15s_2`、`…_150634_15s_1`、`_15s_2` | 292 | 6,851 | 調參與驗收比較，下表的基準分數來自這裡 |
| `holdout` | 152639 | `…_152639_15s_1`、`_15s_2` | 149 | 3,550 | 保留序列，不用於調參；`fusion_distance_stats.py` 會拒絕讀取 |

- GT 列數是 `gt/gt.txt` 的總列數，含 ROI 外的框。stage2 會把 ROI 外的 GT 標成 ignore 再評估，所以下表的 GT 數比這裡少。
- **bus 的軌跡驗證只能靠 holdout**：GT 中的 bus 只有 2 條軌跡。tune 的 `…_150634_15s_2`（id 29，62 列）全部在 ROI 外，評估時被忽略；holdout 的 `…_152639_15s_2`（id 21，42 列，由上往下駛過畫面）有 21 列的中心點在 ROI 內，是唯一能實際評估 bus 的資料。
- 用 `SOURCE_VIDEO_DIR` 指向子集目錄，就能只評估該子集，例如 `./dataset/eval/cctv_all8_subsets/tune/`。不設定時會評估全部 6 段序列。

## 目前的追蹤基準分數

Re-ID 追蹤驗收的基準（2026-10-12）：設定 `runs/reid_domain/accept/configs/baseline_{tune,holdout}.yaml`，由 `acceptance_grid.py configs --name baseline` 以 `config/tracking/botsort_reid_timm.yaml` 為底產生（追蹤參數相同，6 類 ReID 都關閉）。偵測模型為 `yolo11x_dennis.pt`，兩個子集的設定只差 `SOURCE_VIDEO_DIR` 與輸出位置。逐序列 Excel 在 `runs/reid_domain/accept/baseline/{tune,holdout}/`，彙總在 `runs/reid_domain/accept/grid_tune.md` 與 `baseline_holdout.md`。

**tune 子集**（4 段序列）：

| 類別 | HOTA | AssA | IDSW | IDP | GT |
|---|---:|---:|---:|---:|---:|
| person | 87.10 | 87.15 | 0 | 0.978 | 178 |
| car | 88.82 | 92.19 | 2 | 0.996 | 1,355 |
| scooter | 76.68 | 80.73 | 4 | 0.967 | 865 |
| small_truck | 88.70 | 90.66 | 0 | 1.000 | 70 |
| bus | — | — | 0 | — | 0 |
| truck | 93.54 | 93.86 | 0 | 1.000 | 81 |

**holdout 子集**（2 段序列；bus 只能在這裡驗證）：

| 類別 | HOTA | AssA | IDSW | IDP | GT |
|---|---:|---:|---:|---:|---:|
| person | 92.10 | 92.22 | 0 | 1.000 | 56 |
| car | 83.75 | 85.96 | 1 | 0.992 | 729 |
| scooter | 76.41 | 82.85 | 2 | 0.948 | 459 |
| small_truck | 96.72 | 96.80 | 0 | 1.000 | 76 |
| **bus** | **94.97** | **94.97** | **0** | **1.000** | **21** |
| truck | — | — | 0 | — | 0 |

- tune 的 bus 為 0：tune 子集唯一的 bus（`…_150634_15s_2`，GT id 29，62 列）是停在畫面上緣（中心 y 約 39～54、框約 16×23 px）的遠處車輛，中心點全部在 ROI 外，被 stage2 標成 ignore。GT 欄是 TrackEval 輸出的 GT 數（ROI 處理後）。
- holdout 的 bus：`…_152639_15s_2` 的 GT id 21，原有 42 列，其中 21 列的中心點在 ROI 外被忽略，剩下 21 列全部配對成功（TP 21），沒有 ID switch。只有 1 條軌跡、21 幀，分數只能當作單一案例的訊號，不足以代表 bus 類別的整體表現。holdout 沒有 truck 的 GT。
- holdout 的基準最早在 2026-09-28 由 change `train-reid-timm` 執行過（見該 change 的 `verification.md` 3.1）；2026-10-12 重跑的結果逐類別完全相同。holdout 只作為最終確認，不可拿來調參。
- `bytetrack.yaml` 目前沒有保存的評估結果；`botsort_noreid.yaml` 的結果見下一小節。
- 這是 stage2 的 ROI 忽略區域處理後的分數。新設定要和這個基準比較時，請在同一個子集、同一個偵測模型上重跑。

## 開啟 ReID 的比較（2026-10-12，全部 6 段序列）

以主流程 `track_eval.py` 在 `cctv_all8` 全部 6 段序列（tune＋holdout）上跑兩組「ReID 關／開」對照。設定放在 `runs/reid_all6/configs/`，不改動 `config/`；彙總在 `runs/reid_all6/summary_timm.md`、`summary_legacy.md`。

| 組別 | 來源設定 | 差異 |
|---|---|---|
| `timm_off` | `config/tracking/botsort_reid_timm.yaml` 原樣 | 主 baseline，6 類 ReID 關閉 |
| `timm_v3_on` | 同上 | 只改 6 類 `use_reid: true`、`model: models/class_reid_weights/dinov2s_v3/model.pt`；融合維持原本設定（沒有閘門、沒有否決） |
| `legacy_off` | `config/tracking/botsort_noreid.yaml` | 舊情境（`frame_rate: 30`、`gmc_method: ecc`、`track_buffer` 90／60） |
| `legacy_on` | `config/tracking/botsort_reid.yaml` | 舊情境開 ReID（YOLO 當編碼器），`track_buffer` 也改成 120，不是純 ReID 的差異 |

HOTA 與 IDSW：

| 類別 | timm_off | timm_v3_on | legacy_off | legacy_on |
|---|---:|---:|---:|---:|
| person | 88.30 | 88.30 | 88.36 | 88.36 |
| car | 87.05 | 87.06 | 85.68 | 85.68 |
| scooter | 76.58 | 76.52 | 76.24 | 76.09 |
| small_truck | 92.87 | 92.87 | 88.48 | 89.54 |
| bus | 94.97 | 94.97 | 95.14 | 95.14 |
| truck | 93.54 | 93.54 | 92.93 | 92.93 |
| IDSW（car／scooter／small_truck） | 3／6／0 | 3／6／0 | 14／8／4 | 14／8／3 |
| 每幀追蹤時間 | 0.008 s | 0.077 s | 0.839 s | 1.181 s |

結論：**開啟 ReID 沒有讓分數超過 baseline，`botsort_reid_timm.yaml` 維持 6 類 ReID 關閉。**

- `dinov2s_v3`：6 類的 IDSW、IDP、TP 完全不變，依 spec `reid-model` 全部不通過；HOTA 只有 ±0.06 的雜訊級變動，追蹤時間增加約 9.5 倍。原本的融合只在 IoU 已重疊的配對上使用外觀距離，而這些配對靠 IoU 本來就會配對正確。
- 舊設定：開 ReID 後 scooter 的 IDP、AssA 下降；small_truck 的 IDSW 4 → 3，依規則判為通過，但只差 1 次，而且仍不如主 baseline（IDSW 0、HOTA 92.87），判定也用到了 holdout，所以不作為啟用依據。
- 舊設定整體較差，主要來自為 30 fps 設計的追蹤參數（C000025 約 5 fps），不是 ReID 造成的。
- 剩下的 ID switch（car 3、scooter 6）屬於遮擋或長時間遺失。先前用閘門擴大外觀作用範圍反而讓 IDSW 增加，瓶頸在模型與資料（見 `CLAUDE.md` 的 Re-ID 一節）。下一步應先用疊圖影片檢查這些 ID switch 的情境，而且只在 tune 子集上分析。
