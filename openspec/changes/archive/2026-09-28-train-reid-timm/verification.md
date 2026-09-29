# Verification

## 總結

| 項目 | 結果 |
|---|---|
| 程式 | `Custom_ReID` 支援新格式（`safetrajai-reid-v1`：state_dict＋設定，`weights_only=True`，每個框一個 L2 特徵），舊的完整模型 pickle 路徑不變；新增 `safetrajai/reid/training/` 與 `scripts/reid_train/`（打包、訓練、評估）。追蹤器 override 方法未修改（AST 檢查通過）。 |
| 快取 | `v2` 從掛載磁碟讀 156,141 張（74 秒、622 MB）；`v3` 沿用 `v2`，只補讀 1,243 張（使用者決定）。 |
| 模型 | DINOv2 ViT-S/14，`v2`／`v3` 各訓練 30 個 epoch：val mAP 0.7846／0.7830、Rank-1 0.909／0.906（只作參考）。 |
| 模型選擇 | 全部類別開啟時，兩者在調參序列上的追蹤指標完全相同，依 D7 平手規則採用 `v3`。 |
| 逐類別驗收 | 有 GT 的 5 個類別都不變差，但只有 car 有可量測的改善（調參序列 AssA +0.006）；依使用者決定只啟用 car，bus（調參序列無 GT）維持關閉。 |
| 保留序列 | 整體 IDSW 3→3、IDP 0.9775→0.9775，不需關閉；car AssA −0.009。 |
| 產出 | `config/botsort_reid_timm.yaml`（R1 追蹤參數＋只有 car 以 `dinov2s_v3` 開 ReID）；既有 `botsort_reid.yaml` 與 golden 未改變。 |
| 結論 | 模型本身在 val 上可用，但在目前的融合方式下對 C000025 的追蹤**沒有實質改善**，且追蹤時間增加數倍；Re-ID 的效果受限於 `get_dists_class_aware()` 只讓外觀距離作用在 IoU 已重疊的配對上。改進融合方式由後續 change 處理（使用者決定）。 |
| 測試 | `pytest` 249 passed、8 skipped（既有的「script does not iterate classes」）；`openspec validate train-reid-timm --strict` 通過。 |

## 1. 程式修改

- `pytest`：245 passed、8 skipped（skip 為既有的「script does not iterate classes」），tracker override 方法的 AST 檢查通過。
- 新增測試：`tests/test_reid_encoder.py`（每個框一個 L2 特徵、不依賴 pickle 類別路徑、裁切座標與 `ReID` 一致、舊格式仍可載入）、`tests/test_reid_training.py`（排除的裁切不會被打包或讀取、快取往返、快取與 manifest 不一致時報錯、PK 取樣、三元組損失、檢索指標、快取複用只補讀缺少的裁切、以 `resnet18` 跑 1 個 epoch 後由 `Custom_ReID` 載入）。

## 2. 訓練與評估

### 2.1 本機快取

`v3` 已由 `merge-occlusion-fragments` 產生，`v2`、`v3` 都使用。使用者決定 `v2` 從掛載磁碟全讀，`v3` 沿用 `v2` 的快取、只補讀缺少的裁切（`pack_manifest.py --reuse-version v2`），掛載磁碟的讀取量約減半。

| 版本 | manifest 列數 | 打包張數 | 從磁碟讀取 | 耗時 | 快取大小 |
|---|---:|---:|---:|---:|---:|
| v2 | 156,141 | 156,141 | 156,141 | 74.4 秒 | 622 MB |
| v3 | 156,186 | 156,186 | 1,243（其餘沿用 v2） | 1.4 秒 | 624 MB |

快取位置：`~/.cache/safetrajai/reid_packs/{v2,v3}/`。打包後以 `verify_pack` 確認快取的路徑集合與 manifest 完全相同。

### 2.2 冒煙訓練（DINOv2 ViT-S/14）

設定：D1 的預設值（`vit_small_patch14_dinov2.lvd142m`、224、P=32×K=4、骨幹 lr 1e-5／頭部 lr 1e-3、bf16），`v2`，train 與 val 各取 2,000 個身分（`--max-identities 2000`），2 個 epoch；輸出在 scratchpad，未寫入 `models/`。

| epoch | loss | CE | triplet | val mAP | val Rank-1 | 訓練秒數 | 含 val 秒數 | GPU 峰值記憶體 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.635 | 7.729 | 2.906 | 0.402 | — | 4.9 | 9.0 | 6.84 GB |
| 2 | 7.720 | 7.160 | 0.559 | 0.474 | 0.712 | 3.9 | 7.1 | 6.84 GB |

- 損失下降、val mAP 上升，流程正常（2,000 個身分的 val 比全量容易，數字只作流程確認）。
- 每個 epoch 62 iter，約 0.065 秒／iter。全量 `v2` train 19,043 個身分 → 595 iter／epoch，約 40 秒，加上全量 val（約 2 萬張）約 50 秒／epoch；30 個 epoch 約 25 分鐘，`v3` 相近。

### 2.3 全量訓練與 val 評估

`v2`、`v3` 以相同的 D1 設定（30 個 epoch、種子 0）在同一張 RTX 5090 上平行訓練（使用者決定），每個 epoch 約 82～85 秒，各約 43 分鐘。log 在 `runs/reid_train/dinov2s_{v2,v3}.log`，`eval_reid.py` 的結果在 `runs/reid_train/eval_dinov2s_{v2,v3}.json`（與訓練紀錄的最佳 epoch 數值一致）。

| 模型 | 權重 | train 身分／張數 | val 身分／張數 | 最佳 epoch | val mAP | val Rank-1 |
|---|---|---:|---:|---:|---:|---:|
| v2 | `models/class_reid_weights/dinov2s_v2/model.pt` | 19,043／133,081 | 3,121／23,060 | 28 | 0.7846 | 0.9093 |
| v3 | `models/class_reid_weights/dinov2s_v3/model.pt` | 19,101／133,230 | 3,122／22,956 | 27 | 0.7830 | 0.9058 |

各類別（mAP／Rank-1，query 數）：

| 類別 | v2 | v3 |
|---|---|---|
| person（5） | 0.949／1.000 | 0.980／1.000 |
| car（1,456／1,450） | 0.823／0.942 | 0.829／0.946 |
| scooter（1,429） | 0.755／0.908 | 0.747／0.903 |
| small_truck（55／64） | 0.785／0.800 | 0.760／0.703 |
| bus（44） | 0.477／0.455 | 0.504／0.500 |
| truck（132／130） | 0.777／0.758 | 0.762／0.723 |

- 最後 6 個 epoch 的 val mAP 在 0.782～0.785 之間，已收斂。
- `v3` 的 train 身分數比 `v2` 多：碎片合併後，部分原本不足 4 張的碎片合併成 ≥ 4 張，進入 manifest。
- 依 D7，兩者身分定義不同，val 分數不能直接比較；模型選擇以 3.2 的追蹤結果為準。person 的 val 只有 5 個 query，數字不具代表性。

## 3. 追蹤驗收

### 3.1 驗收基準與子集

- **子集**（符號連結，不改動評估資料集）：`dataset/eval/cctv_all8_subsets/tune/`（145630、150634 兩個時段、4 段）與 `dataset/eval/cctv_all8_subsets/holdout/`（152639 時段、2 段），各含 `annotations_all8/`、`mot_annotations_all8/`；以 `paths.SOURCE_VIDEO_DIR` 指定。
- **驗收 config**：由 `config/reid_crops.yaml` 產生（`runs/reid_accept/configs/<name>_<subset>.yaml`）：`dataset: cctv_all8`、開啟 stage2（只跑 track eval）、全部類別 `use_reid: false`、`save_reid_crops: false`；追蹤參數（R1、`sparseOptFlow`、`frame_rate: auto`）不變。`save_txt` 必須為 true（MOT 軌跡由 YOLO label 轉出）。輸出在 `runs/reid_accept/<name>/<subset>/`。
- **彙總方式**（與 `rebuild-reid-dataset` 相同）：IDSW、CLEAR TP 各序列加總；IDP 以 IDTP／IDFP 加總後計算；HOTA、AssA 依 GT 框數加權平均；每幀追蹤時間取各序列 `track avg` 的平均。

基準（ReID 全部關閉）：

| 子集 | 類別 | IDSW | TP | IDP | AssA | HOTA |
|---|---|---:|---:|---:|---:|---:|
| 調參 | person | 0 | 177 | 0.9779 | 87.154 | 87.102 |
| 調參 | car | 2 | 1349 | 0.9963 | 92.187 | 88.824 |
| 調參 | scooter | 4 | 825 | 0.9669 | 80.731 | 76.676 |
| 調參 | small_truck | 0 | 70 | 1.0 | 90.659 | 88.704 |
| 調參 | bus | 0 | 0 | — | — | — |
| 調參 | truck | 0 | 81 | 1.0 | 93.862 | 93.538 |
| 調參 | **整體** | **6** | **2502** | **0.9854** | **87.959** | **84.728** |
| 保留 | person | 0 | 55 | 1.0 | 92.225 | 92.105 |
| 保留 | car | 1 | 724 | 0.9917 | 85.955 | 83.750 |
| 保留 | scooter | 2 | 436 | 0.9476 | 82.853 | 76.406 |
| 保留 | small_truck | 0 | 76 | 1.0 | 96.804 | 96.716 |
| 保留 | bus | 0 | 21 | 1.0 | 94.970 | 94.970 |
| 保留 | truck | 0 | 0 | — | — | — |
| 保留 | **整體** | **3** | **1312** | **0.9775** | **85.911** | **82.496** |

- 兩個子集合計 IDSW 9，與 `rebuild-reid-dataset` 的 R1 結果（C000025 IDSW 9）一致，基準可重現。
- 調參序列沒有 bus 的 GT，保留序列沒有 truck 的 GT；這兩個類別在調參序列上的判定只剩「不變差」的條件（bus 無 GT，任何偵測都會是 FP，但 CLEAR TP 與 IDSW 無從比較）。
- 基準的每幀追蹤時間 0.0129 秒（與 Re-ID 訓練同時執行時量測，僅供參考；候選組會在同一條件下重新量測基準）。

### 3.2 `v2` 與 `v3` 模型的選擇（D7）

全部類別開啟 Re-ID、初始門檻（`botsort_reid.yaml` 的 `appearance_thresh`／`proximity_thresh`：person 0.3／0.3，car 0.1／0.5，scooter 0.1／0.5，small_truck 0.1／0.3，bus 0.1／0.1，truck 0.1／0.6），調參序列；基準在同一時間平行重跑一次，以相同條件量測每幀追蹤時間（`baseline_t`）。

| 組別 | IDSW | TP | IDP | AssA | HOTA | 每幀追蹤時間 |
|---|---:|---:|---:|---:|---:|---:|
| 基準（`baseline_t`） | 6 | 2502 | 0.9854 | 87.959 | 84.728 | 0.0095 秒 |
| 全部開啟，`v2` | 6 | 2502 | 0.9854 | 87.962 | 84.736 | 0.1076 秒 |
| 全部開啟，`v3` | 6 | 2502 | 0.9854 | 87.962 | 84.736 | 0.1084 秒 |

- `v2` 與 `v3` 各類別的指標完全相同（唯一的差異是 car 的 AssA 92.187 → 92.193，兩個模型相同），依 D7「平手時選 `v3`」，**選定 `v3`**（`models/class_reid_weights/dinov2s_v3/model.pt`）進入逐類別驗收。
- 兩個模型對追蹤幾乎沒有影響，原因在融合方式（`get_dists_class_aware`）：外觀距離 `embedding_distance / 2` 只在 IoU 距離 ≤ `1 − proximity_thresh` 的配對上生效，最後取 `min(運動距離, 外觀距離)`。Re-ID 只能降低「本來就重疊」的配對成本，無法接回遮擋後位置已偏移的軌跡；`appearance_thresh` 0.1 只排除外觀距離 > 0.9（cos < −0.8）的配對，幾乎不會觸發。
- 每幀追蹤時間增加約 11 倍（凍結的 `init_track` 對每個框各呼叫一次 encoder）；依使用者決定只記錄、不設上限。

### 3.3 逐類別驗收（調參序列，`v3`）

每組只開啟單一類別的 Re-ID（`model: models/class_reid_weights/dinov2s_v3/model.pt`，`proximity_thresh` 沿用 `botsort_reid.yaml`），`appearance_thresh` 試 `botsort_reid.yaml` 的值與 0.3；person 的原值就是 0.3，兩組相同，所以只跑 1 組，共 11 次（6＋5 兩批平行）。比較對象是同條件重跑的基準 `baseline_t`。

| 組別 | 該類別 IDSW | TP | IDP | AssA | 整體 IDSW／IDP／AssA | 每幀追蹤時間 | 判定 |
|---|---|---|---|---|---|---:|---|
| person 0.3 | 0→0 | 177→177 | 0.9779→0.9779 | 87.154→87.154 | 6／0.9854／87.959 | 0.0289 秒 | 通過（無變化） |
| car 0.1 | 2→2 | 1349→1349 | 0.9963→0.9963 | 92.187→92.193 | 6／0.9854／87.962 | 0.0874 秒 | 通過（AssA +0.006） |
| car 0.3 | 2→2 | 1349→1349 | 0.9963→0.9963 | 92.187→92.193 | 6／0.9854／87.962 | 0.0884 秒 | 通過（AssA +0.006） |
| scooter 0.1 | 4→4 | 825→825 | 0.9669→0.9669 | 80.731→80.731 | 6／0.9854／87.959 | 0.0518 秒 | 通過（無變化） |
| scooter 0.3 | 4→4 | 825→825 | 0.9669→0.9669 | 80.731→80.731 | 6／0.9854／87.959 | 0.0517 秒 | 通過（無變化） |
| small_truck 0.1 | 0→0 | 70→70 | 1.0→1.0 | 90.659→90.659 | 6／0.9854／87.959 | 0.0173 秒 | 通過（無變化） |
| small_truck 0.3 | 0→0 | 70→70 | 1.0→1.0 | 90.659→90.659 | 6／0.9854／87.959 | 0.0125 秒 | 通過（無變化） |
| bus 0.1 | 0→0 | 0→0 | — | — | 6／0.9854／87.959 | 0.0099 秒 | 無 GT，無法判定 |
| bus 0.3 | 0→0 | 0→0 | — | — | 6／0.9854／87.959 | 0.0100 秒 | 無 GT，無法判定 |
| truck 0.1 | 0→0 | 81→81 | 1.0→1.0 | 93.862→93.862 | 6／0.9854／87.959 | 0.0118 秒 | 通過（無變化） |
| truck 0.3 | 0→0 | 81→81 | 1.0→1.0 | 93.862→93.862 | 6／0.9854／87.959 | 0.0118 秒 | 通過（無變化） |

- 依 D5 的字面條件（IDSW 不增加、IDP／AssA／TP 不下降），有 GT 的 5 個類別全部通過，但只有 car 有可量測的改善（AssA +0.006，兩組門檻相同）；其他類別的指標完全不變，原因見 3.2（融合方式讓 Re-ID 只作用在已重疊的配對上）。
- 兩組 `appearance_thresh` 的結果在每個類別都相同，無法依 IDSW／IDP／AssA 區分。
- 調參序列沒有 bus 的 GT，bus 無法以調參序列判定。

**使用者決定（2026-09-28）**：通過判定只是必要條件；只啟用有可量測改善的類別（car），其他類別維持關閉，bus（無 GT）維持關閉；改進融合方式另開 change（已補進 design D5）。car 的兩組門檻結果相同，採用 `botsort_reid.yaml` 的原值 0.1。

**合併驗證**：通過並啟用的只有 car，合併設定與 `c_car_0.1` 相同：調參序列整體 IDSW 6（基準 6）、IDP 0.9854（0.9854）、AssA 87.962（87.959）、TP 2502（2502），整體不變差。

### 3.4 保留序列與最終設定

最終設定（只有 car 以 `v3` 模型啟用 Re-ID，`appearance_thresh` 0.1）與基準在保留序列上同時執行：

| 組別 | 整體 IDSW | TP | IDP | AssA | HOTA | car IDSW | car AssA | 每幀追蹤時間 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 基準（`baseline_t`） | 3 | 1312 | 0.9775 | 85.911 | 82.496 | 1 | 85.955 | 0.0087 秒 |
| 最終設定（`final`） | 3 | 1312 | 0.9775 | 85.907 | 82.496 | 1 | 85.946 | 0.0626 秒 |

- 保留序列的整體 IDSW 不增加、IDP 不下降，依 D5 不需要關閉類別。
- car 的 AssA 在保留序列上略降 0.009（調參序列上是 +0.006），兩者都在單一 ID 片段的差異量級內；整體來說，本模型在目前的融合方式下**對 C000025 的追蹤沒有實質改善**，只有追蹤時間增加（約 7 倍）。
- 產生 `config/botsort_reid_timm.yaml`：以 `config/reid_crops.yaml` 為底（`dataset: cctv_all8`、開啟 stage2、`save_txt: true`、`save_reid_crops: false`），只有 car 設 `use_reid: true`、`model: models/class_reid_weights/dinov2s_v3/model.pt`。以 YAML 比對確認其 `class_tracking`、`tracking_params` 與驗收的 `final` 設定完全相同。
- 驗證：smoke 測試對新 config 的 3 項檢查（`load_config`、類別名稱契約、不引用隔離資產）通過；`git status`／`git diff` 顯示 `config/botsort_reid.yaml` 與 `tests/golden/` 沒有任何變更。
