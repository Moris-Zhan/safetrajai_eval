# safetrajai/reid/ — Re-ID 推論端與訓練說明

本目錄包含推論端與訓練端。舊的 YOLO-as-encoder 訓練端（`yolo_reid_trainer.py`、舊的 `training/`、`scripts/train_reid.py`）已在 change `reorganize-dataset-and-eval-gt` 移除：它訓練出的 projector 在追蹤器的推論路徑上根本不會被用到，訓練資料也已確認無效。需要參考舊實作時，從 git 歷史取回。新的 Re-ID 訓練（timm backbone、ID + triplet）由 change `train-reid-timm` 加在 `training/`，CLI 在 `scripts/reid_train/`；change `align-reid-training-domain` 加上解析度增強、干擾物取樣、推論端緊框與外觀分離度關卡。

## 一、模組總覽

```
safetrajai/reid/
├── encoders.py   推論期特徵抽取：ReID（ultralytics YOLO 當 encoder）與 Custom_ReID（新格式 state_dict＋設定，或舊的完整模型 pickle）
├── model.py      舊格式 Custom_ReID 使用的 ReIDNet 等類別定義；utils/model.py 是它的 pickle 相容 shim
├── fusion_stats.py  外觀距離分布、AUC、TPR@FPR5%、分尺寸回報、雜訊底線與外觀分離度關卡判定（純函式）
└── training/     manifest 資料集與本機快取、解析度增強與縮小版 val（data）、PK 與干擾物取樣（sampler）、
                  骨幹＋BNNeck 與權重格式（model）、CE＋三元組（losses）、val 檢索評估（evaluate）、訓練流程（train）
```

## 二、追蹤器如何選 encoder

`ClassAwareBOTSORT.__init__`（凍結，見 `docs/adr/0004`）依 `class_tracking:` 裡各類別的 `use_reid` 與 `model:` 路徑決定：

```
model 路徑含 "class_reid_weights" ? --yes--> Custom_ReID(model_path)
                                  \--no---> ReID(model_path)   # ultralytics YOLO(path)，取倒數第二層 embedding
```

目前的情境設定中，`botsort_reid.yaml`（與它在 golden 的副本）6 類都走 `ReID`，權重是 detector 本身（`models/yolo/yolo11x.pt`、`models/yolo/yolo11x_dennis.pt`）的 backbone 特徵。`botsort_reid_timm.yaml` 的 6 類都是 `use_reid: false`（等同驗收基準）；候選模型是 `models/class_reid_weights/dinov2s_v3/model.pt`，重新驗收的方式記在檔頭。

## 三、`Custom_ReID` 與 pickle shim

`Custom_ReID` 先以 `torch.load(..., weights_only=True)` 讀取：內容是 `format: safetrajai-reid-v1` 的字典時，依設定建立 timm 模型並載入 state_dict；每個框以 `xywh2xyxy` 後的框**緊框**裁切（`save_one_box` 的 `gain=1.0, pad=0`，與訓練裁切及 `CONTEXT.md` 的「裁切」定義一致），letterbox 到 `input_size`（與訓練、`dinov2_features.letterbox` 相同）、以權重內的 `mean`／`std` 正規化，批次推論後回傳每個框一個 L2 正規化特徵。

- `Custom_ReID(model_path, legacy_crop=True)` 恢復修改前的裁切（`gain=1.02, pad=10`），只給 `fusion_distance_stats.py --legacy-crop` 量測參考；凍結的追蹤器只以 `Custom_ReID(model_path)` 建立，情境設定無法開啟它。
- 緊框改變了新格式權重的特徵值（BREAKING），但目前沒有情境設定啟用新格式權重。

讀取失敗（完整模型 pickle）時退回舊路徑：以 `torch.load(..., weights_only=False)` 載入完整模型 pickle，反序列化時依 `utils.model.<Class>` 找類別，所以 `utils/model.py` 不可刪除或改名（見 `docs/adr/0001`）。舊的 `models/class_reid_weights/{person,scooter}` 權重已隔離到 `models/class_reid_weights/temp/`，情境 config 不得引用；smoke 測試仍會從隔離路徑載入一次，確認 shim 可用。

舊格式的行為維持不變（只處理第一個框、只做 `/255`、不做 resize）；YOLO 的 `ReID` 也不變（golden 只用它）。

## 四、訓練（`training/`、`scripts/reid_train/train_reid.py`）

訓練與 val 只讀 manifest 與本機打包檔 `~/.cache/safetrajai/reid_packs/<version>/`（見 `scripts/reid_data/README.md`）。change `align-reid-training-domain` 新增的參數**預設關閉**：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--downscale-prob` | 0 | 解析度增強：以此機率在 letterbox 前把裁切等比例縮小，最短邊取 `[--downscale-min, --downscale-max]`（預設 8～40px，所有類別相同，不依評估資料逐類別設定）；原本較小的裁切不放大 |
| `--distractor-fraction` | 0 | 干擾物取樣：每批以一個種子身分開始，最多 `round(P×比例)−1` 個名額給它的干擾物 |
| `--cooccur-min-frames` | 10 | 干擾物條件：兩個身分的幀範圍（不擴張）至少重疊的幀數 |
| `--pair-max-frame-gap` | 5 | 干擾物條件：至少一對裁切的幀差上限，且框中心距離 ≤ 半徑（2×`norm_factor`×鏡頭寬度/640） |
| `--crop-root`／`--roi-dir`／`--track-config` | `dataset/reid/crops`、`config/roi`、`config/reid_crops.yaml` | 干擾物取樣讀 `crops/_index/` 的框座標（只讀 metadata）、ROI 檔的畫面寬度、各類別 `norm_factor` |
| `--select-by` | `reduced_val` | 以縮小版 val（val 裁切依 crop_path 雜湊固定縮小）或原始 val（`val`）的 mAP 選 epoch |

- **注意**：`--select-by` 預設是縮小版 val。要重現 `dinov2s_v3` 的設定（例如種子對照 `dinov2s_v3_seed1`），要加 `--select-by val`。
- 干擾物（`sampler.DistractorPKBatchSampler`）先從本 epoch 尚未用過的干擾物選，不足時可重複選入已用過的（同一批不重複）；種子與補位身分在同一 epoch 不重複，批數維持「身分數 // P」。仍不足時以隨機身分補滿，不放寬條件（幀範圍不重疊的身分可能是同一物體的碎片）。`--distractor-fraction 0` 時與 `PKBatchSampler` 逐批相同。
- 比例受干擾物數本身限制：`v3` train 每個身分的干擾物數中位數 5，P=32 時理論上限約 0.196，實測約 0.18。
- 訓練紀錄 `train_record.json` 記錄上述設定、半徑、有干擾物的身分比例，以及每個 epoch 的原始 val／縮小版 val mAP、實際干擾物比例、重複選入數與看到的身分比例。
- 每個訓練任務約需 8.3 GB GPU 記憶體；32 GB 的 GPU 平行四個任務會 OOM，最多同時跑三個。

範例（`dinov2s_v3_both` 的設定）：

```bash
python scripts/reid_train/train_reid.py --version v3 --output models/class_reid_weights/<run> \
  --workers 5 --downscale-prob 0.5 --distractor-fraction 0.5
```

## 五、外觀分離度關卡（`fusion_stats.py`、`scripts/reid_train/fusion_distance_stats.py`）

新模型（或新的推論裁切方式）在追蹤驗收前，先在調參序列（C000025 145630、150634 時段；保留序列由程式拒絕）上量測：

- 正樣本：同一條 GT 軌跡、間隔 5 幀～該類別 `track_buffer` 幀（直接取 `--config`，預設 `botsort_reid_timm.yaml`，ADR 0009 不換算）；另回報「≥ 1 秒」的結果，可與 `reid-center-gate-fusion` 比較。
- 負樣本（干擾物）：同幀、同類別、中心距離 ≤ 2×`norm_factor` 的不同 GT 軌跡；不足 30 對時退回任意配對，且不列入關卡的「其他類別」條件。
- 指標：AUC、TPR@FPR5%，並依較小框的最短邊分 < 16px、16～32px、≥ 32px 回報（只供參考）。
- 判定：參考是 `dinov2s_v3` 以 `--legacy-crop` 量測；雜訊底線是 `dinov2s_v3_seed1` 與 `dinov2s_v3`（都是緊框）的差距。car 與 scooter 的 AUC、TPR@FPR5% 都要比參考高出 max(0.01／0.02, 2×雜訊底線)，其他類別 AUC 下降 ≤ 0.02；多個通過時取 car、scooter 平均 AUC 最高者。val mAP 不作為依據。

```bash
python scripts/reid_train/fusion_distance_stats.py --legacy-crop --out runs/reid_domain/dinov2s_v3_legacy.json
python scripts/reid_train/fusion_distance_stats.py --weights models/class_reid_weights/<run>/model.pt \
  --reference runs/reid_domain/dinov2s_v3_legacy.json \
  --noise-reference runs/reid_domain/dinov2s_v3_seed1_tight.json runs/reid_domain/dinov2s_v3_tight.json \
  --out runs/reid_domain/<run>_tight.json
```

**目前結果（change `align-reid-training-domain`）**：四個候選（`dinov2s_v3` 緊框、`_lowres`、`_distractor`、`_both`）都沒有通過，沒有做追蹤驗收，`botsort_reid_timm.yaml` 不變。lowres／both 的 car、scooter 大幅改善，只因 small_truck AUC 下降約 0.03（43 對干擾物，下降集中在最短邊 < 16px 的 13 對）而不通過。詳細數字見該 change 的 `verification.md`；後續方向是補 small_truck 資料，或事先訂定各尺寸組的樣本充分性門檻。

## 六、已隔離的資產

- `models/yolo-reid/temp/`：舊 YOLO-as-encoder 訓練產出的權重。
- `models/class_reid_weights/temp/`：舊 `ReIDNet` 完整模型權重；`smoke_domain_both`（冒煙訓練）與 `dinov2s_v3_lowres_oom`（OOM 中止的不完整權重）。
- `dataset/reid/temp/`：舊追蹤器產生的虛擬身分裁切與衍生資料。

每個 `temp/DO_NOT_USE.md` 都記錄了原路徑、隔離原因與還原指令。
