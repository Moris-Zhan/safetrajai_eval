# module/reid/ — 自行設計的 Re-ID 模型架構與訓練腳本說明

本文件只涵蓋**自行設計、有訓練程式碼**的那條路徑：YOLO-as-encoder（`yolo_reid_trainer.py` + `training/`）。`encoders.py` 的 `Custom_ReID`（載入 `utils/model.py` 的 `ReIDNet`）是另一條推論用路徑，權重是外部訓練好的產物，訓練程式碼不在本專案範圍內，只在第六節簡短帶過。

## 一、模組總覽

```
module/reid/
├── encoders.py              推論期特徵抽取：ReID（本文件訓練出的模型）／Custom_ReID（外部模型，見第六節）
├── yolo_reid_trainer.py     模型架構本體：ReIDHookManager、ReIDProjector、BatchHardTripletLoss、
│                            save_yolo_reid()、load_reid_model()
└── training/                訓練資料管線 + CLI 訓練腳本
    ├── dataset.py            discover_identities、split_identities、ReIDCropDataset、PKSampler
    └── train.py               CLI 入口，串起整個訓練流程
```

資料來源見 `dataset/reid/README.md`（`track_eval.py` 的 `save_reid_crops` 功能蒐集裁切、`build_reid_split.py` 切 train/val）；本文件只講模型本身怎麼設計、訓練腳本怎麼運作。

## 二、模型架構（`yolo_reid_trainer.py`）

### 2.1 設計原則：不包 wrapper nn.Module

沒有另外定義一個 `class YOLOReID(nn.Module)` 把 YOLO 包起來——直接在 ultralytics YOLO 的原生 `model.model`（`DetectionModel` 或其他 task 對應的模型）上掛 `forward hook`，取倒數第二層輸出當特徵。這樣做的理由：

- **推論端可以直接沿用 ultralytics 的 `YOLO(path)` 載入方式**，不需要額外的 wrapper 類別定義同時存在於訓練與推論兩端（否則兩邊都要 import 同一個 wrapper class，容易因為路徑或版本不同步而載入失敗，`utils/model.py` 的 pickle 相容性限制就是這類問題的前車之鑑，見 `CLAUDE.md`「架構」章節）。
- **projector 直接附掛成 `yolo.model.reid_projector` 屬性**，`torch.save(yolo.model)`／ultralytics 的 `yolo.save()` 會把它一起序列化，不需要另外管理第二份 checkpoint。

### 2.2 前向流程（`ReIDHookManager`）

```
輸入影像 (B,3,H,W)
   │  _pad_to_stride(x, 32)   ── 補齊到 32 的倍數（YOLO backbone 對輸入解析度的隱性要求）
   ▼
YOLO 原生 backbone（yolo.model.model，一個 nn.ModuleList）
   │  forward hook 掛在 model.model[len(model.model)-2]（倒數第二層）
   ▼
_cache_feat()：把該層輸出存到 self._feat（forward hook 的副作用）
   │
   ▼
_pool(feat)：依 feat 維度自動池化
   - (B,C,H,W) → mean over (H,W)
   - (B,seq,C) → mean over seq
   - 其餘原樣回傳
   ▼
ReIDProjector（Linear → BatchNorm1d → ReLU → Linear）
   ▼
F.normalize(embed, p=2, dim=1)   ── L2-normalized embedding (B,D)
```

- **`hook_layer_idx = len(inner.model) - 2`**：倒數第二層，不管換哪個 YOLO 權重（`yolo11x_dennis.pt`、`yolo11x.pt`…）都用同一套邏輯定位，不寫死層數。
- **`feat_dim` 自動 probe**：建構 `ReIDHookManager` 時會用一張 `(1,3,64,64)` 的 dummy tensor 做一次 forward，從 hook 抓到的實際輸出形狀反推 `feat_dim`，再據此建立 `ReIDProjector`。換 YOLO 權重（不同層數/通道數）不需要手動改 projector 的輸入維度。
- **`embed()` 同時服務訓練與推論**：訓練迴圈裡呼叫 `manager.embed(imgs)` 取得帶梯度的 embedding 算 loss；推論時（`load_reid_model()` 之後）呼叫同一個方法，只是 `eval_mode()` 已關閉梯度追蹤語意上的差異。

### 2.3 `ReIDProjector`

```python
nn.Sequential(
    nn.Linear(feat_dim, feat_dim),
    nn.BatchNorm1d(feat_dim),
    nn.ReLU(inplace=True),
    nn.Linear(feat_dim, feat_dim),
)
```

維度不變（輸入輸出都是 `feat_dim`），純粹是一個度量學習用的投影頭，把 YOLO backbone 的偵測用特徵重新映射到一個更適合算距離（triplet loss）的空間。沒有降維，好處是不用猜一個合適的輸出維度；代價是這層本身不小（`feat_dim × feat_dim` 兩層），`yolo11x` 系列的 `feat_dim` 常在 1000+ 量級（實際跑過 `scooter` 是 1280），這部分參數量會直接反映在存檔大小上。

### 2.4 Loss：`BatchHardTripletLoss`

標準 batch-hard triplet mining：同一個 batch 內，對每個 anchor 找「距離最遠的 positive」與「距離最近的 negative」各一個，組成一組 triplet，套 `MarginRankingLoss`（`margin` 預設 0.3）。這解釋了為什麼訓練資料的抽樣方式（見下方 `PKSampler`）需要保證同一個 batch 裡每個身分都有多張影像——否則同一 batch 內某個身分只出現一次，就找不到 positive pair，那個 anchor 會被跳過（`dist_ap`/`dist_an` 收集邏輯裡 `if len(pos_i) == 0 or len(neg_i) == 0: continue`）。

## 三、模型持久化（`save_yolo_reid` / `load_reid_model`）

### 3.1 儲存機制

`save_yolo_reid()` 直接呼叫 ultralytics 原生的 `yolo.save(path)`；因為 `reid_projector` 已經是 `yolo.model` 的屬性（`nn.Module` 子模組），這次存檔會把 backbone 權重和 projector 權重一起序列化成同一個 `.pt`，不需要額外維護第二個 checkpoint 檔案。

### 3.2 ⚠️ 存檔前必須暫時清空 hook（且型別要對）

`ReIDHookManager`、`wandb.watch` 這類工具會在 `nn.Module` 上掛帶 lambda 的 `forward hook`，這些 hook 物件本身無法被 pickle。`save_yolo_reid()` 存檔前會遍歷 `yolo.model.modules()`，暫存並清空每個 submodule 的 `_forward_hooks`/`_forward_pre_hooks`/`_backward_hooks`，存完再原樣還原。

清空時**必須用 `OrderedDict()` 而不是 `{}`**——這是本專案早期一個真實存在過的 bug：用 `{}` 清空會讓存檔內容裡的 hook 容器型別跟著變成一般 `dict`，之後 `load_reid_model()` 重新呼叫 `register_forward_hook()` 時，PyTorch 內部要對這個容器物件建立 `weakref` 會直接拋 `TypeError: cannot create weak reference to 'dict' object`。已修正為 `OrderedDict()`，此修正只影響修正後才存的檔案，舊權重不受影響（詳見 `markdown/EXPERIMENTS.md`）。

### 3.3 讀取（`load_reid_model`）

流程是「先建構一個新的 `ReIDHookManager`（會重新 probe feat_dim、建一個新的 projector、重掛 hook），再用 checkpoint 裡實際存的 `reid_projector` 權重覆寫掉這個暫時建的 projector」，而不是直接反序列化出舊的 hook 物件——因為 hook 本身在存檔前就已經被清空，checkpoint 裡本來就沒有 hook 可以還原，一定要重新掛。

```python
yolo, manager = load_reid_model("models/yolo-reid/scooter_reid.pt", device="cuda")
manager.eval_mode()
embed = manager.embed(imgs)   # (B, D)，L2-normalized
```

## 四、訓練資料管線（`training/dataset.py`）

### 4.1 目錄假設

```
{root}/{class_name}/{identity_folder}/*.jpg
```

`identity_folder` 命名由 `track_eval.py` 的 `save_reid_crops` 產生（`{影片檔名}_track{track_id}`），但 `dataset.py` 本身不依賴這個命名規則——只要是同一個資料夾底下的影像就當作同一身分。

### 4.2 依身分切分，不是依影像切分

`split_identities()` 把整個身分資料夾（而非個別影像）隨機分進 train/val，確保同一個身分的所有照片只會出現在其中一邊——這是刻意的設計，避免同一身分的照片同時出現在 train 和 val 造成資料外洩（模型可能記住某個身分的特定拍攝角度/光線，而不是學到真正的外觀特徵）。身分數只有 1 個時全部進 train，val 為 0（不會報錯）。

### 4.3 `ReIDCropDataset`：`min_images_per_identity` 過濾

身分資料夾裡影像數低於 `min_images_per_identity`（`train.py` 預設 4）的整個身分會被捨棄——張數太少無法在同一 batch 內組出有效的 positive pair（見上方 loss 說明），與其讓這類身分在訓練時被跳過，不如一開始就不計入。

### 4.4 `PKSampler`：P 身分 × K 影像抽樣

```python
DataLoader(dataset, batch_sampler=PKSampler(dataset.labels, P=8, K=4))
```

每個 batch 固定抽 `P` 個身分、每個身分抽 `K` 張影像（不足 `K` 張時用取後放回抽樣補足），這是 batch-hard triplet mining 的標準抽樣方式——只有確保同一 batch 內每個身分都有 ≥1 對照片，才能算出有意義的 hardest positive。`P` 若大於實際身分數會自動 clamp（`self.P = min(P, len(self.unique_labels))`），至少需要 2 個身分才能組出 triplet（否則 `PKSampler.__init__` 直接拋錯）。

## 五、訓練腳本（`training/train.py`）

### 5.1 訓練流程

```
discover_identities(data_root, class_name)
        │
        ▼
split_identities(val_ratio, seed)  ──→ train_dirs / val_dirs
        │
        ▼
ReIDCropDataset × 2（train / val，套 min_images_per_identity 過濾）
        │
        ▼
DataLoader（batch_sampler=PKSampler(P, K)）× 2
        │
        ▼
YOLO(yolo_weights) → ReIDHookManager → freeze_backbone(可選)
        │
        ▼
Adam optimizer（backbone lr × 0.1、projector 用完整 lr）+ CosineAnnealingLR
        │
        ▼
逐 epoch：run_epoch(train) → run_epoch(val) → scheduler.step()
        │                                          │
        │                              EarlyStopping(val_loss) 記錄最佳 state
        ▼
早停或跑完全部 epoch → restore_best() 還原最佳權重
        │
        ▼
manager.remove_hook() → save_yolo_reid()
```

### 5.2 差異化學習率：為什麼要拆 backbone / projector 兩組

```python
projector_ids = {id(p) for p in manager.projector.parameters()}
backbone_params = [p for p in yolo.model.parameters() if id(p) not in projector_ids]
optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": args.lr * 0.1},
    {"params": manager.projector.parameters(), "lr": args.lr},
], weight_decay=args.weight_decay)
```

`reid_projector` 是掛在 `yolo.model` 底下的子模組，`yolo.model.parameters()` 天然就包含 projector 的參數，所以要用 `id()` 集合排除法把兩組參數分開，才能分別給不同學習率——backbone 是已經訓練過的偵測權重，用完整 `lr` 微調容易破壞原本學到的特徵，因此打七折（`lr * 0.1`）；projector 是從零初始化的投影頭，用完整 `lr` 讓它能快速收斂。

### 5.3 `--freeze-backbone` 與 projector 的例外

`ReIDHookManager.freeze_backbone(True)` 會把 `yolo.model.parameters()`（含 projector）全部設成 `requires_grad=False`，然後**再把 projector 的參數明確設回 `True`**——因為 projector 是 `yolo.model` 的子模組，天真地凍結「整個 backbone」會連 projector 一起凍住。開 `--freeze-backbone` 時實際上永遠只訓練 projector，不管有沒有加這個旗標，projector 都會是可訓練的；差別只在 backbone 本身要不要跟著微調。

### 5.4 `EarlyStopping`：同時記錄 backbone 與 projector 的最佳狀態

依 val loss 是否比目前最佳值進步超過 `min_delta` 判斷，若連續 `patience` 個 epoch 沒進步就早停。跟一般只存單一 checkpoint 檔案的做法不同，這裡是把「當下最佳的 `yolo.model.state_dict()`」與「當下最佳的 `manager.projector.state_dict()`」都存在記憶體裡（`EarlyStopping.best_backbone_state` / `best_projector_state`，先搬到 CPU 再 clone 避免佔用 GPU 記憶體），訓練結束後用 `restore_best()` 一次還原兩者再存檔——確保存下來的是「backbone 和 projector 同一個 epoch 的最佳組合」，而不是兩者各自在不同 epoch 的最佳值拼在一起。

### 5.5 CLI 旗標

| 旗標 | 預設 | 說明 |
|---|---|---|
| `--class-name` | 必填 | 對應 `dataset/reid/crop_dataset/{class}/` |
| `--data-root` | `dataset/reid/crop_dataset` | 訓練資料根目錄 |
| `--yolo-weights` | `models/yolo/yolo11x_dennis.pt` | 當 encoder 起點的 YOLO 權重 |
| `--output` | `models/yolo-reid/{class}_reid.pt` | 輸出權重路徑 |
| `--epochs` | 50 | |
| `--P` / `--K` | 8 / 4 | 每 batch 身分數 × 每身分影像數 |
| `--lr` / `--weight-decay` | 1e-4 / 1e-4 | projector 的學習率（backbone 實際是這個值 × 0.1） |
| `--margin` | 0.3 | triplet loss margin |
| `--img-size` | 128 | 訓練用影像 resize 邊長 |
| `--val-ratio` | 0.2 | 依身分切分的驗證集比例 |
| `--min-images` | 4 | 身分資料夾影像數低於此值整個捨棄 |
| `--freeze-backbone` | 關閉 | 只訓練 projector（見 5.3） |
| `--patience` | 10 | Early stopping 耐心值，`<=0` 關閉 |
| `--seed` | 42 | train/val 身分切分的隨機種子 |
| `--device` | 自動偵測 cuda/cpu | |
| `--wandb` | 關閉 | 開啟才會呼叫 `wandb.init()`，需先 `wandb login` 或設 `WANDB_MODE=offline` |

一次只練一個類別的模型（呼應 `config/*.yaml` 的 `class_tracking:` 逐類別指定 `model:` 路徑的慣例），要練多個類別就跑多次、換 `--class-name` 和 `--output`。

## 六、與 `encoders.py::ReID` 的關係、跟 `Custom_ReID` 的差異

訓練出的權重（`models/yolo-reid/{class}_reid.pt`）直接被 `encoders.py::ReID` 載入——`ReID.__init__` 用 ultralytics `YOLO(model_path)` 讀權重，走的是 ultralytics 自己的 `embed=` 推論參數（不是本文件的 `ReIDHookManager`），但因為存檔時 backbone 權重本來就在同一個 `.pt` 裡，數值上是一致的（`reid_projector` 這個附掛屬性不會影響 ultralytics 自己的推論路徑，只有透過 `load_reid_model()` 明確讀取才會用到）。`module/tracker/class_aware_botsort.py` 依 `class_tracking:` 裡 `model:` 路徑字串判斷走哪條 encoder（不含 `class_reid_weights` → `ReID`）。

`encoders.py::Custom_ReID` 是完全不同的另一條路徑：載入 `utils/model.py` 定義的 `ReIDNet`（`torch.save()` 整物件序列化，pickle 相容性限制見 `CLAUDE.md`「架構」章節），權重是外部訓練好的產物，**本專案沒有它的訓練程式碼**，不在本文件範圍內，也不在目前的重新設計計畫裡（見 `markdown/ARCHITECTURE.md`）。

## 七、已驗證

用 `scooter` 類別（`dataset/reid/crop_dataset/scooter/`）跑過 smoke test（5 epoch、`--freeze-backbone`）：loss 穩定下降，`save_yolo_reid()` 存檔後，`load_reid_model()` 與 `encoders.py::ReID` 兩條讀取路徑都能正確還原並產生 L2-normalized embedding。實際數據與完整驗證記錄見 `markdown/EXPERIMENTS.md`。
