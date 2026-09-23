# safetrajai/reid/ — Re-ID 推論端說明

本目錄只剩推論端。舊的 YOLO-as-encoder 訓練端（`yolo_reid_trainer.py`、`training/`、`scripts/train_reid.py`）已在 change `reorganize-dataset-and-eval-gt` 移除：它訓練出的 projector 在追蹤器的推論路徑上根本不會被用到，訓練資料也已確認無效。需要參考舊實作時，從 git 歷史取回。新的 Re-ID 訓練（timm backbone、ID + triplet）由後續 change 處理。

## 一、模組總覽

```
safetrajai/reid/
├── encoders.py   推論期特徵抽取：ReID（ultralytics YOLO 當 encoder）與 Custom_ReID（完整模型 pickle）
└── model.py      Custom_ReID 使用的 ReIDNet 等類別定義；utils/model.py 是它的 pickle 相容 shim
```

## 二、追蹤器如何選 encoder

`ClassAwareBOTSORT.__init__`（凍結，見 `docs/adr/0004`）依 `class_tracking:` 裡各類別的 `use_reid` 與 `model:` 路徑決定：

```
model 路徑含 "class_reid_weights" ? --yes--> Custom_ReID(model_path)
                                  \--no---> ReID(model_path)   # ultralytics YOLO(path)，取倒數第二層 embedding
```

目前的情境設定中，只有 `botsort_reid.yaml`（與它在 golden 的副本）開啟 ReID，6 類都走 `ReID`，權重是 detector 本身（`models/yolo/yolo11x.pt`、`models/yolo/yolo11x_dennis.pt`）的 backbone 特徵。沒有任何情境使用 `Custom_ReID`。

## 三、`Custom_ReID` 與 pickle shim

`Custom_ReID` 以 `torch.load(..., weights_only=False)` 載入完整模型 pickle，反序列化時依 `utils.model.<Class>` 找類別，所以 `utils/model.py` 不可刪除或改名（見 `docs/adr/0001`）。舊的 `models/class_reid_weights/{person,scooter}` 權重已隔離到 `models/class_reid_weights/temp/`，情境 config 不得引用；smoke 測試仍會從隔離路徑載入一次，確認 shim 可用。

`Custom_ReID` 目前只做 `/255` 正規化、不做 resize，推論端前處理和訓練端不一致。後續 HF／timm 模型接入時，會連同前處理契約一起重新定義。

## 四、已隔離的資產

- `models/yolo-reid/temp/`：舊 YOLO-as-encoder 訓練產出的權重。
- `models/class_reid_weights/temp/`：舊 `ReIDNet` 完整模型權重。
- `dataset/reid/temp/`：舊追蹤器產生的虛擬身分裁切與衍生資料。

每個 `temp/DO_NOT_USE.md` 都記錄了原路徑、隔離原因與還原指令。
