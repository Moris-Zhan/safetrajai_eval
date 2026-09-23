---
paths:
  - 'module/reid/**/*.py'
  - 'utils/model.py'
  - 'dataset/reid/**/*.py'
---

# Re-ID 開發規範

## 支援範圍

- 僅保留兩條推論路徑：`ReID` 使用 YOLO 作 embedding encoder；`Custom_ReID` 載入 `models/class_reid_weights/` 的自訓模型。
- 不得重新引入已移除的 SOTA_ReID、FastReID 或 TransReID 套件、權重與設定。
- Re-ID 啟用是 tracker YAML 的逐類別行為，Re-ID 模組不得硬編碼 person、scooter 或其他類別策略。

## 權重與資料契約

- `utils/model.py` 不可搬移或改名。既有 `best_full_model.pt` 是完整 PyTorch pickle，反序列化時依賴 `utils.model` 的模組路徑。
- 若必須遷移 Custom_ReID，先制定舊權重轉成 `state_dict` 或以新模組路徑重新儲存的遷移方案；不得直接改路徑後假設舊權重可載入。
- crop 資料結構為 `dataset/reid/crop_dataset/{class}/{identity}/*.jpg`。identity 必須在 train／validation 間互斥；逐類別與跨類別混合 split 是不同資料集模式，不能混用。

## 訓練與驗證

- 新訓練流程採 YOLO-as-encoder、projector、batch-hard triplet loss 與 PK sampling。修改 hook 層、embedding 維度、sampler、loss 或 checkpoint 格式時，保持載入格式相容。
- 訓練 smoke test 至少確認 loss 可計算與下降、checkpoint 可由 `load_reid_model()` 載入、輸出為 L2-normalized embedding。
- 評估 Re-ID 時分別呈現 person、scooter 等受影響類別的 HOTA／AssA／IDSW；不要僅以整體平均分數做結論。
