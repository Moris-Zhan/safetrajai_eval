# 保留 `utils/model.py` 作為 pickle 相容 shim

`models/class_reid_weights/*/best_full_model.pt` 是以 `torch.save` 存成的完整模型 pickle，載入時會依 `utils.model.<Class>` 這個模組路徑找類別。專案重構後，實作移到 `safetrajai/reid/model.py`，但 `utils/model.py` 刻意保留成只負責 re-export 的 shim，並在 pyproject 裡登記為頂層 package，讓舊權重在任何工作目錄下都能載入。它看起來是多餘的檔案，但刪掉後所有逐類別 Re-ID 權重都會無法載入。

## Considered Options

- **一次性轉存成 state_dict**：可以徹底擺脫舊路徑，但會改動沒有版控保護的權重檔（`models/` 在 gitignore 裡），一旦出錯就無法回復，所以這次不採用。之後如果要移除 shim，必須先用舊路徑載入、轉存成 state_dict，確認新權重可用之後才能刪除。
