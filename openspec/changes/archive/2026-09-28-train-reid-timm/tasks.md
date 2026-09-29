# Tasks

## 1. 程式修改

- [x] 1.1 `safetrajai/reid/encoders.py`：`Custom_ReID` 依 D3 支援新格式（`weights_only=True` 讀取、建立 timm 模型、每個框以 `xywh2xyxy` 裁切、letterbox、正規化、批次推論、L2 正規化），並保留舊的完整模型 pickle 路徑；以單元測試涵蓋「每個框一個特徵」、「不依賴 pickle 類別路徑」、「舊格式仍可載入」與裁切座標和 `ReID` 一致作為驗證（測試用 `resnet18`、不下載預訓練權重）
- [x] 1.2 新增 `safetrajai/reid/training/`：manifest 資料集（只讀 manifest 列出的裁切）、D2 的本機打包快取（寫入與讀取、張數與 manifest 一致）、PK 取樣、骨幹＋BNNeck＋分類頭、交叉熵＋batch-hard 三元組；以單元測試涵蓋「被排除的裁切不會被訓練」、快取往返、PK 取樣每批 P×K 張作為驗證
- [x] 1.3 新增 `scripts/reid_train/pack_manifest.py`、`train_reid.py`、`eval_reid.py`（D1、D4）：訓練輸出 `model.pt` 與 `train_record.json`；以合成資料、`resnet18` 跑 1 個 epoch 的冒煙測試，驗證權重可由 `Custom_ReID` 載入、訓練紀錄包含資料集版本與種子、`--help` 正常
- [x] 1.4 跑 `pytest` 確認全部通過，且 tracker override 方法的 AST 檢查仍通過

## 2. 訓練與評估

- [x] 2.1 確認可用的資料集版本（`v3` 已由 `merge-occlusion-fragments` 產生則 `v2`、`v3` 都用，否則只用 `v2`）；以 `pack_manifest.py` 建立各版本的本機快取，記錄打包張數與 manifest 列數一致、讀取張數、耗時與快取大小，寫入 `verification.md`
- [x] 2.2 以 D1 的設定在 DINOv2 ViT-S/14 上做小規模冒煙訓練（例如 1 個 epoch、部分身分），確認損失下降、GPU 記憶體與每個 epoch 的耗時；依結果估計全量訓練時間，向使用者回報後再進行全量訓練
- [x] 2.3 全量訓練 `v2`（與 `v3`）的 Re-ID 模型（背景執行），產生各自的 `models/class_reid_weights/<run>/model.pt` 與 `train_record.json`；以 `eval_reid.py` 回報各自 val 的整體與各類別 mAP、Rank-1，寫入 `verification.md`

## 3. 追蹤驗收

- [x] 3.1 建立驗收基準（`reid_crops.yaml` 的追蹤設定、`cctv_all8`、ReID 全部關閉）與調參序列、保留序列兩個子集（符號連結），在兩個子集上跑基準；結果寫入 `verification.md`
- [x] 3.2 依 D7 以全部類別開啟 Re-ID、初始門檻，在調參序列上比較 `v2` 與 `v3` 模型的整體 IDSW、IDP、AssA，選定進入驗收的模型（只有 `v2` 時直接採用）；結果寫入 `verification.md`
- [x] 3.3 依 D5 在調參序列上對 6 個類別各試 2 組 `appearance_thresh`（共 12 次，可平行），以 IDSW、IDP、AssA、CLEAR TP 判定並記錄每幀追蹤時間；通過的類別一起開啟做合併驗證；結果寫入 `verification.md`
- [x] 3.4 在保留序列上跑最終設定，回報與基準的比較；若整體 IDSW 增加或 IDP 下降，依 D5 逐一關閉類別；產生 `config/botsort_reid_timm.yaml`，以 config 載入測試通過、golden 情境與既有 `botsort_reid.yaml` 未改變作為驗證；結果寫入 `verification.md`

## 4. 文件與收尾

- [x] 4.1 更新 CLAUDE.md 的 Re-ID 章節（新權重格式、`Custom_ReID` 兩種格式、`botsort_reid_timm.yaml`、ReID 預設關閉的說明依驗收結果修正）與常用指令、`scripts/reid_data/README.md`（訓練讀取 manifest 與本機快取）
- [x] 4.2 撰寫 `verification.md` 的總結；跑 `pytest` 與 `openspec validate --strict` 確認通過
