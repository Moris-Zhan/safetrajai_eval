# Design

## Context

動機見 proposal.md，術語（Re-ID 模型、基底模型、工具模型、調參序列、保留序列、資料集版本）見 `CONTEXT.md`。現況與限制：

- **資料**：`dataset/reid/splits/v2/manifest.csv`（22,164 個身分、156,141 張；train 19,043／val 3,121 個身分）；`merge-occlusion-fragments` 完成後另有 `v3`。裁切平均約 7 KB，位於掛載磁碟（CLAUDE.md 禁止大量 I/O）。
- **追蹤器（ADR 0004，凍結）**：
  - `ClassAwareBOTSORT.__init__` 對 `use_reid` 的類別，路徑含 `class_reid_weights` 時建立 `Custom_ReID(model_path)` 並讀取 `.model`，否則建立 `ReID(model_path)`（YOLO）。
  - `init_track` 對每個偵測框各呼叫一次 encoder（傳入單一框），取回傳的第一個特徵。
  - `get_dists_class_aware()`（可修改）以 `embedding_distance / 2` 與 `appearance_thresh`、`proximity_thresh` 融合 IoU 距離。
- **推論端**：`Custom_ReID` 目前只載入完整模型 pickle，只處理第一個框，輸入為原始尺寸、未正規化。`ReID`（YOLO）以 `xywh2xyxy` 轉換框後裁切；追蹤器傳入的框格式由凍結的方法決定，新 encoder 必須沿用相同的轉換。
- **環境**：conda env `track_eval`，timm 1.0.15、torch 2.7.1（CUDA）；RTX 5090 32 GB；本機磁碟約 2 TB 可用、記憶體 248 GB。

## Goals / Non-Goals

**Goals:**
- 一個 6 類共用、可重現訓練的 Re-ID 模型，權重不依賴 pickle 類別路徑。
- 追蹤器不改 override 方法即可使用新模型。
- 以 C000025 的追蹤指標決定哪些類別啟用 Re-ID。

**Non-Goals:**
- 不追求 val 的 mAP 最大化；val 只作參考。
- 不調整追蹤器的非 Re-ID 參數；不做跨鏡頭對應。

## Decisions

### D1. 骨幹、輸入與損失

- 骨幹：timm `vit_small_patch14_dinov2.lvd142m`（與污染偵測同一預訓練權重），`img_size=224`，取 CLS 特徵（384 維），接 BNNeck；推論時輸出 BNNeck 後的特徵並做 L2 正規化。
- 輸入：letterbox 到 224×224（等比例縮放、黑邊），與 `detect_contamination.py` 相同，讓人與車輛共用同一輸入。
- 損失：身分分類交叉熵（label smoothing 0.1，緩解同一實體碎片互為負樣本的問題）＋ batch-hard 三元組（margin 0.3）。
- 取樣：每批 P=32 個身分 × K=4 張（身分張數不足 K 時重複取樣）；身分標籤在全部類別中統一編號（6 類共用分類頭）。
- 優化：AdamW，骨幹學習率 1e-5、頭部 1e-3，cosine 衰減，warmup 1 個 epoch；混合精度；固定亂數種子。預設 30 個 epoch，以 val mAP 選最佳 epoch（只用於挑權重，不作為啟用依據）。
- 資料集版本：`v2` 與 `v3` 各以相同設定訓練一次（`v3` 不存在時只訓練 `v2`），兩者的基底模型都是 DINOv2 ViT-S/14。
- 資料增強：水平翻轉、輕微色彩抖動、random erasing；不做會改變長寬比的裁切。

### D2. 只讀一次的訓練資料快取

- 每個資料集版本第一次訓練前，依 manifest 從掛載磁碟**各讀一次**全部裁切的 JPEG 位元組，打包成本機快取（單一資料檔＋索引，存放在 repo 之外的本機路徑，預設 `~/.cache/safetrajai/reid_packs/<version>/`）。之後每個 epoch 只讀本機快取。
- `v2` 的打包檔約 1.1 GB；打包時記錄張數，與 manifest 列數比對。
- **替代方案**：每個 epoch 直接從 `dataset/reid/crops/` 讀。**不採用的原因**：30 個 epoch 約 470 萬次小檔讀取，違反禁止大量 I/O 的規定。
- **替代方案**：把 letterbox 後的像素存成 memmap。**不採用的原因**：約 23 GB，而且改變輸入尺寸就要重建；JPEG 位元組在訓練時解碼即可。

### D3. 權重格式與 `Custom_ReID`

- 權重檔 `models/class_reid_weights/<run>/model.pt`：`{"format": "safetrajai-reid-v1", "arch", "embed_dim", "input_size", "mean", "std", "manifest_version", "state_dict"}`；同目錄的 `train_record.json` 記錄資料集版本、超參數、種子、各 epoch 的 val 指標與選用的 epoch。
- `Custom_ReID.__init__` 以 `torch.load(..., weights_only=True)` 讀取；內容是含 `format` 的字典時建立 timm 模型並載入 `state_dict`，否則沿用既有的完整模型 pickle 路徑（`weights_only=False`，ADR 0001）。
- `Custom_ReID.__call__`（新格式）對全部偵測框各自以 `xywh2xyxy` 轉換後裁切（與 `ReID` 相同）、letterbox、正規化，一次批次推論，回傳每個框一個 L2 正規化特徵。舊格式的行為不變。
- **替代方案**：TorchScript 或 ONNX。**不採用的原因**：timm ViT 的匯出需要額外驗證，而 state_dict＋設定已足以避免 pickle 類別路徑的問題。

### D4. val 評估

- 以 manifest 的 val：每個身分以固定種子取 1 張作 query，其餘作 gallery；gallery 包含所有 val 身分（跨類別），計算 mAP 與 Rank-1，並分類別回報。
- val 身分都來自 C000054 的 0629、同一序列內，分數偏樂觀；只用於挑 epoch 與比較不同訓練設定。

### D5. 追蹤驗收：逐類別啟用

- **驗收基準**（使用者決定）：以 `config/reid_crops.yaml` 的追蹤設定為底（R1 參數：四輪車 `center_dist_weight` 加重；`gmc_method: sparseOptFlow`；`frame_rate: auto`），`dataset` 改為 `cctv_all8`、開啟 stage2、全部類別 `use_reid: false`。它在 C000025 的 IDSW 為 9（`botsort_noreid.yaml` 為 26），是目前最好的追蹤設定；ReID 要在這個基礎上再改善才有意義。
- **調參序列與保留序列**（使用者決定，術語見 `CONTEXT.md`）：C000025 的 145630、150634 兩個時段（4 段）為調參序列，152639 時段（2 段）為保留序列。以時段切分，避免同一時段前後兩段的同一批車輛同時出現在兩邊。以符號連結組出兩個子集，分別以 `SOURCE_VIDEO_DIR` 指定，不改動評估資料集。
- **候選**：與基準相同，只把單一類別的 `use_reid` 改為 true、`model` 指向選定的 Re-ID 模型；`appearance_thresh` 試兩組（`botsort_reid.yaml` 的值與 0.3），`proximity_thresh` 沿用 `botsort_reid.yaml`。每類別 2 組、共 12 次，只在調參序列上執行，可平行。
- **判定**（使用者決定）：該類別在調參序列上 IDSW 不增加，且 IDP、AssA、CLEAR TP 都不下降時通過；兩組都通過時選 IDSW 較少者，再依 IDP、AssA 決定。
- **只啟用有改善的類別**（實作後使用者決定，2026-09-28）：通過判定是啟用的必要條件而非充分條件。IDSW、IDP、AssA、CLEAR TP 相對基準全部不變的類別不啟用（Re-ID 讓每幀追蹤時間增加數倍，卻沒有可量測的效果）；調參序列上沒有 GT、無法判定的類別（bus）維持關閉。兩組門檻結果完全相同時，採用 `botsort_reid.yaml` 的原值。
- **已知限制**：`get_dists_class_aware` 以 `min(運動距離, 外觀距離)` 融合，而且外觀距離只在 IoU 距離 ≤ `1 − proximity_thresh` 的配對上生效，Re-ID 無法接回遮擋後位置已偏移的軌跡。改進融合方式由後續 change 處理，不在本 change 範圍內。
- **合併驗證**：通過的類別一起開啟，先在調參序列上確認整體不變差（若變差，逐一關閉對整體貢獻最小的類別）。
- **保留序列**：只回報基準與最終設定的成績；若最終設定在保留序列的整體 IDSW 增加或 IDP 下降，逐一關閉類別直到不再變差（使用者決定：保留序列只否決整體，不否決個別類別，因為單一類別在 30 秒內只有少數身分）。
- 結果寫成 `config/botsort_reid_timm.yaml`（驗收基準加上通過類別的 `use_reid`、`model`、`appearance_thresh`）。
- 每幀追蹤時間只記錄、不設上限（使用者決定：離線評估不需要即時）。

### D7. `v2` 與 `v3` 模型的選擇

- 兩個模型以相同的初始門檻（`botsort_reid.yaml` 的值）、全部類別開啟 Re-ID，在調參序列上比較整體 IDSW、IDP、AssA；較好者進入 D5 的逐類別驗收，平手時選 `v3`。
- 不以 val mAP 比較：`v2` 與 `v3` 的身分定義不同（`v3` 有碎片合併），val 分數不能直接比較。
- 兩者的 val 指標與比較結果都寫進 `verification.md`，作為碎片合併是否有益的證據。

### D6. 程式位置

- `safetrajai/reid/training/`：manifest 資料集與快取、PK 取樣、模型（骨幹＋BNNeck＋分類頭）、損失、評估函式。
- `scripts/reid_train/train_reid.py`、`scripts/reid_train/eval_reid.py`、`scripts/reid_train/pack_manifest.py`。
- 單元測試以小型合成資料與不需下載的小骨幹（例如 timm `resnet18`、`pretrained=False`）驗證流程，不在測試中下載 DINOv2。

## Risks / Trade-offs

- **[風險] 逐框呼叫的 ViT 推論拖慢追蹤** → 凍結的 `init_track` 每個框呼叫一次；只在通過驗收的類別啟用，並在驗收時記錄每幀追蹤時間。
- **[風險] val 偏樂觀** → val 不作為啟用依據，驗收以 C000025 追蹤為準。
- **[風險] person（28 個身分）與 small_truck 資料少** → 共用模型可從其他類別學到通用外觀特徵；驗收不通過就維持關閉。
- **[取捨] 同一台公車可能跨日同時出現在 train 與 val** → 已知且接受（`rebuild-reid-dataset` design），只影響 val 分數。
- **[風險] 打包快取佔用本機磁碟** → 每個版本約 1 GB，放在 repo 外，可隨時刪除重建。

## Migration Plan

1. 實作 D2、D3、D6 與單元測試；以小樣本做訓練冒煙測試。
2. 打包快取、全量訓練（背景執行，先告知使用者耗時）、val 評估。
3. 逐類別追蹤驗收，產生 `config/botsort_reid_timm.yaml`。
4. 更新 CLAUDE.md、README。

**還原方式**：新權重與快取可直接刪除；`Custom_ReID` 保留舊格式路徑；新情境設定不影響既有情境與 golden；程式以 git revert 還原。
