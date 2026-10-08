# Verification

## 1. 程式修改

- `pytest`：398 passed、11 skipped，tracker override 方法的 AST 檢查通過。
- `Custom_ReID` 新格式改用緊框（`gain=1.0, pad=0`）；`legacy_crop=True` 保留 `gain=1.02, pad=10`，只供量測參考。
- 新增與修改的測試：
  - `tests/test_reid_encoder.py`：緊框裁切（中心 (100, 80)、寬 40、高 60 → x 80～120、y 50～110，像素逐一相同）、`legacy_crop` 仍放大加邊；修改前兩個測試失敗、修改後通過，舊格式測試不變。
  - `tests/test_reid_training.py`：解析度增強（被選中時最短邊落在範圍內、原本較小的裁切不放大、機率 0 與原本相同、縮小版 val 每次相同且尺寸有變化）；以縮小版 val 選 epoch（spec 情境：第 20 個 epoch 原始 val 最高、第 25 個 epoch 縮小版最高 → 選 25）；兩者都開啟訓練 1 個 epoch 的冒煙測試（訓練紀錄的設定、半徑、每個 epoch 兩種 val mAP 與實際干擾物比例，權重可由 `Custom_ReID` 載入）。
  - `tests/test_reid_distractor_sampler.py`：半徑依鏡頭寬度換算（C000054 car 320px）、同時出現且鄰近才算干擾物、碎片（幀範圍不重疊）不算、距離超過半徑不算、重疊不足或不同類別／序列不算；每批 P×K、同一 epoch 身分不重複、比例 0.5 時第一批至少一半是種子與它的干擾物、干擾物不足時不退回、比例 0 時與 `PKBatchSampler` 逐批相同；裁切索引只讀 metadata、缺列報錯、ROI 取寬度。
  - `tests/test_reid_fusion_stats.py`：AUC（完全分開＝1、完全相同＝0.5）、TPR@FPR5%、正樣本間隔以保留期限為上限（scooter `track_buffer` 10、幀率 5：8 幀列入、25 幀只列入 ≥ 1 秒）、保留期限不依追蹤器或幀率換算（ByteTrack、幀率 30、`track_buffer` 15 → 15 幀；非整數報錯）、分尺寸歸組（12px 與 30px → < 16px）、關卡情境（scooter AUC 下降、改善太小、雜訊底線較大、只有尾端沒改善、其他類別下降超過 0.02、多個通過只取平均 AUC 最高者、沒有通過）。

## 2. 參考、訓練與關卡

### 2.1 參考與緊框候選（`dinov2s_v3`）

量測設定：`config/botsort_reid_timm.yaml`（保留期限：四輪車 15 幀、person／scooter 10 幀；干擾物半徑 2×`norm_factor`），調參序列 4 段（約 5fps）。輸出在 `runs/reid_domain/dinov2s_v3_legacy.json`（參考）與 `dinov2s_v3_tight.json`（候選）。

**回歸確認**：參考的「≥ 1 秒」正樣本與干擾物的百分位數及配對數，與 `reid-center-gate-fusion` 的 `runs/reid_accept/fusion/fusion_distance_stats.json` 在 6 個類別逐位相同（例如 car 正樣本 P95 0.4064、干擾物 P5 0.2102）。

| 類別 | 干擾物 | 參考 AUC | 參考 TPR@FPR5% | 緊框 AUC | 緊框 TPR@FPR5% | 參考 AUC（≥ 1 秒） | 緊框 AUC（≥ 1 秒） |
|---|---|---|---|---|---|---|---|
| car | hard | 0.9418 | 0.7412 | 0.9370 | 0.6769 | 0.8539 | 0.8575 |
| scooter | hard | 0.9130 | 0.6639 | 0.9366 | 0.7137 | 0.8466 | 0.8852 |
| person | hard | 0.9576 | 0.8320 | 0.9515 | 0.7889 | 0.8785 | 0.8958 |
| small_truck | hard（43 對） | 0.9874 | 0.9614 | 0.9852 | 0.9492 | 0.8420 | 0.8509 |
| truck | any_fallback | 0.9792 | 0.9013 | 0.9717 | 0.9029 | 0.9204 | 0.9165 |
| bus | 無配對 | — | — | — | — | — | — |

- 保留期限內的正樣本比「≥ 1 秒」容易分開（car AUC 0.94 對 0.85）。
- 緊框讓 scooter 變好（AUC +0.024、TPR@FPR5% +0.050），car 的尾端變差（TPR@FPR5% −0.064）。關卡判定等 2.4 有雜訊底線後一起做。

### 2.2 冒煙訓練（兩者都開啟）

指令：`train_reid.py --version v3 --output models/class_reid_weights/smoke_domain_both --epochs 2 --max-identities 2000 --workers 5 --downscale-prob 0.5 --distractor-fraction 0.5`（log：`runs/reid_domain/smoke_domain_both.log`）。

| epoch | loss | CE | triplet | 訓練秒數 | epoch 秒數 | GPU 峰值 | val mAP | 縮小版 val mAP | 干擾物比例 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.40 | 7.73 | 3.67 | 4.8 | 11.9 | 6.84 GB | 0.3545 | 0.2742 | 0.009 |
| 2 | 7.85 | 7.21 | 0.64 | 3.8 | 9.9 | 6.84 GB | 0.4115 | 0.3269 | 0.010 |

- 損失下降；縮小版 val 比原始 val 低約 0.08，符合預期。
- 冒煙只取 2000 個身分（隨機、跨序列），打散了同時出現的關係，干擾物比例不具代表性。另以全量 v3 train（19,101 個身分）計算干擾物關係：前置計算 2.0 秒；有干擾物的身分 91.5%（car 96.8%、scooter 94.8%、person 8.7%、truck 29.6%、small_truck 17.3%、bus 3.5%）；每個身分的干擾物數中位數 5（P25 3、P75 9、P90 16）。
- **全量時實際干擾物比例只有約 9%**（3 個 epoch：0.089、0.093、0.095；上限是 15/32 ≈ 0.47）：干擾物在同一 epoch 被先用掉後就不能再選，後面的種子多半已沒有未用過的干擾物。D3 已預期「不足時以隨機身分補滿」，但比例遠低於設定值。
- 全量估計：單獨一個任務每個 epoch 約 74 秒訓練＋兩次 val 約 16 秒（`dinov2s_v3` 實測 74.4／82.5 秒，多一次縮小版 val）；四個任務平行在同一張 GPU（每個約 7 GB，共約 28 GB），預估每個 epoch 約 2～3 分鐘，30 epoch 共約 1～1.5 小時。
