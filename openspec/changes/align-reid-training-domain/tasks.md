# Tasks

## 1. 程式修改

- [x] 1.1 先改測試：`tests/test_reid_encoder.py` 的裁切測試改為要求緊框（spec「緊框裁切」：中心 (100, 80)、寬 40、高 60 → x 80～120、y 50～110），另加 `legacy_crop=True` 時仍為 `gain=1.02, pad=10`；確認在修改前失敗，再修改 `Custom_ReID`（D2），確認通過、舊格式測試不變
- [x] 1.2 `training/data.py` 加入解析度增強與縮小版 val（D1）；以單元測試驗證被選中時最短邊落在範圍內、原本較小的裁切不放大、機率 0 時輸出與原本相同、縮小版 val 同一張裁切每次結果相同
- [x] 1.3 新增干擾物取樣器（D3）：由裁切索引取得 manifest 裁切的序列、幀與框座標，由 ROI 檔取得鏡頭寬度，訓練前一次算好干擾物關係；以合成資料的單元測試驗證每批 P×K、同一 epoch 身分不重複、比例 0.5 時符合條件的種子至少一半身分是它的干擾物、幀範圍不重疊的碎片不會被選入（spec「碎片不會被選為干擾物」）、同時出現但距離超過半徑的身分不會被選入（spec「同時出現但距離太遠的身分不算干擾物」）、半徑依鏡頭寬度換算、比例 0 時批次結構與 `PKBatchSampler` 相同
- [x] 1.4 `train.py`／`train_reid.py` 加入 `--downscale-prob`、`--downscale-min`、`--downscale-max`、`--distractor-fraction`、`--cooccur-min-frames`、`--pair-max-frame-gap`（預設關閉；干擾物取樣需要 `--crop-root` 下的 `_index/` 與 `config/roi/`）與縮小版 val 選 epoch，寫入訓練紀錄（含每個 epoch 兩種 val mAP 與實際干擾物比例）；以合成資料冒煙測試（兩者都開啟、1 個 epoch）驗證紀錄內容、權重可由 `Custom_ReID` 載入
- [x] 1.5 `fusion_stats.py` 加入 AUC、TPR@FPR5%、兩種正樣本間隔（≥ 1 秒換算的幀數；最小間隔幀數～保留期限幀數，取 config 的逐類別 `track_buffer`；`--config` 預設改為 `botsort_reid_timm.yaml`）、分尺寸回報與關卡判定（含雜訊底線）（D4）；`fusion_distance_stats.py` 加入 `--reference`、`--legacy-crop`、`--noise-reference`；以單元測試驗證 AUC（完全分開＝1、完全相同＝0.5）、TPR@FPR5%、間隔上限與保留期限取法（spec「正樣本間隔以保留期限為上限」「保留期限不依追蹤器或幀率換算」）、分尺寸歸組（spec「分尺寸回報」）與 spec 的關卡情境（改善太小、雜訊底線較大、只有尾端沒改善、多個通過只取最好、沒有通過）
- [x] 1.6 跑 `pytest`，確認全部通過且 tracker override 方法的 AST 檢查仍通過

## 2. 參考、訓練與關卡

- [x] 2.1 量測參考（`dinov2s_v3`，`--legacy-crop`）與第一個候選（`dinov2s_v3`，緊框），並與 `reid-center-gate-fusion` 的舊數字（≥ 1 秒）並列；關卡判定等 2.4 有雜訊底線後一起做；結果寫入 `verification.md`
- [x] 2.2 冒煙訓練（兩者都開啟、2 個 epoch、部分身分），確認損失下降、每個 epoch 耗時、GPU 記憶體與實際干擾物比例；估計四個訓練任務平行的全量時間，回報使用者後再進行
- [ ] 2.3 平行訓練 `dinov2s_v3_lowres`、`dinov2s_v3_distractor`、`dinov2s_v3_both` 與種子對照 `dinov2s_v3_seed1`（背景執行），以 `eval_reid.py` 回報各自原始與縮小版 val 指標；結果寫入 `verification.md`
- [ ] 2.4 量測 `dinov2s_v3_seed1`（緊框）算出雜訊底線；對三個訓練候選與 2.1 的緊框候選依 spec 判定關卡（含雜訊底線），選出進入追蹤驗收的候選（或記錄沒有候選通過）；並列各候選的分尺寸結果；結果寫入 `verification.md`

## 3. 追蹤驗收（只在 2.4 有候選通過時）

- [ ] 3.1 依 D6 以通過的候選跑調參序列的 40 組網格與同批基準，逐類別依 `reid-model` spec 判定並記錄每幀追蹤時間；沒有候選通過關卡時記錄跳過；結果寫入 `verification.md`
- [ ] 3.2 有類別通過時做合併驗證與保留序列，更新 `config/botsort_reid_timm.yaml`；沒有類別通過時 config 維持全部關閉；以 smoke 的 config 檢查通過、`config/botsort_reid.yaml` 與 golden configs 未改變作為驗證；結果寫入 `verification.md`

## 4. 文件與收尾

- [ ] 4.1 更新 CLAUDE.md 的 Re-ID 章節（推論端緊框、訓練新參數與指令、外觀分離度關卡、結果）與 `src/safetrajai/reid/HANDOFF.md`
- [ ] 4.2 撰寫 `verification.md` 的總結；跑 `pytest` 與 `openspec validate align-reid-training-domain --strict` 確認通過
