# Claude Session 交接紀錄

- Session ID：`91d9c55d-5875-4e4b-b54e-5f13f41dec8c`
- 整理日期：2026-10-08
- 專案：`/mnt/moris/Dcoker_Project/Project/safetrajai_eval`
- 原始紀錄：`/root/.claude/projects/-mnt-moris-Dcoker-Project-Project-safetrajai-eval/91d9c55d-5875-4e4b-b54e-5f13f41dec8c.jsonl`
- 對話主要時段：2026-10-08 03:36～07:58 UTC（台北時間 11:36～15:58）。
- 本文件是對話與產物的交接摘要，不是逐字稿。歷史指令僅作背景；本次只整理文件，未續跑訓練、修改設定、提交或封存 change。

## 1. 接手時最重要的結論

目前工作是 OpenSpec change `align-reid-training-domain`。程式實作、四個模型訓練與外觀分離度關卡判定均已完成；**沒有候選通過關卡**。目前 `tasks.md` 為 **11/15 完成**，尚未勾選 3.1、3.2、4.1、4.2。

使用者最後明確回覆「先照建議繼續往下走」，選擇：

1. 依既定 spec 判定不通過，記錄跳過第 3 節追蹤驗收。
2. `config/botsort_reid_timm.yaml` 維持不變，不啟用新的 Re-ID 候選。
3. 完成第 4 節文件與驗證收尾，記錄 small_truck 樣本限制，作為未來 change 的依據。

這個選擇已在原 session 確認，不是仍待回答的問題。當時未選擇修改關卡規則，也未選擇額外對 `both` 跑參考用追蹤驗收。

**中斷位置：** Claude 準備檢查 config 差異、執行 config smoke test 並讀取既有 Re-ID HANDOFF。該 Bash 呼叫先被 `Fact-Forcing Gate` hook 阻擋，沒有成功執行；接著 Claude 顯示週用量上限，對話中斷。因此不能把這次預定檢查或第 4 節當成已完成。

## 2. 背景與本次完成的工作

Re-ID 資料製作已完成，現有 `dataset/reid/splits/` 包含 v1、v2、v3；原 session 盤點 v3 為 22,223 個身分、156,186 張裁切，train 子集為 19,101 個身分。本 change 聚焦訓練與 C000025 評估鏡頭的輸入條件對齊。

已完成：

- 修正過時的 OpenSpec artifact，遵循 ADR 0009：`track_buffer` 直接代表幀數，不依追蹤器或 fps 換算。
- 分離度工具預設使用 `config/botsort_reid_timm.yaml`；C000025 四輪車保留 15 幀，person／scooter 10 幀。`reid_crops.yaml` 的 30／20 幀屬於其他來源條件，不應拿來校準 C000025。
- 新格式 `Custom_ReID` 推論使用緊框（gain=1.0、pad=0），`legacy_crop=True` 保留舊裁切（gain=1.02、pad=10）作為量測參考；舊格式測試不變。
- 加入解析度增強、固定的縮小版 validation 與選 epoch 邏輯。
- 加入干擾物取樣器、裁切索引與 ROI metadata 讀取，以及訓練統計。
- 加入 AUC、TPR@FPR5%、保留期限內正樣本、分尺寸回報、雜訊底線與關卡判定。
- 將過長的 `reid-model` 關卡需求拆成量測與判定兩條，歷史上的 strict 驗證失敗已修復。

主要程式位置：

- `src/safetrajai/reid/encoders.py`
- `src/safetrajai/reid/training/{data,sampler,train}.py`
- `src/safetrajai/reid/fusion_stats.py`
- `scripts/reid_train/{train_reid,eval_reid,fusion_distance_stats}.py`
- `tests/test_reid_encoder.py`、`tests/test_reid_training.py`、`tests/test_reid_distractor_sampler.py`、`tests/test_reid_fusion_stats.py`

## 3. 已確認的實驗與設計決策

- 干擾物可以在同一 epoch 重複選入；先選尚未用過的，不足再選已用過的，不另外設重複次數上限。同一批身分不重複，種子與補位不重複使用。
- 每個 epoch 批數維持「身分數 // P」，以保持各候選訓練量一致；紀錄重複干擾物數與看到的身分比例。
- P=32、K=4、30 epoch、一般種子 0、`--workers 5`。不為提高比例而改 P 或放寬干擾物定義。
- 原本預估約 47% 的干擾物比例不可達。可重複取樣後實測約 17～19%，理論平均上限約 19.6%，原因是每個身分的干擾物中位數只有 5 個。使用者已接受約 18%，任務 1.7 門檻改為接近理論上限。
- 三個訓練候選以縮小版 val 選 epoch；`seed1` 必須使用 `--select-by val`，與既有 v3 相同，避免選 epoch 方法污染雜訊底線。
- val mAP 只供訓練觀察，不能代替分離度關卡或追蹤驗收。
- 冒煙與 OOM 不完整權重隔離至 `models/class_reid_weights/temp/`，用途記於 `temp/DO_NOT_USE.md`。

## 4. 訓練結果與資源限制

四個任務最初同時執行，但 lowres 在第 2 epoch 發生 CUDA OOM。每個任務實際約需 8.25～8.4 GB，四個超過 32 GB GPU 容量。其餘三個完成後，lowres 已單獨重跑完成；接手時不要因看到舊 OOM log 就再次啟動訓練。

| 模型 | 最佳 epoch | 原始 val mAP | Rank-1 | 縮小版 val mAP | 平均干擾物比例 |
|---|---:|---:|---:|---:|---:|
| dinov2s_v3（既有參考） | 27 | 0.7830 | 0.9058 | — | — |
| dinov2s_v3_seed1 | 26 | 0.7821 | 0.9039 | 0.4777 | — |
| dinov2s_v3_lowres | 27 | 0.7721 | 0.9010 | 0.7128 | — |
| dinov2s_v3_distractor | 26 | 0.7924 | 0.9094 | 0.5005 | 0.183 |
| dinov2s_v3_both | 29 | 0.7798 | 0.9033 | 0.7205 | 0.183 |

時間紀錄有一處需在收尾時修正：`verification.md` 仍寫 lowres 13:59～14:35，但 session 的完成通知時間是 **06:24 UTC／14:24 台北時間**。14:35 是先前估計，不能當成實際完成時間。

## 5. 外觀分離度關卡結果

參考模型為 `dinov2s_v3` 舊裁切；所有候選以緊框量測。種子對照與既有 v3 都使用緊框計算雜訊底線。

| 候選 | car AUC | car TPR@FPR5% | scooter AUC | scooter TPR@FPR5% | small_truck AUC | 判定 |
|---|---:|---:|---:|---:|---:|---|
| v3 舊裁切參考 | 0.9418 | 0.7412 | 0.9130 | 0.6639 | 0.9874 | 參考 |
| v3 緊框 | 0.9370 | 0.6769 | 0.9366 | 0.7137 | 0.9852 | car 退步 |
| lowres | 0.9710 | 0.8762 | 0.9443 | 0.7656 | 0.9568 | small_truck 下降約 0.031 |
| distractor | 0.9440 | 0.6969 | 0.9427 | 0.7475 | 0.9791 | car 改善不足且 TPR 下降 |
| both | 0.9725 | 0.8866 | 0.9473 | 0.7866 | 0.9546 | small_truck 下降約 0.033 |

- 雜訊底線：car AUC 0.0006、TPR 0.0102；scooter AUC 0.0020、TPR 0.0140。
- 最小改善量：兩類 AUC 均為 0.01；car TPR 0.0204，scooter TPR 0.028。
- `select_best` 回傳 None；lowres 與 both 雖改善 car、scooter，仍違反其他類別 AUC 下降不得超過 0.02 的規則。
- small_truck 只有 43 對干擾物，剛超過 30 對門檻；退步集中於最短邊 <16px 的組，該組只有 13 對（AUC 約 0.998 → 0.951）。16～32px 組沒有下降。
- 解析度增強對極小框有明顯效果，例如 car <16px 的 AUC：緊框 v3 約 0.940 → both 約 0.972。但這不代表追蹤 IDSW 已改善，因本輪未進行追蹤驗收。
- 未來可另開 change 補 small_truck 資料，或事先設計各尺寸組的樣本充分性門檻；本輪不因看到結果才放寬規則。

## 6. Git 與驗證狀態

本次整理時確認最新三個 commit 仍為：

| Commit | 內容 |
|---|---|
| `bdcb1a0` | `docs(openspec): add align-reid-training-domain change` |
| `964260b` | `feat(reid): align Re-ID training with the evaluation camera domain` |
| `f723ba0` | `feat(reid): allow distractors to repeat within an epoch` |

與此 change 有關的未提交修改仍為 `tasks.md` 與 `verification.md`。工作區另有 Markdown、skills-lock、tools 等不相關修改及未追蹤檔案，後續提交應先檢查並明確限定範圍。

本次 `git diff --stat -- config tests/golden` 無輸出，表示這些路徑目前沒有工作區差異；這不等於已重新執行 config smoke test。

歷史驗證證據：

- 初次提交前：368 passed、11 skipped。
- 第 1 節實作後：398 passed、11 skipped，tracker override AST 檢查通過。
- 干擾物重複取樣修改後：399 passed、11 skipped。
- 需求拆分後 `openspec validate align-reid-training-domain --strict` 通過。
- **最終第 4.2 驗證尚未完成。本次撰寫交接文件未重新執行上述測試。**

## 7. 待續工作順序

1. 讀取目前 change 的 `tasks.md`、`verification.md`、design 與兩份 delta spec，確認工作區是否有後續更新。
2. 完成 3.1 的「無候選通過，跳過 40 組追蹤網格」紀錄；不要啟動該網格。
3. 完成 3.2 的設定維持不變紀錄，檢查 `botsort_reid_timm.yaml`、`botsort_reid.yaml` 與 golden configs，執行 config smoke test，再更新任務狀態。
4. 完成 4.1：更新 `CLAUDE.md` 的 Re-ID 章節及 `src/safetrajai/reid/HANDOFF.md`，涵蓋緊框、新參數與訓練指令、分離度關卡和最終不通過結論。
5. 完成 4.2：補 `verification.md` 最終總結、修正 lowres 完成時間；檢查前段「同一 epoch 身分不重複」的舊描述，明確區分初版與 1.7 後的規則，避免矛盾。
6. 在原使用的 `track_eval` conda 環境執行完整 `pytest` 與 strict OpenSpec 驗證，根據實際結果勾選完成。第 2.3 雖已勾選，若需要獨立 `eval_reid.py` 執行證據，應先查原 log／訓練紀錄，勿把訓練 val 數字誤稱為本次重新評估。
7. 檢查最終 diff，回報並依後續授權處理 commit／archive；此 session 沒有完成最終提交或封存的證據。

## 8. 優先閱讀的文件與產物

- [任務清單](../openspec/changes/align-reid-training-domain/tasks.md)
- [完整驗證紀錄](../openspec/changes/align-reid-training-domain/verification.md)
- [設計](../openspec/changes/align-reid-training-domain/design.md)
- [提案](../openspec/changes/align-reid-training-domain/proposal.md)
- [reid-model delta spec](../openspec/changes/align-reid-training-domain/specs/reid-model/spec.md)
- [reid-fusion delta spec](../openspec/changes/align-reid-training-domain/specs/reid-fusion/spec.md)
- [既有 Re-ID HANDOFF](../src/safetrajai/reid/HANDOFF.md)（第 4.1 的待更新目標，與本文件不同）
- 訓練紀錄：`models/class_reid_weights/dinov2s_v3_{lowres,distractor,both,seed1}/train_record.json`。
- 分離度 JSON：`runs/reid_domain/dinov2s_v3_legacy.json`、`dinov2s_v3_tight.json` 及四個新模型的 `*_tight.json`；本次已確認這些路徑存在。
- 訓練與量測 log：`runs/reid_domain/`。
- 隔離說明：`models/class_reid_weights/temp/DO_NOT_USE.md`。
