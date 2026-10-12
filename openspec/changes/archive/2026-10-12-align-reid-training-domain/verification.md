# Verification

## 1. 程式修改

- `pytest`：398 passed、11 skipped，tracker override 方法的 AST 檢查通過。
- `Custom_ReID` 新格式改用緊框（`gain=1.0, pad=0`）；`legacy_crop=True` 保留 `gain=1.02, pad=10`，只供量測參考。
- 新增與修改的測試：
  - `tests/test_reid_encoder.py`：緊框裁切（中心 (100, 80)、寬 40、高 60 → x 80～120、y 50～110，像素逐一相同）、`legacy_crop` 仍放大加邊；修改前兩個測試失敗、修改後通過，舊格式測試不變。
  - `tests/test_reid_training.py`：解析度增強（被選中時最短邊落在範圍內、原本較小的裁切不放大、機率 0 與原本相同、縮小版 val 每次相同且尺寸有變化）；以縮小版 val 選 epoch（spec 情境：第 20 個 epoch 原始 val 最高、第 25 個 epoch 縮小版最高 → 選 25）；兩者都開啟訓練 1 個 epoch 的冒煙測試（訓練紀錄的設定、半徑、每個 epoch 兩種 val mAP 與實際干擾物比例，權重可由 `Custom_ReID` 載入）。
  - `tests/test_reid_distractor_sampler.py`：半徑依鏡頭寬度換算（C000054 car 320px）、同時出現且鄰近才算干擾物、碎片（幀範圍不重疊）不算、距離超過半徑不算、重疊不足或不同類別／序列不算；每批 P×K、同一 epoch 身分不重複（1.6 時的初版規則；1.7 起只有種子與補位身分在同一 epoch 不重複，干擾物可重複選入，見下方 1.7）、比例 0.5 時第一批至少一半是種子與它的干擾物、干擾物不足時不退回、比例 0 時與 `PKBatchSampler` 逐批相同；裁切索引只讀 metadata、缺列報錯、ROI 取寬度。
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

### 1.7 干擾物可重複選入（2.2 後的決定）

- 冒煙權重已隔離到 `models/class_reid_weights/temp/smoke_domain_both/`（記錄於 `temp/DO_NOT_USE.md`）。
- `pytest`：399 passed、11 skipped；新增「種子不重複、干擾物用完後仍可重複選入、同一批不重複」測試，比例 0 時仍與 `PKBatchSampler` 逐批相同。
- 全量 `v3` train（P=32、比例 0.5、15 個干擾物名額）：

| epoch | 干擾物比例 | 重複選入 | 看到的身分比例 |
|---|---|---|---|
| 0 | 0.1887 | 1600 | 0.9147 |
| 1 | 0.1823 | 1556 | 0.9170 |
| 2 | 0.1702 | 1436 | 0.9233 |

- **比例仍未接近 0.47**：上限由干擾物數本身決定（中位數 5）。以「min(干擾物數, 名額) / P」計算的理論上限：P=32 為 0.196、P=16 為 0.290、P=8 為 0.317。重複選入已達到 P=32 的上限，要再提高只能改變干擾物的定義或 P。
- 使用者決定：接受約 0.18（P=32 的理論上限約 0.196），不改 P、不放寬干擾物定義，以維持四個候選只差一個變因。

### 2.3 訓練

`v3`、與 `dinov2s_v3` 相同的其他設定（30 epoch、P=32、K=4、種子 0）、`--workers 5`。distractor、both、seed1 平行（12:51～13:59，每個 epoch 約 130～180 秒、每個任務約 8.4 GB）；lowres 原本也平行，但第 2 個 epoch GPU 記憶體不足中止（四個任務共需約 33 GB，超過 32 GB；不完整的權重已隔離為 `temp/dinov2s_v3_lowres_oom`），之後單獨重跑（13:59 開始，完成通知於 14:24；時間皆為台北時間，原本記錄的 14:35 是開始前的估計）。log 在 `runs/reid_domain/`。

| 模型 | 選 epoch 方式 | 最佳 epoch | 原始 val mAP | Rank-1 | 縮小版 val mAP | 平均干擾物比例 |
|---|---|---|---|---|---|---|
| `dinov2s_v3`（既有） | 原始 val | 27 | 0.7830 | 0.9058 | — | — |
| `dinov2s_v3_seed1` | 原始 val | 26 | 0.7821 | 0.9039 | 0.4777 | — |
| `dinov2s_v3_lowres` | 縮小版 val | 27 | 0.7721 | 0.9010 | 0.7128 | — |
| `dinov2s_v3_distractor` | 縮小版 val | 26 | 0.7924 | 0.9094 | 0.5005 | 0.183 |
| `dinov2s_v3_both` | 縮小版 val | 29 | 0.7798 | 0.9033 | 0.7205 | 0.183 |

- 解析度增強讓縮小版 val 從約 0.48 升到約 0.71～0.72，原始 val 只降約 0.01；干擾物取樣對原始 val 略有幫助。val mAP 不作為關卡依據。

### 2.4 雜訊底線與關卡

雜訊底線（`dinov2s_v3_seed1` 與 `dinov2s_v3`，都以緊框量測）：car AUC 0.0006、TPR@FPR5% 0.0102；scooter AUC 0.0020、TPR@FPR5% 0.0140。最小改善量：car AUC 0.01、TPR@FPR5% 0.0204；scooter AUC 0.01、TPR@FPR5% 0.028（TPR 以 2 倍雜訊底線為準）。

| 候選（緊框） | car AUC | car TPR | scooter AUC | scooter TPR | person AUC | small_truck AUC | 關卡 | 未通過原因 |
|---|---|---|---|---|---|---|---|---|
| 參考（`dinov2s_v3` 舊裁切） | 0.9418 | 0.7412 | 0.9130 | 0.6639 | 0.9576 | 0.9874 | — | — |
| `dinov2s_v3` | 0.9370 | 0.6769 | 0.9366 | 0.7137 | 0.9515 | 0.9852 | 未通過 | car AUC −0.005、TPR −0.064 |
| `dinov2s_v3_lowres` | 0.9710 | 0.8762 | 0.9443 | 0.7656 | 0.9445 | 0.9568 | 未通過 | small_truck AUC −0.031 |
| `dinov2s_v3_distractor` | 0.9440 | 0.6969 | 0.9427 | 0.7475 | 0.9511 | 0.9791 | 未通過 | car AUC +0.002、TPR −0.044 |
| `dinov2s_v3_both` | 0.9725 | 0.8866 | 0.9473 | 0.7866 | 0.9464 | 0.9546 | 未通過 | small_truck AUC −0.033 |

- **依 spec 沒有候選通過**（`select_best` 回傳 None）。
- lowres 與 both 的 car、scooter 都遠超過最小改善量（both：car AUC +0.031、TPR +0.145；scooter AUC +0.034、TPR +0.123），只因 small_truck AUC 下降超過 0.02 而未通過。small_truck 只有 43 對干擾物（剛超過 30 對的門檻）；下降集中在最短邊 < 16px 的組（只有 13 對干擾物：0.998 → 0.951），16～32px 組沒有下降（0.987 → 0.987）。
- 分尺寸：car < 16px 的 AUC 由 0.940（緊框 `dinov2s_v3`）升到 0.970（lowres）／0.972（both），解析度增強主要改善極小的框。
- 「≥ 1 秒」的 AUC（與 `reid-center-gate-fusion` 可比）：car 0.858 → 0.921（both）、scooter 0.885 → 0.899（both）。
- 干擾物取樣單獨使用時效果小（car 未達最小改善量），與解析度增強合併時比只用解析度增強略好。

## 3. 追蹤驗收

### 3.1 追蹤網格：跳過

2.4 沒有候選通過外觀分離度關卡（`select_best` 回傳 None），依 spec「沒有候選通過」與 D6，不進行追蹤驗收：40 組網格（A 閘門 20 組、C 否決 10 組、原融合方式寬／嚴各 1 組 × 5 類）與同批基準都沒有執行，也沒有每幀追蹤時間的量測。使用者確認依既定 spec 判定，不修改關卡規則，也不對 `both` 另跑參考用的追蹤驗收。

### 3.2 設定維持不變

沒有類別通過，`config/botsort_reid_timm.yaml` 維持 6 類 `use_reid: false`，不啟用任何新候選。

- 以本 change 的第一個 commit（`bdcb1a0`）之前為基準，`git diff --stat 51f8281 HEAD -- config tests/golden` 無輸出：`config/botsort_reid_timm.yaml`、`config/botsort_reid.yaml` 與 `tests/golden/configs/` 的 5 份設定都沒有改變。
- smoke 的 config 檢查：`tests/test_smoke.py` 156 passed、9 skipped（全部 config 載入、`datasets.yaml` 的 eval／source 鏡頭不重疊、config 不引用 `temp/`、tracker override 方法的 AST 檢查都通過；環境見 4.2）。

## 4. 文件與收尾

### 4.1 文件

- `CLAUDE.md`：常用指令加上訓練新參數（`--downscale-prob`、`--distractor-fraction`、`--select-by`）與 `fusion_distance_stats.py` 的 `--legacy-crop`／`--reference`／`--noise-reference`；Tracker 章節記錄本 change 的結論；Re-ID 章節改寫 `Custom_ReID` 的緊框與 `legacy_crop`、列出四個未通過的實驗權重、新增「外觀分離度關卡」條目（條件、結果、`--select-by` 預設與干擾物比例上限）。
- `src/safetrajai/reid/HANDOFF.md`：修正舊敘述「`botsort_reid_timm.yaml` 只有 car 開啟 ReID」（實際 6 類都關閉）；新增 `fusion_stats.py`、緊框與 `legacy_crop`、訓練新參數表、干擾物取樣規則、外觀分離度關卡與本次結果、新隔離的權重。
- 本文件：1 節的取樣器測試描述註明「同一 epoch 身分不重複」是 1.6 的初版規則（1.7 起只限種子與補位）；2.3 的 lowres 完成時間改為 14:24（14:35 是估計）。

### 4.2 驗證

環境：雲端 container（無 GPU）以 Miniforge 建立 conda env `track_eval`（Python 3.10.22；依 `requirement.txt` 的版本：torch 2.7.1（CPU 版，原機為 `+cu128`）、torchvision 0.22.1、ultralytics 8.3.141、opencv-python 4.11.0.86、timm 1.0.15；其餘依 `pyproject.toml`），`pip install -e . --no-deps`。這個環境沒有 `dataset/`、`models/`、`runs/`。

- `pytest`：**398 passed、12 skipped**（74 秒）。與原機 1.7 後的 399 passed、11 skipped 相比，多 skip 的一個是 `tests/test_smoke.py` 的 pickle shim 載入測試（「quarantined Re-ID weight not present」：本環境沒有 `models/class_reid_weights/temp/`）；其餘 skip 原因與原機相同（8 個「script does not iterate classes」、3 個「not a tracking scenario」）。tracker override 方法的 AST 檢查通過。自 `f723ba0`（原機最後一次測試的程式）之後，`src/`、`tests/`、`scripts/`、`config/`、`pyproject.toml` 沒有任何差異。
- `openspec validate align-reid-training-domain --strict`（OpenSpec 1.14.1）：通過。
- 2.3 表中的 val 指標來自原機的訓練紀錄與 log；本環境沒有權重與打包檔，未重新執行 `eval_reid.py`。

## 總結

- 程式（緊框、解析度增強、干擾物取樣、縮小版 val 選 epoch、外觀分離度量測與關卡）完成，單元測試涵蓋 spec 情境。
- 四個候選都沒有通過外觀分離度關卡：緊框 `dinov2s_v3` 與 `distractor` 的 car 未達最小改善量；`lowres` 與 `both` 的 car、scooter 大幅改善（both：car AUC +0.031、TPR@FPR5% +0.145；scooter AUC +0.034、TPR +0.123），但 small_truck AUC 下降約 0.03，超過 0.02 的限制。
- 依 spec 不做追蹤驗收，`config/botsort_reid_timm.yaml` 維持 6 類關閉；沒有修改關卡規則。
- 限制與後續：small_truck 只有 43 對干擾物，下降集中在最短邊 < 16px 的 13 對，結論對極小框的樣本量很敏感。後續可另開 change 補 small_truck 資料，或事先訂定各尺寸組的樣本充分性門檻再重新判定；也可依分尺寸結果考慮追蹤器的最小框限制。外觀分離度改善不代表追蹤 IDSW 會改善，仍需追蹤驗收確認。
