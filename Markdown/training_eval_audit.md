# 各模組訓練／評估流程盤點

盤點日期：2026-11-09

盤點範圍：`utils/modules/safetrajai/` 下的功能子套件，以及對應的 `scripts/` 操作腳本（兩者合併視為一條完整的訓練/評估管道）。

定義採廣義：
- 「訓練」包含有可學習參數的模型訓練，也包含沒有可學習參數但需要「調參」或「人工校準」的流程（例如 tracking 的 grid search、collision 的 homography 人工對應點）。
- 「評估」包含正式的驗收流程，也包含手動執行的驗證/比對腳本。

本次盤點只列現況與缺口，不包含補強建議或實作規劃。

## 盤點結果

| 模組（含對應腳本） | 訓練（廣義） | 評估（廣義） | 明顯缺口 |
|---|---|---|---|
| **`detection/`**（YOLO11） | **無**——只用外部預訓練權重（`yolo11n.pt`／`yolo11x.pt`／`yolo11x_dennis.pt`），repo 內無微調腳本 | 間接——靠 stage2 `evaluation/` 的 det_eval 驗證下游指標，非自身驗收 | 換權重後沒有固定驗收流程；`models/yolo-cls/yolo26m-cls.pt` 與 `models/yolo-reid/`（除 `temp/`）已用 grep 確認零程式碼引用，疑似孤立/待淘汰檔案 |
| **`tracking/`**＋`scripts/grid_search/`＋`tests/golden/` | 有（調參＝廣義訓練）——3 支腳本逐類別調 `match_thresh` 等 | 有——grid search 以 `HOTA.AssA_mean` 評分＋golden 零容忍回歸比對 | 見下方「Grid Search 實際執行覆蓋度」小節——只有 scooter 有留存的真實執行結果，其餘 5 類別×3 個 tracker 組合都找不到已執行證據，部分組合（BoTSORT+ReID 的 car/bus/truck/small_truck）甚至未被設計成可搜尋 |
| **`reid/`**＋`scripts/reid_train/`＋`scripts/reid_data/` | **完整**——manifest → `pack_manifest.py` → `train_reid.py`（PK sampler＋CE+triplet） | **完整分層**——`eval_reid.py`（參考用）→ `fusion_distance_stats.py`（外觀分離度關卡）→ `acceptance_grid.py`（追蹤驗收）；`reid_data/` 的資料品質把關也算上游評估 | 整個 repo 沒有 CI 自動化（無 `.github/workflows`，全靠人工跑 pytest/golden）；模型版本靠手動資料夾命名（`dinov2s_v3` 等），無正式 model registry |
| **`evaluation/`** | — | **本身就是評估引擎** | 不適用（它是評估工具，不是被評估對象） |
| **`collision/`**＋`scripts/homography/` | 無資料驅動調參——homography 矩陣/衝突門檻靠既有矩陣或人工對應點，沿用 `ignore/Sub_Module/trajectoryanalyst2` 固定定義 | 有，但手動——`validate_homography.py`／`check_homography_scale.py`／`geocode_cameras.py` 都是手動執行，無像 reid 那樣的正式驗收 spec | 沒有系統性的跨鏡頭驗收報表；指標本身也明記「只能當技術訊號」 |
| `roi.py`／`secrets.py`／`paths.py`／`classes.py`／`pipeline/`／`visualization/` | 不適用 | 不適用 | 確認完全不需要（純工具/編排，已逐一核實非假設） |

## 結構性缺口彙總

1. `detection/` 完全沒有微調/驗收流程。
2. 整個 repo 沒有任何 CI 自動化，驗證全靠人工執行。
3. `collision/`／homography 的驗證偏手動，沒有正式驗收 spec（相較 reid 的 `acceptance_grid.py`）。
4. `tracking/` 的 grid search 覆蓋度已追查（見下方小節）：6 類別×3 個 tracker 中，只有 scooter 有留存的真實執行結果，其餘組合缺乏執行證據，部分甚至未被設計成可搜尋。
5. `yolo-cls`、`yolo-reid` 兩個模型權重已確認零程式碼引用，疑似孤立檔案。

## Grid Search 實際執行覆蓋度

補充調查日期：2026-11-09。repo 沒有逐次執行紀錄（`runs/`／`ignore/runs/` 被 gitignore，三支腳本歷史上都沒有留存 `--result-csv`），以下是由間接證據推論出的覆蓋度。

**唯一有留存真實執行結果的組合：`scooter`**

`ignore/runs/grid_search/best_params_scooter.txt`（2026-07-24，score = `HOTA.AssA_mean` 0.4737）記錄了一次完整跑完的結果，`use_reid: True`、`norm_factor: 200.0`。`norm_factor=200` 只出現在 BoTSORT／BoTSORT+ReID 的搜尋網格（`[150, 200, 250]`），不在 ByteTrack 的網格（`[80, 100, 120]`）裡，可確定這次是對 **scooter × BoTSORT(+ReID)** 跑的。三份情境設定檔（`bytetrack.yaml`／`botsort_noreid.yaml`／`botsort_reid.yaml`）的 `scooter` 參數完全一致，且與這份結果檔一致——代表這組參數只被真正搜尋過一次，之後手動複製到另外兩份情境設定，並非三個 tracker 各自獨立調參的結果。

**其他 5 個類別（person／car／bus／truck／small_truck）：找不到任何已執行證據**

- 三份情境設定檔裡這 5 個類別的參數三份完全一致，且 `grid_search_botsort_reid.py` 的程式碼註解明寫：這些數值「與 `config_BotSort_0413.py`（refactor 前的舊版設定）固定值完全一致」——即沿用舊設定，不是調參產物。
- `grid_search_botsort_reid.py` 裡 `car`／`bus`／`small_truck`／`truck` 的 `PARAM_GRIDS` 全部是單一值（例如 `"match_thresh": [0.7]`），沒有候選範圍可搜——這幾個組合在這支腳本裡連「設計上要搜」都不成立。
- `grid_search_botsort.py` 對這 4 個類別也只有 ReID 相關的 `proximity_thresh`／`appearance_thresh` 有真正多值範圍，核心匹配參數（`match_thresh` 等）同樣是固定單值。
- `grid_search_bytetrack.py` 是唯一一支「6 類別都有真正多值搜尋網格」的腳本，但沒有任何留存結果能證明它對 person/car/bus/truck/small_truck 真的被執行過並套用結果。

| 類別 | ByteTrack | BoTSORT（無 ReID） | BoTSORT + ReID |
|---|---|---|---|
| scooter | 網格有設計；配置值＝BoTSORT+ReID 結果複製 | 網格有設計；同上複製 | **有留存執行證據**（2026-07-24） |
| person | 網格有設計，無留存執行證據 | 網格有設計，無留存執行證據 | 網格有設計，無留存執行證據 |
| car／bus／truck／small_truck | 網格有設計，無留存執行證據 | 核心參數未設計成搜尋（固定值＋只搜 ReID 參數） | 完全未設計成搜尋（全固定值＝沿用舊 config） |

## 現成訓練／評估資料集盤點（已在 repo 下）

補充調查日期：2026-11-09。

| 模組 | 現成訓練資料集 | 現成評估資料集 | 備註 |
|---|---|---|---|
| **`reid/`** | **完整**——`dataset/reid/crops/{6類}/`（逐 track 裁切）＋`dataset/reid/splits/{v1,v2,v3}/manifest.csv`（v3 共 156,187 筆）＋品質把關清單（`exclusions.csv`／`auto_exclusions.csv`／`fragment_merges.csv`） | **完整**——`dataset/reports/reid/`（`contamination_crops.csv`／`class_consistency.csv`／`appearance_outliers.csv`／`motion_flags.csv`／`fragment_features.npz` 等稽核報表，供資料品質與分離度評估） | 唯一訓練＋評估資料都齊全、且版本化（v1/v2/v3）的模組 |
| **`detection/`** | **無**——repo 內沒有任何偵測訓練用標註資料集，只找到預訓練/外部權重 | 有（間接）——與 tracking 共用 `dataset/eval/cctv_all8/annotations_all8/`（6 段人工標註影片的偵測 GT） | 這份是「評估」GT，不是訓練資料；YOLO 權重若要微調，目前沒有現成資料可用 |
| **`tracking/`** | 不適用（無可學習參數） | **有，且分兩用途**——`dataset/eval/cctv_all8/mot_annotations_all8/`（MOT GT，stage2 評估用）＋`dataset/eval/cctv_all8_subsets/{tune,holdout}/`（grid search 調參用 tune、驗收用 holdout，各自內含 annotations_all8/mot_annotations_all8） | tune/holdout 子集的存在，表示當初確實規劃了「調參–驗收」資料切分，但多數類別×tracker 沒有留下已執行證據（見上方小節） |
| **`evaluation/`** | 不適用 | 本身即消費上面 `cctv_all8` 的 GT，不是獨立資料集 | — |
| **`collision/`**（homography／conflict） | 無 | **只有校準素材，非正式資料集**：`models/homography/matrices/`（既有矩陣）、`models/homography/checkpoints/`（僅 `C000297` 一組 check points）、`config/homography_checkpoints/`（僅 `C000025`／`C000297` 兩支鏡頭有設定）、`dataset/raw/homography_videos/{C000043,C000054,C000085}/`（校準用影片） | **沒有「衝突事件」的人工標註 GT**——TTC/PET/衝突指標本身沒有現成資料可驗證對錯，這點與 CLAUDE.md「只能當技術訊號」的限制一致 |
| `roi.py`／`secrets.py`／`paths.py`／`classes.py`／`pipeline/`／`visualization/` | 不適用 | 不適用 | 純工具/編排模組，無對應資料集 |

**重點發現**：
1. `reid/` 是唯一訓練與評估資料都完整齊全的模組。
2. `detection/` 完全沒有訓練資料集（只有評估 GT），呼應前面「完全沒有微調流程」的缺口——沒資料，所以也沒流程。
3. `tracking/` 有 tune/holdout 資料切分，但目前的 grid search 執行覆蓋度很低，代表資料備好了、流程卻沒真正跑滿。
4. `collision/`（衝突事件分析）沒有任何人工標註的事件級 GT，只有幾何校準用的素材，這是比「流程缺口」更根本的「資料缺口」。

## 補強優先順序（成本／效益判斷）

補充調查日期：2026-11-09。判斷依據：**成本**＝要補齊這個缺口需要新建多少東西（資料／程式碼／人力),**效益**＝補齊後實際影響的範圍與重要性。排序由「最划算」到「最不划算」。

| 順序 | 缺口 | 成本 | 效益 | 為什麼是這個排序 |
|---|---|---|---|---|
| 1 | **隔離孤立模型權重**（`models/yolo-cls/`、`models/yolo-reid/` 除 `temp/`） | 近乎零——只是 `mv` 到 `temp/` 並記錄 `DO_NOT_USE.md`，不用新建任何資料或程式碼 | 小但立即——避免之後誤用零引用的權重、減少混淆 | 成本幾乎是 0，沒有理由不先做，屬於「順手清理」 |
| 2 | **補齊 `tracking/` grid search 覆蓋**（person／car／bus／truck／small_truck × 3 個 tracker） | 低～中——`scripts/grid_search/` 三支腳本、`config/tracking/*.yaml`、`dataset/eval/cctv_all8_subsets/{tune,holdout}/` 全部已經存在，只差：(a) 把 `grid_search_botsort.py`／`grid_search_botsort_reid.py` 裡 car/bus/truck/small_truck 的固定單值網格改回真正的候選範圍，(b) 實際執行並留存 `--result-csv` | 高——tracking 是整條管線的核心，目前 5/6 類別的參數是直接沿用 refactor 前的舊設定，從未驗證是否最佳；影響 HOTA/AssA，也間接影響下游 collision／counting 的輸入品質 | 基礎設施（腳本＋資料＋設定）已經 100% 到位，缺的只是執行與一處小改動，是目前投入產出比最高的一項 |
| 3 | **建立基本 CI 自動化**（至少跑 `pytest tests/test_smoke.py`） | 中——需要新建 CI workflow，但不需要新資料；要注意 golden 比對依賴掛載磁碟上的 `dataset/`／`models/`，不易搬進一般 CI runner，實際能自動化的範圍可能只到 smoke test | 中～高——跨所有模組的回歸防護網，目前完全靠人工記得要跑 pytest/golden | 單次建置成本中等，但長期每次改動都受益，優先度次於「資料和腳本已齊備只缺執行」的 tracking 缺口 |
| 4 | **將 `collision/`／homography 驗證正式化成驗收 spec** | 中～高——`validate_homography.py`／`check_homography_scale.py` 已存在，但要仿照 reid 的 `acceptance_grid.py` 設計正式門檻與報表；而且現有 check points 只覆蓋 `C000025`／`C000297` 兩支鏡頭，範圍本來就窄 | 中——只影響有 homography 的 4 支鏡頭（C000025/C000043/C000054/C000085）的衝突指標可信度 | 影響範圍比 tracking 窄，且要先有更多鏡頭的 check points 才能把 spec 做紮實，成本效益不如前兩項 |
| 5 | **`detection/` 建立微調訓練資料與流程** | 高——repo 內完全沒有偵測訓練用標註資料，必須從頭標註或另外取得資料集，還要新建訓練/驗收腳本 | 不確定——目前沒有證據顯示預訓練 YOLO（`yolo11x_dennis.pt`）在 `cctv_all8` 評估 GT 上的表現不足，微調的必要性未被證實 | 成本最高、效益卻還沒被證實有需求，應該等有具體證據顯示偵測品質不足後才投入 |
| 6 | **`collision/` 建立衝突事件人工標註 GT** | 最高——需要人工在影片上標註真實的近距離接近/衝突事件，主觀且耗時，且目前只有 4 支鏡頭有 homography 可用公尺單位 | 低～中——CLAUDE.md 已明確定位這些指標「只能當技術訊號，不能當正式安全結論」，代表專案本身對形式化驗證的需求程度還不高 | 成本最高，而且專案自身的定位已經限縮了這份資料能帶來的效益上限，目前最不划算 |

## 目前實際使用的 Re-ID 權重

補充調查日期：2026-11-09。

**目前實際有在運作的流程裡沒有任何 Re-ID 權重真的被啟用**，分三層：

1. **預設流程**（`track_eval.py` 不帶 `--config`）走 `config/tracking/bytetrack.yaml` → ByteTrack，完全沒有 Re-ID 能力。
2. **目前的驗收基準情境**（`config/tracking/botsort_reid_timm.yaml`，R1 參數）——6 個類別的 `use_reid` 全部是 `false`（檔頭註解：「追蹤結果等同驗收基準」）。檔案裡提到唯一的候選訓練權重是 `models/class_reid_weights/dinov2s_v3/model.pt`（DINOv2 ViT-S/14，資料集版本 v3，由 `Custom_ReID` 載入），但它沒有通過追蹤驗收——在 car、scooter 上會讓 IDSW 明顯變差，40 組（閘門／否決／原融合 × 5 類）實驗沒有任何類別通過 spec `reid-model` 的啟用條件，所以目前停用。
3. **舊的 `config/tracking/botsort_reid.yaml`**（情境 4，非目前主線）——6 類 `use_reid: true`，但 `model` 欄位指向單純的 YOLO 偵測權重（`models/yolo/yolo11x.pt`／`yolo11x_dennis.pt`），走的是 `ReID`（YOLO backbone 當 encoder）這條舊路徑，不是本地訓練的 `dinov2s_v3`。這份設定還能跑，但不是目前建議使用的版本。

**結論**：目前沒有正式在用的 Re-ID 權重；唯一存在且訓練完整的自訂權重是 `dinov2s_v3`，但因未通過追蹤驗收而關閉。

## 目前實際使用的偵測權重

補充調查日期：2026-11-09。已用 `grep -rn "det_model_path" config/` 逐一確認全部 11 份情境設定檔（`tracking/`、`grid_search/`、`collision/`、`reid/reid_crops.yaml` 等）。

**結論**：目前所有情境都指向同一個偵測權重：

```
models/yolo/yolo11x_dennis.pt
```

- 這個權重**不在本專案任何訓練流程的產出範圍內**——README 標註「不在 HF，向專案維護者取得」，屬於外部提供的權重，repo 裡沒有對應的訓練腳本或訓練紀錄。
- `models/yolo/yolo11x.pt`（標準版）只出現在少數舊情境（`botsort_reid.yaml`／`botsort_noreid.yaml`）的 `class_tracking_params.<class>.model` 欄位，但那是 **Re-ID 融合的 backbone 來源**（`ReID` 類別把 YOLO 當 encoder），不是偵測模型；真正負責偵測推論的 `det_model_path` 一律是 `yolo11x_dennis.pt`。
- 與前面「盤點結果」一致：`detection/` 模組本身完全沒有微調訓練流程，目前用的就是這顆外部取得、未在 repo 內訓練過的權重。

## 未來計畫：以 `yolo11x_dennis.pt` 為起始模型精煉並製作偵測資料集

記錄日期：2026-11-09（使用者提出，尚未開始實作）。

使用者計畫以目前在用的偵測權重 `models/yolo/yolo11x_dennis.pt` 當作起始模型（seed model），用來精煉（refine）並製作偵測訓練資料集。這對應前面「補強優先順序」第 5 項（`detection/` 建立微調訓練資料與流程）——目前 repo 內完全沒有偵測訓練用的標註資料集，這個計畫就是要補上這個資料缺口。

以下是經過 `/grill-me`（grilling skill）逐輪確認後的共識（2026-11-09）：

1. **來源影片**：`config/datasets.yaml` 已登記的 `reid_sources` 9 支鏡頭（C000011、C000012、C000033、C000043、C000048、C000054、C000056、C000085、C000256）。
2. **精煉方式**：用 `yolo11x_dennis.pt` 自動產生偽標籤，再人工修正——初期採**更高比例的人工逐框覆核**（前幾批每張都看，理由：偵測框的位置誤差／漏檢／誤檢會直接變成訓練標籤的雜訊，比 `reid/` 那種寬鬆的二元排除判斷後果更大），穩定後再視情況拉高自動化比例。repo 內目前沒有任何偽標籤/自動標註的先例可參考，需要全新設計。
3. **類別範圍**：全部 6 類（person／car／scooter／small_truck／bus／truck）一起做。
4. **Pilot 規模**：9 支來源鏡頭各取 1～2 個檔案（約 10～15 個檔案），不是一次用滿 `reid_sources` 全部 39 個檔案。
5. **畫面取樣頻率**：每支影片每 N 秒（建議 1～2 秒）取 1 張畫面，比照 `dataset/reid/` 裁切「每條軌跡每 0.5 秒最多 1 張」的慣例精神，而非逐幀或固定抽樣張數。
6. **Pilot 資料量目標**：逐類別門檻——每個類別至少要有一定數量的標註框（例如每類至少 100 個框）；`bus`／`truck` 這類出現頻率較低的類別若不足，之後再加碼取樣頻率或鏡頭補足，不是只看總畫面張數。
7. **標註/修正工具**：引入 **CVAT**。已確認 repo 內沒有任何現成的「畫框＋選類別」標註工具（`scripts/picker/make_picker.py` 四種模式都只是點選座標點），部署方式與具體操作留到真正開始標註時再決定。
8. **資料集存放結構**：新開 `dataset/detection/`，仿照 `dataset/reid/` 的版本化慣例（`images/`／`labels/`／`splits/<version>/manifest.csv`），不直接沿用現有 `dataset/eval/`／`dataset/raw/` 的扁平結構。
9. **資料集登記**：在 `config/datasets.yaml` 新增一個獨立項目（例如 `detection_source`），即使來源鏡頭跟 `reid_sources` 重疊也不直接共用，讓偵測與 Re-ID 兩條資料管線各自獨立演化。
10. **驗收方式**：精煉後的模型先用現有 `dataset/eval/cctv_all8/annotations_all8/` 這份人工 GT 做 det_eval 比較（precision/recall/mAP），不夠再決定是否擴充評估集。

**背景事實**（調查於 2026-11-09，供執行時參考）：
- 目前 `dennis640`（即 `yolo11x_dennis.pt`）在 `cctv_all8` 6 段序列上的逐類別平均 mAP@50:95：scooter 0.78（最弱）、person 0.84、bus 0.88、car 0.89、small_truck 0.90、truck 0.91（來源：`runs/rebuild-reid-dataset/tool_model/dennis640/trackeval_output/*/det_metrics_*_new.xlsx`，與候選模型 `yolo11x1280` 的比較留存結果）。
- `reid_sources` 9 支鏡頭共 39 個檔案，分佈不均：8 支鏡頭各只有 1 個檔案，`C000054` 一支就有 31 個。
