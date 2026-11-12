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

### 補齊進度（2026-11-09 起，優先順序第 2 項執行中）

**ByteTrack 階段：已完成**（全部 6 類別、完整 6 段 `cctv_all8` 序列，`--params-subset match_thresh,second_match_thresh,unconfirmed_match_thresh`，搜尋範圍沿用各類別既有的真實網格）。詳細逐序列數據與兩種分數的完整比較見獨立報告：`Markdown/grid_search/bytetrack_all6_rerun/bytetrack_all6_report.md`（腳本與設定備份在同一資料夾）。

**⚠️ 重要更正**：grid search 回傳的分數（`safetrajai/grid_search/core.py::score_params()`）是「排除 0 分序列後取第 25 百分位數」，**不是平均值**；實際用逐序列算術平均重新驗算後，person／scooter 兩類在 grid search 分數上打平或進步，但平均準確率反而下降——這兩類最終**不採用** grid search 找到的參數。

| 類別 | GS分數(基準→最佳) | 平均準確率(基準→最佳) | 建議新參數 (match/second/unconfirmed) | 結論 |
|---|---|---|---|---|
| small_truck | 0.7180→0.8115 | 0.8268→0.8836 | 0.7 / 0.45 / 0.85 | 兩指標都明顯改善，**建議更新** |
| car | 0.7947→0.7966 | 0.8144→0.8152 | 0.7 / 0.5 / 0.85 | 改善幅度極小，可不改 |
| truck | 0.8373→0.8373 | 0.7208→0.7208 | 現值已最佳 | 不用改 |
| bus | 0.8369→0.8369 | 0.8409→0.8409 | 現值已最佳 | 不用改（僅 1 段序列有 GT） |
| person | 0.7444→0.7444 | 0.8563→**0.7907** | — | GS 打平但平均準確率變差，**不採用** |
| scooter | 0.6756→0.6856 | 0.7789→**0.7597** | — | GS 進步但平均準確率變差（指標矛盾），**不採用** |

**資料覆蓋度的重要發現**：`cctv_all8` 只有單一鏡頭（C000025）共 6 段序列，**沒有任何一段同時包含全部 6 個類別**。逐序列 × 逐類別 ROI 內可評估 GT 統計（6 段合計）：person 234、car 2084、scooter 1324、small_truck 146（4/6 段有）、truck 81（3/6 段有）、**bus 僅 21 個，且全部來自單一段序列**（`152639_15s_2`）。這代表 bus／truck／small_truck 的調參結論統計強度明顯弱於 person／car／scooter，是資料集本身的結構性限制，不是切分或程式的問題。

**BoTSORT（no ReID）階段：已完成**（同樣全 6 段序列，`scripts/grid_search/grid_search_botsort.py` 的 car/bus/truck/small_truck 已從固定單值改成真實候選範圍）。詳細報告：`Markdown/grid_search/botsort_noreid_all6_rerun/botsort_noreid_all6_report.md`。

| 類別 | GS分數(基準→最佳) | 平均準確率(基準→最佳) | 建議新參數 (match/second/unconfirmed) | 結論 |
|---|---|---|---|---|
| car | 0.8317→0.8641 | 0.8603→0.8942 | 0.8 / 0.5 / 0.85 | 兩指標都改善，**建議更新** |
| small_truck | 0.7871→0.9106 | 0.8598→0.9252 | 0.8 / 0.45 / 0.85 | 兩指標都明顯改善，**建議更新** |
| truck | 0.8352→0.8490 | 0.8790→0.8882 | 0.7 / 0.45 / 0.85 | 兩指標都改善，**建議更新** |
| bus | 0.9497→0.9497 | 0.9514→0.9514 | 現值已最佳 | 不用改（僅 1 段序列有 GT） |
| person | N/A（基準參數不在原 ReID 導向網格內） | 0.8856→0.8856 | — | 逐序列分數與基準逐位元相同，完全不敏感，不用改 |
| scooter | N/A（同上） | 0.7945→**0.7815** | — | ⚠️ 平均準確率反而下降，**不採用** |

**BoTSORT＋ReID 階段：已完成**（同樣全 6 段序列，`scripts/grid_search/grid_search_botsort_reid.py` 的 car/bus/truck/small_truck 已從固定單值改成真實候選範圍；ReID 權重沿用 `models/class_reid_weights/dinov2s_v3/model.pt`）。詳細報告：`Markdown/grid_search/botsort_reid_all6_rerun/botsort_reid_all6_report.md`。

門檻參數（match/second/unconfirmed）的 grid search 結論：

| 類別 | GS分數(基準→最佳) | 平均準確率(基準→最佳) | 結論 |
|---|---|---|---|
| person | N/A→0.756847 | 0.8856→0.8856 | 打平／不敏感，不採用 |
| car | 0.834227→0.835340 | 0.8941→**0.8789** | ⚠️ GS 分數微幅改善但平均準確率反而下降，**不採用** |
| scooter | N/A→0.689818 | 0.7933→**0.7771** | ⚠️ 平均準確率反而下降，**不採用新門檻**（ReID 開關見下） |
| small_truck | 0.869323→0.869323 | 0.9252→0.9252 | 打平，不用改 |
| bus | 0.903620→0.903620 | 0.9514→0.9514 | 打平（僅 1 段序列有 GT），不用改 |
| truck | 0.846728→0.846728 | 0.8882→0.8882 | 與基準完全相同，不用改 |

門檻參數沿用 BoTSORT（no ReID）已驗收值即可，grid search 沒有找到任何真正更優的新組合。

**ReID 開關驗收**（CLAUDE.md `spec reid-model`：與驗收基準比較不變差，且 IDSW 減少或 IDP 上升）：

| 類別 | ΔAssA | IDSW(基準→ReID) | IDP(基準→ReID) | 判定 |
|---|---|---|---|---|
| person | 0 | 0→0 | 0.9902→0.9902 | 無差異，不啟用 |
| car | -0.0001 | 3→3 | 0.9935→0.9935 | 無差異，不啟用 |
| scooter | **+0.0118** | 6→8 | 0.8997→**0.9480** | AssA 升、IDP 升 → **符合啟用條件** |
| small_truck | 0 | 0→0 | 1.0000→1.0000 | 無差異，不啟用 |
| bus | 0 | 0→0 | 1.0000→1.0000 | 無差異，不啟用 |
| truck | 0 | 0→0 | 1.0000→1.0000 | 無差異，不啟用 |

**結論：僅 scooter 符合 ReID 啟用條件**（使用原始基準門檻 0.9/0.4/0.8，非本次 grid search 找到的門檐），其餘 5 類維持 `use_reid: false`——這是本專案目前為止第一次有類別通過 ReID 驗收。scooter 的 IDSW 增加量不大（6→8，個位數差異）但仍建議視為初步結果，待未來有新資料或新 Re-ID 模型時重新確認。

至此 ByteTrack／BoTSORT(no ReID)／BoTSORT(+ReID) 三個 tracker 的全 6 類別 grid search 覆蓋度補齊工作已完成。ByteTrack（small_truck）與 BoTSORT(no ReID)（car／small_truck／truck）驗收通過的參數已寫回 `config/tracking/bytetrack.yaml`、`config/tracking/botsort_noreid.yaml`。

**`config/tracking/botsort_reid_timm.yaml` 已依其檔頭要求，用 `scripts/reid_train/acceptance_grid.py` 以它自己的 R1 參數（`center_dist_weight` 四輪車加重、`track_buffer` 15/15/15/15/10/10、`gmc_method: sparseOptFlow`、`frame_rate: auto`）正式重新驗收完畢（2026-11-10，`cctv_all8` 全 6 序列，tune/holdout 兩子集涵蓋全部）**：baseline（6 類全關）對逐類別開啟 ReID（`model: models/class_reid_weights/dinov2s_v3/model.pt`）共 14 次 `track_eval.py` 執行，結果**沒有任何類別**符合 spec `reid-model` 的啟用條件——person／scooter／small_truck／truck 的 IDSW、IDP、AssA、HOTA、CLEAR TP 全部與基準逐位元相同；car 的 AssA 只在 tune 子集微升、IDSW／IDP 未變，holdout 子集 AssA 反而下降；bus／truck 分別在 tune／holdout 子集沒有 GT。依「沒有類別符合時全部關閉」，`botsort_reid_timm.yaml` **維持現狀不需改動**（6 類 `use_reid: false`）。上方「僅 scooter 符合 ReID 啟用條件」的結論，驗證的是 `botsort_noreid.yaml` 結構的基準（`track_buffer` 60～90 幀），與 `botsort_reid_timm.yaml` 的 R1 基準（`track_buffer` 僅 10～15 幀）不同，`track_buffer` 愈短，軌跡愈少進入需要外觀比對的「遺失中」狀態，可能是兩者結論不同的主因；兩者的結論互不適用於對方的設定檔。完整方法與逐類別數據見 `Markdown/grid_search/botsort_reid_timm_acceptance/botsort_reid_timm_acceptance_report.md`。

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

| 順序 | 狀態 | 缺口 | 成本 | 效益 | 為什麼是這個排序 |
|---|---|---|---|---|---|
| 1 | ✅ 已完成（2026-11-09） | **隔離孤立模型權重**（`models/yolo-cls/`、`models/yolo-reid/` 除 `temp/`） | 近乎零——只是 `mv` 到 `temp/` 並記錄 `DO_NOT_USE.md`，不用新建任何資料或程式碼 | 小但立即——避免之後誤用零引用的權重、減少混淆 | 成本幾乎是 0，沒有理由不先做，屬於「順手清理」 |
| 2 | ✅ 已完成（2026-11-10） | **補齊 `tracking/` grid search 覆蓋**（person／car／bus／truck／small_truck × 3 個 tracker） | 低～中——`scripts/grid_search/` 三支腳本、`config/tracking/*.yaml`、`dataset/eval/cctv_all8_subsets/{tune,holdout}/` 全部已經存在，只差：(a) 把 `grid_search_botsort.py`／`grid_search_botsort_reid.py` 裡 car/bus/truck/small_truck 的固定單值網格改回真正的候選範圍，(b) 實際執行並留存 `--result-csv` | 高——tracking 是整條管線的核心，目前 5/6 類別的參數是直接沿用 refactor 前的舊設定，從未驗證是否最佳；影響 HOTA/AssA，也間接影響下游 collision／counting 的輸入品質 | 基礎設施（腳本＋資料＋設定）已經 100% 到位，缺的只是執行與一處小改動，是目前投入產出比最高的一項 |
| 3 | ✅ 已完成（2026-11-11，CI run 9761da6 綠燈） | **建立基本 CI 自動化**（至少跑 `pytest tests/test_smoke.py`） | 中——需要新建 CI workflow，但不需要新資料；要注意 golden 比對依賴掛載磁碟上的 `dataset/`／`models/`，不易搬進一般 CI runner，實際能自動化的範圍可能只到 smoke test | 中～高——跨所有模組的回歸防護網，目前完全靠人工記得要跑 pytest/golden | 單次建置成本中等，但長期每次改動都受益，優先度次於「資料和腳本已齊備只缺執行」的 tracking 缺口 |
| 4 | 🟡 spec 骨架已完成，門檻待定（2026-11-11，見下方「#4 執行說明」） | **將 `collision/`／homography 驗證正式化成驗收 spec** | 中～高——`validate_homography.py`／`check_homography_scale.py` 已存在，但要仿照 reid 的 `acceptance_grid.py` 設計正式門檻與報表；而且現有 check points 只覆蓋 `C000025`／`C000297` 兩支鏡頭，範圍本來就窄 | 中——只影響有 homography 的 4 支鏡頭（C000025/C000043/C000054/C000085）的衝突指標可信度 | 影響範圍比 tracking 窄，且要先有更多鏡頭的 check points 才能把 spec 做紮實，成本效益不如前兩項 |
| 5 | 🟡 進行中（pilot 資料已產生並匯入 CVAT，待人工覆核，2026-11-12，見下方「#5 執行說明」） | **`detection/` 建立微調訓練資料與流程** | 高——repo 內完全沒有偵測訓練用標註資料，必須從頭標註或另外取得資料集，還要新建訓練/驗收腳本 | 不確定——目前沒有證據顯示預訓練 YOLO（`yolo11x_dennis.pt`）在 `cctv_all8` 評估 GT 上的表現不足，微調的必要性未被證實 | 成本最高、效益卻還沒被證實有需求，應該等有具體證據顯示偵測品質不足後才投入 |
| 6 | ⬜ 未開始 | **`collision/` 建立衝突事件人工標註 GT** | 最高——需要人工在影片上標註真實的近距離接近/衝突事件，主觀且耗時，且目前只有 4 支鏡頭有 homography 可用公尺單位 | 低～中——CLAUDE.md 已明確定位這些指標「只能當技術訊號，不能當正式安全結論」，代表專案本身對形式化驗證的需求程度還不高 | 成本最高，而且專案自身的定位已經限縮了這份資料能帶來的效益上限，目前最不划算 |

**#3 執行說明（2026-11-11）**：新增 `.github/workflows/smoke-test.yml`，push／PR 時自動跑 `pytest tests/test_smoke.py`；golden 比對因依賴掛載磁碟上的 `dataset/`／`models/`，維持手動執行不變。沿用這次補強計畫的 Markdown 報告慣例、不重建已被 commit `68523d8` 移除的 `openspec/`。詳細做法與範圍見 `Markdown/ci/smoke_test_ci_setup.md`。

**#3 後續（2026-11-11）**：第一次真實 CI run（`42986579661`）跑了，而且**失敗**了——38 個測試全部死在同一個根因：`pyproject.toml` 的 `dependencies` 沒宣告 `timm`，但 `reid/training/model.py` 無條件 `import timm`，而這個檔案透過 `reid/encoders.py` → `tracking/class_aware_botsort.py` → `tracking/__init__.py` 被整個 `safetrajai.tracking` 套件無條件引入，所以只要 import `tracking`（幾乎任何進入點都會）就一定需要 `timm`，不管任何情境設定的 `use_reid` 是不是全部關閉。本機 conda env `track_eval` 手動裝過 `timm`，不在 `pyproject.toml` 宣告範圍內，才沒提早發現。已把 `timm` 加進 `pyproject.toml` 的核心 `dependencies`（不是放進 optional extra——因為它在 import 層級就是必要的，不是只有訓練 Re-ID 才需要），並用「比對專案原始碼實際 import 的套件 vs. `pyproject.toml` 宣告」的方式做了一次全面稽核，確認沒有其他類似的隱性依賴（`pillow`／`wandb` 雖然宣告但沒有被直接 import，屬於無害的既有宣告，不是缺口）。推送修正（commit `9761da6`）後的下一次 CI run 已確認轉綠（`status=completed, conclusion=success`），這項正式打勾。

**#4 執行說明（2026-11-11）**：新增 `scripts/homography/acceptance_check.py`，讀取 `validate_homography.py` 既有輸出的 `summary.csv`，套用誤差門檻判定每支鏡頭是否驗收通過，輸出 CSV＋Markdown 報表（做法仿照 `scripts/reid_train/acceptance_grid.py` 的 `summarize`）。刻意不改動 `validate_homography.py`／`validate_camera()` 本身的 `validated` 欄位語意，避免連動影響 golden 比對。誤差門檻（`ACCEPTANCE_RMSE_M`）刻意留 `None`（待定），符合使用者「先不定閾值」的決定；點數合格但門檻未定的鏡頭會回報 `threshold_tbd`，不會冒充通過或失敗。目前的首次報表：C000025／C000043 因點數不足回報 `too_few_checkpoints`，C000054（2.64m）／C000085（20.22m）回報 `threshold_tbd`。詳細設計與完整判定邏輯見 `Markdown/homography/acceptance_spec.md`。

**#4 執行說明（2026-11-11）**：部署選點器網頁服務（`scripts/picker/start_picker_service.sh`），使用者手動替 `conflict_analysis` 實際依賴的 4 支鏡頭中原本缺 check points 的 3 支（C000043／C000054／C000085）各補了一組驗證點。重跑 `validate_homography.py` 後發現：`validate_camera()` 目前「驗證」只檢查點數 ≥ 4（`MIN_CHECKPOINTS`），完全沒有誤差門檻，這個弱點被這次的真實資料直接印證——`C000054` 雖然點數足夠被標成 `validated=True`，但逐點比對後發現 `cctv_points`／`sat_points` 兩份清單的對應順序記反（前兩組與後兩組對調），官方算出的 RMSE 高達 749px（≈51.6m）；用來源影片首幀與衛星圖個別裁切、不依賴矩陣本身獨立核對地標後確認是順序問題，修正順序後重跑，真實 RMSE 降到 38.35px（≈2.64m），已直接修正 `config/homography_checkpoints/C000054.yaml`。`C000043` 的 4 組點裡，有 1 組（原第 3 組）經獨立核對後確認兩側根本是不同路口、不是同一地標，判定為誤選並移除，剩 3 組（RMSE 144.37px≈9.95m，因點數 < 4 被標記 `too_few_checkpoints`，不影響其餘 3 組仍是目前最準的驗證資料）。`C000085` 的 6 組點用窮舉配對（Hungarian 演算法）排除了「只是順序記反」的可能——最佳排列也只能把總誤差降低 12%，且只有其中 2 組點明顯改善，其餘 4 組不管怎麼配對都是 269～419px（≈18.5～29m）的大誤差，判定為真實的點位精度不足或矩陣在此路口精度較差，誠實記錄現狀，不強行調整。驗收門檻（可接受的誤差量，公尺或像素）本項仍未定案，待之後視需要再決定；詳細調查過程見 `Markdown/homography/checkpoint_validation_2026-11-11.md`。

**#5 執行說明（2026-11-12）**：依「未來計畫」第 1～7、8、9 點執行了 pilot 階段的取樣、偽標籤、人工覆核前的準備工作：

- **取樣與偽標籤**：新增 `scripts/detection_data/sample_and_pseudolabel.py`，對 `PILOT_FILES` 清單（9 支鏡頭各 1 個檔案，`C000054` 取時間跨度最大的頭尾 2 個檔案，共 10 個檔案）每 1.5 秒取樣一張畫面，用目前在用的偵測權重（`botsort_reid_timm.yaml` 的 `det_model_path`／`iou`／`imgsz`）推論產生偽標籤。輸出 3308 幀到 `dataset/detection/images/`／`labels/`（YOLO 格式，無框的背景幀寫 0-byte 空檔）與 `dataset/detection/splits/pilot_v1/manifest.csv`，仿照 `dataset/reid/crops` 的版本化慣例（對應第 8 點）。逐類別框數：person 1754、car 34882、scooter 24464、small_truck 2184、bus 437、truck 3824，全部超過 pilot 門檻（100，對應第 6 點）。
- **資料集登記**：`config/datasets.yaml` 新增獨立的 `detection_source` 項目（對應第 9 點），鏡頭與 `reid_sources` 重疊但不共用同一個 dataset id。
- **CVAT 部署**：依「手動任務做成網頁服務」慣例，在本機用 Docker Compose 自架 CVAT（`ignore/cvat_deploy/cvat/`，clone 自 `cvat-ai/cvat`，gitignored，不計入版控），對應第 7 點。過程中發現兩個實務問題：
  1. `/mnt/moris` 是 NFS 掛載，docker daemon 看不到同一份內容，無法用 bind mount 的「share 目錄」方式匯入影像；改用 CVAT 官方 HTTP 上傳 API（`cvat-sdk`，裝在獨立的 `ignore/cvat_deploy/cvat_client_venv/`，不動 conda env `track_eval`）。
  2. `cvat-sdk` 的上傳邏輯內部會對每個檔案路徑呼叫 `Path.resolve()`，若用 symlink 省空間，會被解析回原始真實路徑，導致原本刻意取的唯一檔名失效、暴露出 `dataset/detection/images/<seq>/<frame_idx>.jpg` 這種跨影片共用的 `frame_idx` 檔名（例如多支影片都有 `000000.jpg`），造成上傳時的資料庫唯一約束衝突。改用真正複製檔案（非 symlink）、檔名改成 `<seq_idx:02d>_<frame_idx:06d>.jpg`（`seq_idx` 為該 seq 在 `manifest.csv` 中首次出現的序號）才解決。
- **建立專案與匯入**：建立 CVAT 專案 `safetrajai detection pilot v1`，標籤依 `CLASS_NAMES`（`utils/modules/safetrajai/classes.py`）順序（person／car／scooter／small_truck／bus／truck），單一 task `pilot_v1` 依 manifest 順序上傳全部 3308 幀（對應使用者 2026-11-11 決定：單一 task、一次匯入全部）。先匯出該 task（空的）YOLO 1.1 格式標註以確認 CVAT 預期的 zip 結構（`obj.data`／`obj.names`／`train.txt`／`obj_train_data/<name>.txt`），再用既有偽標籤依同樣結構打包匯入。匯入後核對標註框總數為 67545，與逐類別框數加總（1754+34882+24464+2184+437+3824）完全一致，確認匯入無誤。
- **全片重新追蹤並改用 MOT 格式匯入**（2026-11-12 追加）：接著確認「CVAT 的 Automatic Annotation 能不能批次用演算法預先 track `pilot_v1` 的物件、輸出 MOT (Multiple Object Tracking) 格式」。查證 CVAT 原始碼（`cvat/apps/lambda_manager/views.py`）確認：批次自動標註只接受 `detector`／`reid` 兩種 nuclio function kind，`tracker` kind 只能用在互動式單物件追蹤（AI Tools 逐幀點選延伸），天生不支援批次多物件追蹤；而且 `pilot_v1` 的取樣幀之間間隔 21 個原始影格（每 1.5 秒取樣一次），超過目前追蹤設定的 `track_buffer`（10～15 幀），直接對取樣幀跑追蹤也不會有連續 ID。
  改用「對原始連續影片重新跑完整追蹤、再對齊回取樣幀」的做法：新增 `scripts/detection_data/track_pilot_videos.py`，沿用既有的 `VideoSequenceTracker`／`save_mot_format()`（跟主 pipeline stage1 同一套邏輯），對 `PILOT_FILES` 這 10 支原始影片各跑一次完整逐幀追蹤（共 57211 幀，約 35 分鐘），輸出到 `runs/cvat_pilot_v1_tracking/`。用一個小型測試 task 先實測驗證 CVAT「MOT 1.1」匯入格式的語意（`gt.txt` 的 frame 欄位是任務內 1-based 絕對幀序，不是檔名衍生值；track 之間的幀距超過 `frame_step` 時，匯入器會自動補插 `outside=True` 的消失標記），再把 `pilot_v1` manifest 裡每個取樣幀對齊回完整追蹤輸出的對應影格，track ID 依 `(seq, class, 原始 track_id)` 重新配成全域唯一（因為專案的 track_id 契約是「只在單一序列、單一類別內唯一」），清空並以 MOT 1.1 格式重新匯入 `pilot_v1` task，取代掉原本無 track ID 的 67545 個偵測框。
  匯入後核對：13450 條 track（58718 個真實關鍵幀框＋73632 個含自動補的 outside 標記的 shape 總數），抽查單一 track 的框位置逐幀平滑移動，確認 track ID 連續有意義。逐類別真實框數比原本純偵測版本略少（因為 BoTSort 的追蹤確認邏輯比單幀門檻過濾更嚴格）：person 1754→1384（-21%）、car 34882→31000（-11%）、scooter 24464→21068（-14%）、small_truck 2184→1341（-39%，降幅明顯最大）、bus 437→398（-9%）、truck 3824→3527（-8%），全部仍遠高於 pilot 門檻（100）。
  已知但使用者決定先不處理的瑕疵：(1) 36%（4874/13450）的 track 只有 1 個真實關鍵幀，跟舊版無 track ID 的偵測框資訊量相同；(2) 10 支影片在 CVAT 任務裡首尾相接，有 184 筆「track 結束」的 outside 標記因匯入器的自動補幀邏輯落在下一支不相關影片的第一幀上（不影響任何畫面顯示，純粹是資料裡多一個不顯示的收尾標記）；(3) small_truck 降幅特別大的原因還沒查，先讓人工覆核去發現問題再回頭查。舊版「純偵測、無 track ID」的標註已於匯入前用 `task.remove_annotations()` 清空，CVAT 裡不再保留；來源的偽標籤檔案（`dataset/detection/labels/`）本身沒有被動到，需要的話可以重建對照版本。
- **尚未完成**：CVAT 內的人工覆核本身還沒開始（已可透過瀏覽器開啟進行，現在看到的是帶 track ID 的 MOT 版本）；「未來計畫」第 10 點（切分版本、訓練、對 `cctv_all8` 跑 det_eval 驗收比較）也都還沒動工。CVAT 容器與相關 staging／追蹤輸出檔案目前只存在於本機、未提交版控（`ignore/`／`runs/` 慣例）。

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
