# NEED.md

待辦與未滿足需求清單。以下項目是整理程式碼時發現的落差，非正式規劃，請專案負責人確認優先順序與是否要處理。

## Re-ID 訓練資料 / 模型 / 訓練方式重新設計（2026-07-24，進行中）
- [x] `track_eval.py` 新增 `save_reid_crops` 功能（`config/*.yaml` 的 `save params.save_reid_crops`，預設關閉，`config/botsort_save_reid_crops.yaml` 開啟）：`VideoSequenceTracker.run()` 逐幀把追蹤到的物件裁切存到 `{reid_crop_dir}/{class}/{影片檔名}_track{track_id}/{frame:06d}.jpg`，已用真實影片驗證輸出結構正確、預設關閉時完全不影響既有 pipeline（HOTA/AssA 數值與開關前逐位元組相同，且確認未寫入任何檔案到 `dataset/reid_dataset/`）；`config/botsort_save_reid_crops.yaml` 也已實際跑過一次，`dataset/reid_dataset/` 底下新增 6 個類別資料夾、共 3830 張裁切影像，舊的 `train/` 扁平資料完全未受影響。用法見 `USAGE.md`。
- [x] 2026-07-24：**設定檔再次整併，統一放進 `config/`**——`module/tracker/track_config_*.yaml`（剩餘的 `BotSort_0413`、`BotSort_noreid_0413`）與根目錄 `config.yaml` 全數移除／併入，改成 `config/` 底下固定 6 個情境設定檔（`bytetrack.yaml`、`botsort_noreid.yaml`、`botsort_save_reid_crops.yaml`、`botsort_reid.yaml`、`grid_search_bytetrack.yaml`、`grid_search_botsort.yaml`），每個都是完整獨立可用的 config，不再需要 `--track-config` 額外指定。新增 `stages.run_stage1`/`run_stage2` 開關（`TrackingEvalPipeline.run()` 據此決定要不要跑該階段），`botsort_save_reid_crops.yaml` 用這個機制只跑 stage1。`track_config.yaml`（base 版）與 `track_config_new.yaml`（另一個 ReID 變體，`track_buffer`/調參值跟現有 6 個情境檔不完全一樣）已歸檔到 `archive/track_config_variants/`。已用完整 pipeline（`config/bytetrack.yaml`，HOTA/AssA 逐位元組比對前後一致）、`config/botsort_save_reid_crops.yaml`（確認 stage2 正確跳過、crops 正確寫入）、3 支 grid search `--dry-run`（各自新預設 config）驗證。
- [x] 2026-07-24：**新增 `dataset/reid/build_reid_split.py`，把 `crop_dataset/` 依身分切 train/val**：`--mode category`（6 類別各自切，輸出 `dataset/reid/reid_category_dataset/{class}/{train,val}/{identity}/`）與 `--mode flat`（6 類別身分混合切一次，加類別前綴避免撞名，輸出 `dataset/reid/reid_dataset/{train,val}/{class}_{identity}/`）。兩種模式跑完都會產生 `numbers.txt`（每身分張數）+ `res.txt`（`tree -h` 目錄結構）。已用實際資料驗證：209 身分（bus 1／car 107／person 7／scooter 79／small_truck 11／truck 4）、3830 張影像，`--val-ratio 0.2 --seed 42` 下 bus 唯一身分正確分進 train（val=0）不報錯。同時把 `dataset/crop_dataset`、`dataset/reid_category_dataset`、`dataset/reid_dataset` 統一搬進 `dataset/reid/` 底下（分別變成 `dataset/reid/crop_dataset`、`dataset/reid/reid_category_dataset`、`dataset/reid/reid_dataset`），`config/*.yaml` 的 `reid_crop_dir` 一併改指向 `./dataset/reid/crop_dataset`（跟 `save_reid_crops` 蒐集的 class-organized 原始裁切結構相符，解決先前 `reid_crop_dir` 預設值與切分輸出目錄同名的潛在撞名問題）。詳見 `dataset/reid/README.md`。
- [x] 2026-07-24：**三大流程端到端驗證 + 修正 `cfg.base_dir` bug + `video/` 補資料**：完整跑過 `save_reid_crops` 蒐集、`module/reid/training/train.py` 訓練 smoke test（loss 下降、`load_reid_model()`/`encoders.py::ReID` 兩條路徑皆能還原）、`grid_search_botsort_reid.py` 真實試驗（非 `--dry-run`）。驗證 grid search 時發現 `module/grid_search/{grid_search_bytetrack,grid_search_botsort,grid_search_botsort_reid}.py` 共 6 處引用不存在的 `cfg.base_dir`（`AppConfig` 實際欄位是 `cfg.OUTPUT_ROOT`，`--dry-run` 不會走到這段程式碼才一直沒被抓到），已全部修正。另新增 `config/botsort_save_reid_crops_video.yaml` + `video/` 底下 5 支不同路口原始影片（各取前 60 秒放進 `dataset/video_seqs/`，`det_corner` 關閉），補足 `bus`/`person`/`truck`/`small_truck` 身分數過少的問題；`dataset/reid/crop_dataset/` 最終 1185 身分／33399 張，`bus` 仍只有 5 身分（兩批來源影片都幾乎沒拍到公車）。詳見 `VERIFY.md`、`dataset/reid/README.md`。
- [ ] **舊資料 `dataset/reid/crop_dataset/train/{train,val}/{identity}/` 與新資料 `{class}/{影片}_track{id}/` 是兩套不同結構**：舊資料是扁平的（不分類別，446 個身分資料夾，各自命名為純數字如 `000351`），新的 crop 蒐集功能改成先分類別再分身分。需要決定：(a) 舊資料要不要轉換/歸類到新結構下，(b) 新舊資料是否共用同一套 Dataset 讀取邏輯，(c) 訓練資料的 train/val 切分方式（舊資料是資料夾層級切好的，新蒐集的資料現在可以用 `build_reid_split.py` 切）。
- [ ] **`bus` 身分數過少（目前 5 個，遠低於其他類別）**：現有兩批來源影片（中華三路_民生二路 6 支序列 + `video/` 5 支不同路口影片）路口車流裡都幾乎沒拍到公車。需要另外找有公車經過的路口影片，用 `config/botsort_save_reid_crops.yaml` 或 `botsort_save_reid_crops_video.yaml`（換 `SOURCE_VIDEO_DIR`／`video/` 來源）補蒐集，否則 `bus` 類別的 Re-ID 訓練資料量會嚴重不足。
- [ ] **Re-ID 資料讀取、model 架構、訓練方式重新設計中**：原本的 `module/reid/training/train_utils.py`（已移除，見上方「移除 reid_models」相關項目）用 `torchvision.datasets.ImageFolder` 讀取扁平身分資料夾、`ReIDHookManager`（`module/reid/yolo_reid_trainer.py`）掛 YOLO 倒數第二層當 encoder；新設計要以 `dataset/reid/crop_dataset/{class}/{id}/` 這個分類別結構為準，讀取邏輯、model 架構（是否延用 YOLO-as-encoder，或改用 `utils/model.py` 的 `ReIDNet`，或兩者分開依類別訓練）、訓練迴圈都需要重新規劃，尚未定案。
- [ ] **2026-09-03（五模組＋track_eval.py 重構 brainstorm 過程中補充，待後續 module/reid/ 專屬 brainstorm 處理）**：(a) Detection 產生偵測結果時要能選擇是否同步產生對應的 Re-ID 訓練資料（裁切）——現有 `save_reid_crops` 開關（`config/*.yaml` 的 `save params.save_reid_crops`）是這個需求的既有實作，`module/detection/` 或 `track_eval.py` 重構時需保留此可選性，不可假設 Re-ID 裁切一定要跟偵測綁死；(b) 需要規劃 Re-ID 資料的過濾機制（哪些裁切品質不足、遮蔽過多、或不具訓練價值的樣本要濾掉，不納入訓練資料）——目前 `ReidCropCollector`（見 `2026-09-03-track-eval-pipeline-refactor-design.md`）只做多邊形區域過濾，沒有品質過濾機制，細節待 module/reid/ 專屬 brainstorm 定案。

## 待確認 / 待清理
- [x] 2026-07-23：專案已依 Detection / Tracker / Re-ID 三大模組重組，`trackers/` → `module/tracker/`，`utils/track_eval/` + `utils/eval_utils.py` → `evaluation/`。詳見 `CLAUDE.md`「架構」章節與 `Project.md`。
- [x] 2026-07-23：`utils/train_utils.py` 已搬到 `module/reid/training/train_utils.py`（歸檔性質，仍無腳本引用，孤兒程式碼原樣搬移）。`utils/model.py` **維持原路徑不動**——搬移過程中發現 `models/class_reid_weights/{person,scooter}/best_full_model.pt` 是用 `torch.save()` 整物件序列化，pickle 內容寫死 `utils.model.xxx` 模組路徑，且會被 `module/tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支在啟用對應設定時載入，並非真正孤兒；不可搬移，否則載入時會 `ModuleNotFoundError`。（2026-07-24 更新：`module/reid/training/train_utils.py` 本身已移除，ReID 訓練資料與訓練方式正在重新設計，見下方新項目。）
- [x] 2026-07-23：`trackers/trackeval/`（重複副本）已歸檔到 `archive/trackers_trackeval_duplicate/`，根目錄 `trackeval/` 是實際使用的版本。
- [x] 2026-07-23：`past/`、`trackers/past_conf/`、`trackers/past_py_conf/` 已歸檔到 `archive/`（`past_track_eval_orign.py`、`trackers_past_conf/`、`trackers_past_py_conf/`），保留供對照。
- [x] 2026-07-24：**移除 `reid_models/`、`SOTA_ReID`、`models/reid_model_weights/`**——使用者確認只需要保留兩條 Re-ID 路徑：`ReID`（YOLO 本身當 encoder，`models/yolo-reid/*.pt`）與 `Custom_ReID`（自己訓練的模型，`models/class_reid_weights/*/best_full_model.pt`），vendored 的 `SOTA_ReID`（FastReID/TransReID）不需要。移除範圍：`reid_models/` 整包（含 208 個 git 追蹤檔案 + 52 個因 `.gitignore` 的 `models/` 規則誤判而從未被版控的檔案）、`models/reid_model_weights/`（1.9GB）、`module/reid/encoders.py` 的 `SOTA_ReID` class 與死掉的 `load_reid_model` import、`class_aware_botsort.py` 的 `reid_model_weights` 分支與現在也用不到的 `import torch`、`grid_search_botsort_reid.py` 的 `PARAM_GRIDS` 裡對應的 3 個權重路徑。已用直接建構 `ClassAwareBOTSORT`（模擬 `person`→`ReID`、`scooter`→`Custom_ReID` 兩條保留路徑）與完整 pipeline 煙霧測試驗證，HOTA/AssA 數值與移除前逐位元組相同。
- [x] 2026-07-24：**發現 `.gitignore` 的 `models/` 規則有誤判**：因為沒加前導 `/`，這條規則同時吃掉了 `reid_models/modeling/third_party_models/{ABDNet,DeepPersonReid}/models/` 這兩個巧合同名的巢狀子目錄，導致裡面 52 個檔案從一開始就沒被版控（隨這次 `reid_models/` 整包移除一併清掉，若之後又新增巧合叫 `models/` 的子目錄要注意這個規則的副作用）。
- [x] 2026-07-24：**發現多個現有環境缺依賴**——嘗試在 `safetrajai` conda 環境載入 `SOTA_ReID` 權重時依序缺 `prettytable`、`easydict`、`kornia`、`fvcore`；嘗試 `Custom_ReID` 權重時缺 `scikit-learn`。皆已在本次驗證中臨時安裝以完成追蹤，但這些套件都不在 `requirement.txt` 裡。`SOTA_ReID` 隨本次移除已不相關；`scikit-learn` 是 `Custom_ReID` 這條保留路徑（透過 `utils/model.py`）的真實依賴，應該補進 `requirement.txt`。
- [ ] `archive/` 底下的舊碼是否還需要保留，或可進一步刪除，留給專案負責人決定。
- [x] 2026-07-23：**修正搬移遺留的 bug**——`module/grid_search/*.py`（三支）搬進 `module/grid_search/` 後，內部 `PROJECT_ROOT = Path(__file__).resolve().parent` 沒有跟著調整巢狀深度，導致直接執行會 `ModuleNotFoundError: No module named 'tracker'`（原本在根目錄時 `.parent` 剛好等於 repo root，搬進兩層深的子目錄後就不對了）。已改成 `.resolve().parents[2]` 並實際跑 `--dry-run` 驗證三支都正常。這類「靜態檢查通不出來、只有真的執行腳本才會發現」的搬移遺留問題，之後搬檔案要記得對每支可獨立執行的腳本補跑一次最小驗證（不只 py_compile / import 靜態掃描）。
- [x] 2026-07-23：**config.yaml 與 track_config 合併**——`module/tracker/track_config_ByteTrack_0411.yaml`（`track_eval.py` 實際使用的預設版）內容合併進 `config.yaml` 新的 `class_tracking:` 章節，原檔已刪除。`--track-config` 三支 CLI（`track_eval.py`、`module/grid_search/*.py` 三支）預設改成 `None`，不指定時會用 `args.track_config or args.config` 落到 `config.yaml` 本身。`module/tracker/track_config.yaml`（base 版，含 ReID/track_buffer 欄位）與其餘 3 個實驗變體（`BotSort_0413`、`BotSort_noreid_0413`、`new`）維持獨立檔案，用 `--track-config` 指定時整份覆寫 `class_tracking:` 章節。已用 diff 驗證合併前後 6 支序列的 HOTA/DetA/AssA 數值逐位元組相同，3 支 grid search `--dry-run` 也驗證正常（grid search 自己的 `PARAM_GRIDS` 每次試驗都會覆寫 ReID 相關欄位，不受精簡版預設影響）。

- [x] 2026-09-03（決定，尚未實作）：**`track_buffer` 單位在 BoTSORT 與 ByteTrack 兩個 tracker 不一致**（撰寫 `module/tracker/class_aware_botsort_flow.md` / `class_aware_bytetrack_flow.md` 時發現）：`class_aware_botsort.py` 把 `track_buffer` 當作「秒」（`cls_max_time_lost = track_buffer × frame_rate`），`class_aware_bytetrack.py` 仍當作「30fps 基準幀數」（`cls_max_time_lost = (frame_rate/30) × track_buffer`）。同一個 `track_buffer: 2.0` 在 30fps 下，BoTSORT 算出 60 幀存活時間，ByteTrack 只有 2 幀——兩者相差 30 倍。`module/tracker/track_config.yaml`（base 版）目前每個類別都設 `track_buffer: 2.0~3.0`，這組值原本應該是設計給 BoTSORT 用的；若拿去跑 ByteTrack，lost track 幾乎立即被清除。**使用者已決定：統一以秒為基數**（ByteTrack 改用 BoTSORT 現行公式），這是行為變更，`config/bytetrack.yaml` 現有 `track_buffer` 數值在新語意下的實際存活時間會拉長約 30 倍，需要用改動前後的 HOTA/AssA/IDSW 記錄實際影響。設計與驗證計畫見 `2026-09-03-tracker-dedup-design.md`（與 `module/tracker/` 去重複一起實作），尚未實作。

## 流程缺口
- [x] 2026-07-23：已導入 git 版本控制（見 commit `7061f97` 起）。實驗版本仍靠 `track_config_*.yaml` 檔名區分，尚未改用 git branch/tag 管理，可再評估。
- [ ] 沒有自動化測試，也沒有「基準指標」可供迴歸比對（見 `TEST.md` 待補充）；評估是否要建立至少一組基準 HOTA/AssA 數值。
- [ ] 沒有 `requirements.txt` / `pyproject.toml`，環境重建靠人工記憶依賴套件；建議補上。
- [ ] **Detection 模組缺訓練腳本**：`module/detection/` 目前只有推論用的 `Detector`（`module/detection/detector.py`），沒有 YOLO 訓練腳本——這本來就不在專案原始範圍內（`models/yolo/*.pt` 是外部訓練產物）。待日後有專門的 YOLO 訓練腳本，再一併整理進 `module/detection/`（此項由使用者於 2026-07-23 明確要求先記錄、暫不生成佔位腳本）。
- [ ] Tracker / Re-ID 模組同樣沒有「測試腳本」，目前是用 `module/grid_search/*.py`（含 `grid_search_botsort_reid.py`）兼作超參數搜尋與驗證，沒有獨立的單元測試或迴歸測試腳本；是否要另外補上，待評估。
- [x] 2026-07-23：`grid_search_botsort_reid.py` 原本放在 `module/reid/`，經確認本質是 tracker 超參數搜尋（不 import 任何 `reid.*` 模組），已改放到 `module/grid_search/grid_search_botsort_reid.py`，與另外兩支 grid search 腳本放一起。

## 待驗證（見 VERIFY.md 對應段落）
- [ ] ReID on/off 的量化數據補齊（目前只有程式碼註解結論，無 Excel/CSV 佐證）。
- [ ] FPS/解析度比較尚未整理出整體結論摘要（原始數據 `fps_compare/` 已於 2026-07-23 移除，如需重新整理需從 git 歷史 commit `7061f97` 取回或重新跑一次比較）。
- [ ] `det_corner` 偵測區域多邊形設定對指標的影響。
- [ ] `fuse_score` 開關（目前全域關閉）對各類別的影響。
- [ ] `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據。

## 目錄整理建議（2026-07-23，待確認後執行，本次尚未搬移任何檔案）
現況全貌見 `Project.md`「資料與輸出目錄」。以下依風險排序，`git mv` 前一律先確認：

| 項目 | 現況 | 建議目標 | 風險 / 需同步更新 |
|---|---|---|---|
| ~~`botsort_noreid/`~~ | ~~根目錄，GMC 比較報告~~ | **已於 2026-07-23 移除並取消版控**（使用者確認不需保留），結論保留在 `VERIFY.md`，原始數據可從 git 歷史（commit `7061f97`）取回 | 已完成 |
| ~~`fps_compare/`~~ | ~~根目錄，FPS 比較報告~~ | **已於 2026-07-23 移除並取消版控**（使用者確認不需保留），結論（待補充部分）保留在 `VERIFY.md`，原始數據可從 git 歷史（commit `7061f97`）取回 | 已完成 |
| ~~`html/`（index1x3.html、index2x3.html）~~ | ~~根目錄，同步播放比較視覺化工具~~ | **已於 2026-07-23 移除並取消版控**（使用者確認不需保留），可從 git 歷史（commit `7061f97`）取回 | 已完成 |
| `runs/track_eval(0411_ByteTrack)/` 等歷史封存實驗＋對應 `.zip` | 混在 `runs/` 底下，與「目前即時輸出」`runs/track_eval/` 放在一起 | 移到 `archive/experiment_runs/` 或 `output/archive/`，讓 `runs/` 只保留 `config/*.yaml` 的 `OUTPUT_ROOT` 即時輸出 | 中。總量約 1.5GB，實體搬移耗時；已 gitignore 不影響版控，但搬移後若有腳本用相對路徑讀取這些歷史結果需另外確認（目前掃描沒有程式碼引用） |
| `runs/detect/`（train、val1-9…） | ultralytics 自動產生的暫存輸出 | 不建議搬，可視需要清空 | 低。重新執行 `evaluation/pipeline.py` 的 `det_model.val()` 會自動重建 |
| `dataset/` 根目錄下 4 支散落 mp4 | 與兩個結構化資料集資料夾平行放置，無程式碼引用 | 若確認非現行流程需要，移到 `dataset/samples/` 或清除；若不確定用途則先不動 | 中。屬於 `datasets/raw` 唯讀範疇，需你確認這 4 支影片是否還在使用 |
| ~~`runs.zip`（520MB）~~ | ~~根目錄，與展開後的 `runs/` 內容幾乎重複~~ | **已於 2026-07-23 刪除**（未版控，使用者確認不需保留，無法從 git 復原） | 已完成 |
| `dataset.zip`（465MB） | 根目錄，內容與展開後的 `dataset/` 幾乎重複 | 確認是否為必要備份；若是建議移出工作目錄（如另外的備份位置），不建議留在 repo 工作目錄 | 高，需你決定。體積大且無法判斷是否為唯一備份來源，**不會主動刪除** |
| `wandb/`（74MB） | 根目錄，舊 ReID 訓練 run 的 log | 不建議手動搬移 | wandb 套件依執行時 cwd 自動產生位置，手動搬了下次訓練還是會在原處重建；要規範化須在訓練腳本設定 `WANDB_DIR`，不是檔案搬移能解決的 |
| `models/` | 已依 yolo/yolo-cls/yolo-reid/class_reid_weights 分類（`reid_model_weights/` 已於 2026-07-24 移除） | 現況已算整齊，可選配補一份 `registry.yaml` 記錄各權重對應的訓練來源/實驗 | 低，錦上添花，非必要 |
| `requirement.txt`（未版控） | 根目錄，pip install 環境建置腳本 | 位置正確（根目錄慣例），內容其實是 shell 腳本非標準 requirements 格式，可考慮改名如 `setup_env.sh` 更準確；是否納入版控由你決定 | 低 |
| `note.txt`（未版控） | 根目錄，個人便條 | 不動，不在整理範圍 | — |

## 待補充
> 此檔案是根據現有程式碼與文件整理出的「已知落差」，不包含尚未提出的新功能需求。若有新的功能/優化需求，請直接加在下方，並標註提出人與日期。
