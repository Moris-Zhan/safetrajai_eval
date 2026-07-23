# NEED.md

待辦與未滿足需求清單。以下項目是整理程式碼時發現的落差，非正式規劃，請專案負責人確認優先順序與是否要處理。

## 待確認 / 待清理
- [x] 2026-07-23：專案已依 Detection / Tracker / Re-ID 三大模組重組，`trackers/` → `module/tracker/`，`utils/track_eval/` + `utils/eval_utils.py` → `evaluation/`。詳見 `CLAUDE.md`「架構」章節與 `Project.md`。
- [x] 2026-07-23：`utils/train_utils.py` 已搬到 `module/reid/training/train_utils.py`（歸檔性質，仍無腳本引用，孤兒程式碼原樣搬移）。`utils/model.py` **維持原路徑不動**——搬移過程中發現 `models/class_reid_weights/{person,scooter}/best_full_model.pt` 是用 `torch.save()` 整物件序列化，pickle 內容寫死 `utils.model.xxx` 模組路徑，且會被 `module/tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支在啟用對應設定時載入，並非真正孤兒；不可搬移，否則載入時會 `ModuleNotFoundError`。
- [x] 2026-07-23：`trackers/trackeval/`（重複副本）已歸檔到 `archive/trackers_trackeval_duplicate/`，根目錄 `trackeval/` 是實際使用的版本。
- [x] 2026-07-23：`past/`、`trackers/past_conf/`、`trackers/past_py_conf/` 已歸檔到 `archive/`（`past_track_eval_orign.py`、`trackers_past_conf/`、`trackers_past_py_conf/`），保留供對照。
- [ ] `reid_models/`（vendored fast-reid 風格套件）同樣因 pickle 相容性**固定留在根目錄**，未搬進 `module/reid/`；若未來要重組，需先用舊路徑載入權重再用新路徑重新存檔，而非直接搬檔案。
- [ ] `archive/` 底下的舊碼是否還需要保留，或可進一步刪除，留給專案負責人決定。
- [x] 2026-07-23：**修正搬移遺留的 bug**——`module/grid_search/*.py`（三支）搬進 `module/grid_search/` 後，內部 `PROJECT_ROOT = Path(__file__).resolve().parent` 沒有跟著調整巢狀深度，導致直接執行會 `ModuleNotFoundError: No module named 'tracker'`（原本在根目錄時 `.parent` 剛好等於 repo root，搬進兩層深的子目錄後就不對了）。已改成 `.resolve().parents[2]` 並實際跑 `--dry-run` 驗證三支都正常。這類「靜態檢查通不出來、只有真的執行腳本才會發現」的搬移遺留問題，之後搬檔案要記得對每支可獨立執行的腳本補跑一次最小驗證（不只 py_compile / import 靜態掃描）。
- [x] 2026-07-23：**config.yaml 與 track_config 合併**——`module/tracker/track_config_ByteTrack_0411.yaml`（`track_eval.py` 實際使用的預設版）內容合併進 `config.yaml` 新的 `class_tracking:` 章節，原檔已刪除。`--track-config` 三支 CLI（`track_eval.py`、`module/grid_search/*.py` 三支）預設改成 `None`，不指定時會用 `args.track_config or args.config` 落到 `config.yaml` 本身。`module/tracker/track_config.yaml`（base 版，含 ReID/track_buffer 欄位）與其餘 3 個實驗變體（`BotSort_0413`、`BotSort_noreid_0413`、`new`）維持獨立檔案，用 `--track-config` 指定時整份覆寫 `class_tracking:` 章節。已用 diff 驗證合併前後 6 支序列的 HOTA/DetA/AssA 數值逐位元組相同，3 支 grid search `--dry-run` 也驗證正常（grid search 自己的 `PARAM_GRIDS` 每次試驗都會覆寫 ReID 相關欄位，不受精簡版預設影響）。

- [ ] **`track_buffer` 單位在 BoTSORT 與 ByteTrack 兩個 tracker 不一致**（撰寫 `module/tracker/class_aware_botsort_flow.md` / `class_aware_bytetrack_flow.md` 時發現）：`class_aware_botsort.py` 把 `track_buffer` 當作「秒」（`cls_max_time_lost = track_buffer × frame_rate`），`class_aware_bytetrack.py` 仍當作「30fps 基準幀數」（`cls_max_time_lost = (frame_rate/30) × track_buffer`）。同一個 `track_buffer: 2.0` 在 30fps 下，BoTSORT 算出 60 幀存活時間，ByteTrack 只有 2 幀——兩者相差 30 倍。`module/tracker/track_config.yaml`（base 版）目前每個類別都設 `track_buffer: 2.0~3.0`，這組值原本應該是設計給 BoTSORT 用的；若拿去跑 ByteTrack，lost track 幾乎立即被清除。這不是本次重組造成的問題，是既有程式碼的既有落差，建議專案負責人確認：(a) 兩個 tracker 的 `track_buffer` 語意本來就要分開設計，或 (b) 其中一個是漏改的 bug，需要統一。詳見兩份 flow 文件的「注意事項」章節。

## 流程缺口
- [x] 2026-07-23：已導入 git 版本控制（見 commit `7061f97` 起）。實驗版本仍靠 `track_config_*.yaml` 檔名區分，尚未改用 git branch/tag 管理，可再評估。
- [ ] 沒有自動化測試，也沒有「基準指標」可供迴歸比對（見 `TEST.md` 待補充）；評估是否要建立至少一組基準 HOTA/AssA 數值。
- [ ] 沒有 `requirements.txt` / `pyproject.toml`，環境重建靠人工記憶依賴套件；建議補上。
- [ ] **Detection 模組缺訓練腳本**：`module/detection/` 目前只有推論用的 `Detector`（`module/detection/detector.py`），沒有 YOLO 訓練腳本——這本來就不在專案原始範圍內（`models/yolo/*.pt` 是外部訓練產物）。待日後有專門的 YOLO 訓練腳本，再一併整理進 `module/detection/`（此項由使用者於 2026-07-23 明確要求先記錄、暫不生成佔位腳本）。
- [ ] Tracker / Re-ID 模組同樣沒有「測試腳本」，目前是用 `module/grid_search/*.py`（含 `grid_search_botsort_reid.py`）兼作超參數搜尋與驗證，沒有獨立的單元測試或迴歸測試腳本；是否要另外補上，待評估。
- [x] 2026-07-23：`grid_search_botsort_reid.py` 原本放在 `module/reid/`，經確認本質是 tracker 超參數搜尋（不 import 任何 `reid.*` 模組），已改放到 `module/grid_search/grid_search_botsort_reid.py`，與另外兩支 grid search 腳本放一起。

## 待驗證（見 VERIFY.md 對應段落）
- [ ] ReID on/off 的量化數據補齊（目前只有程式碼註解結論，無 Excel/CSV 佐證）。
- [ ] `fps_compare/` 現有數據尚未整理出整體結論摘要。
- [ ] `det_corner` 偵測區域多邊形設定對指標的影響。
- [ ] `fuse_score` 開關（目前全域關閉）對各類別的影響。
- [ ] `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據。

## 目錄整理建議（2026-07-23，待確認後執行，本次尚未搬移任何檔案）
現況全貌見 `Project.md`「資料與輸出目錄」。以下依風險排序，`git mv` 前一律先確認：

| 項目 | 現況 | 建議目標 | 風險 / 需同步更新 |
|---|---|---|---|
| ~~`botsort_noreid/`~~ | ~~根目錄，GMC 比較報告~~ | **已於 2026-07-23 移除並取消版控**（使用者確認不需保留），結論保留在 `VERIFY.md`，原始數據可從 git 歷史（commit `7061f97`）取回 | 已完成 |
| `fps_compare/` | 根目錄，FPS 比較報告（已版控，104KB） | `output/experiments/fps_comparison/` | 低。僅 `VERIFY.md` FPS 段落的路徑要同步改 |
| `html/`（index1x3.html、index2x3.html） | 根目錄，同步播放比較視覺化工具（已版控，16KB），**尚未搬移** | 併入 `visualize/`（`visualize_gt_on_video.py`／`visualize_gt_on_class_video.py` 已於 2026-07-23 搬進 `visualize/`，`.html` 兩支還留在根目錄 `html/`，尚待確認是否一併搬入） | 低。無程式碼引用 |
| `runs/track_eval(0411_ByteTrack)/` 等歷史封存實驗＋對應 `.zip` | 混在 `runs/` 底下，與「目前即時輸出」`runs/track_eval/` 放在一起 | 移到 `archive/experiment_runs/` 或 `output/archive/`，讓 `runs/` 只保留 `config.yaml` 的 `OUTPUT_ROOT` 即時輸出 | 中。總量約 1.5GB，實體搬移耗時；已 gitignore 不影響版控，但搬移後若有腳本用相對路徑讀取這些歷史結果需另外確認（目前掃描沒有程式碼引用） |
| `runs/detect/`（train、val1-9…） | ultralytics 自動產生的暫存輸出 | 不建議搬，可視需要清空 | 低。重新執行 `evaluation/pipeline.py` 的 `det_model.val()` 會自動重建 |
| `dataset/` 根目錄下 4 支散落 mp4 | 與兩個結構化資料集資料夾平行放置，無程式碼引用 | 若確認非現行流程需要，移到 `dataset/samples/` 或清除；若不確定用途則先不動 | 中。屬於 `datasets/raw` 唯讀範疇，需你確認這 4 支影片是否還在使用 |
| `dataset.zip`（465MB）、`runs.zip`（520MB） | 根目錄，內容與展開後的 `dataset/`／`runs/` 幾乎重複 | 確認是否為必要備份；若是建議移出工作目錄（如另外的備份位置），不建議留在 repo 工作目錄 | 高，需你決定。體積大且無法判斷是否為唯一備份來源，**不會主動刪除** |
| `wandb/`（74MB） | 根目錄，舊 ReID 訓練 run 的 log | 不建議手動搬移 | wandb 套件依執行時 cwd 自動產生位置，手動搬了下次訓練還是會在原處重建；要規範化須在訓練腳本設定 `WANDB_DIR`，不是檔案搬移能解決的 |
| `models/` | 已依 yolo/yolo-cls/yolo-reid/reid_model_weights/class_reid_weights 分類 | 現況已算整齊，可選配補一份 `registry.yaml` 記錄各權重對應的訓練來源/實驗 | 低，錦上添花，非必要 |
| `requirement.txt`（未版控） | 根目錄，pip install 環境建置腳本 | 位置正確（根目錄慣例），內容其實是 shell 腳本非標準 requirements 格式，可考慮改名如 `setup_env.sh` 更準確；是否納入版控由你決定 | 低 |
| `note.txt`（未版控） | 根目錄，個人便條 | 不動，不在整理範圍 | — |

## 待補充
> 此檔案是根據現有程式碼與文件整理出的「已知落差」，不包含尚未提出的新功能需求。若有新的功能/優化需求，請直接加在下方，並標註提出人與日期。
