# VERIFY.md

紀錄已經做過的實驗驗證與結論，避免重複測試或誤改已驗證過的設定。每筆結論都附上資料來源，之後有新結論請照同樣格式補充。

## ReID 開關（person vs. scooter）
- **結論**：開啟 ReID 對 person 的追蹤有幫助，但會損害 scooter 追蹤並增加 IDSW（ID switch）。
- **目前設定**：`module/tracker/__init__.py` 的 `DEFAULT_TRACKER_CONFIG['with_reid']` 預設 `False`。
- **來源**：`module/tracker/__init__.py` 程式碼註解；細節推論見 `module/tracker/HANDOFF.md` 需求 4。
- **待補充**：實際數據（各類別 HOTA/IDSW 變化量）目前只存在於程式碼註解描述，沒有對應的 Excel/CSV 可查— 需要的話應重新用 `module/grid_search/grid_search_botsort_reid.py` 針對 person / scooter 各跑一次 ReID on/off 對照並存檔。

## GMC（Generalized Motion Compensation）方法比較 — BoTSORT + NoReID
- **結論**：以全部 6 類別 × 6 序列的 `mean_AssA` 平均比較，`sift` 排名最佳，`ecc`、`none`、`sparseOptFlow`、`orb` 依序在後，但彼此差距極小（第三位小數才有差異），**實務上五種 GMC 方法表現接近，非決定性差異**。
- **來源**：原始報告 `botsort_noreid/`（逐類別、逐序列明細 `.md`、對應 `.xlsx`、`gmc_*.txt` 各方法原始結果）已於 2026-07-23 移除，結論已確認保留於此，如需追溯原始數據可從 git 歷史（commit `7061f97`）取回。
- **目前設定**：`config.yaml` 的 `tracking_params.gmc_method: ecc`。

## FPS / 推論解析度對 ByteTrack 準確度的影響
- **來源**：`fps_compare/fps_bytetrack_comparison_by_class_and_sequence.md`、`fps_bytetrack_comparison_by_sequence.md`（imgsz=640，比較 10/20/30 fps）、對應 `.xlsx`。
- **結論**：待補充 — 檔案內已有逐序列逐類別數據，尚未整理出整體結論摘要。

## 追蹤參數查表機制（class-aware matching）
- **結論**：確認「逐類別分組匹配」相較原版 Ultralytics 全域統一匹配，可解決跨類別誤配問題（如 person 配到 scooter track）。此為架構性修改，已落地於 `module/tracker/class_aware_botsort.py` / `class_aware_bytetrack.py`。
- **來源**：`module/tracker/HANDOFF.md`（完整原理與程式碼對照表）。
- **待補充**：沒有找到「改前 vs 改後」的量化比較數據（例如跨類別誤配次數），若需要作為佐證應另外統計。

## 模組重組第一階段（Detection / Tracker / Re-ID + 共用 evaluation/）— 2026-07-23
- **結論**：搬移後先用 `python3 -m py_compile` 對所有改動檔案做語法檢查、並寫了一支 AST 靜態掃描腳本逐一比對每個 `from detection/tracker/reid/evaluation/utils import X` 是否都能解析到實際存在的模組與符號，兩項皆通過。之後在 conda `safetrajai` 環境（`ultralytics==8.3.141`、`torch==2.7.1+cu128`、`opencv-python`，見專案 `requirement.txt`）補齊系統 `libgl1` 相依後，於 RTX 5090 上實際跑了一次 `TEST.md` 的完整 pipeline 煙霧測試：
  ```bash
  python track_eval.py --config config.yaml --track-config tracker/track_config_ByteTrack_0411.yaml
  ```
  （此指令為當時路徑，`tracker/` 已於下方第二階段再搬進 `module/tracker/`）6 支序列全部跑完 stage1（tracking）+ stage2（TrackEval），無 traceback。檢查點全過：
  - 逐類別 tracker txt（`.../trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt`）在有偵測到的類別皆非空（如 person 在無偵測的序列裡是 0 行，且與該序列 TrackEval 印出的 person HOTA=0 一致，非漏算）。
  - `trackeval_output/` 下 6 個 `trackeval_metrics_*_MULTI_new.xlsx` 全數產生，HOTA/DetA/AssA 數值正常（非 NaN、非全 0）。
  - `save: true` 下 6 支 `Demo.mp4`（約 2–2.5MB，mp4v 編碼；系統無 `ffmpeg` 故 H.264 轉檔略過並保留原始 mp4v，屬既有行為非本次改動造成）全數產生。
  - `DET skipped`（`config.yaml` 的 `run_det_eval: false`）行為與改動前一致。
- **來源**：`/tmp/.../scratchpad/track_eval_run.log`（本次 smoke test 完整輸出）；本次重組的 git commit（`track_eval.py` OOP 重構 + 模組搬移）。

## 模組重組第二階段（Detection / Tracker / Re-ID / grid_search 統一搬進 `module/`）— 2026-07-23
- **結論**：`detection/`、`tracker/`（含原本巢狀的 `grid_search/`）、`reid/` 搬進 `module/` 底下並列，`grid_search/` 同時從 `tracker/` 巢狀子目錄升級為與其他三者同層的獨立模組（`module/grid_search/`）。搬移後：
  - `python3 -m py_compile` 對整個 active 程式碼樹掃過，0 錯誤。
  - AST 靜態掃描腳本重跑一次，確認每個 `from module.detection / module.tracker / module.reid / evaluation / utils import X` 都能解析到實際模組與符號。
  - 三支 `module/grid_search/*.py`（`grid_search_botsort.py`、`grid_search_bytetrack.py`、`grid_search_botsort_reid.py`）各自用 `--dry-run --target-class scooter` 實際執行，皆正確列出參數組合數，無 `ModuleNotFoundError`。
  - 完整跑一次 `python track_eval.py --config config.yaml --track-config module/tracker/track_config_ByteTrack_0411.yaml`，6 支序列全過，輸出的 HOTA/DetA/AssA 數值與第一階段驗證時完全一致，無 traceback。
- **搬移過程中發現並修正的 bug**：`tracker/grid_search/*.py`（第一階段搬移時）內部 `PROJECT_ROOT = Path(__file__).resolve().parent` 沒有隨巢狀深度調整，導致直接執行會 `ModuleNotFoundError: No module named 'tracker'`——只有靜態檢查是抓不到的，必須實際執行腳本才會發現。已修正為 `.resolve().parents[2]`（本次 `module/grid_search/` 深度不變，此路徑無需再調整）。詳見 `NEED.md`。
- **來源**：本次重組的 git commit（`module/` 搬移 + import 路徑修正 + `PROJECT_ROOT` bug 修正）；`--dry-run` 與完整 pipeline 執行的終端輸出。

## pickle 相容性 — `reid_models/` 與 `utils/model.py` 不可搬移
- **結論**：`models/reid_model_weights/**/*.pth`、`models/class_reid_weights/*/best_full_model.pt` 皆為 `torch.save()` 整物件序列化（非 state_dict），內容以 `strings` 掃描確認寫死了 `reid_models.modeling....`、`utils.model.xxx` 的完整模組路徑。將 `reid_models/` 搬進 `reid/vendored/`、或將 `utils/model.py` 搬進 `module/reid/training/` 都會讓 `torch.load(weights_only=False)` 在還原時找不到原模組而失敗。已於本次重組中驗證並還原。
- **來源**：`strings models/reid_model_weights/**/*.pth models/class_reid_weights/**/*.pt` 掃描結果；`module/tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支、`module/grid_search/grid_search_botsort_reid.py`（原 `reid/grid_search_botsort_reid.py`，中間曾放在 `tracker/grid_search/`，現位於 `module/grid_search/`）對 `models/class_reid_weights/{person,scooter}/best_full_model.pt` 的引用。
- **目前設定**：`reid_models/` 留在根目錄；`utils/model.py` 留在 `utils/model.py`。

## 待補充
> 以下項目目前沒有找到對應的驗證紀錄，若已經做過請補上來源；若還沒做，建議列進 `NEED.md`：
> - `config.yaml` 的 `det_corner` 偵測區域多邊形設定對指標的影響
> - `fuse_score` 開關（目前全域關閉）對各類別的影響
> - `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據
