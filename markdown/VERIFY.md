# VERIFY.md

紀錄已經做過的實驗驗證與結論，避免重複測試或誤改已驗證過的設定。每筆結論都附上資料來源，之後有新結論請照同樣格式補充。

## 三大流程（save_reid_crops / Re-ID 訓練 / grid search ReID）端到端驗證 + `video/` 補資料 — 2026-07-24
- **結論**：針對搬移後的 `dataset/reid/` 與 `config/` 結構，完整驗證三條流程：
  1. `python track_eval.py --config config/botsort_save_reid_crops.yaml`：`stage2 skipped` 正常、`reid_category_dataset/`／`reid_dataset/` 不受影響。**觀察**：BoTSORT + GMC `ecc` 重跑同一批影片不是逐位元組決定性的（同一批 6 支序列重跑，身分數/裁切數與前次不同），這是既有追蹤器行為，非本次改動造成。
  2. `python module/reid/training/train.py --class-name scooter --epochs 5 --freeze-backbone`（smoke test）：loss 穩定下降（train 0.537→0.335、val 0.424→0.335），存檔後 `load_reid_model()` 與 `encoders.py::ReID` 兩條路徑皆能正確還原並產生 L2-normalized 1280 維 embedding。
  3. `python module/grid_search/grid_search_botsort_reid.py --target-class scooter --max-videos 1 --params-subset match_thresh`：修正下方 bug 後完整跑完 2 組試驗，CSV／`best_params_scooter.txt` 正常輸出。
- **發現並修正的 bug**：`module/grid_search/{grid_search_bytetrack,grid_search_botsort,grid_search_botsort_reid}.py` 各自有 2 處引用 `cfg.base_dir`，但 `AppConfig`（`evaluation/config_loader.py`）從未定義過這個屬性（只有文件字串裡的示範程式碼誤寫成 `cfg.base_dir`，實際欄位是 `cfg.OUTPUT_ROOT`）。這個 bug 讓 3 支 grid search 腳本只要不是 `--dry-run` 就必定在收集影片路徑那步 crash（`AttributeError: AppConfig has no attribute 'base_dir'`）——先前只驗證過 `--dry-run`（不會走到這段程式碼），這次要跑真正的一組試驗才發現。已修正全部 6 處為 `cfg.OUTPUT_ROOT`，同時修正 `evaluation/config_loader.py` 文件字串裡的示範程式碼。
- **`video/` 補資料**：新增 `config/botsort_save_reid_crops_video.yaml`，來源改用 `video/` 底下 5 支不同路口原始影片（各取前 60 秒放進 `dataset/video_seqs/annotations_video/`），`det_corner` 關閉。目的是補足既有 6 支序列裡 `bus`/`person`/`truck`/`small_truck` 身分數過少的問題。`dataset/reid/crop_dataset/` 最終統計（兩批來源合計）：1185 身分、33399 張裁切（`person` 35、`car` 531、`scooter` 336、`small_truck` 183、`bus` 5、`truck` 95）。`bus` 仍明顯偏少——這兩批來源影片路口都幾乎沒拍到公車，需要額外找有公車出現的影片才能補足，詳見 `dataset/reid/README.md`。
- **來源**：本次驗證的實際終端輸出（未另外存檔）；`dataset/reid/README.md`「目前資料量」段落。

## Re-ID 資料集搬進 `dataset/reid/` + `reid_crop_dir` 語意修正 — 2026-07-24
- **結論**：`dataset/crop_dataset`、`dataset/reid_category_dataset`、`dataset/reid_dataset` 統一搬進 `dataset/reid/` 底下（分別變成 `dataset/reid/crop_dataset`、`dataset/reid/reid_category_dataset`、`dataset/reid/reid_dataset`），`dataset/crop_dataset/README.md`、`build_reid_split.py` 一併搬到 `dataset/reid/` 這一層（不留在 `crop_dataset/` 裡，因為兩者描述/操作的是三個平行目錄，放在共同父層更合理）。同時把 `config/*.yaml` 的 `reid_crop_dir` 從 `./dataset/reid_dataset` 改指向 `./dataset/reid/crop_dataset`——這是語意修正非單純改名：`save_reid_crops` 寫入的結構是 `{class}/{影片}_track{id}/*.jpg`，跟 `crop_dataset` 的結構相符，跟 `reid_dataset`（`build_reid_split.py --mode flat` 的切分輸出，`{split}/{class}_{identity}/`）結構不符，先前用舊名稱會讓兩者撞在同一個目錄下。`evaluation/config_loader.py` 的預設值、`track_eval.py` 的 `VideoSequenceTracker` 預設參數、`module/reid/training/train.py` 的 `--data-root` 預設值都同步更新。
- **驗證**：`python dataset/reid/build_reid_split.py --mode category --overwrite` 與 `--mode flat --overwrite` 在新路徑下重新跑過，結果與搬移前一致（209 身分、3830 張影像，bus 唯一身分正確分進 train）；`python3 -m py_compile` 對 `build_reid_split.py`、`track_eval.py`、`evaluation/config_loader.py`、`module/reid/training/{train,dataset}.py` 全部通過。
- **來源**：本次搬移的實際執行輸出（終端紀錄，未另外存檔）；`dataset/reid/README.md`。

## 設定檔整併至 `config/`（6 個情境檔）— 2026-07-24
- **結論**：`config.yaml` + `module/tracker/track_config_*.yaml` 全數整併成 `config/` 底下 6 個獨立情境檔（`bytetrack.yaml`、`botsort_noreid.yaml`、`botsort_save_reid_crops.yaml`、`botsort_reid.yaml`、`grid_search_bytetrack.yaml`、`grid_search_botsort.yaml`），新增 `stages.run_stage1`/`run_stage2` 開關。實際驗證：
  - `python track_eval.py`（預設 `config/bytetrack.yaml`）：6 支序列 HOTA/DetA/AssA 數值與整併前逐位元組相同。
  - `python track_eval.py --config config/botsort_save_reid_crops.yaml`：`stages.run_stage2: false` 正確跳過 stage2（無 TrackEval 輸出）；BOTSORT tracker 正確啟用；`dataset/reid_dataset/` 新增 6 個類別資料夾、共 3830 張裁切影像，舊的 `train/` 扁平資料完全未受影響。
  - `module/grid_search/grid_search_bytetrack.py`、`grid_search_botsort.py`、`grid_search_botsort_reid.py` 三支 `--dry-run`（各自新預設 config）皆正常列出參數組合。
- **來源**：本次整併的 git commit；`config/botsort_save_reid_crops.yaml` 的實際執行輸出（終端紀錄，未另外存檔）。

## ReID 開關（person vs. scooter）
- **結論**：開啟 ReID 對 person 的追蹤有幫助，但會損害 scooter 追蹤並增加 IDSW（ID switch）。
- **目前設定**：`module/tracker/__init__.py` 的 `DEFAULT_TRACKER_CONFIG['with_reid']` 預設 `False`。
- **來源**：`module/tracker/__init__.py` 程式碼註解；細節推論見 `module/tracker/HANDOFF.md` 需求 4。
- **待補充**：實際數據（各類別 HOTA/IDSW 變化量）目前只存在於程式碼註解描述，沒有對應的 Excel/CSV 可查— 需要的話應重新用 `module/grid_search/grid_search_botsort_reid.py` 針對 person / scooter 各跑一次 ReID on/off 對照並存檔。

## GMC（Generalized Motion Compensation）方法比較 — BoTSORT + NoReID
- **結論**：以全部 6 類別 × 6 序列的 `mean_AssA` 平均比較，`sift` 排名最佳，`ecc`、`none`、`sparseOptFlow`、`orb` 依序在後，但彼此差距極小（第三位小數才有差異），**實務上五種 GMC 方法表現接近，非決定性差異**。
- **來源**：原始報告 `botsort_noreid/`（逐類別、逐序列明細 `.md`、對應 `.xlsx`、`gmc_*.txt` 各方法原始結果）已於 2026-07-23 移除，結論已確認保留於此，如需追溯原始數據可從 git 歷史（commit `7061f97`）取回。
- **目前設定**：`config/*.yaml` 的 `tracking_params.gmc_method: ecc`。

## FPS / 推論解析度對 ByteTrack 準確度的影響
- **來源**：原始報告 `fps_compare/`（`fps_bytetrack_comparison_by_class_and_sequence.md`、`fps_bytetrack_comparison_by_sequence.md`，imgsz=640，比較 10/20/30 fps、對應 `.xlsx`）已於 2026-07-23 移除，如需追溯可從 git 歷史（commit `7061f97`）取回。
- **結論**：待補充 — 移除前檔案內已有逐序列逐類別數據，尚未整理出整體結論摘要；現原始數據已不在工作目錄，需要的話應從 git 歷史取回或重新跑一次比較。

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
  - `DET skipped`（`config/*.yaml` 的 `eval_flags.run_det_eval: false`）行為與改動前一致。
- **來源**：`/tmp/.../scratchpad/track_eval_run.log`（本次 smoke test 完整輸出）；本次重組的 git commit（`track_eval.py` OOP 重構 + 模組搬移）。

## 模組重組第二階段（Detection / Tracker / Re-ID / grid_search 統一搬進 `module/`）— 2026-07-23
- **結論**：`detection/`、`tracker/`（含原本巢狀的 `grid_search/`）、`reid/` 搬進 `module/` 底下並列，`grid_search/` 同時從 `tracker/` 巢狀子目錄升級為與其他三者同層的獨立模組（`module/grid_search/`）。搬移後：
  - `python3 -m py_compile` 對整個 active 程式碼樹掃過，0 錯誤。
  - AST 靜態掃描腳本重跑一次，確認每個 `from module.detection / module.tracker / module.reid / evaluation / utils import X` 都能解析到實際模組與符號。
  - 三支 `module/grid_search/*.py`（`grid_search_botsort.py`、`grid_search_bytetrack.py`、`grid_search_botsort_reid.py`）各自用 `--dry-run --target-class scooter` 實際執行，皆正確列出參數組合數，無 `ModuleNotFoundError`。
  - 完整跑一次 `python track_eval.py --config config.yaml --track-config module/tracker/track_config_ByteTrack_0411.yaml`，6 支序列全過，輸出的 HOTA/DetA/AssA 數值與第一階段驗證時完全一致，無 traceback。
- **搬移過程中發現並修正的 bug**：`tracker/grid_search/*.py`（第一階段搬移時）內部 `PROJECT_ROOT = Path(__file__).resolve().parent` 沒有隨巢狀深度調整，導致直接執行會 `ModuleNotFoundError: No module named 'tracker'`——只有靜態檢查是抓不到的，必須實際執行腳本才會發現。已修正為 `.resolve().parents[2]`（本次 `module/grid_search/` 深度不變，此路徑無需再調整）。詳見 `NEED.md`。
- **來源**：本次重組的 git commit（`module/` 搬移 + import 路徑修正 + `PROJECT_ROOT` bug 修正）；`--dry-run` 與完整 pipeline 執行的終端輸出。

## pickle 相容性 — `utils/model.py` 不可搬移（`reid_models/` 部分已隨 SOTA_ReID 移除，見下方）
- **結論**：`models/class_reid_weights/*/best_full_model.pt` 是 `torch.save()` 整物件序列化（非 state_dict），內容以 `strings` 掃描確認寫死了 `utils.model.xxx` 的完整模組路徑。將 `utils/model.py` 搬到別處會讓 `torch.load(weights_only=False)` 在還原時找不到原模組而失敗。已於本次重組中驗證並還原。
- **來源**：`strings models/class_reid_weights/**/*.pt` 掃描結果；`module/tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支、`module/grid_search/grid_search_botsort_reid.py`（原 `reid/grid_search_botsort_reid.py`，中間曾放在 `tracker/grid_search/`，現位於 `module/grid_search/`）對 `models/class_reid_weights/{person,scooter}/best_full_model.pt` 的引用。
- **目前設定**：`utils/model.py` 留在 `utils/model.py`。

## 移除 `reid_models/` / `SOTA_ReID` / `models/reid_model_weights/` — 2026-07-24
- **結論**：使用者確認 Re-ID 只需保留兩條路徑（`ReID`：YOLO 本身當 encoder；`Custom_ReID`：自己訓練的模型），vendored 的 `SOTA_ReID`（FastReID/TransReID）不需要。移除前用 `pickletools` + 實際 `torch.load()` 追蹤確認：`SOTA_ReID` 唯一的呼叫來源是 `module/grid_search/grid_search_botsort_reid.py` 的 `PARAM_GRIDS`（只有 `person`/`scooter` 兩個類別列了 `reid_model_weights` 路徑，且只列 `_wrap.pth` 版本，非 wrap 版三個檔案完全沒被引用）；`module/reid/encoders.py` 裡 `load_reid_model` 的 import 也是死的（唯一使用處是已被註解掉的舊版 `ReID` class）。
- **意外發現**：`SOTA_ReID` 在乾淨的 `safetrajai` conda 環境完全無法載入，依序缺 `prettytable`、`easydict`、`kornia`、`fvcore` 四個套件（都不在 `requirement.txt`），且光是 `import reid_models` 就會連帶拉入 `reid_models.data.*`（11 個訓練用資料集類別）與 `reid_models.evaluate.*`，這些對推論用途完全不需要。移除後改用直接建構 `ClassAwareBOTSORT`（模擬 `person`→`ReID`、`scooter`→`Custom_ReID`）與完整 pipeline 驗證兩條保留路徑皆正常，HOTA/AssA 數值與移除前一致。`Custom_ReID` 這條保留路徑則缺 `scikit-learn`（`utils/model.py` 的依賴），建議補進 `requirement.txt`。
- **來源**：本次移除的 git commit；`.gitignore` 誤判分析（`models/` 規則無前導 `/`，連帶影響 `reid_models/` 巢狀同名子目錄的版控狀態）；詳見 `NEED.md`。

## 待補充
> 以下項目目前沒有找到對應的驗證紀錄，若已經做過請補上來源；若還沒做，建議列進 `NEED.md`：
> - `config/*.yaml` 的 `det_corner` 偵測區域多邊形設定對指標的影響
> - `fuse_score` 開關（目前全域關閉）對各類別的影響
> - `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據
