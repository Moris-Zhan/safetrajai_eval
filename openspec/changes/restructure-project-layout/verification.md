# 驗證紀錄：restructure-project-layout

- 分支：`refactor/project-structure`（基準 `BASE_COMMIT` = `9c6414b`）
- 環境：conda env `track_eval`（Python 3.10、torch 2.7.1+cu128、ultralytics 8.3.141、setuptools 83）、NVIDIA GeForce RTX 5090
- 日期：2026-09-23

## Golden 比對總表

容忍度由未改動程式碼連跑兩次（`before1`、`before2`）量測，結果完全一致，容忍度為 0（`tests/golden/TOLERANCE.md`）。之後每個 Phase 都與 `before1` 比對：

| run | 程式碼狀態 | 執行方式 | 結果 |
|---|---|---|---|
| `before2` | `9c6414b`（未改動） | 序列 | identical |
| `after_p1` | Phase 1（`062c511`） | 序列 | identical |
| `after_p2` | Phase 2（`2504e3e`），入口改為 `scripts/` | 平行 | identical |
| `after_p3` | Phase 3（`cfdd430`），經 `python track_eval.py` wrapper | 平行 | identical |

每次比對涵蓋：MOT 軌跡 360 檔、YOLO labels 1764 檔、TrackEval 摘要與 Excel 126 檔、grid search 2 檔、衝突事件 44 檔、Re-ID 裁切 3830 檔（路徑集合）、homography 驗證 4 檔、其他 2214 檔。每次執行後也確認 `dataset/`、`models/`、`config/` 沒有被寫入。

## Requirement 對照

### 可安裝套件且不依賴執行期路徑修補

| Scenario | 證據 |
|---|---|
| 安裝後可從任意工作目錄 import | 6.1：`env -i` 乾淨 shell 依 CLAUDE.md 重新安裝後，於 `/tmp` 下 `safetrajai` 解析到 `src/safetrajai/`、`trackeval` 解析到 `third_party/trackeval/`，pytest 96 passed／1 skipped |
| 安裝不變動既有依賴版本 | 1.2、3.2、6.1 各次安裝前後比對 `pip list`，torch／ultralytics／numpy／scipy／opencv 版本不變 |
| 原始碼不含路徑修補 | `tests/test_smoke.py::test_no_sys_path_mutation` 掃描 `src/`、`scripts/`、`utils/`、`track_eval.py` |

### 專案根目錄的解析只有單一來源

| Scenario | 證據 |
|---|---|
| 預設設定檔路徑正確 | `DEFAULT_TRACK_CONFIG_PATH = CONFIG_DIR / "bytetrack.yaml"`（`safetrajai/paths.py`）；`grep "parents\["` 只剩 `paths.py`。搬移後原本的 `parents[2]` 會解析到 `src/`，已由此修正 |
| 資料集對照表路徑正確 | `_DATASETS_YAML_PATH = CONFIG_DIR / "datasets.yaml"`；smoke 對全部 `config/*.yaml` 呼叫 `load_config()`（含 `paths.dataset` 查表）；golden 5 個情境皆經由 `cctv_all8` 查表且輸出一致 |

### 主入口指令相容

| Scenario | 證據 |
|---|---|
| 舊指令照常執行 | `after_p3` 全程經 `python track_eval.py --config ...` 執行，與 `before1` identical |
| 未安裝時的提示 | 4.2：`pip uninstall safetrajai` 後執行 `python track_eval.py`，印出 `safetrajai is not installed. Run: conda activate track_eval && pip install -e . --no-deps` 並 exit 1；重新安裝後恢復 |

### 既有 Re-ID 完整模型權重可載入

| Scenario | 證據 |
|---|---|
| 載入既有權重 | 3.8：於 `/tmp` 以 `torch.load(..., weights_only=False)` 載入 `person`、`scooter` 的 `best_full_model.pt` 成功（pickle 引用 `utils.model.{ReIDNet,CameraEmbedding,OcclusionRepair}`）；smoke `test_utils_model_is_reexport_shim` |

**限制**：`config/botsort_reid.yaml` 各類別的 `model` 都是 YOLO ReID 權重，golden 情境不會走到 `Custom_ReID`／`best_full_model.pt`，因此「追蹤結果與重構前等價」這半句沒有 golden 證據，只驗證到權重可載入且類別與 shim 一致。另外，原 `track_eval` env 缺 `scikit-learn`，`utils.model` 在重構前就無法 import；1.2 依使用者決定補裝後才能載入。

### 類別 ID 契約單一來源

| Scenario | 證據 |
|---|---|
| MOT 類別 ID 推導 | `safetrajai/classes.py::MOT_CLASS_ID = {name: id + 1}`；smoke `test_class_constants_match_legacy_values`（值與順序皆等於舊字面值） |
| 設定檔載入的類別名稱與契約一致 | smoke `test_track_config_class_names_match_contract`（每個含 `class_tracking:` 的 config） |
| Re-ID 資料工具的類別順序不變 | `CLASSES = sorted(CLASS_NAMES.values())`；smoke `test_reid_data_class_order_unchanged` 斷言等於舊清單 |

### 重構前後輸出等價

| Scenario | 證據 |
|---|---|
| stage1 軌跡等價 | 上方總表 `mot_tracks`、`yolo_labels` |
| stage2 指標等價 | `trackeval`（`*_summary.txt` 與 Excel 逐值） |
| stage3 事件等價 | `conflict_events`（`ttc_events`／`pet_events` JSON、CSV 逐筆逐欄） |
| Re-ID 裁切蒐集等價 | `reid_crops`（3830 個相對路徑集合一致；內容未逐位元比對） |
| grid search 與 homography 驗證等價 | `grid_search`（結果 CSV，不含 `elapsed_sec`）、`homography`（`summary.csv`、stats Excel） |

### Tracker override 方法本體不變

| Scenario | 證據 |
|---|---|
| 方法本體比對 | smoke `test_tracker_override_bodies_unchanged` 以 AST 比對 `BASE_COMMIT` 與現行的 `__init__`／`update`／`remove_duplicate_stracks`（兩類）與 `init_track`（BoTSORT），並以負向測試確認能偵測本體改動；另以 `git diff BASE_COMMIT` 確認文字層級只有 `class_aware_botsort.py` 的 2 行 import 變動，`class_aware_bytetrack.py` 完全未變 |

## 其他驗證

- 純搬移 commit（`2df462e`、`fc0dc15`、`5bfc57b`）皆以 `git diff --numstat` 確認 0 行改動的 rename；`git log --follow` 可追到搬移前的歷史（例如 `class_aware_botsort.py` → `trackers/`、`homography/points.py` → `module/collision/homography_points.py`）。
- Phase 3 拆分（4.1）：`make_min_size_quality_filter`、`ReidCropCollector`、`VideoWriterWithFfmpeg`、`VideoSequenceTracker`、`TrackingEvalPipeline` 與 `main()` 皆與拆分前 AST 相同。
- Phase 5（文件同步）：23 個 `.py` 去掉 docstring 後 AST 與前一版相同，唯一的程式碼差異是 `tracking/config.py` 找不到 YAML 時錯誤訊息中的目錄名稱。

## 偏離與決定（皆已記入 tasks.md／design.md）

- golden configs 除 `OUTPUT_ROOT` 外也改 `reid_crop_dir`，避免寫入正式資料集 `dataset/reid/crop_dataset`（1.3）。
- 補裝 `scikit-learn`（1.2，使用者決定）。
- `run_all.sh` 改為平行執行（使用者決定），單次 golden 約 32 → 11 分鐘，`after_p2`／`after_p3` 以平行版跑出 identical。
- design D5b：`homography.py` 成為 `collision/homography/__init__.py`（使用者決定）；D5c：`homography_*` 去前綴、`visualization/drawing.py`、`validate_homography.py` 無 argparse 故 smoke 只做 import。
- `.claude/settings.local.json`（未進版控）移除 18 條舊路徑的一次性權限，homography 驗證權限改為新路徑。

## 分支摘要

`git diff --stat -M master...HEAD`：141 files changed, 2787 insertions(+), 1079 deletions(-)；95 個 rename、30 個新增、16 個修改（大部分新增行數來自 `tests/`、`src/safetrajai/pipeline/` 拆分與 openspec 文件）。
