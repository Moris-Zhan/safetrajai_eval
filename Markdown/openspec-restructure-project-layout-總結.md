# OpenSpec 變更總結：restructure-project-layout（專案結構重構）

> 狀態：**已完成並歸檔**（archived at `2026-09-23`）
> 來源：`openspec/changes/archive/2026-09-23-restructure-project-layout/`
> 對應 spec：`openspec/specs/project-structure/spec.md`

## 目前 OpenSpec 進度總覽

檢查 `openspec/changes/` 目錄後確認：**目前沒有任何進行中（未歸檔）的 change**，唯一存在的 change `restructure-project-layout` 已完整走完 Phase 0～Phase 3、文件同步與收尾驗證共 6 大階段，`tasks.md` 全部項目皆已勾選（`[x]`），且已產出 `verification.md` 驗證紀錄並歸檔至 `openspec/changes/archive/2026-09-23-restructure-project-layout/`，同步生成 `openspec/specs/project-structure/spec.md` 作為正式規格。

因此，本次任務即為：**將這份已完成的 spec 整理成文件**，輸出於此。

## 為什麼要做這次重構（Why）

重構前的專案沒有可安裝的套件結構：至少 7 支腳本各自用 `sys.path.insert` 與 `Path(__file__).parents[N]` 推算專案根目錄；主入口 `track_eval.py`（736 行）同時承擔 CLI、Re-ID 裁切蒐集、影片寫出與 pipeline 編排；類別 ID 契約在 5 處重複定義；視覺化程式碼分散三處；根目錄混雜與本專案無關的工具腳本。這種結構讓「搬動或新增模組時靜默讀錯路徑」的風險很高，且當時沒有任何測試能攔住行為改變。

## 重構做了什麼（What Changes）

### Phase 0：回歸防線（在舊結構上完成，不搬任何程式碼）
- 建立 `tests/golden/`：5 份情境 config（bytetrack、botsort_noreid、botsort_reid、collision_detect、botsort_save_reid_crops）、`run_all.sh`（平行跑 5 情境＋grid search＋homography 驗證）、`compare.py`（MOT txt、TrackEval 摘要、grid search CSV、ttc/pet events、Re-ID crops 路徑集合逐項比對）。
- 建立 `tests/test_smoke.py`（舊結構版）並記錄基準 commit 到 `tests/golden/BASE_COMMIT`。
- 以舊程式碼連跑兩次（`before1`、`before2`）量測自比容忍度，結果**完全一致，容忍度為 0**，記錄於 `tests/golden/TOLERANCE.md`。

### Phase 1：清理（仍在舊結構上）
- 移除無呼叫者的 `translate_path`／`normalize_string` 與 `argostranslate` 依賴。
- 兩支 `.sh` 工具腳本移至 `tools/`。

### Phase 2：套件化與搬移
- 新增 `pyproject.toml`，程式碼移入 `src/safetrajai/`；於 conda env `track_eval` 以 `pip install -e . --no-deps` 安裝。
- vendored `trackeval/` 整包搬到 `third_party/trackeval/`（內容不變，純 rename）。
- 新增 `safetrajai/paths.py` 作為唯一 `PROJECT_ROOT` 來源，移除所有 `sys.path` hack。
- Tracker 檔案整檔搬移至 `safetrajai/tracking/`，**只改檔案層級 import，override 方法本體一字不動**（AST 比對驗證）。
- `utils/model.py` 改為 pickle 相容 shim，實作移至 `safetrajai/reid/model.py`（既有 Re-ID 權重仍可 `torch.load(weights_only=False)` 成功載入）。
- 新增 `safetrajai/classes.py` 作為類別 ID 契約單一來源，取代原本 5 處重複定義。
- CLI 腳本集中到 `scripts/`（grid search、homography 驗證、Re-ID 訓練與資料稽核、GT 疊圖等）。**Breaking**：舊路徑（`module/grid_search/*.py` 等）失效，改用 `scripts/` 新路徑。

### Phase 3：拆分主入口
- `track_eval.py` 內容逐字拆到 `safetrajai/pipeline/{tracking_pipeline, sequence_tracker, reid_crops, video_io}.py` 與 `safetrajai/cli/run_pipeline.py`。
- 根目錄 `track_eval.py` 改為相容 wrapper：已安裝時行為不變；未安裝時印出安裝指令並 `exit(1)`。
- 新增 `scripts/run_pipeline.py`、`scripts/run_evaluation.py`。

### 文件同步
- 更新 `CLAUDE.md`（架構圖、常用指令、安裝步驟）、`.claude/commands/*`、`.claude/settings.local.json`、各 `HANDOFF.md`／`*_flow.md`／README、`config/*.yaml` 開頭註解，術語對齊 glossary（碰撞 → 衝突事件、影片 → 序列、tag → 鏡頭）。

### 明確不做的事
config 分層拆檔、Re-ID 權重轉存 state_dict、任何演算法或參數調整、修改 vendored trackeval 內容。

## 重構後的專案架構樹狀圖

以下為目前實際的目錄結構（已排除 `.gitignore` 所列的 `dataset/`、`models/`、`runs/`、`video/`、`Sub_Module/`、`ignore/`、`PaperSurvey/` 等大型資料/權重/實驗輸出目錄，以及 `__pycache__`、`.git` 等暫存目錄；`third_party/trackeval/` 只列到子模組層級，內部檔案眾多且為唯讀 vendored 程式碼，不逐一展開）：

```text
.
├── track_eval.py                  # 相容入口 wrapper → safetrajai.cli.run_pipeline.main()
├── pyproject.toml                 # 可安裝套件定義（src/ 佈局）
├── CLAUDE.md
├── CONTEXT.md                     # 術語表（鏡頭、序列、軌跡、身分、衝突事件…）
├── requirement.txt                # 舊 pip 腳本，非依賴權威來源
│
├── config/                        # 情境設定檔：每檔自包含（paths/models/stages/eval_flags/tracking_params + class_tracking:）
│   ├── bytetrack.yaml
│   ├── botsort_noreid.yaml
│   ├── botsort_reid.yaml
│   ├── botsort_save_reid_crops.yaml
│   ├── botsort_save_reid_crops_video.yaml
│   ├── collision_detect.yaml
│   ├── grid_search_bytetrack.yaml
│   ├── grid_search_botsort.yaml
│   ├── datasets.yaml               # dataset id 對照表
│   ├── homography_points/          # 人工點選座標（以序列為鍵）
│   └── homography_checkpoints/     # 已求得的鏡頭矩陣
│
├── scripts/                        # CLI 入口，統一取代舊的 module/*.py 散落腳本
│   ├── run_pipeline.py             # 相當於 python track_eval.py
│   ├── run_evaluation.py           # 只跑 stage2
│   ├── grid_search_bytetrack.py
│   ├── grid_search_botsort.py
│   ├── grid_search_botsort_reid.py
│   ├── validate_homography.py
│   ├── train_reid.py
│   ├── visualize_gt_on_video.py
│   ├── visualize_gt_on_class_video.py
│   └── reid_data/                  # Re-ID 資料切分與稽核
│       ├── build_reid_split.py
│       ├── check_class_consistency.py
│       ├── generate_identity_review.py
│       └── quarantine_flagged.py
│
├── src/safetrajai/                 # 可安裝套件本體（pip install -e . --no-deps）
│   ├── paths.py                    # 唯一 PROJECT_ROOT／CONFIG_DIR 來源
│   ├── classes.py                  # 類別 ID 契約（CLASS_NAMES、MOT_CLASS_ID）
│   ├── cli/
│   │   └── run_pipeline.py         # argparse + main()
│   ├── pipeline/                   # tracking_pipeline／sequence_tracker／reid_crops／video_io
│   ├── detection/                  # Detector：單幀 YOLO 推論 + 逐類別信心門檻
│   ├── tracking/                   # class-aware ByteTrack／BoTSORT（唯一入口 create_tracker()）
│   │   ├── class_aware_botsort.py  # override 方法本體凍結（ADR-0004）
│   │   ├── class_aware_bytetrack.py
│   │   ├── config.py
│   │   └── matching.py
│   ├── reid/                       # ReID（YOLO 當 encoder）與 Custom_ReID 兩條路徑
│   │   ├── model.py                # 完整模型 pickle 的真正實作
│   │   └── training/
│   ├── grid_search/
│   │   └── core.py                 # grid search 共用邏輯
│   ├── collision/                  # stage3：TTC/PET 衝突事件、熱區圖
│   │   └── homography/             # 轉換／點選／驗證子套件
│   ├── evaluation/                 # stage2 共用：config_loader／dataset_layout／mot_converter／trackeval_runner／excel_export
│   └── visualization/
│       └── drawing.py
│
├── utils/
│   └── model.py                    # pickle 相容 shim（不可刪除／改名，見 ADR-0001）
│
├── third_party/
│   └── trackeval/                  # vendored TrackEval（6 類別對照，唯讀，見 ADR-0002）
│       ├── datasets/
│       ├── metrics/
│       └── baselines/
│
├── tests/
│   ├── test_smoke.py               # import 全模組／全 config／類別契約／tracker AST 不變性／--help
│   └── golden/                     # 黃金比對（回歸防線）
│       ├── run_all.sh
│       ├── compare.py
│       ├── BASE_COMMIT
│       ├── TOLERANCE.md
│       └── configs/
│
├── tools/                          # 與追蹤評估流程無關的小工具腳本
│
├── docs/
│   └── adr/                        # 關鍵決策紀錄（Architecture Decision Records）
│
├── openspec/                       # 規格驅動的變更紀錄
│   ├── changes/archive/2026-09-23-restructure-project-layout/
│   └── specs/project-structure/spec.md
│
└── Markdown/                       # 本次任務輸出的整理文件目錄
    └── openspec-restructure-project-layout-總結.md
```

> 註：根目錄下原本殘留一個重構前 `evaluation/` 套件留下的 `evaluation/__pycache__/` 編譯快取（已 gitignore、無 `.py` 原始檔），已於整理文件時一併清除。

## 新增規格：`project-structure`（不變式契約）

正式規格見 `openspec/specs/project-structure/spec.md`，定義了 6 條 Requirement：

| Requirement | 內容摘要 |
|---|---|
| 可安裝套件且不依賴執行期路徑修補 | conda env `track_eval` 一次 editable install 後可從任意目錄 import；原始碼不得有 `sys.path` 修補 |
| 專案根目錄的解析只有單一來源 | 需要定位 repo 內檔案（`config/` 等）一律透過 `safetrajai/paths.py`，不依賴呼叫端目錄深度 |
| 主入口指令相容 | `python track_eval.py --config <path>` 行為與重構前相同；未安裝時印出安裝指令並非零狀態結束 |
| 既有 Re-ID 完整模型權重可載入 | `models/class_reid_weights/*/best_full_model.pt` 不修改權重檔即可載入，`utils.model` 持續可解析 |
| 類別 ID 契約單一來源 | `person=0、car=1、scooter=2、small_truck=3、bus=4、truck=5` 只定義一次，MOT 用的 1-based ID 由此推導 |
| Tracker override 方法本體不變 | `ClassAwareBOTSORT`／`ClassAwareBYTETracker` 的 override 方法本體與重構前完全相同（AST 比對） |

## 驗證結果（verification.md 摘要）

- 分支：`refactor/project-structure`，基準 commit `9c6414b`。
- 環境：conda env `track_eval`（Python 3.10、torch 2.7.1+cu128、ultralytics 8.3.141、setuptools 83）、RTX 5090。

**Golden 比對總表**（容忍度為 0，逐一與 `before1` 比對）：

| run | 程式碼狀態 | 結果 |
|---|---|---|
| `before2`（未改動，自比基準） | `9c6414b` | identical |
| `after_p1` | Phase 1（`062c511`） | identical |
| `after_p2` | Phase 2（`2504e3e`），入口改為 `scripts/` | identical |
| `after_p3` | Phase 3（`cfdd430`），經 `python track_eval.py` wrapper | identical |

每次比對涵蓋：MOT 軌跡 360 檔、YOLO labels 1764 檔、TrackEval 摘要與 Excel 126 檔、grid search 2 檔、衝突事件 44 檔、Re-ID 裁切 3830 檔（路徑集合）、homography 驗證 4 檔，共約 2214+ 其他檔案；每次執行後皆確認 `dataset/`、`models/`、`config/` 未被寫入。

**已知限制**：`config/botsort_reid.yaml` 各類別的 `model` 皆為 YOLO ReID 權重，golden 情境未實際走到 `Custom_ReID`／`best_full_model.pt`，因此「ReID 追蹤結果與重構前等價」只驗證到「權重可載入且類別與 shim 一致」，未有 golden 輸出證據。

**分支摘要**：`git diff --stat -M master...HEAD` → 141 files changed, 2787 insertions(+), 1079 deletions(-)；95 個 rename、30 個新增、16 個修改。

## 對後續開發的影響

- 舊 CLI 路徑（`module/grid_search/*.py`、`python -m module.collision.run_homography_validation`、`module/reid/training/train.py`、`dataset/reid/*.py`、`visualize/*.py`）**已失效**，一律改用 `scripts/` 下對應腳本，詳見 `CLAUDE.md` 的「常用指令」章節。
- 安裝方式固定為 conda env `track_eval` + `pip install -e . --no-deps`。
- 新增模組定位路徑一律用 `safetrajai.paths`，不要再用 `Path(__file__).parents[N]` 或修改 `sys.path`。
- Tracker（`ClassAwareBOTSORT`／`ClassAwareBYTETracker`）override 方法本體仍是**不可更動**的紅線（見 `docs/adr/0004`），任何未來改動都會被 smoke 測試的 AST 比對攔下。

---
*本文件由 Claude Code 依 `openspec/changes/archive/2026-09-23-restructure-project-layout/` 下的 `proposal.md`、`tasks.md`、`verification.md` 與 `openspec/specs/project-structure/spec.md` 整理而成，內容以上述原始檔案為準。*
