# Design

## Context

動機見 proposal.md。影響做法的現況與限制：

- 環境只能使用 conda env `track_eval`（Python 3.10、pip 26、setuptools 83、torch 2.7.1+cu128、ultralytics 8.3.141），env 內沒有 `pytest`，也沒有 `tabulate`（trackeval 載入時的 BURST 警告是既有狀況）。
- 目前 `import trackeval` 能成功，只是因為從 repo 根目錄執行，cwd 剛好在 `sys.path` 上。
- `Path(__file__).parents[N]` 出現在 7 處，N 從 1 到 3 不等；`module/tracker/config.py` 還會在自己所在的目錄找 config。
- `track_eval.py:60` 把 repo 的上一層加入 `sys.path`，但沒有任何 import 用到它。
- 預設資料集 `cctv_all8` 只有 6 支 15 秒片段，全量跑一次的成本低。
- 專案規則：tracker override 方法不可更動、`utils/model.py` 的模組路徑必須維持、vendored trackeval 不修改、`config/*.yaml` 的區塊名稱是穩定介面。

## Goals / Non-Goals

**Goals:**
- 每個 Phase 結束時，黃金比對都要通過；任一 commit 都可以單獨回退（revert）。
- 搬移以 `git mv` 為主，保留檔案歷史（`git log --follow` 仍然可追）。

**Non-Goals:**
- 不改善既有已知限制，例如 `save_mot_format()` 的 `C000025_*` glob、`track_buffer` 的單位落差、缺少 `tabulate`。
- 不為搬移後的模組新增功能或補型別註記，只修改搬移必需的部分。
- 不把人工對應點（`config/homography_points/`）的查找鍵從**序列**改成**鏡頭**；雖然依領域模型應以鏡頭為鍵，但這會改變行為，要另開 change 處理（見 `CONTEXT.md` 的 Flagged ambiguities）。
- 不把 `collision` 改名為 `conflict`（見 `docs/adr/0003`）。

## Decisions

### D1：`src/` 佈局＋editable install（`--no-deps`）
採用 `src/safetrajai/`，在 `track_eval` env 執行 `pip install -e . --no-deps`。`pyproject.toml` 宣告核心依賴（`ultralytics==8.3.141`，其餘不鎖版本），`pytest` 列為 dev 依賴，但安裝一律使用 `--no-deps`，避免 pip 把 CUDA 版 torch 換成 PyPI 的版本。
- 替代方案：根目錄平鋪、不安裝。這樣仍然需要 path 處理，第三方目錄也無法乾淨地 import，因此不採用。
- `requirement.txt` 保持原樣，CLAUDE.md 註明它不是依賴的權威來源。

### D2：`third_party/trackeval` 透過同一份 pyproject 對應
在 setuptools 設定 `package-dir`，讓 `safetrajai*` 對應到 `src/`、`trackeval*` 對應到 `third_party/`，一次安裝就能 import 兩者，vendored 目錄內也不必新增任何檔案。
- 實作第一步先驗證 setuptools 83 的 editable 模式能正確處理這種多根目錄對應（包含子套件 `trackeval.metrics` 和 `trackeval.datasets`）。如果不行，就改用備案：採上游結構 `third_party/TrackEval/trackeval/`，另外放一份最小 pyproject，分兩次安裝。這個備案只影響安裝步驟，不影響 spec。

### D3：`safetrajai/paths.py` 作為唯一根目錄
`PROJECT_ROOT = Path(__file__).resolve().parents[2]`（`src/safetrajai/paths.py` → repo）。editable install 下 `__file__` 仍然指向原始碼位置，所以結果穩定。所有 `parents[N]`、`DEFAULT_TRACK_CONFIG_PATH`、`_DATASETS_YAML_PATH` 都改由這裡取得。
- `tracker/config.py` 原本「在套件目錄旁找 config」的 fallback 保留原本的語意（搬家前後那個目錄底下都沒有 YAML，所以行為不變）。
- 替代方案：改用 cwd，會讓從其他目錄呼叫時悄悄出錯，因此不採用。

### D4：類別常數
新增 `safetrajai/classes.py`，內容為 `CLASS_NAMES: dict[int, str]`（0-based）與推導出來的 `MOT_CLASS_ID = {name: id + 1}`。`mot_format.NAMES` 和 `trackeval_runner.class_name_to_class_id` 改成引用這裡；Re-ID 資料工具改用 `sorted(CLASS_NAMES.values())`。`tracking/config.py::CLASS_NAMES`（執行期從 YAML 載入、會被原地修改的 dict）不動，改由 smoke 測試斷言它和 `classes.py` 一致。

### D5：Tracker 整檔搬移
用 `git mv module/tracker/* src/safetrajai/tracking/`，只改檔案層級的 import 敘述（例如 `from module.reid.encoders import ...` → `from safetrajai.reid.encoders import ...`）。`HANDOFF.md` 和 `*_flow.md` 一起搬。驗收時用 AST 比對 `__init__`、`update`、`remove_duplicate_stracks`、`init_track` 的本體（以 `ast.dump` 比較 `FunctionDef.body`），這項檢查納入 smoke 測試，並以 git 裡搬移前的檔案作為參照。

### D6：`utils/model.py` 改為 shim
實作搬到 `safetrajai/reid/model.py`，`utils/model.py` 只保留 re-export（`from safetrajai.reid.model import *` 加上明確列出的類別名稱）。`utils/` 仍然是頂層 package，repo 根目錄只要在 path 上就能被解析。為了讓安裝後在任意 cwd 也能 unpickle，pyproject 會把 `utils` 列為額外的頂層 package（只包含 `utils/model.py`）。
- 替代方案：轉存 state_dict，會改動沒有版控保護的權重檔，因此不採用。

### D7：入口
- 根目錄的 `track_eval.py` 改成 wrapper：`try: from safetrajai.cli.run_pipeline import main`，`except ModuleNotFoundError` 時印出安裝指令並 `sys.exit(1)`。實際的 CLI 放在 `scripts/run_pipeline.py`；wrapper 和 script 共用同一個 `main()`，因此 `main()` 放在套件內，而不是 `scripts/`，讓 wrapper 可以 import。
- 其他 CLI 直接搬到 `scripts/`，不保留 wrapper。
- `scripts/` 本身不是 package，每支腳本只 import `safetrajai`，不自行處理 path。

### D5b：`collision/homography/` 子套件
`module/collision/homography.py` 的內容整檔搬為 `safetrajai/collision/homography/__init__.py`，其餘 5 支 `homography_*.py` 搬進同一子套件（`run_homography_validation.py` 則移到 `scripts/`）。如此 `from . import homography as homography_mod`、`homography_mod.compute_homography()` 等既有呼叫不必改寫，子套件內原本的 `from .homography import ...` 改為 `from . import ...`。
- 替代方案：改名為 `homography/core.py`，呼叫端全部要跟著改，import 改動較多，不採用。

### D8：Phase 3 拆分 `track_eval.py`
| 新位置 | 內容 |
|---|---|
| `safetrajai/pipeline/reid_crops.py` | `make_min_size_quality_filter`、`ReidCropCollector` |
| `safetrajai/pipeline/video_io.py` | `VideoWriterWithFfmpeg` |
| `safetrajai/pipeline/sequence_tracker.py` | `VideoSequenceTracker` |
| `safetrajai/pipeline/tracking_pipeline.py` | `TrackingEvalPipeline` |
| `safetrajai/cli/run_pipeline.py` | argparse＋`main()` |

只搬不改：類別和方法的內容逐字保留，只調整 import。`evaluation/pipeline.py::main()` 移到 `scripts/run_evaluation.py`，`run_eval_pipeline` 則留在套件內。

### D9：黃金比對（Phase 0）
- `tests/golden/configs/*.yaml`：複製 5 個情境，只改 `OUTPUT_ROOT` 為 `./runs/refactor_golden/<run>/...`（`runs/` 已在 gitignore 裡）。
- `tests/golden/run_all.sh <run>`：依序跑 5 個情境、grid search（`--max-videos 1` 加單組 `--params-subset`，結果寫到獨立的 `--result-csv`）以及 homography 驗證。
- `tests/golden/compare.py <runA> <runB>`：
  - MOT txt：逐行比對，數值欄位計算最大絕對誤差。
  - TrackEval 摘要與 grid search CSV：比對指標值。
  - ttc／pet events：比對事件數量，以及逐筆的欄位值。
  - Re-ID crops：比對相對路徑的集合。
- 流程：舊程式碼跑 `before1`、`before2`，由 `compare` 算出容忍度；重構後的每個 Phase 都跑 `after_pN`，再和 `before1` 比對。如果 `before1` 和 `before2` 完全一致，容忍度就是 0。
- 比對腳本和 golden configs 必須在搬移任何程式碼**之前** commit，確保 `before` 的輸出來自未改動的程式碼。

### D10：smoke 測試（`tests/test_smoke.py`，pytest）
- 以 `pkgutil.walk_packages` import 所有 `safetrajai.*` 和 `utils.model`。
- 對 `config/*.yaml` 逐一呼叫 `load_config()`；如果檔案含 `class_tracking:`，也呼叫 `load_track_config()`。
- 檢查類別常數的一致性（見 D4）。
- 做 tracker override 方法的 AST 不變性檢查（見 D5）。
- 掃描 `src/`、`scripts/`、`utils/`、`track_eval.py`，確認沒有 `sys.path` 修改。
- `scripts/*.py --help` 都能以 0 狀態結束，涵蓋訓練、視覺化和 Re-ID 資料工具的 smoke 驗證。

## Risks / Trade-offs

- [某條很少跑到的 import 路徑到執行期才出錯] → D10 import 全部模組，加上 D9 涵蓋 ReID、stage3、crops、grid search、homography 這些分支。
- [GPU 推論的非確定性讓比對出現誤報] → D9 用兩次自比量出容忍度。
- [setuptools 無法處理多根目錄的 editable 對應] → D2 的備案，只影響安裝步驟。
- [unpickle 在安裝後、非 repo cwd 的情況下找不到 `utils.model`] → D6 把 `utils` 列入 pyproject 的 packages；smoke 測試在非 repo cwd 下實際載入一個權重來驗證。
- [文件和 slash command 沒同步，下一個 session 照舊路徑操作] → 文件更新列為各 Phase 的明確任務，最後以 grep 檢查舊路徑已經沒有殘留。
- [搬移讓 `git blame` 失去連續性] → 一律使用 `git mv`，而且搬移和修改 import 分成不同 commit（先純搬移，再修 import），讓 git 能偵測到 rename。
- [grid search 或基準執行覆寫既有輸出] → golden configs 的 `OUTPUT_ROOT`，以及 grid search 的 `--result-csv`，都指到 `runs/refactor_golden/`。

## Migration Plan

1. 開分支 `refactor/project-structure`。
2. Phase 0：commit 比對工具和 golden configs → 用舊程式碼跑 `before1`、`before2` → 記錄容忍度。
3. Phase 1 到 3：依 tasks.md 逐步進行，每個 Phase 結束都跑 smoke 測試和 golden 比對，通過才進下一個 Phase。
4. 開發者操作：`conda activate track_eval && pip install -e . --no-deps && pip install pytest`。
5. 回退：每個 Phase 都是獨立的 commit 群組，可以用 `git revert` 回到上一個 Phase；因為是 editable install，回退後只要重新執行一次 `pip install -e . --no-deps`，或 `pip uninstall safetrajai`。
