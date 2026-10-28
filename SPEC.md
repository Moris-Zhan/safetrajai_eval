# 規格：safetrajai_eval 精簡重構（Refactor Spec）

> 狀態：已完成 grilling（Q1–Q25 全數決定），待使用者審閱確認後才進入 Plan／Tasks／Implement。
> 本檔只定義「要做什麼、不做什麼、如何驗證」，不含任何程式碼變更。

## 1. 目標（Objective）

在**不改變功能行為**的前提下，精簡第一方程式碼與文件中的冗餘與重複、整理目錄架構，並讓共用邏輯集中到單一位置。

「行為不變」的定義：
- 所有 CLI 的參數名稱、預設值、`--help` 輸出相同。唯一允許的差異：config 分資料夾（T4）後，`--config` 預設路徑字串改變。
- `config/*.yaml` 的區塊名稱與語意不變（檔案位置會改，見 T4）。
- `import safetrajai.…` 與 `import utils.model` 的路徑全部不變。
- stage1／stage2／stage3 的輸出檔案格式與內容不變（含 `combo_id`、CSV 欄位順序）。
- `pytest` 全數通過；測試只允許「機械式路徑更新」（見第 6 節），斷言不改。

成功標準見第 9 節。

## 2. 技術環境（Tech Stack）

- Python ≥ 3.10，套件 `safetrajai`（搬動後原始碼根目錄為 `utils/modules/`）。
- 只使用 conda env `track_eval`；安裝一律 `pip install -e . --no-deps`。
- 核心依賴以 `pyproject.toml` 為準。
- 測試：`pytest`。重構前基準線：**739 passed, 11 skipped**（約 43 秒，`main` 上量得）。

## 3. 指令（Commands）

```bash
conda activate track_eval

# 快速回歸（每個任務 commit 前都要跑）
pytest

# 行為基準：動手前先擷取到 scratchpad（不放進 repo），每個任務後逐位元 diff
python scripts/grid_search_<bytetrack|botsort|botsort_reid>.py --help
python scripts/grid_search_<bytetrack|botsort|botsort_reid>.py --target-class <class> --dry-run
#   <class> 取 person／car／scooter／small_truck／bus／truck 六類，加上 _default 路徑
python track_eval.py --help
python scripts/run_evaluation.py --help

# golden 比對：本輪不跑（行為不變，且耗時 16 分鐘、大量讀寫掛載磁碟）；若 pytest 或 --help 比對出現任何差異，先停下來問使用者
tests/golden/run_all.sh <run>
python tests/golden/compare.py conflict1 <run>
```

## 4. 專案結構（Project Structure）

### 目標結構（重構後）

```
dataset/                  唯一的資料根目錄（gitignore）：eval/、raw/、reid/、reports/
  raw/videos/<camera>/    Re-ID 蒐集來源影片（不動）
  raw/homography_videos/<camera>/   由 video/ 搬入，供 pick_aid／validate_homography 使用（不屬於任何 dataset 登記，不被 Re-ID 蒐集掃描）
utils/
  model.py                pickle 相容 shim（不動，ADR 0001）
  modules/safetrajai/     第一方套件（由 src/safetrajai 搬入，套件名不變）
  modules/_vendor/trackeval/   vendored TrackEval（由 third_party/trackeval 搬入，匯入名稱仍是 trackeval，內容不改，Q57-b）
scripts/                  CLI 腳本，依功能分資料夾（Q53）
  pipeline/               run_pipeline.py、run_evaluation.py
  grid_search/            grid_search_bytetrack.py、grid_search_botsort.py、grid_search_botsort_reid.py
  homography/             validate_homography.py、migrate_homography_config.py、geocode_cameras.py、check_homography_scale.py、download_satellite.py
  visualization/          visualize_gt_on_video.py、visualize_gt_on_class_video.py
  picker/                 make_roi_picker.py（自 roi/ 搬入，內容不改）；新功能再加 make_picker.py（見 SPEC-features.md）
  reid_data/  reid_train/ （不動）
track_eval.py             根目錄相容入口（不動）
tests/                    smoke、單元測試、golden 比對
config/
  datasets.yaml           全域 dataset 登記表（不動）
  roi/  homography/  homography_checkpoints/   以鏡頭為鍵（不動）
  tracking/               bytetrack、botsort_noreid、botsort_reid、botsort_reid_timm
  grid_search/            grid_search_bytetrack、grid_search_botsort
  reid/                   reid_crops、reid_split、fragment_merge、botsort_save_reid_crops、botsort_save_reid_crops_video
  collision/              collision_detect、conflict_analysis
（third_party/ 目錄於 T2d 移除；vendored TrackEval 見 utils/modules/_vendor/trackeval/）
（docs/ 已由使用者移到 ignore/docs/，含 adr/ 與 道安文件/；ADR 不再放在 repo 內，Q60-a）
Markdown/                 使用者教學文件，依功能模組分資料夾（小寫英文 kebab-case 檔名，內容繁體中文）
  USAGE.md                總入口：簡介、各文件目錄連結、常見情境速查、已知限制
  getting-started/        install.md、data-layout.md
  pipeline/               main-flow.md、stages.md
  roi/                    roi-guide.md
  grid_search/            grid-search.md
  reid/                   reid-data-tools.md
  validation/             validation-and-viz.md
  testing/                testing.md
  workflow/               GITHUB_PUSH_WORKFLOW.md、HF_UPLOAD_WORKFLOW.md、Roboflow.md
  config/                 overview.md、tracking-scenarios.md、reid-configs.md、collision-configs.md、per-camera-configs.md
```

模組就近放置的文件（`tracking/HANDOFF.md`、`reid/HANDOFF.md`、`scripts/reid_data/README.md`、`config/homography_checkpoints/README.md`）不搬進 `Markdown/`（Q28）。之後新增的模組教學文件放進對應資料夾，沒有對應資料夾時再新建。

### 搬離或移除

- `src/` 目錄（內容搬到 `utils/modules/`）。
- `openspec/`：使用者已手動移到 `ignore/`；本輪提交這批刪除。
- `Markdown/` 中其餘 4 個檔案（`HANDOF.MD`、`claude_copy.md`、兩份 `openspec-*總結.md`）：使用者已刪除；本輪提交這批刪除。
- `.claude/commands/opsx/`、`.cursor/commands/opsx-*`、`.cursor/skills/openspec-*` 共 18 個檔案（Q16 決定移除）。
- `docs/`（`adr/` 11 份與 `道安文件/`）、`.codex/`（3 檔）、`專案架構重構.txt`：使用者已手動移到 `ignore/`（gitignore）；本輪提交這批刪除（歷史仍在 git log）。ADR 原文留在 `ignore/docs/adr/`，不在 repo 內（Q60-a）。
- `third_party/`：T2d 把 `trackeval/` 搬到 `utils/modules/_vendor/trackeval/` 後移除空目錄。
- `config/homography_points/`（只剩一份過時 README，程式碼已不讀取）。
- `video/`（3 支 mp4，約 470 MB）：搬入 `dataset/raw/homography_videos/`。其中 C000043、C000085 與 `dataset/raw/videos/<camera>/` 的同名檔案大小不同（97 MB 對 8 MB、61 MB 對 6 MB），C000054 大小相同；一律保留、不去重、不覆蓋，也不併入 `raw/videos/`（會被 Re-ID 蒐集誤掃）。

## 5. 程式風格（Code Style）

沿用既有風格，不引入新慣例。重構時新增的程式碼遵守：
- 函式簽章全數加型別註記；變數用描述性名稱。
- 偏好函式式；三行相似程式碼不抽 helper（避免過早抽象）。
- 程式碼註解用英文；Markdown 與面向使用者的說明用繁體中文。
- 不為顯而易見的程式碼加註解，不為私有函式加 docstring。

範例（抽出共用 helper 時的目標形式）：

```python
def read_video_size(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height
```

## 6. 測試策略（Testing Strategy）

- 每個任務完成後跑 `pytest`；失敗即還原，不修改測試去配合重構。
- **測試檔允許的唯一修改類型：機械式路徑更新**，不改任何斷言：
  - T2：`tests/test_smoke.py:32,35` 的 `src/safetrajai/…` 改為 `utils/modules/safetrajai/…`，`:177` 的掃描根目錄 `"src"` 改為 `"utils/modules"`。
  - T2d：`test_smoke.py` 的 `test_no_sys_path_mutation` 掃描迴圈略過路徑含 `_vendor` 的檔案；斷言不變。
  - T4：`test_reid_dataset_pipeline.py`、`test_reid_fragment_merge.py`、`test_roi_and_layout.py`、`test_reid_manifest.py`、`test_reid_fusion.py` 內寫死的 `CONFIG_DIR / "<name>.yaml"` 改為新子資料夾路徑；`test_readme_coverage.py:22` 的 `glob("*.yaml")` 改為 `rglob("*.yaml")`，維持「README 必須提到每個 config 檔名」的覆蓋。
  - 其餘測試檔內的 openspec 註解（`test_smoke.py:1`、`test_grid_search_core.py:93`）不動。
- CLI 重構（B2）：重構前擷取的 `--help` 與各類別 `--dry-run` 輸出逐位元 diff。
- 抽出共用 helper：先寫特徵化測試（characterization test）鎖定現行行為，再抽取。
- Excel 輸出（B7、B9）：以合成的假資料直接呼叫 `export_*` 函式，逐格比對儲存格值；不讀 `dataset/`，不跑 stage2，特徵化測試留在 repo 內當回歸保護。
- 套件搬動（T2）額外驗證：`import utils.model` 仍可解析、`tracking/config.py` 以 `__file__` 為基準的相對路徑解析正常、`PROJECT_ROOT` 解析結果不變。
- scripts 搬動（T2c）：`pytest` 內 smoke 測試對全部 `scripts/**/*.py` 跑 `--help`；`test_readme_coverage.py` 以 `rglob` 只比對檔名，搬動不影響；`reid_data/`、`reid_train/` 路徑不變，寫死這兩個目錄的測試不動。
- `USAGE.md` 拆分（T3b）：只做機械式搬移（整節剪貼、重新編號、修連結），不改寫句子、不補新內容、不刪內容（第 0 節除外）。驗證：拆分後各檔內容合併，扣掉編號與連結行，應與原檔（扣掉第 0 節）逐行一致。
- 文件變更不需要測試，但要確認沒有其他檔案引用被刪除的路徑。

## 7. 範圍與任務順序（Scope）

### 7.1 任務順序（Q25、Q56）

每個任務一個 commit；commit 前跑 `pytest` 並出示 diff 摘要。

**實際執行順序**（先做所有結構搬動，最後才同步文件，避免文件改兩遍）：
T0 → T1 → T2 → T2d → T2b → T2c → T4 → T3 → T3b → T4b → T5 → T6 → T7 → T8 → T9。
T2c、T4 的 commit 之間，文件裡的指令路徑暫時是舊的（`pytest` 不受影響，覆蓋測試只比對檔名）；整條分支合併前由 T3 補齊。T3 一次對齊 `src`、`scripts`、`config` 三種新路徑。

新功能（網頁選點器、計數線、鳥瞰影片、盲區）不在本規格範圍，另見 `SPEC-features.md`，於 T9 之後才開始（Q36）。

| # | 任務 | 內容 | 驗證 |
|---|---|---|---|
| T0 | 建分支與規格 | 從 `main` 開新分支 `refactor/simplify-codebase`；提交 `SPEC.md`、`SPEC-features.md`、`tasks/plan.md`、`tasks/todo.md`；只 `git add` 自己改的檔案，不用 `git add -A`，不碰 `skills-lock.json` 與其他未追蹤檔 | — |
| T1 | 移除已搬走的內容 | `git add -u -- openspec Markdown docs .codex "專案架構重構.txt"` 提交刪除（ADR 移出 repo 屬 Q60-a）；`git rm` 18 個 opsx／openspec 工具檔（另一個 commit）；移除 `.gitignore:43-44` | `pytest` |
| T2 | 搬動套件 | `git mv src/safetrajai utils/modules/safetrajai`；更新 `pyproject.toml`（package-dir、find 的 where 改 `utils/modules`）、`paths.py`（`parents[2]` → `parents[3]`）、`track_eval.py`、`test_smoke.py` 3 行；重裝 editable | `pytest`、`import utils.model`、`--help` 前後比對 |
| T2d | 搬移 vendored TrackEval | `git mv third_party/trackeval utils/modules/_vendor/trackeval`，**不改名、不改任何 vendored 檔案內容**；`pyproject.toml` 改為 `trackeval = "utils/modules/_vendor/trackeval"` 並移除 `where` 中的 `third_party`；`tests/test_smoke.py` 的 `test_no_sys_path_mutation` 掃描迴圈加「路徑含 `_vendor` 就略過」（5 個 vendored 檔案含 `sys.path.insert`，斷言不變）；新增上游 MIT `LICENSE` 放在 `_vendor/trackeval/`（內容必須與上游一字不差；抓不到原文就不放，記入待辦，不自己編寫）；重裝 editable。兩個 commit：(1) 搬動＋pyproject＋測試豁免；(2) LICENSE | `pytest`、`python -c "import trackeval; print(trackeval.__file__)"` 指向新路徑、`git diff -M --stat` 顯示 vendored 檔案純 rename（相似度 100%）、確認 `trackeval_runner.py` 顯式指定所有輸入輸出路徑（不依賴 `trackeval.utils.get_code_path()`） |
| T2b | 資料目錄整理 | 把 `video/` 三支 mp4 搬到 `dataset/raw/homography_videos/<camera>/`（同一 NFS 掛載內 rename，不讀內容）；`pick_aid.py:107,111,115` 的 3 個路徑與 `.gitignore:8`（`video/`）同步更新。流程：先貼出 `mkdir`＋3 個 `mv` 與還原指令 → 使用者同意後執行 → 只比對路徑清單、檔案數與 `stat` 大小。資料搬動不進 git，單一 commit 只含 `pick_aid.py` 與 `.gitignore`，資料搬動的指令與結果記在 commit message | 路徑清單與大小前後相同；`USABLE_INTERSECTIONS` 4 個路徑 `Path.exists()` 皆為真；`pytest`。不跑 `validate_homography.py`（會解碼影片） |
| T2c | scripts 分資料夾 | 依 4.1 `git mv` 腳本（`roi/` 併入 `picker/`）；不保留舊路徑相容層；兩個 commit：(1) 搬動腳本，並機械式更新 `tests/golden/run_all.sh` 的 2 個預設路徑（第 10、11、40、41 行，不改任何邏輯）；(2) 更新 `config/homography/*.yaml` 檔頭註解中的腳本路徑（約 29～32 個檔案，只改註解） | `pytest`（smoke 的 `scripts/**/*.py --help` 涵蓋全部腳本）、`bash -n tests/golden/run_all.sh` 且兩個預設路徑的檔案存在、每個被改的 yaml 在修改前後 `yaml.safe_load` 結果相同、`load_camera_config` 載入結果不變 |
| T3 | 功能型文件同步路徑 | 先 `git mv` 三份流程文件到 `Markdown/workflow/`（commit 一）；再更新 `CLAUDE.md`、`README.md`（含 `:544` 指向 `workflow/HF_UPLOAD_WORKFLOW.md`）、`Markdown/USAGE.md`、`Markdown/workflow/*.md`、`scripts/reid_data/README.md`、`.claude/commands/tracker-change.md`、`tracking/HANDOFF.md`、`reid/HANDOFF.md` 內的 `src/safetrajai` 路徑與 openspec 引用（commit 二）；只改失效的路徑與引用，其餘內容不動 | `pytest`（含 README 覆蓋測試） |
| T3b | 拆分 `USAGE.md` | 依 4.1 對照拆成各模組資料夾；第 0 節（兩次變更的重點）整節刪除，內容留在 git 歷史；各檔內章節重新從 1 編號；`README.md:500` 改指向 `Markdown/roi/roi-guide.md`，`USAGE.md` 內 2 處「見 5.2」交叉引用改成相對連結；排在 T2、T3 之後，讓 diff 只含搬移。一個 commit | 拆分前後正文行集合比對、`pytest`（含 README 覆蓋測試） |
| T4 | config 分資料夾 | 依 4.1 搬 yaml；改 `--config` 預設值（`cli/run_pipeline.py`、`run_evaluation.py`、3 支 grid search 腳本）、`config_loader` 與 `paths` 內路徑（文件中的指令範例留到 T3 一次對齊）；同步更新第 6 節列出的測試路徑；移除 `config/homography_points/`；不保留舊路徑相容層 | `pytest`、`--help` 只有預設路徑字串不同、`--dry-run` 組合數一致 |
| T4b | config 使用教學 | 新增 `Markdown/config/`：`overview.md`、`tracking-scenarios.md`、`reid-configs.md`、`collision-configs.md`、`per-camera-configs.md`。任務導向（我想做什麼 → 改哪個檔、哪個鍵 → 用哪個指令驗證）；情境檔以「與 `bytetrack.yaml` 的差異」描述；與 `grid_search/grid-search.md`、`roi/roi-guide.md` 重疊處用連結。這是新寫的文件，不是搬移。一個 commit | 文件中的每個鍵都能在 loader 或 yaml 找到；指令範例以 `--help`／`--dry-run` 驗證可執行；`pytest` |
| T5 | A 級零風險清理 | A1–A8，每個子項一個 commit | `pytest`、`--help` 與 `--dry-run` 比對 |
| T6 | B 級去重複 | B2（新增 `grid_search/cli.py`）→ B3／B4 → B1 → B5 → B6–B10；B7／B9 先補特徵化測試 | `pytest`＋特徵化測試 |
| T7 | 視覺化顏色修正 | Q4(b)：獨立 commit，標明這是行為變更；先寫失敗測試再修 | 新增測試 |
| T8 | 雜項 | `git mv ".claude/agents/project-manger copy.md" ".claude/agents/qa-engineer.md"`，內容不動 | `pytest` |
| T9 | 交付報告 | 淨減行數、D 級清單、未跑 golden 的聲明 | — |

### 7.2 A 級：零風險清理（T5）

| 編號 | 項目 | 位置 |
|---|---|---|
| A1 | 刪除被註解掉的舊 `PARAM_GRIDS`（約 80 行） | `scripts/grid_search_botsort.py` |
| A2 | 修正失效 docstring：檔名 `tracker_grid_search.py`、不存在的 `run_class_aware_tracking`、`--n-jobs`、`--video-dir`；help 的預設值錯字（botsort 的 `--gmc-method` 實為 `sift`、`--max-videos` 實為 6） | 三支 `grid_search_*.py` |
| A3 | 修正 `score_params()` docstring（實際是第 25 百分位並排除 0 分，非平均值） | `grid_search/core.py` |
| A4 | 刪除未使用參數 `trial_dir`；`generate_config_patch()` 內相同的 `bool`／`else` 分支合併；函式內重複 import 移到頂部 | `grid_search/core.py` |
| A5 | 修正過時敘述：「兩階段主流程」、`_annotate_frame` 引用的 `utils.visualization`、`target_mp4s = mp4s[:]` 的空操作 | `tracking_pipeline.py`、`sequence_tracker.py` |
| A6 | 刪除註解掉的程式碼與貼上的舊結果表：`pipeline.py` 尾段（約 40 行）；`excel_export.py` 末段被註解掉的 `print_eval_summary`（範圍實作時確認） | `evaluation/pipeline.py`、`evaluation/excel_export.py` |
| A7 | `_autosize_columns` 迴圈內的 `import` 移到檔案頂部 | `evaluation/utils.py:89` |
| A8 | 失效引用清理（取代原本「改指向 openspec spec」的做法，因 `openspec/` 已移除）：刪除程式碼註解與設定註解中的「見 openspec／docs/superpowers…」片語，其餘句子保留。涵蓋 `docs/superpowers`（`config/datasets.yaml:4`、`evaluation/dataset_layout.py:6`、`grid_search/core.py:6,11`）與 `openspec`（`reid/dataset_index.py:1`、`config/reid_crops.yaml:13`、`tracking/class_aware_botsort_flow.md:122`）。`datasets.yaml` 與 `reid_crops.yaml` 只改註解。另含不存在的 `markdown/reference_code/`、`markdown/reference_results/` 引用共 7 處（`collision/output.py:11`、`collision/detection_core.py:4`、`collision/homography/__init__.py:4`、`compare_video.py:7`、`heatmap.py:6,134`、`validation.py:180`），同樣只刪引用片語，其餘句子保留。另含 `docs/adr` 引用（ADR 已移出 repo，Q60-a）：`config/roi/*.yaml` 10 檔註解、`scripts/reid_data/` 6 檔、`scripts/roi/make_roi_picker.py`、`collision/trajectory_adapter.py`、`evaluation/{config_loader,ignore_region,mot_format,trackeval_runner}.py`、`reid/auto_exclusion.py`、`roi.py`、`reid/HANDOFF.md`、`tracking/class_aware_botsort_flow.md`、`scripts/reid_data/README.md`；同樣只刪「見 docs/adr/…」片語。**排除**：`utils/model.py`、`tracking/class_aware_*.py`（凍結）、`_vendor/trackeval/`、`tests/` 與 `tests/golden/`。CLAUDE.md 等功能型文件中的 ADR 引用在 T3 處理：凍結區規則改為直接寫出理由，不再指向 ADR 檔 | 見左 |

不改：測試檔註解、`tests/golden/` 底下的說明（凍結範圍）。

### 7.3 B 級：消除重複邏輯（T6）

| 編號 | 項目 | 說明 |
|---|---|---|
| B1 | 抽出共用的 label 寫出邏輯 | `sequence_tracker._process_frame` 與 `core._run_tracking_for_video` 各有一份相同的 YOLO label 輸出區塊 |
| B2 | 抽出共用的 grid search CLI | 新增 `grid_search/cli.py`（`core.py` 不動，已 796 行）；三支腳本 `main()` 後段約 100 行逐字相同；各腳本只保留 `PARAM_GRIDS`、預設值與 botsort 專屬參數（如 `--conf`，由各腳本呼叫時傳入） |
| B3 | 抽出影片尺寸讀取 helper | `tracking_pipeline`（2 處）、`core.score_params`、`run_stage3` 共 4 處重複 |
| B4 | 抽出序列檔名縮短邏輯 | 寫死 `" 中華三路 民生二路(東北角)_"` 的 `replace` 在 `sequence_tracker.py`、`core.py` 各一份，必須維持同一結果 |
| B5 | 合併兩支視覺化腳本的共用部分 | `CLASS_COLORS`、`_color_for_id`、`parse_seqinfo`、`draw_gt_on_frame`；**先逐位元保留現行行為**，顏色修正走 T7 |
| B6 | `stage3` 的 `_ensure_stage1_output_exists()` 後緊接重複掃描 `list_stage1_sequences` | 保留 soft-skip 行為 |
| B7 | `key_means` 與 `key_mid` 兩段近乎相同（只差後綴） | `excel_export.py:240-262`；儲存格內容須逐格相同 |
| B8 | 預設類別清單寫死，與 `classes.py` 契約重複 | `excel_export.py:174`；順序相同，改由 `classes.py` 推導；實作前先確認匯出方式 |
| B9 | DET 曲線輸出：巢狀 `downsample_idx` 與重複的 `len(data)` 分支 | `excel_export.py:354-384`；先寫特徵化測試 |
| B10 | `file_md5` 與 `md5_file` 功能重疊 | `evaluation/utils.py:96,107`；傳回值型別不同，需先確認呼叫端，不能直接合併；確認後若無法安全合併則不做 |

### 7.4 文件與雜項處理

| 項目 | 決定 |
|---|---|
| 功能型文件（Q18、Q24） | 本輪維護：T3 清單內的文件；`Markdown/` 依 4.1 建立目錄層次並拆分 `USAGE.md`（T3b，Q26–Q31） |
| 其他文件 | **延後**到之後另開的文件盤點，本輪不碰：已移至 `ignore/` 的 `docs/`、`Handoff/`、`config/homography_points/README.md`（目錄本身在 T4 移除）、`專案架構重構.txt` |
| 根目錄 `locale.sh`、`codegraph.json`、`專案架構重構.txt`、`skills-lock.json` | 全部保留，本輪不刪除（`skills-lock.json` 有使用者的未提交修改，不屬於本次重構） |
| `.claude/agents/project-manger copy.md` | 實為 `qa-engineer` agent（檔名錯誤），T8 改名，內容不動 |
| `.claude`／`.cursor` opsx 檔 | 不再「保留」：Q16 決定移除（T1） |
| README／USAGE／CLAUDE 內容重疊 | 不處理（無複製貼上，README 受覆蓋測試約束） |

### 7.5 D 級：疑似死碼（全部保留）

以下 9 個符號在全部 `.py` 中只出現在自己的定義處。因為要求「保留所有既有功能與介面」，**全部保留**；清單寫進 T9 交付報告，由使用者日後決定。

- `save_homography`、`transform_points_batch`、`warp_and_overlay`（`collision/homography/__init__.py`）
- `list_available_tags`、`load_prebuilt_homography`、`save_verification_image`（`collision/homography/points.py`）
- `merge_map`（`reid/fragment_merge.py:253`）
- `keep_mot_gt_valid_ids`（`evaluation/mot_converter.py:413`）
- `clip_xywh`（`evaluation/utils.py:192`）

動態呼叫、notebook 或外部腳本無法由靜態掃描排除，所以不刪。

### 7.6 E 級：長函式拆分（本輪排除）

`ul_track_labels_to_mot`（219 行）、`run_eval_pipeline`（160）、`run_trackeval_mot`（151）、`config_loader._load`（128）、`train`（124）等。需要大量特徵化測試加 golden 比對，列為日後獨立 change。

### 7.7 刻意不處理（附理由）

- `class_aware_botsort.py` 與 `class_aware_bytetrack.py` 的逐字重複（`remove_duplicate_stracks`、`_center_distance`）：ADR 0004 凍結區，不去重。
- `collision/__init__.py` 與 `conflict/__init__.py` 的 6 行「首幀熱區圖」相似區塊：三行相似優於過早抽象。
- `evaluation/utils.get_video_wh()`：解碼第一幀取尺寸，語意與 `CAP_PROP_*` 不同，不併入 B3。
- `reid/model.py` 與 `reid/training/model.py` 同名異用途：改名會影響 shim，不改。
- `tests/golden/configs/` 的凍結副本：不合併、不隨 T4 移動。
- 架構優化幅度（Q10）：分層本身已清楚，僅做小範圍職責整理（`grid_search/cli.py`）加上 T2、T4 的目錄整理，不再搬動其他套件。

## 8. 邊界（Boundaries）

### 一定要做（Always）
- 一個任務一個 commit（conventional commits，英文）；commit 前先跑 `pytest` 並出示 diff 摘要。
- 在新分支 `refactor/simplify-codebase` 上作業；只 `git add` 自己改的檔案，絕不使用 `git add -A`。
- 動手前先把基準輸出（`--help`、`--dry-run`）存到 scratchpad。
- 保留 `--model` 參數（三支腳本都未使用，只把 help 改為「目前無作用」）。
- 保留 botsort `PARAM_GRIDS` 中 `model` 候選重複列出的 `yolo11x.pt`（去重會改變 `--dry-run` 組合數與 `combo_id`，影響 `--resume`）。
- 保留各腳本目前不一致的寫死值：bytetrack `imgsz=1280`／`gmc="none"`、botsort_reid `imgsz=1280`／`gmc="sparseOptFlow"`、botsort 由 CLI 提供（`gmc` 預設 `sift`）。
- 重構 `VideoWriterWithFfmpeg`、`score_params` 等既有怪異行為時只搬移、不修正（fps 寫死 5、25 百分位排除 0 分）。
- 新增、改名或刪除 `scripts/` 腳本或 `config/**/*.yaml` 時，同步更新 `README.md`。

### 先問再做（Ask first）
- 跑 golden 比對（本輪預設不跑；`pytest` 或 `--help` 比對出現任何差異時先停下來問）。
- 本規格未列出的檔案刪除。
- 新增或移除依賴；本規格 T2 以外的 `pyproject.toml` 變更。
- 任何會改變輸出內容的修正（視覺化顏色修正只在 T7 做，須標明為行為變更）。
- 任何超出第 6 節「機械式路徑更新」的測試修改。

### 絕對不做（Never）
- 修改 `ClassAwareBOTSORT`／`ClassAwareBYTETracker` 中 override 的方法（ADR 0004、0009），也不搬動或抽成共用函式；檔案可以搬位置，只能改檔案層級的 import。
- 修改或格式化 vendored TrackEval（`utils/modules/_vendor/trackeval/`）的任何既有檔案內容；唯一允許的是 T2d 的 `git mv` 與新增 `LICENSE`。刪除、改名或搬動 `utils/model.py`。
- 更動 `config` 的區塊名稱，或覆寫既有基準情境。
- 對 `dataset/`、`models/`、`runs/` 做全量讀取（`sha256sum`、`du`、`find -exec cat` 等）。
- 刪除、覆蓋或去重 `video/` 的影片；把影片放進 `dataset/raw/videos/<camera>/`；未先給使用者看過指令與還原方式就執行任何資料搬動。
- 為了讓測試通過而修改斷言、跳過或刪除既有測試。
- 修改評估資料集的人工 GT。
- 修改 `tests/golden/` 底下的基準設定、`BASE_COMMIT`、容忍度或說明文件，也不把 golden config 與 `config/` 合併。唯一例外：T2c 對 `tests/golden/run_all.sh` 的 2 個預設腳本路徑做機械式更新（Q54）。
- 未經使用者逐項同意，刪除 D 級疑似死碼。
- 建立新的 README 或額外文件（`SPEC.md` 與 `tasks/` 除外）。
- 提交 `skills-lock.json` 的未提交修改，或其他不屬於本次重構的未追蹤檔（`Handoff/`、`docs/道安文件/`、`report-*.html`、`tools/favorites install.v18.sh`）。

## 9. 成功標準（Success Criteria）

1. `pytest` 全數通過（基準 739 passed, 11 skipped；新增特徵化測試後 passed 數只增不減）；既有測試檔只有第 6 節列出的機械式路徑更新，斷言零修改。
2. 三支 grid search 腳本：重構前後 `--help` 逐位元相同（T4 之後允許 `--config` 預設路徑字串不同，其餘逐字相同）；每個類別與 `_default` 的 `--dry-run` 輸出逐位元相同。
3. 第一方程式碼淨減少行數，且無新增跨模組循環 import；`scripts/**/*.py --help` 的 smoke 測試維持通過。
4. `import utils.model` 與 `from safetrajai… import …` 在任意工作目錄下仍可解析；`PROJECT_ROOT` 解析結果與搬動前相同。
5. 每個任務（T1–T8）與每個 A／B 子項有獨立 commit，可單獨還原。
6. 第 8 節「絕對不做」清單零違反（以 `git diff --stat` 確認未觸及凍結檔案，並確認 tracker override 的 AST 檢查仍通過）。
7. `git grep -E "docs/superpowers|openspec|markdown/reference_"` 在 `utils/modules/`、`config/`、`scripts/` 內零命中；`git grep "src/safetrajai"` 在 T3 清單的文件內零命中。
7a. T2b 完成後：`video/` 目錄不存在；`dataset/raw/homography_videos/` 內 3 支檔案的 `stat` 大小與搬動前相同；`git grep -n "video/C0"` 在程式碼內零命中；`dataset/raw/videos/` 內容與搬動前相同。
7e. T2d 完成後：`third_party/` 不存在；`import trackeval` 解析到 `utils/modules/_vendor/trackeval/`；vendored 檔案內容與搬動前逐檔相同（`git diff -M` 相似度 100%）；`git grep "third_party"` 在 `utils/modules/`、`config/`、`scripts/`、`pyproject.toml` 內零命中；`git grep -E "docs/adr"` 在 `utils/modules/`（扣除 `utils/model.py`、`tracking/class_aware_*.py`、`_vendor/`）、`config/`、`scripts/` 內零命中。
7d. T2c 完成後：`scripts/` 根目錄下沒有 `.py` 檔案；`bash -n tests/golden/run_all.sh` 通過且預設路徑的檔案存在；`config/homography/*.yaml` 在修改前後 `safe_load` 結果逐檔相同；`git grep -E "scripts/(grid_search_|validate_homography|geocode_cameras|check_homography_scale|migrate_homography_config|download_satellite|run_pipeline|run_evaluation|visualize_gt)"` 在 `utils/modules/`、`config/`、`scripts/` 內零命中。
7c. T4b 完成後：`Markdown/config/` 的 5 份文件存在；文件中列出的 config 鍵與指令都經過驗證；文件對既有模組文件的連結皆有效。
7b. T3b 完成後：`Markdown/` 的檔案結構與 4.1 一致；`USAGE.md` 與各模組文件之間、`README.md` 對 `Markdown/` 的連結全部指向存在的檔案；拆分前後正文行集合比對通過。
8. golden 本輪未跑，交付報告明確標示「未跑」及理由。

## 10. 已決定事項與殘餘待確認（Decisions）

| 編號 | 決定 |
|---|---|
| Q1 | 新分支 `refactor/simplify-codebase`，一任務一 commit，精確 `git add` |
| Q2 | D 級 9 個符號全部保留，列入交付報告 |
| Q3 | E 級長函式拆分整輪排除 |
| Q4 | (b) B5 先逐位元保留，顏色修正為獨立 commit（T7），標明行為變更 |
| Q5 | `project-manger copy.md` 改名為 `qa-engineer.md`；根目錄雜項全部保留 |
| Q6 | 原選「改寫 README」，被 Q13／Q18 取代：延後到文件盤點，目錄於 T4 移除 |
| Q7 | A8 改為「刪除引用片語」（openspec 已移除） |
| Q8 | 共用 CLI 放新檔 `grid_search/cli.py` |
| Q9 | Excel 驗證用合成資料特徵化測試 |
| Q10 | 架構幅度：小範圍職責整理，加 T2、T4 目錄整理 |
| Q11–Q14 | 套件搬動、`test_smoke.py` 路徑更新、只維護功能型文件、搬動最先做且不跑 golden（Q11-a 的位置被 Q21 取代） |
| Q15 | 提交 openspec／Markdown 刪除，與工具檔移除分成兩個 commit |
| Q16 | 移除 opsx／openspec 工具檔 18 個 |
| Q17 | 註解引用只刪片語；測試與 golden 說明不改 |
| Q18／Q24 | 功能型文件維護清單如 T3；其他文件延後盤點；`Markdown/` 只保留 4 份 |
| Q19／Q21 | 套件最終位置 `utils/modules/safetrajai/`；`utils/model.py` 留原處；`scripts`、`track_eval.py` 不搬出原目錄（`third_party/trackeval` 由 Q57 改搬） |
| Q20／Q23 | config 只做 (a) 清冗餘並依功能分資料夾；不保留舊路徑相容層；掃描「已無人讀取的鍵」並列清單給使用者確認後才刪 |
| Q22 | 測試允許機械式路徑更新（含 `rglob`），斷言不改 |
| Q25 | 任務順序如 7.1；動手前先存 `--help`／`--dry-run` 基準到 scratchpad |
| Q26 | `Markdown/` 依功能模組分資料夾；`USAGE.md` 留根目錄當總入口；三份流程文件搬到 `workflow/`；只建立有檔案要放的資料夾 |
| Q27 | (b) 這輪就拆 `USAGE.md`（取代 Q18 的「拆分延後」） |
| Q28 | 就近放置的模組文件不搬進 `Markdown/` |
| Q29 | 拆分對照如 4.1；各檔內重新編號；`README.md:500` 與 `USAGE.md` 內交叉引用改連結 |
| Q30 | 第 0 節整節刪除（內容留在 git 歷史） |
| Q31 | 只做機械式搬移；T3 先 `git mv`，T3b 一個 commit；以正文行集合比對驗證 |
| Q32 | (a) `video/` 搬到 `dataset/raw/homography_videos/<camera>/`；只處理 `dataset/` 與 `video/`，`models/`、`runs/`、`Sub_Module/`、`ignore/` 不動；全部保留不去重 |
| Q33 | 資料搬動先貼指令與還原方式、同意後才執行；只驗路徑清單、檔案數、`stat` 大小；排在 T2 之後（T2b） |
| Q34 | `Markdown/config/` 5 份任務導向教學文件（新增任務 T4b，排在 T4 之後） |
| Q35 | 任務順序：T0 → T1 → T2 → T2b → T3 → T3b → T4 → T4b → T5 → T6 → T7 → T8 → T9 |
| 回覆 | config 會依功能模組分資料夾（T4，沿用 Q23 的對照） |
| Q36–Q52 | `Sub_Module/` 盤點後的新功能（選點器、計數線、鳥瞰影片、盲區）全部寫入 `SPEC-features.md`，重構 T9 後才開始；`Sub_Module/` 本身不動 |
| Q53 | `scripts/` 依功能分資料夾（對照如 4.1）；`roi/` 併入 `picker/`；不保留舊路徑相容層 |
| Q54 | 允許機械式更新 `tests/golden/run_all.sh` 的 2 個預設腳本路徑 |
| Q55 | 一併更新 `config/homography/*.yaml` 檔頭註解中的腳本路徑，以 `safe_load` 前後相同驗證 |
| Q56 | 執行順序改為先結構搬動、後文件同步（見 7.1），T3 一次對齊三種新路徑 |
| Q57 | (b) vendored TrackEval 搬到 `utils/modules/_vendor/trackeval/`，不改名、不改內容；`pyproject.toml` 以 `trackeval = "utils/modules/_vendor/trackeval"` 對映（T2d）。不選改名／併入 `safetrajai` 命名空間：必須修改 8 個含絕對匯入 `from trackeval…` 的 vendored 檔案，違反唯讀 |
| Q58 | 允許 `test_no_sys_path_mutation` 略過 `_vendor/`（機械式，斷言不變）；搬完驗證 runner 不依賴 `get_code_path()` |
| Q59 | 補上游 MIT `LICENSE`，內容須與上游一字不差，抓不到就不放 |
| Q60-a | 使用者有意把 `docs/`（ADR 與 道安文件）移出 repo：ADR 不再版控；程式碼與 config 註解中的 `docs/adr` 引用比照 openspec 清理（A8）；功能型文件的凍結區規則改為直接寫理由；新功能不新增 ADR 0012，決策記在 `SPEC-features.md` |

殘餘待實作時確認（不阻擋規格）：
1. A6 `excel_export.py` 末段被註解掉的程式碼的確切範圍。
2. B8 的 `classes.py` 匯出方式。
3. B10 是否能安全合併（呼叫端確認後決定，不能就不做）。
4. T4 中「已無人讀取的 config 鍵」掃描結果，需使用者確認後才刪除。
