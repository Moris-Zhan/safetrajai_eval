# 任務清單：safetrajai_eval 精簡重構

> **狀態：已全部完成並合併進 `main`（2026-10-29，merge commit `8b5eb4e`）。** 結果與交付細節見 `tasks/delivery-report.md`；新功能的計畫在 `tasks/plan-features.md`、`tasks/todo-features.md`。

> 依據：`SPEC.md`、`tasks/plan.md`。每個任務一個 commit（英文 conventional commits，結尾附 Co-Authored-By 與 Claude-Session 兩行）；commit 前跑 `pytest` 並出示 diff 摘要；只 `git add` 自己改的檔案。
> 規模：XS 1 檔／S 1–2 檔／M 3–5 檔。標示「機械式」的任務檔案數較多，但只做字串替換或 `git mv`，以 `git grep` 驗證取代逐檔審閱。

通用驗證（每個任務都做）：
- [x] `pytest`：739 passed, 11 skipped（新增特徵化測試後 passed 數只增不減；收尾時 790 passed, 11 skipped）
- [x] `git diff --stat` 確認未觸及凍結檔案（`tracking/class_aware_*.py` 的 override 方法、`utils/model.py`、vendored TrackEval 內容、`tests/golden/` 基準）

---

## 階段 0：準備

### T0：建分支、提交規格、擷取基準
- [x] 從 `main` 建 `refactor/simplify-codebase`
- [x] 擷取基準輸出到 scratchpad（不放進 repo）：三支 grid search 的 `--help`、六類加 `_default` 的 `--dry-run`、`track_eval.py --help`、`scripts/run_evaluation.py --help`、`PROJECT_ROOT` 與 `CONFIG_DIR` 解析結果
- [x] commit：`SPEC.md`、`SPEC-features.md`、`tasks/plan.md`、`tasks/todo.md`
- 驗收：分支存在；基準檔在 scratchpad；commit 只含這四個檔案
- 依賴：無｜規模：S

---

## 階段 1：結構搬動（行為不變）

### T1a：提交已搬走內容的刪除
- [x] `git add -u -- openspec Markdown docs .codex "專案架構重構.txt"`，確認暫存清單只含刪除
- [x] commit：`chore: remove openspec, legacy docs and ADRs moved to ignore/`
- 驗收：`git status` 不再顯示這些路徑；`skills-lock.json` 與其他未追蹤檔仍未暫存
- 驗證：`pytest`
- 依賴：T0｜規模：機械式

### T1b：移除 opsx／openspec 工具檔與 gitignore 殘留
- [x] `git rm` 18 個檔案：`.claude/commands/opsx/*`（6）、`.cursor/commands/opsx-*`（6）、`.cursor/skills/openspec-*`（6）
- [x] 移除 `.gitignore:43-44`（`openspec/changes/...` 兩行）
- [x] commit：`chore: remove opsx tooling and openspec gitignore entries`
- 驗收：`git ls-files | grep -iE "opsx|openspec"` 無輸出
- 依賴：T1a｜規模：機械式

### T2：搬動套件 `src/safetrajai` → `utils/modules/safetrajai`
- [x] `git mv src/safetrajai utils/modules/safetrajai`，移除空的 `src/`
- [x] `pyproject.toml`：`[tool.setuptools.package-dir]` 的 `"" = "src"` 改 `"utils/modules"`；`packages.find.where` 的 `src` 改 `utils/modules`
- [x] `utils/modules/safetrajai/paths.py`：`parents[2]` → `parents[3]`，註解更新
- [x] `track_eval.py` 內的路徑說明
- [x] `tests/test_smoke.py`：第 32、35 行 `src/safetrajai/…` → `utils/modules/safetrajai/…`；第 177 行掃描根目錄 `"src"` → `"utils/modules"`
- [x] 重裝：`pip install -e . --no-deps`（只動 conda env `track_eval`）
- 驗收：
  - [x] `PROJECT_ROOT`、`CONFIG_DIR` 解析結果與 T0 基準相同
  - [x] `import utils.model`、`import safetrajai` 在 `/tmp` 目錄下可解析
  - [x] `--help` 與 `--dry-run` 與基準逐位元相同
  - [x] `tracking/config.py` 以 `__file__` 為基準的相對路徑解析正常（跑 `pytest tests/test_track_config*`）
- 依賴：T1b｜規模：M（改 4 個檔案＋大量 rename）

### T2d：搬動 vendored TrackEval
- [x] 提醒使用者：補 LICENSE 需要一次對外網路請求
- [x] commit 1：`git mv third_party/trackeval utils/modules/_vendor/trackeval`（不改名、不改內容）；`pyproject.toml` 的 `trackeval = "utils/modules/_vendor/trackeval"`、`where` 移除 `third_party`；`tests/test_smoke.py` 的 `test_no_sys_path_mutation` 迴圈略過路徑含 `_vendor` 的檔案；移除空的 `third_party/`；重裝 editable
- [x] commit 2：補上游 MIT `LICENSE` 到 `utils/modules/_vendor/trackeval/LICENSE`（內容須與上游一字不差；抓不到就不放，記入待辦）（已完成：`e53ce62`）
- 驗收：
  - [x] `git diff -M --stat`：vendored 檔案相似度 100%
  - [x] `python -c "import trackeval; print(trackeval.__file__)"` 指向新路徑
  - [x] `evaluation/trackeval_runner.py` 顯式指定所有輸入輸出路徑，不依賴 `trackeval.utils.get_code_path()`
  - [x] `git grep third_party` 在 `utils/modules/`、`config/`、`scripts/`、`pyproject.toml` 零命中
- 依賴：T2｜規模：S（改 2 個檔案＋rename）

### T2b：搬動 `video/` → `dataset/raw/homography_videos/<camera>/`
- [x] **先把將執行的指令與還原指令貼給使用者，同意後才執行**（`mkdir` ＋ 3 個 `mv`；同一掛載內 rename，不讀內容）
- [x] 記錄搬動前 `stat` 大小：C000043 97,456,550、C000054 311,221,801、C000085 61,385,808
- [x] 更新 `utils/modules/safetrajai/collision/homography/pick_aid.py` 的 3 個路徑；移除 `.gitignore:8`（`video/`）
- [x] commit（只含 `pick_aid.py` 與 `.gitignore`）：資料搬動的指令與結果寫進 commit message
- 驗收：
  - [x] `video/` 不存在；新位置 3 個檔案 `stat` 大小與搬動前相同
  - [x] `dataset/raw/videos/` 內容與搬動前相同（只比對路徑清單與檔案數）
  - [x] `USABLE_INTERSECTIONS` 4 個路徑 `Path.exists()` 皆為真
  - [x] 不跑 `validate_homography.py`（會解碼影片）
- 依賴：T2｜規模：XS（2 個檔案）

### T2c：`scripts/` 依功能分資料夾
- [x] commit 1：`git mv` 依 SPEC 4.1 對照搬動（`pipeline/`、`grid_search/`、`homography/`、`visualization/`、`picker/`，`roi/` 併入 `picker/`）；機械式更新 `tests/golden/run_all.sh` 第 10、11、40、41 行的 2 個預設路徑
- [x] commit 2：更新 `config/homography/*.yaml` 檔頭註解中的腳本路徑（約 29～32 個檔案，只改註解）
- 驗收：
  - [x] `scripts/` 根目錄無 `.py`
  - [x] smoke 對全部 `scripts/**/*.py` 的 `--help` 通過
  - [x] `bash -n tests/golden/run_all.sh` 通過，兩個預設路徑的檔案存在
  - [x] 每個被改的 yaml 在修改前後 `yaml.safe_load` 結果相同；`load_camera_config` 載入結果不變
- 依賴：T2b｜規模：機械式

### T4：`config/` 依功能分資料夾
- [x] 盤點：掃描 config 中「已無人讀取的鍵」，**列出清單給使用者確認**後才刪（未確認的一律保留）
- [x] commit 1：`git mv` 依 SPEC 4.1 搬 yaml（`tracking/`、`grid_search/`、`reid/`、`collision/`）；更新 `--config` 預設值（`cli/run_pipeline.py`、`scripts/pipeline/run_evaluation.py`、3 支 grid search）、`config_loader` 與 `paths` 內路徑；同步更新第 6 節列出的 6 個測試檔的路徑與 `test_readme_coverage.py` 的 `rglob`；`README.md` 檔名覆蓋維持；移除 `config/homography_points/`
- [x] commit 2（若有使用者確認的鍵）：刪除未使用的鍵
- 驗收：
  - [x] `--help` 只有預設路徑字串不同；`--dry-run` 組合數與基準一致
  - [x] smoke 載入全部 config 通過
  - [x] `tests/golden/configs/` 凍結副本未動
- 依賴：T2c｜規模：機械式（搬動）＋M（程式碼與測試）

### 檢查點 A（T4 完成後）
- [x] `pytest` = 739 passed, 11 skipped；三種 import 在任意工作目錄可解析
- [x] `--help`／`--dry-run` 與基準相同（T4 之後允許 `--config` 預設路徑字串不同）
- [x] `git diff main --stat` 未觸及凍結檔案；tracker override 的 AST 檢查通過
- [x] **與使用者一起審閱再繼續**（使用者確認後進入階段 2）

---

## 階段 2：文件

### T3a：把三份流程文件搬進 `Markdown/workflow/`
- [x] `git mv` `GITHUB_PUSH_WORKFLOW.md`、`HF_UPLOAD_WORKFLOW.md`、`Roboflow.md` 到 `Markdown/workflow/`
- 驗收：三個檔案在新位置
- 依賴：檢查點 A｜規模：XS

### T3b：同步功能型文件的路徑與引用
- [x] 更新 `CLAUDE.md`、`README.md`（含 `:544` 指向 `workflow/HF_UPLOAD_WORKFLOW.md`）、`Markdown/USAGE.md`、`Markdown/workflow/*.md`、`scripts/reid_data/README.md`、`.claude/commands/tracker-change.md`、`tracking/HANDOFF.md`、`reid/HANDOFF.md`：`src/safetrajai` → `utils/modules/safetrajai`、`third_party/trackeval` → `_vendor/trackeval`、`video/` → `dataset/raw/homography_videos/`、`scripts/…` 與 `config/…` 新路徑、openspec 與 `docs/adr` 引用
- [x] `CLAUDE.md` 中凍結區規則改為直接寫出理由，不再指向 ADR 檔；opsx 指令與 `openspec/` 段落移除
- 驗收：
  - [x] SPEC 成功標準 7、7e 的 `git grep` 零命中（`utils/modules/`、`config/`、`scripts/`）
  - [x] `git grep "src/safetrajai"` 在這批文件內零命中
  - [x] `pytest`（含 README 覆蓋）
- 依賴：T3a｜規模：機械式（約 10 個檔案；分兩個 commit：路徑、ADR／openspec 引用）

### T3c：拆分 `Markdown/USAGE.md`
- [x] 依 SPEC 4.1 拆成 `getting-started/`（`install.md`、`data-layout.md`）、`pipeline/`（`main-flow.md`、`stages.md`）、`roi/roi-guide.md`、`grid_search/grid-search.md`、`reid/reid-data-tools.md`、`validation/validation-and-viz.md`、`testing/testing.md`；`USAGE.md` 留總入口、速查、已知限制
- [x] 第 0 節整節刪除；各檔內章節重新從 1 編號；`README.md:500` 改指向 `Markdown/roi/roi-guide.md`；`USAGE.md` 內「見 5.2」2 處改為相對連結
- 驗收：
  - [x] 拆分後各檔合併、扣掉編號與連結行，與原檔（扣掉第 0 節）逐行一致
  - [x] `Markdown/` 結構與 SPEC 4.1 一致；所有 `README`／`USAGE` 對 `Markdown/` 的連結都指向存在的檔案
  - [x] `pytest`（含 README 覆蓋）
- 依賴：T3b｜規模：M（一個 commit）

### T4b：新增 `Markdown/config/` 使用教學
- [x] 新增 `overview.md`、`tracking-scenarios.md`、`reid-configs.md`、`collision-configs.md`、`per-camera-configs.md`
- [x] 任務導向（我想做什麼 → 改哪個檔、哪個鍵 → 用哪個指令驗證）；情境檔以「與 `bytetrack.yaml` 的差異」描述；與 `grid_search/grid-search.md`、`roi/roi-guide.md` 重疊處用連結
- 驗收：
  - [x] 文件中的每個鍵都能在 loader 或 yaml 找到（寫腳本交叉檢查）
  - [x] 指令範例以 `--help`／`--dry-run` 驗證可執行
  - [x] `pytest`
- 依賴：T3c｜規模：M（5 個新檔案，一個 commit）

### 檢查點 B（T4b 完成後）
- [x] SPEC 成功標準 7、7a–7e 逐項通過
- [x] **與使用者一起審閱再繼續**

---

## 階段 3：清理與去重

### T5：A 級零風險清理（每項一個 commit）

- [x] **A1**：刪除 `scripts/grid_search/grid_search_botsort.py` 中被註解掉的舊 `PARAM_GRIDS`（約 80 行）｜XS｜驗證：`--dry-run` 與基準相同
- [x] **A2**：修正三支 grid search 的失效 docstring（檔名 `tracker_grid_search.py`、不存在的 `run_class_aware_tracking`／`--n-jobs`／`--video-dir`）與 help 預設值（botsort `--gmc-method` 實為 `sift`、`--max-videos` 實為 6）；`--model` 的 help 註明「目前無作用」｜S｜驗證：`--help` 只在這些字串不同，且修正後與實際預設一致
- [x] **A3**：修正 `score_params()` docstring（第 25 百分位並排除 0 分）｜XS
- [x] **A4**：刪除 `_run_tracking_for_video` 未使用的 `trial_dir` 參數；合併 `generate_config_patch` 內相同的 `bool`／`else` 分支；函式內重複 import 移到頂部｜S｜驗證：既有 `test_grid_search_core.py`
- [x] **A5**：修正過時敘述（「兩階段主流程」、`_annotate_frame` 引用的 `utils.visualization`）；刪除 `target_mp4s = mp4s[:]` 空操作｜S
- [x] **A6**：刪除 `evaluation/pipeline.py` 尾段註解掉的程式碼與貼上的舊結果表（約 40 行）；`excel_export.py` 末段註解掉的 `print_eval_summary`（先回報確切範圍）｜S
- [x] **A7**：`evaluation/utils.py:89` 迴圈內的 `from openpyxl.utils import get_column_letter` 移到頂部｜XS
- [x] **A8**：失效引用清理（`docs/superpowers`、`openspec`、`markdown/reference_*`、`docs/adr`）：只刪引用片語，其餘句子保留；**排除** `utils/model.py`、`tracking/class_aware_*.py`、`_vendor/`、`tests/`。分三個 commit：`config/`（`datasets.yaml`、`reid_crops.yaml`、`roi/*.yaml`）、`scripts/`、`utils/modules/`｜機械式｜驗證：SPEC 成功標準 7；smoke 的 tracker AST 檢查通過

通用驗證：每項 `pytest`＋`--help`／`--dry-run` 與基準比對。

- [x] T5 完成後 golden（2026-10-29，`runs/refactor_golden/t5_check_20261029`，基準 `runs/refactor_golden_archive/conflict1`）：9 類別全部 identical，差異與最大絕對差皆為 0。
  比對指令：`python tests/golden/compare.py runs/refactor_golden_archive/conflict1 t5_check_20261029`（舊 run 已搬到 `refactor_golden_archive/`，文件中的指令字面待 T9 同步）。

### T6：B 級消除重複邏輯（每項一個 commit；先寫特徵化測試再抽取）

- [x] **B2**：新增 `grid_search/cli.py`，抽出三支腳本 `main()` 後段約 100 行共用邏輯；各腳本只保留 `PARAM_GRIDS`、預設值與 botsort 專屬參數；保留 `imgsz=1280`／`gmc` 等寫死值、botsort `model` 候選重複的 `yolo11x.pt`、`--model`｜M｜驗證：`--help` 與六類加 `_default` 的 `--dry-run` 逐位元相同
- [x] **B3**：影片尺寸讀取 helper（`tracking_pipeline` 2 處、`core.score_params`、`run_stage3`）；不併入 `get_video_wh()`｜S
- [x] **B4**：序列檔名縮短邏輯（`" 中華三路 民生二路(東北角)_"` 的 replace，`sequence_tracker.py`、`core.py`）｜S｜驗證：新增特徵化測試
- [x] **B1**：共用的 label 寫出邏輯（`sequence_tracker._process_frame` 與 `core._run_tracking_for_video`）；**先確認是否動到 pipeline 行為路徑，是則停下來問使用者是否跑 golden**｜M
- [x] **B5**：合併兩支視覺化腳本的共用部分（`CLASS_COLORS`、`_color_for_id`、`parse_seqinfo`、`draw_gt_on_frame`），逐位元保留現行行為｜M
- [x] **B6**：`stage3` 的 `_ensure_stage1_output_exists()` 後的重複掃描，保留 soft-skip 行為｜S
- [x] **B7**：`excel_export.py:240-262` 的 `key_means`／`key_mid` 以後綴參數化｜S｜驗證：合成資料逐格比對
- [x] **B8**：預設類別清單改由 `classes.py` 推導（先確認匯出方式）｜XS
- [x] **B9**：`excel_export.py:354-384` DET 曲線的巢狀 `downsample_idx` 與重複分支｜S｜驗證：合成資料逐格比對
- [x] **B10**：`file_md5` 與 `md5_file`：先確認呼叫端，傳回值型別不同，無法安全合併就不做並回報｜XS｜結果：評估後不做（回傳型別不同，不改呼叫端無法合併）

### 檢查點 C（T6 完成後）
- [x] `pytest` 全過，新增特徵化測試；`--help`／`--dry-run` 與基準相同（790 passed, 11 skipped；golden 於 T5 後與 B1 後各跑一次，皆 identical）
- [x] 第一方程式碼淨減少行數；無新增循環 import（第一方 .py 17,202 → 16,680 行，淨 −522；tests +614 行；追蹤檔案 446 → 320）
- [x] **與使用者一起審閱再繼續**

### T7：視覺化類別顏色修正（行為變更）
- [x] 先寫失敗測試：`visualize_gt_on_video.py` 的 `CLASS_COLORS` 以 0 起算、GT `class_id` 以 1 起算，目前每個類別拿到上一個類別的顏色
- [x] 修正對照；commit message 標明這是行為變更
- 驗收：新測試通過；輸出影片顏色變更寫入 commit message
- 依賴：B5｜規模：S

### T8：雜項
- [x] `git mv ".claude/agents/project-manger copy.md" ".claude/agents/qa-engineer.md"`，內容不動
- 驗收：`pytest`
- 依賴：無｜規模：XS

---

## 階段 4：交付

### T9：交付報告
- [x] 淨減行數（`git diff main --shortstat`）（淨 −522 行，見 `tasks/delivery-report.md`）
- [x] D 級 9 個疑似死碼清單（保留，供使用者日後決定）
- [x] 聲明 golden「未跑」及理由；若中途有任何輸出差異需列出（實際已跑兩次且 identical，報告第 5 節說明與規格的差異）
- [x] 殘餘待辦：上游 LICENSE 是否補上、`Sub_Module.md` 過時、文件盤點（`docs/` 已移出 repo）、`SPEC-features.md` 的啟動
- 依賴：全部

### 檢查點 D（完成）
- [x] SPEC 第 9 節成功標準逐項通過（逐項結果見報告第 2 節）
- [x] 「絕對不做」清單零違反
- [x] 使用者核准合併（2026-10-29「核准合併，開始 SPEC-features」）；之後才開始 `SPEC-features.md`
