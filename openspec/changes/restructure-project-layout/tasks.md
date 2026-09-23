# Tasks

## 1. Phase 0：回歸防線（在舊結構上完成，不搬任何程式碼）

- [x] 1.1 從 `master` 開分支 `refactor/project-structure`，驗證 `git branch --show-current` 輸出該分支名
- [x] 1.2 在 `track_eval` env 安裝 `pytest`，驗證 `pip list` 中 `torch`、`ultralytics` 版本與安裝前相同（另依使用者決定安裝 `scikit-learn`：`utils/model.py` 頂層 import sklearn，原 env 缺它導致 `best_full_model.pt` 無法 unpickle；安裝後 torch／ultralytics／numpy／scipy／opencv 版本不變）
- [x] 1.3 建立 `tests/golden/configs/` 下 5 份情境 config（bytetrack、botsort_noreid、botsort_reid、collision_detect、botsort_save_reid_crops），與 `config/` 原檔相比只有 `OUTPUT_ROOT` 與 `reid_crop_dir` 不同（`reid_crop_dir` 預設寫入正式資料集 `dataset/reid/crop_dataset`，必須一併導向 run 目錄）；以 `diff` 驗證差異僅此兩行
- [x] 1.4 撰寫 `tests/golden/run_all.sh <run>`：依序跑 5 個情境、grid search（`--max-videos 1`＋單組 `--params-subset`＋獨立 `--result-csv`）、homography 驗證，輸出全部落在 `runs/refactor_golden/<run>/`；以 `--dry-run` 或 echo 模式驗證不會寫到 `runs/refactor_golden/` 以外
- [x] 1.5 撰寫 `tests/golden/compare.py <runA> <runB>`（MOT txt 逐行＋數值欄位最大絕對誤差、TrackEval 摘要與 grid search CSV 指標、ttc/pet events 逐筆比對、Re-ID crops 相對路徑集合）；以同一個 run 自比驗證輸出全為 0 差異
- [x] 1.6 撰寫 `tests/test_smoke.py`（舊結構版）：import 全部專案模組、對 `config/*.yaml` 呼叫 `load_config()`／`load_track_config()`；記錄基準 commit hash 到 `tests/golden/BASE_COMMIT`；以 `pytest tests/test_smoke.py` 通過驗證
- [x] 1.7 commit Phase 0 工具（在任何搬移之前），驗證 `git log -1 --stat` 只含 `tests/` 檔案
- [x] 1.8 以舊程式碼執行 `run_all.sh before1`、`run_all.sh before2`，並以 `compare.py before1 before2` 產出容忍度報告存成 `tests/golden/TOLERANCE.md`；驗證報告涵蓋全部比對項目

## 2. Phase 1：清理（仍在舊結構上）

- [x] 2.1 刪除 `utils/visualization.py` 的 `translate_path`、`normalize_string` 與 `argostranslate` import；以 `grep -rn argostranslate` 無結果（排除 vendored／ignored 目錄）驗證
- [x] 2.2 `git mv` 兩支 `.sh` 到 `tools/`，並確認 `skills-lock.json` 由哪個工具讀取、保持原位；驗證根目錄已無 `.sh` 檔
- [x] 2.3 跑 smoke 測試與 `run_all.sh after_p1`＋`compare.py before1 after_p1`，驗證差異在容忍度內；commit Phase 1

## 3. Phase 2：套件化與搬移

- [x] 3.1 建立 `pyproject.toml`（setuptools、`package-dir` 對應 `src/` 與 `third_party/`、額外頂層 package `utils`、依賴宣告、dev 依賴 pytest）與空的 `src/safetrajai/__init__.py`；以最小原型驗證 setuptools 83 editable 模式可同時 import `safetrajai`、`trackeval.metrics`、`trackeval.datasets`，失敗則改用 design D2 備案
- [x] 3.2 `git mv trackeval third_party/trackeval`（內容不變），執行 `pip install -e . --no-deps`；在 `/tmp` 下啟動 Python 驗證 `trackeval.__file__` 指向 `third_party/trackeval/`，且 `git diff --stat -M` 顯示純 rename
- [x] 3.3 純搬移 commit：`git mv` `module/{detection,tracker→tracking,reid,grid_search/core.py,collision}`、`evaluation/`、`utils/visualization.py→visualization/` 到 `src/safetrajai/`，`homography_*` 收進 `collision/homography/`（`homography.py` 成為該子套件的 `__init__.py`，見 design D5b），`utils/model.py` 實作移到 `safetrajai/reid/model.py`；驗證 `git show --stat -M` 全部被辨識為 rename
- [x] 3.4 純搬移 commit：CLI 腳本 `git mv` 到 `scripts/`（grid_search_*、validate_homography、train_reid、`reid_data/*`＋README、`visualize_gt_*`），`evaluation/pipeline.py::main()` 抽到 `scripts/run_evaluation.py`；驗證 rename 偵測
- [x] 3.5 新增 `safetrajai/paths.py`（`PROJECT_ROOT`），取代全部 `parents[N]`、`DEFAULT_TRACK_CONFIG_PATH`、`_DATASETS_YAML_PATH`；以 `grep -rn "parents\[" src scripts` 只剩 `paths.py` 驗證
- [x] 3.6 改寫全部 import 為 `safetrajai.*`、刪除所有 `sys.path` 修改（含 `track_eval.py` 加入 repo 上一層的那行）；以 `grep -rn "sys.path" src scripts utils track_eval.py` 無結果驗證
- [x] 3.7 tracker 檔案只改檔案層級 import；以 AST 比對 `BASE_COMMIT` 版本與新版的 `__init__`／`update`／`remove_duplicate_stracks`／`init_track` 本體完全相同驗證
- [x] 3.8 `utils/model.py` 改為 re-export shim；在 `/tmp` 下實際 `torch.load` 一個 `best_full_model.pt`（`weights_only=False`）驗證可載入
- [x] 3.9 新增 `safetrajai/classes.py`（0-based `CLASS_NAMES`、`MOT_CLASS_ID = id + 1`），替換 `mot_format.NAMES`、`trackeval_runner.class_name_to_class_id`、`scripts/reid_data/` 三份 `CLASSES`（改 `sorted(CLASS_NAMES.values())`）；以 smoke 斷言新舊值完全相同驗證
- [x] 3.10 `.gitignore` 移除 `dataset/reid/` 例外規則；以 `git status` 確認 `dataset/reid/` 無新增追蹤或遺失追蹤的檔案
- [x] 3.11 更新 `tests/test_smoke.py` 為新結構版：walk import `safetrajai.*`＋`utils.model`、全 config 載入、類別一致性（含 YAML 載入的 `CLASS_NAMES`）、tracker AST 不變性、`sys.path` 掃描、`scripts/*.py --help` 皆 0 退出；`pytest` 全數通過
- [x] 3.12 跑 `run_all.sh after_p2`（以 `scripts/` 新路徑）＋`compare.py before1 after_p2`，驗證差異在容忍度內；commit Phase 2

## 4. Phase 3：拆分主入口

- [x] 4.1 將 `track_eval.py` 內容逐字拆到 `safetrajai/pipeline/{reid_crops,video_io,sequence_tracker,tracking_pipeline}.py` 與 `safetrajai/cli/run_pipeline.py`（argparse＋`main()`），只調整 import；以 AST 比對各類別／函式本體與拆分前相同驗證
- [x] 4.2 新增 `scripts/run_pipeline.py` 呼叫 `safetrajai.cli.run_pipeline.main`；根目錄 `track_eval.py` 改為 wrapper，未安裝時印出安裝指令並 `sys.exit(1)`；在未安裝的暫時 env（或 `pip uninstall` 後）驗證提示訊息與非零狀態，再重新安裝
- [x] 4.3 跑 smoke 測試與 `run_all.sh after_p3`（經 `python track_eval.py` 入口）＋`compare.py before1 after_p3`，驗證差異在容忍度內；commit Phase 3

## 5. 文件同步

- [ ] 5.1 更新 CLAUDE.md（架構圖、常用指令、安裝步驟、conda env、`requirement.txt` 非權威來源、`save_mot_format` 等既有限制的新路徑）；開頭加上「術語見 `CONTEXT.md`、關鍵決策見 `docs/adr/`」，並將用語對齊 glossary（碰撞 → 衝突事件、影片 → 序列、tag → 鏡頭）；以 `grep -nE "碰撞偵測|碰撞風險|\btag\b" CLAUDE.md` 只剩刻意說明對應關係的行驗證
- [ ] 5.2 更新 `.claude/commands/{run-pipeline,tracker-change,tune-config}.md` 與 `.claude/settings.local.json` 的路徑
- [ ] 5.3 更新 `tracking/HANDOFF.md`、`reid/HANDOFF.md`、`*_flow.md`、`scripts/reid_data/README.md`、`config/homography_*/README.md`，以及 `config/*.yaml` 開頭的用法註解（只改註解）
- [ ] 5.4 以 `grep -rnE "module/|module\.(tracker|reid|collision|grid_search|detection)|from evaluation|dataset/reid/.*\.py|visualize/" --include=*.md --include=*.yaml --include=*.json --include=*.py .`（排除 vendored／ignored／openspec archive）驗證舊路徑無殘留；commit 文件更新

## 6. 收尾驗證

- [ ] 6.1 在乾淨 shell（`conda activate track_eval`、repo 外 cwd）依 CLAUDE.md 的安裝步驟重做一次安裝並跑 `pytest`，驗證文件步驟可重現
- [ ] 6.2 彙整各 Phase 的 compare 報告與 `git diff --stat master...HEAD` 摘要，確認 spec 全部 requirement 皆有對應驗證證據
