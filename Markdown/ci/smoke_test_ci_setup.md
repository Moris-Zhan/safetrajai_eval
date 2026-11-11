# 建立基本 CI 自動化（`training_eval_audit.md` 補強優先順序 #3）

日期：2026-11-11。

## 背景

`Markdown/training_eval_audit.md` 盤點發現整個 repo 沒有任何 CI 自動化，驗證全靠人工記得要跑 `pytest`／golden 比對。補強優先順序把這項排第 3——`validate_homography.py` 那類手動腳本（#4）與偵測訓練資料（#5）成本更高，而 CI 只差「新建 workflow」，不需要新資料。

本項的執行方式沿用這次補強計畫一路使用的慣例（Markdown 報告＋直接改動），不透過 openspec：repo 在 commit `68523d8`（"chore: remove openspec, legacy docs and ADRs moved to ignore/"）已刻意移除 `openspec/` 整個目錄與流程，這次經與使用者確認，維持移除後的狀態，不重新引入。

## 範圍：只自動化 smoke 測試，不含 golden 比對

CLAUDE.md 已記錄 golden 比對（`tests/golden/run_all.sh` + `compare.py`）依賴掛載磁碟上的 `dataset/`／`models/`（gitignored，一般 CI runner 不會有這些檔案），而且單次約 16 分鐘，不適合放進一般 GitHub Actions runner。這次只把 `pytest tests/test_smoke.py`（幾秒內跑完，涵蓋全部模組 import、全部 config 載入、類別 ID 契約、tracker override 方法的 AST 不變性等）接上 CI；golden 比對仍是 CLAUDE.md 既有規範的手動步驟，不變。

## 做法

新增 `.github/workflows/smoke-test.yml`：

- 觸發條件：push 到 `main`、所有 pull request。
- `actions/checkout@v4` 搭配 `fetch-depth: 0`——`tests/test_smoke.py` 的 `test_tracker_override_bodies_unchanged`／`test_frozen_methods_reject_unlisted_changes` 兩個測試用 `git show <BASE_COMMIT>:<path>` 讀取凍結基準版本的 tracker 原始碼，淺層 checkout 會讀不到該 commit。
- `actions/setup-python@v5`，Python 3.10（符合 `pyproject.toml` 的 `requires-python = ">=3.10"`），開啟 pip cache。
- `pip install -e ".[dev]"`：一般 CI runner 沒有既有的 CUDA 版 torch 需要保護，所以不像本機 conda env `track_eval` 那樣加 `--no-deps`；直接讓 pip 依 `pyproject.toml` 的 `dependencies` 正常解析（含 `torch`／`ultralytics`／`opencv-python` 等），同時裝上 `dev` extra 的 `pytest`。
- `pytest tests/test_smoke.py -v`。

## 驗證

- 推送前已在本機 conda env `track_eval` 重新跑過 `pytest tests/test_smoke.py`，確認目前程式碼本身是綠的（265 passed, 8 skipped），排除「CI 一接上就先壞在既有 bug」的可能性。
- 無法在本機模擬一個乾淨的 pip 安裝環境驗證 workflow YAML 本身（CLAUDE.md 規定本機只能用 conda env `track_eval`、不建其他 venv），所以 workflow 的實際正確性要看它第一次在 GitHub Actions 上真正跑過的結果；若失敗，依錯誤訊息調整（常見風險：`pip install -e ".[dev]"` 解析到的 torch/ultralytics 版本組合在 CPU-only runner 上 import 失敗）。

## 後續

若之後要把 golden 比對也自動化，需要先解決「資料不在 git、不易搬進一般 runner」的問題（例如改用有掛載對應磁碟的自架 runner），這個範圍比目前這次大，留給未來視需求再評估。
