# 測試與回歸驗證

| 工具 | 指令 | 用途 |
|---|---|---|
| smoke／單元測試 | `pytest` | 模組 import、全部 config 載入、類別 ID 契約、tracker AST 不變性、`sys.path` 掃描、`scripts/**/*.py --help`、ROI／忽略區域、Re-ID 資料流程。幾秒內跑完，任何結構調整後都要跑 |
| golden 執行 | `tests/golden/run_all.sh <run>` | 5 個情境＋grid search＋homography 驗證，平行執行約 16 分鐘，輸出到 `runs/refactor_golden/<run>/` |
| golden 比對 | `python tests/golden/compare.py runs/refactor_golden_archive/conflict1 <run>` | 與基準 `conflict1`（放在 `runs/refactor_golden_archive/`）比對，容忍度 0，任何非零差異都代表行為改變 |

- 基準在 2026-09-23 由 `before1` 改為 `after1`，2026-09-24 再改為 `reid1`（裁切情境改在來源鏡頭 C000043 上蒐集），原因見 `tests/golden/TOLERANCE.md`。
- 改動演算法時，要實際跑受影響的 stage 再比對輸出；比較追蹤效果時，看同一基準下的 HOTA、AssA、IDSW，再用疊圖影片檢查 ID switch 和跨類別誤配。
- 全量序列、grid search 和重跑 stage1 都很耗時，也可能覆寫既有輸出，執行前要先確認範圍。

## 測試分層與單元測試對照

測試分三層。前兩層都在 `track_eval` 環境、repo 根目錄執行 `pytest`，不需要資料集、權重或 GPU；第三層才需要完整資料。

| 層級 | 指令 | 耗時 | 前提 | 何時跑 |
|---|---|---|---|---|
| smoke | `pytest tests/test_smoke.py` | 秒級 | 安裝即可 | 任何結構調整後 |
| 單元 | `pytest`（含 smoke 與下列單元測試） | 秒級～分鐘 | 不需要資料／權重 | 提交前 |
| golden 比對 | `tests/golden/run_all.sh <run>`，再 `python tests/golden/compare.py <基準> <run>` | 約 16 分鐘 | 完整資料（`hf-restore.md` 第 2 節）、YOLO 權重 | 改動演算法或 stage 行為後 |

**smoke**（`tests/test_smoke.py`）：全部模組可 import、全部 config 可載入、類別 ID 契約、tracker override 方法的 AST 不變性、`sys.path` 掃描、有 argparse 的腳本 `--help` 可執行。其中「被隔離的 Re-ID 權重仍可載入」在權重檔不存在時會被略過（`skipif`），不是失敗。另有少數參數組合不適用（例如不迭代類別的腳本、不是追蹤情境的設定）而被略過，`pytest` 結尾出現 `skipped` 屬正常。

**單元測試**依領域分組：

| 領域 | 測試檔 |
|---|---|
| Re-ID 資料與訓練 | `tests/test_reid_*.py`（manifest、切分、污染標記、自動排除、抽查、碎片合併、編碼、融合、外觀距離統計、訓練、驗收） |
| Homography／georef | `tests/test_homography_*.py`、`test_geocode_cameras.py`、`test_satellite_download.py`、`test_scale_match.py`、`test_secrets.py` |
| 衝突指標 | `tests/test_conflict_metrics.py` |
| ROI／資料集路徑／忽略區域 | `tests/test_roi_and_layout.py`、`test_ignore_region.py` |
| tracker 保留期限 | `tests/test_tracker_retention.py` |
| 追蹤設定與 grid search | `tests/test_track_config.py`（設定載入與驗證、全域參數更新、查表退回預設、載入順序契約）、`test_grid_search_core.py`（參數組合、AssA 萃取、trial 參數修補與還原、resume） |
| 文件 | `tests/test_readme_coverage.py`：`scripts/` 的每支腳本與 `config/` 的每份設定都必須出現在本 README；新增入口卻沒寫文件時會失敗，確實不適合列入的才寫進該檔的 `EXEMPT`（要附理由） |

**golden 比對**確認重構或調整後的輸出與基準逐位元一致：`run_all.sh` 以 6 個情境＋grid search＋homography 驗證跑一次完整流程，輸出在 `runs/refactor_golden/<run>/`（先用 `--dry-run` 看會執行哪些步驟）。基準自比的容忍度為 0，任何非零差異都代表行為改變；基準名稱、更換基準的原因與預期差異見 [`tests/golden/TOLERANCE.md`](../../tests/golden/TOLERANCE.md)。全量執行耗時，也會寫入 `runs/`，執行前先確認範圍。
