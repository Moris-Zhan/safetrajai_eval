# 測試與回歸驗證

| 工具 | 指令 | 用途 |
|---|---|---|
| smoke／單元測試 | `pytest` | 模組 import、全部 config 載入、類別 ID 契約、tracker AST 不變性、`sys.path` 掃描、`scripts/**/*.py --help`、ROI／忽略區域、Re-ID 資料流程。幾秒內跑完，任何結構調整後都要跑 |
| golden 執行 | `tests/golden/run_all.sh <run>` | 5 個情境＋grid search＋homography 驗證，平行執行約 16 分鐘，輸出到 `runs/refactor_golden/<run>/` |
| golden 比對 | `python tests/golden/compare.py reid1 <run>` | 與基準 `reid1` 比對，容忍度 0，任何非零差異都代表行為改變 |

- 基準在 2026-09-23 由 `before1` 改為 `after1`，2026-09-24 再改為 `reid1`（裁切情境改在來源鏡頭 C000043 上蒐集），原因見 `tests/golden/TOLERANCE.md`。
- 改動演算法時，要實際跑受影響的 stage 再比對輸出；比較追蹤效果時，看同一基準下的 HOTA、AssA、IDSW，再用疊圖影片檢查 ID switch 和跨類別誤配。
- 全量序列、grid search 和重跑 stage1 都很耗時，也可能覆寫既有輸出，執行前要先確認範圍。
