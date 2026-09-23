---
paths:
  - 'module/grid_search/**/*.py'
---

# Grid Search 開發規範

## 搜尋流程

- 三支腳本分別處理 ByteTrack、BoTSORT 與 BoTSORT Re-ID；它們是 tracker 超參數搜尋，不是通用 sklearn estimator。
- 每個 trial 的順序固定為：複製原始參數 → patch `CLASS_TRACKING_PARAMS` → tracking → MOT 格式輸出 → TrackEval → 擷取 `HOTA.AssA_mean` → 寫 CSV → 還原參數。
- `HOTA.AssA_mean` 是關聯性分數，不是 HOTA 本身；輸出與結論必須正確命名。

## 狀態與可重現性

- 不論 trial 成功、跳過或拋出例外，都必須還原 `CLASS_TRACKING_PARAMS`，避免下一組使用殘留參數。
- CSV 必須保留參數組合、config、評估資料範圍、狀態、錯誤與分數，讓結果能被重跑與比較。
- `--resume` 只能跳過完全相同且已完成的組合；不得因部分記錄或失敗 trial 而錯誤略過。

## 執行與驗證

- 新增維度前先以 `--dry-run` 計算組合數；全量搜尋或多影片搜尋前先確認成本。最小驗證使用 `--max-videos 1` 或受限參數子集。
- 腳本必須可從專案根目錄直接執行；更動路徑時驗證 `PROJECT_ROOT` 與 imports，因為純 py_compile 無法發現執行期路徑錯誤。
- 至少抽取一組完成 trial，用相同 config 手動跑一次 pipeline，確認 CSV 的 `AssA_mean` 與 TrackEval／Excel 結果一致。
