---
paths:
  - 'track_eval.py'
  - 'config/**/*.yaml'
  - 'visualize/**/*.py'
---

# 專案與 Pipeline 規範

## 三階段流程

- Stage 1：逐影片建立 Detector 與 tracker，輸出逐幀 label、MOT 軌跡與可選 Demo 影片。
- Stage 2：將 Stage 1 軌跡整理為 TrackEval 結構，選擇性執行 DET mAP，並匯出逐類別 Excel。
- Stage 3：讀取 Stage 1 的逐類別 MOT 軌跡，計算 TTC／PET，輸出事件 JSON、CSV、legacy log 與熱區圖。
- `stages.run_stage1`、`run_stage2`、`run_stage3` 控制三階段；`eval_flags.run_det_eval` 與 `run_track_eval` 僅控制 Stage 2 內的評估子工作。不要混用它們的語意。

## 情境設定

- 每個 `config/*.yaml` 都是完整、自包含的實驗情境，包含一般設定與 `class_tracking`。新實驗建立新檔，勿覆蓋既有基準情境。
- 固定類別與 ID：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`。變更此契約須同步修改所有下游資料與輸出。
- `SOURCE_VIDEO_DIR`、模型路徑、`OUTPUT_ROOT`、`run_det_eval_dir`、`run_track_eval_dir` 是可重現實驗的必要紀錄；改動時應保留能追溯的 config 名稱與輸出位置。
- 完整影片、重跑 Stage 1、grid search 與輸出目錄都可能耗時且覆寫既有結果；執行前先確認範圍。

## 維護界線

- `archive/` 是歷史對照碼，`trackeval/` 是 vendored 套件，參考資產不屬現行流程；未明確要求時不要修改或格式化。
- 修改跨模組輸入、輸出、設定欄位或 CLI 行為時，同步更新本規則對應內容，確保規則本身是唯一可依賴的行為摘要。

## I/O Contract（輸入輸出格式）

- 輸入是 YAML 情境設定、影片序列、模型權重與既有 Stage 1 產物；輸出根目錄由 `OUTPUT_ROOT` 決定。
- Stage 1 輸出追蹤軌跡，Stage 2 輸出 TrackEval／Excel，Stage 3 輸出 TTC／PET 事件；下游只能讀取上一階段已定義的產物。

## Stable Interface（不可破壞的介面）

- `python track_eval.py --config <path>` 是主 CLI 介面。
- YAML 的 `paths`、`models`、`stages`、`eval_flags`、`tracking_params`、`class_tracking` 區塊名稱與六類編碼不可任意改名。

## Design Principles（設計原則）

- 用完整 config 與明確 stage 開關重現每次實驗。
- 將偵測、追蹤、評估與風險分析分層，避免跨層讀取非正式中間資料。

## Domain Knowledge（專案知識）

- 這是離線 CCTV 交通追蹤評估專案，不是線上服務；處理成本主要來自影片推論與評估。
- 同一 tracker ID 不跨影片唯一；各類別的追蹤品質與最佳參數也不應假定相同。

## Dependencies（依賴）

- 主流程依賴 Python、PyTorch、Ultralytics、OpenCV、NumPy、PyYAML、openpyxl 與 vendored TrackEval。
- 執行 GPU 推論前確認 PyTorch、CUDA 與模型權重相容，避免在未確認環境中更換版本。

## AI Modification Rules（給 AI 的規則）

- 不要自動執行全量影片、覆寫 `runs/` 或改動模型／資料集路徑；先說明影響並取得確認。
- 修改 config 或主流程後，至少驗證受影響 stage 與產出路徑；不可只以程式可啟動宣稱完成。
