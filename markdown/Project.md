# Project.md

## 專案名稱
safetrajai_eval — CCTV 交通物件偵測與多物件追蹤（MOT）評估流程

## 目標
針對路口 CCTV 影像，用 YOLO11 偵測六類交通參與者（person、car、scooter、small_truck、bus、truck），再用自行修改的 class-aware BoTSORT / ByteTrack 做多物件追蹤，最後用 TrackEval 計算 HOTA / CLEAR / Identity 等指標，輸出 Excel 報表，作為追蹤參數調優與模型選擇的依據。

## 範圍
- **包含**：偵測 → 追蹤 → 評估 的完整 pipeline、追蹤超參數 grid search、GT/追蹤結果疊圖視覺化。
- **不包含**：YOLO 偵測模型本身的訓練（`models/yolo/` 下的權重是外部訓練好的產物）、線上服務化（無 API/後端）。
- `utils/model.py`、`utils/train_utils.py` 是 ReID 模型訓練用的工具函式，目前沒有任何腳本引用，屬於孤兒程式碼，不算現行流程的一部分。

## 技術棧
- Python + PyTorch / torchvision
- Ultralytics（YOLO11 偵測、`BYTETracker` / `BOTSORT` 基底類別）
- OpenCV、NumPy、SciPy
- TrackEval（vendored，於 `trackeval/`）
- openpyxl（Excel 報表輸出）、PyYAML（設定檔）
- 無 `requirements.txt`／`pyproject.toml`、無 CI、非 git 版本控制（實驗版本靠檔名區分）

## 核心流程
```
CCTV 影片
   │
   ▼
YOLO11 偵測 (models/yolo/yolo11x_dennis.pt)
   │
   ▼
class-aware 追蹤 (trackers/)  ── 依類別分組匹配，逐類別可設不同閾值/ReID開關
   │
   ▼
MOT 格式追蹤結果 (utils/track_eval/mot_converter.py)
   │
   ▼
TrackEval 評估 (trackeval/)  ── HOTA / CLEAR / Identity，逐類別計算
   │
   ▼
Excel 報表 (utils/track_eval/excel_export.py)
```

主要入口：`track_eval.py`（完整流程）、`tracker_grid_search_*.py`（超參數搜尋）、`visualize_gt_on_*.py`（結果視覺化）。詳細架構與指令見根目錄 `CLAUDE.md`。

## 已知限制
- 無自動化測試，改動需手動跑 `track_eval.py` 驗證輸出（見 `TEST.md`）。
- 無版本控制，實驗版本與參數調整靠 `trackers/track_config_*.yaml` 檔名區分，容易混淆。
- ReID 目前預設關閉：實驗顯示開啟後對 person 有幫助，但會損害 scooter 追蹤並增加 IDSW（見 `VERIFY.md`）。

## 待補充
> 以下為新建骨架，需要專案負責人補充：專案的實際使用者/利害關係人、交付週期、與上游（偵測模型訓練）/下游（如有應用系統）的介接方式。
