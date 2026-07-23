# Project.md

## 專案名稱
safetrajai_eval — CCTV 交通物件偵測與多物件追蹤（MOT）評估流程

## 目標
針對路口 CCTV 影像，用 YOLO11 偵測六類交通參與者（person、car、scooter、small_truck、bus、truck），再用自行修改的 class-aware BoTSORT / ByteTrack 做多物件追蹤，最後用 TrackEval 計算 HOTA / CLEAR / Identity 等指標，輸出 Excel 報表，作為追蹤參數調優與模型選擇的依據。

## 範圍
- **包含**：偵測 → 追蹤 → 評估 的完整 pipeline、追蹤超參數 grid search、GT/追蹤結果疊圖視覺化。
- **不包含**：YOLO 偵測模型本身的訓練（`models/yolo/` 下的權重是外部訓練好的產物）、線上服務化（無 API/後端）。
- `utils/model.py`、`reid/training/train_utils.py` 是 ReID 模型訓練用的工具函式，目前沒有任何腳本引用，屬於孤兒程式碼，不算現行流程的一部分（`utils/model.py` 因既有權重檔的 pickle 相容性限制不可搬移，詳見下方架構章節）。

## 技術棧
- Python + PyTorch / torchvision
- Ultralytics（YOLO11 偵測、`BYTETracker` / `BOTSORT` 基底類別）
- OpenCV、NumPy、SciPy
- TrackEval（vendored，於 `trackeval/`）
- openpyxl（Excel 報表輸出）、PyYAML（設定檔）
- 無 `requirements.txt`／`pyproject.toml`、無 CI；已導入 git 版本控制，但實驗版本仍靠檔名區分（如 `track_config_BotSort_0413.yaml`）

## 核心流程
```
CCTV 影片
   │
   ▼
YOLO11 偵測 (detection/detector.py → models/yolo/yolo11x_dennis.pt)
   │
   ▼
class-aware 追蹤 (tracker/)  ── 依類別分組匹配，逐類別可設不同閾值/ReID開關（依賴 reid/ 做外觀特徵比對）
   │
   ▼
MOT 格式追蹤結果 (evaluation/mot_converter.py, evaluation/mot_format.py)
   │
   ▼
TrackEval 評估 (trackeval/)  ── HOTA / CLEAR / Identity，逐類別計算（由 evaluation/trackeval_runner.py 呼叫）
   │
   ▼
Excel 報表 (evaluation/excel_export.py)
```

主要入口：`track_eval.py`（完整流程，`TrackingEvalPipeline` 類別）、`tracker/grid_search/*.py` + `reid/grid_search_botsort_reid.py`（超參數搜尋）、`visualize_gt_on_*.py`（結果視覺化）。詳細架構與指令見根目錄 `CLAUDE.md`。

## 專案架構（2026-07 重組）

依 Detection / Tracker / Re-ID 三大模組拆分，另有橫跨兩者的共用 `evaluation/` 模組：

```
safetrajai_eval/
├── track_eval.py          CLI 入口：TrackingEvalPipeline（stage1 tracking → stage2 TrackEval）
├── config.yaml             全域路徑/評估設定
│
├── detection/               【Detection 模組】
│   └── detector.py          Detector 類別：YOLO 推論 + 逐類別信心過濾
│                             （訓練腳本缺口，見下方「已知限制」）
│
├── tracker/                 【Tracker 模組】（原 trackers/）
│   ├── class_aware_botsort.py / class_aware_bytetrack.py   class-aware 追蹤核心
│   ├── config.py             逐類別參數查表（match_thresh、track_buffer、use_reid…）
│   ├── matching.py, track_config*.yaml, HANDOFF.md
│   └── grid_search/          超參數 grid search（botsort、bytetrack）
│
├── reid/                     【Re-ID 模組】
│   ├── encoders.py           ReID / SOTA_ReID / Custom_ReID（推論期特徵抽取，被 tracker/ 引用）
│   ├── yolo_reid_trainer.py  YOLO ReID 訓練用 hook/projector
│   ├── grid_search_botsort_reid.py   含 ReID 開關的 grid search
│   └── training/              孤兒訓練工具（train_utils.py，目前無腳本呼叫）
│
├── evaluation/               【共用】DET 評估 + TrackEval 評估的 pipeline 邏輯
│   ├── config_loader.py, dataset_prep.py, mot_converter.py
│   ├── trackeval_runner.py, excel_export.py, mot_format.py, pipeline.py
│
├── utils/                    通用小工具
│   ├── visualization.py       畫面標註（draw_polygon_and_save 等）
│   └── model.py                ReIDNet 定義（不可搬移，見下方 pickle 相容性）
│
├── trackeval/                 vendored TrackEval 第三方套件
├── reid_models/                vendored fast-reid 風格套件（不可搬移，見下方 pickle 相容性）
├── archive/                    歸檔：past_conf/、past_py_conf/、trackers_trackeval_duplicate/ 等舊碼
└── visualize_gt_on_*.py        GT / 追蹤結果疊圖視覺化
```

**pickle 相容性限制**：`reid_models/` 與 `utils/model.py` 的模組路徑被既有權重檔（`models/reid_model_weights/**`、`models/class_reid_weights/**`）以 `torch.save()` 整物件序列化寫死在檔案內，`torch.load(weights_only=False)` 還原時需要能用原路徑 `import`。這兩者因此**固定留在原本的位置**，不隨模組重組搬動；詳見 `CLAUDE.md`「架構」章節。

## 已知限制
- 無自動化測試，改動需手動跑 `track_eval.py` 驗證輸出（見 `TEST.md`）。
- 實驗版本與參數調整靠 `tracker/track_config_*.yaml` 檔名區分，容易混淆。
- ReID 目前預設關閉：實驗顯示開啟後對 person 有幫助，但會損害 scooter 追蹤並增加 IDSW（見 `VERIFY.md`）。
- **Detection 模組缺訓練腳本**：YOLO 偵測模型訓練本來就不在此專案範圍內（`models/yolo/*.pt` 為外部訓練產物），`detection/` 目前只有推論用的 `Detector`。待未來有專門的 YOLO 訓練腳本再一併整理進 `detection/`（見 `NEED.md`）。

## 待補充
> 以下為新建骨架，需要專案負責人補充：專案的實際使用者/利害關係人、交付週期、與上游（偵測模型訓練）/下游（如有應用系統）的介接方式。
