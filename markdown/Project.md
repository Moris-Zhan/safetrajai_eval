# Project.md

## 專案名稱
safetrajai_eval — CCTV 交通物件偵測與多物件追蹤（MOT）評估流程

## 目標
針對路口 CCTV 影像，用 YOLO11 偵測六類交通參與者（person、car、scooter、small_truck、bus、truck），再用自行修改的 class-aware BoTSORT / ByteTrack 做多物件追蹤，最後用 TrackEval 計算 HOTA / CLEAR / Identity 等指標，輸出 Excel 報表，作為追蹤參數調優與模型選擇的依據。

## 範圍
- **包含**：偵測 → 追蹤 → 評估 的完整 pipeline、追蹤超參數 grid search、GT/追蹤結果疊圖視覺化。
- **不包含**：YOLO 偵測模型本身的訓練（`models/yolo/` 下的權重是外部訓練好的產物）、線上服務化（無 API/後端）。
- `utils/model.py`、`module/reid/training/train_utils.py` 是 ReID 模型訓練用的工具函式，目前沒有任何腳本引用，屬於孤兒程式碼，不算現行流程的一部分（`utils/model.py` 因既有權重檔的 pickle 相容性限制不可搬移，詳見下方架構章節）。

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
YOLO11 偵測 (module/detection/detector.py → models/yolo/yolo11x_dennis.pt)
   │
   ▼
class-aware 追蹤 (module/tracker/)  ── 依類別分組匹配，逐類別可設不同閾值/ReID開關（依賴 module/reid/ 做外觀特徵比對）
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

主要入口：`track_eval.py`（完整流程，`TrackingEvalPipeline` 類別）、`module/grid_search/*.py`（超參數搜尋，含 ReID 開關版本）、`visualize_gt_on_*.py`（結果視覺化）。詳細架構與指令見根目錄 `CLAUDE.md`。

## 專案架構（2026-07 重組，2026-07-23 再次搬移至 `module/`）

Detection / Tracker / Re-ID / grid_search 四個模組統一放在 `module/` 底下、彼此同一層；`evaluation/` 是橫跨 Detection eval 與 Tracker eval 的共用模組，不屬於這四者之一，留在根目錄：

```
safetrajai_eval/
├── track_eval.py                CLI 入口：TrackingEvalPipeline（stage1 tracking → stage2 TrackEval）
├── config.yaml                   全域路徑/評估設定
│
├── module/
│   ├── detection/                 【Detection 模組】
│   │   └── detector.py            Detector 類別：YOLO 推論 + 逐類別信心過濾
│   │                               （訓練腳本缺口，見下方「已知限制」）
│   │
│   ├── tracker/                   【Tracker 模組】（原 trackers/ → tracker/ → module/tracker/）
│   │   ├── class_aware_botsort.py / class_aware_bytetrack.py   class-aware 追蹤核心
│   │   ├── config.py               逐類別參數查表（match_thresh、track_buffer、use_reid…）
│   │   └── matching.py, track_config*.yaml, HANDOFF.md
│   │
│   ├── reid/                       【Re-ID 模組】
│   │   ├── encoders.py             ReID / SOTA_ReID / Custom_ReID（推論期特徵抽取，被 module/tracker/ 引用）
│   │   ├── yolo_reid_trainer.py    YOLO ReID 訓練用 hook/projector
│   │   └── training/                孤兒訓練工具（train_utils.py，目前無腳本呼叫）
│   │
│   └── grid_search/                【grid_search 模組】超參數 grid search（botsort、bytetrack、botsort_reid）
│                                     原巢狀在 tracker/grid_search/ 下，後獨立成與其他三者同層的模組
│
├── evaluation/                   【共用】DET 評估 + TrackEval 評估的 pipeline 邏輯，不屬於 module/ 四者之一
│   ├── config_loader.py, dataset_prep.py, mot_converter.py
│   ├── trackeval_runner.py, excel_export.py, mot_format.py, pipeline.py
│
├── utils/                        通用小工具
│   ├── visualization.py           畫面標註（draw_polygon_and_save 等）
│   └── model.py                    ReIDNet 定義（不可搬移，見下方 pickle 相容性）
│
├── trackeval/                     vendored TrackEval 第三方套件
├── reid_models/                    vendored fast-reid 風格套件（不可搬移，見下方 pickle 相容性）
├── archive/                        歸檔：past_conf/、past_py_conf/、trackers_trackeval_duplicate/ 等舊碼
└── visualize_gt_on_*.py            GT / 追蹤結果疊圖視覺化
```

**pickle 相容性限制**：`reid_models/` 與 `utils/model.py` 的模組路徑被既有權重檔（`models/reid_model_weights/**`、`models/class_reid_weights/**`）以 `torch.save()` 整物件序列化寫死在檔案內，`torch.load(weights_only=False)` 還原時需要能用原路徑 `import`。這兩者因此**固定留在原本的位置**，不隨模組重組搬動；詳見 `CLAUDE.md`「架構」章節。

## 資料與輸出目錄（現況全貌，2026-07-23 補齊）

上方「專案架構」只涵蓋程式碼模組。以下是根目錄下**非程式碼**的資料/輸出/實驗記錄目錄，先前未被記錄進本文件：

```
safetrajai_eval/
├── dataset/                    原始資料（大型，已 gitignore）
│   ├── 中華三路_民生二路(東北角)yolo驗證用_all8/   正式評估資料集（annotations_all8/、mot_annotations_all8/ 等）
│   ├── reid_dataset/            Re-ID 訓練用資料集
│   └── *.mp4（4 支散落在 dataset/ 根目錄，非上述兩個結構化資料夾內）  來源不明的測試片段，目前無程式碼引用
├── dataset.zip                 dataset/ 的壓縮備份（465MB，內容與 dataset/ 幾乎重複）
│
├── models/                     權重檔（大型，已 gitignore），已依類型分子目錄：
│   ├── yolo/                    YOLO 偵測權重（yolo11x_dennis.pt 等）
│   ├── yolo-cls/、yolo-reid/     YOLO 分類 / YOLO-as-ReID-encoder 權重
│   ├── reid_model_weights/       SOTA_ReID 用（mgn_sbs_*、vit_transreid，pickle 相依 reid_models/，見上方限制）
│   └── class_reid_weights/       Custom_ReID 逐類別權重（person/、scooter/，pickle 相依 utils/model.py，見上方限制）
│
├── runs/                        pipeline 輸出（大型，已 gitignore），目前混雜三種性質不同的內容：
│   ├── track_eval/               目前這次執行的即時輸出（= config.yaml 的 OUTPUT_ROOT，會被下次執行覆寫）
│   ├── detect/                   ultralytics `model.val()`/`predict()` 自動產生的暫存輸出（train、val1-9…），非本專案手動管理，重新執行會自動重建
│   └── track_eval(0411_ByteTrack)/、track_eval(0413_BotSort_noreid)/ 等   人工封存的歷史實驗結果（含展開目錄與對應 .zip，命名以實驗日期＋tracker/設定區分）
├── runs.zip                     runs/ 的壓縮備份（520MB，內容與 runs/ 幾乎重複）
│
├── wandb/                       Weights & Biases 訓練 log（74MB，已 gitignore），對應舊 ReID 訓練流程（`module/reid/training/train_utils.py`，目前孤兒），由 wandb 套件依執行時 cwd 自動產生
│
├── fps_compare/                  ByteTrack FPS/解析度比較實驗報告（已版控）── 對應 VERIFY.md「FPS / 推論解析度」段落
├── html/                         2 支獨立 HTML 視覺化工具（index1x3.html、index2x3.html：多支影片同步播放比較），已版控，無程式碼引用
│
├── .claude/                      Claude Code 本機設定（settings.local.json），非專案邏輯的一部分
└── requirement.txt               環境建置腳本（pip install 指令集合，非標準 requirements.txt 格式；未版控）
```

> `botsort_noreid/`（BoTSORT+NoReID 的 GMC 方法比較實驗報告）已於 2026-07-23 移除並取消版控；結論保留在 `VERIFY.md`「GMC 方法比較」段落，原始數據如需要可從 git 歷史（commit `7061f97`）取回。

**現況觀察**（依此給出下方「整理建議」）：
- `fps_compare/`、`html/` 是程式碼以外「散落在根目錄」的實驗報告/工具，且已版控、體積小、無程式碼引用，是重組風險最低、最值得優先處理的部分。
- `runs/` 底下混雜「目前即時輸出」「ultralytics 自動暫存」「人工封存的歷史實驗」三種性質，職責不清。
- `dataset.zip`／`runs.zip` 與展開後的目錄內容重複，合計近 1GB，用途待確認（備份快照或可清理的舊檔）。
- `dataset/` 根目錄下 4 支散落 mp4 未歸入任何結構化子資料夾。

具體搬移建議見 `NEED.md`「目錄整理建議」。

## 已知限制
- 無自動化測試，改動需手動跑 `track_eval.py` 驗證輸出（見 `TEST.md`）。
- 實驗版本與參數調整靠 `module/tracker/track_config_*.yaml` 檔名區分，容易混淆。
- ReID 目前預設關閉：實驗顯示開啟後對 person 有幫助，但會損害 scooter 追蹤並增加 IDSW（見 `VERIFY.md`）。
- **Detection 模組缺訓練腳本**：YOLO 偵測模型訓練本來就不在此專案範圍內（`models/yolo/*.pt` 為外部訓練產物），`module/detection/` 目前只有推論用的 `Detector`。待未來有專門的 YOLO 訓練腳本再一併整理進 `module/detection/`（見 `NEED.md`）。

## 待補充
> 以下為新建骨架，需要專案負責人補充：專案的實際使用者/利害關係人、交付週期、與上游（偵測模型訓練）/下游（如有應用系統）的介接方式。
