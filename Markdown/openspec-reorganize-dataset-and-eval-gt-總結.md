# OpenSpec 變更總結：reorganize-dataset-and-eval-gt（資料分層與評估 GT 修正）

> 狀態：**已完成並歸檔**（archived at `2026-09-23`）
> 來源：`openspec/changes/archive/2026-09-23-reorganize-dataset-and-eval-gt/`
> 對應 spec：`openspec/specs/dataset-layout/spec.md`、`openspec/specs/evaluation-roi-filtering/spec.md`

## 目前 OpenSpec 進度總覽

檢查 `openspec/changes/` 後確認：

| Change | 狀態 | tasks 完成度 |
|---|---|---|
| `restructure-project-layout` | 已歸檔 | 32/32（已於 `Markdown/openspec-restructure-project-layout-總結.md` 整理） |
| `reorganize-dataset-and-eval-gt` | 已歸檔 | 33/33（本文件） |
| `rebuild-reid-dataset` | **進行中，尚未開始執行** | 0/約 26（`tasks.md` 全部項目仍為 `[ ]`，包含 Phase 1「來源鏡頭的 ROI」在內，程式碼層面 `src/safetrajai/reid/dataset_index.py`／`manifest.py`、`scripts/roi/`、`config/roi/`、`config/reid_split.yaml` 等產出物均尚未建立） |

因此，目前唯一未完成的 change 是 `rebuild-reid-dataset`，且連 Phase 1（ROI 點選）都還沒開始，沒有可整理成文件的已完成階段。本次任務改為補齊前一個已完成、但尚未產出 Markdown 總結的 change：**`reorganize-dataset-and-eval-gt`**。

## 為什麼要做這次變更（Why）

下一步要重建 Re-ID 訓練資料並以 HF／timm 在本地訓練模型，但當時 `dataset/` 與 `models/` 混著已確認無效的舊 Re-ID 資料與有疑慮的權重，而且**評估用的人工 GT 本身有問題**：stage2 使用的 `annotations_all8/labels` 是被「範圍限縮」改過的版本（7 欄、帶 conf），逐框比對只有 7976／10401 框對得上，IoU 中位數僅 0.83，多出 1339 框，且與 `mot_annotations_all8` 的 MOT GT 對不起來——也就是說 detection 評估其實不是在跟真正的人工 GT 比較。在啟動 Re-ID 資料重建之前，必須先把資料的權威來源、有疑慮資產的隔離方式，以及評估範圍限縮的機制定清楚。

## 這次變更做了什麼（What Changes / Decisions）

### D1. 評估資料集改名，沿用舊的內部命名

`dataset/中華三路_民生二路(東北角)yolo驗證用_all8/` → `dataset/eval/cctv_all8/`；`annotations_all8(orign)` → `annotations_all8`；`mot_annotations_all8(orign)` → `mot_annotations_all8`。改名只搬移目錄、不改內容，`dataset_layout.py` 與 `datasets.yaml` 沿用既有命名慣例，只換 `root`。限縮版與重複版（`annotations_all8(compasate)`、`mot_annotations_all8(Ver2)` 等）移入 `temp/`。

### D2. 原始影片依鏡頭歸位，新增 raw 版面

影片統一搬到 `dataset/raw/videos/<camera>/<seq>.mp4`（來源含 `video_seqs`、`bus_video/`、根目錄散落的 C000025 mp4；`bus_video/` 中 4 支與 `video_seqs` 內容相同的影片只留一份）。`datasets.yaml` 新增 `role`（`eval`／`source`）、`cameras`（明確清單）、`layout`（`annotated` 或 `raw`）三個欄位；`video_seqs` 更名為 `reid_sources`（`role: source`，涵蓋 C000043／054／085／294／297），`cctv_all8` 標為 `role: eval`（C000025）。評估鏡頭與訓練來源鏡頭互斥，由 smoke 測試檢查。

### D3. 隔離一律用 `mv`，並附 `DO_NOT_USE.md`

有疑慮的權重與資料（`models/yolo-reid/*`、`models/class_reid_weights/{person,scooter}`、`dataset/reid/` 全部內容、`all8` 的限縮版與重複版）一律搬到所在目錄下的 `temp/`，不刪除；每個 `temp/` 內的 `DO_NOT_USE.md` 記錄原路徑、隔離日期、原因與還原指令，且**先寫紀錄再搬移**。`config/` 與 `tests/golden/configs/` 對 `temp/` 的引用由 smoke 測試攔截。

### D4. ROI 以鏡頭為鍵的檔案與解析

新增 `config/roi/<camera>.yaml`（`camera`、`image_size`、`polygon`），C000025 由原本 config 頂層的 `det_corner` 移植而來。新增單一 ROI 解析函式：依序列名稱第一個空白前的鏡頭編號查找；沒有 ROI 檔時回傳整張影像並提示；`image_size` 與實際影片不符時報錯（不自動縮放）。情境 config 若仍帶頂層 `det_corner` 會直接載入失敗。

### D5. stage1 不再過濾軌跡，下游自行套用 ROI

`save_mot_format()` 移除多邊形過濾，7 欄軌跡檔保留全部追蹤結果。stage3（衝突事件）與 Re-ID 裁切蒐集改為讀入軌跡後、以同一 ROI 解析函式做中心點硬過濾；**過濾必須在依 `track_id` 分組之前逐列進行**，否則會改變軌跡出現順序、進而改變 PET 事件（golden 驗證時實際踩到這個坑，修正後 24 份事件檔與舊基準完全一致）。

### D6. ignore 匹配放在 TrackEval 之前的本專案前處理（UA-DETRAC 式）

新增 ignore 前處理模組，接在 GT shadow 準備之後、TrackEval 執行之前：
1. GT 中心點在 ROI 外者標為 ignore。
2. 預測與全部 GT（含 ignore）以 IoU ≥ 0.5 做一對一最佳匹配（匈牙利演算法）。
3. 匹配到 ROI 內 GT 的預測保留。
4. 其餘預測中，匹配到 ignore GT 或中心點在 ROI 外者移除（不計 TP／FP）。
5. 移除全部 ignore GT。

只改 stage2 輸出目錄中的副本，不修改來源或 vendored TrackEval。最初實作採 MOT17 式（只移除匹配到 ignore GT 的預測），golden 實測顯示 ROI 外未匹配的預測大量變成 FP（bytetrack car 的 FP 60→551，HOTA 78.3→69.3），因此改採 UA-DETRAC 式，使 **FP 只會來自 ROI 內**；代價與風險已寫入 `specs/evaluation-roi-filtering/spec.md`。

### D7. detection 評估改在 stage2 輸出目錄組一份唯讀來源的資料集

在 `<stage2 輸出>/det_eval/<seq>/` 建立人工 GT 影像的 hardlink（symlink 會被 ultralytics 解析、把 `.cache` 寫回來源，故不可用）與 5 欄 label 複本，搭配對應 `data.yaml` 餵給 `det_model.val()`。不再使用 stage1 的追蹤標籤充當 GT，也不再需要幀號 +1 對齊。報表與終端摘要明確標示「ROI：未套用（全畫面）」。

### D8. `botsort_reid.yaml` 暫代權重

small_truck／scooter 原本引用已隔離的 `yolo-reid/yolo11x-reid-ultra.pt`，暫時改用 `models/yolo/yolo11x.pt`（與 person／truck 相同的 detector backbone），`use_reid` 維持 `true`，等 `rebuild-reid-dataset` 產出新模型後再替換。

### D9. 淘汰 YOLO-as-encoder 訓練端（Breaking）

刪除 `scripts/train_reid.py`、`src/safetrajai/reid/yolo_reid_trainer.py`、`src/safetrajai/reid/training/`；推論端 `encoders.ReID`／`encoders.Custom_ReID` 與 `utils/model.py` shim 保留，供後續 HF／timm 路線接入。

### D10. golden 基準重建

以新程式碼跑 `after1`／`after2` 自比確認差異為 0，再與舊基準 `before1` 比對，逐項歸類「必須相同」與「預期變動」，記錄於 `verification.md`；`TOLERANCE.md` 同步記錄基準由 `before1` 切換為 `after1` 的日期與原因。

### 明確不做的事

重新產生 Re-ID 裁切、HF 訓練腳本、`Custom_ReID` 支援 timm（留給 `rebuild-reid-dataset`）；detection 標註或訓練；修改 `third_party/trackeval/`；修改 tracker override 方法本體；調整追蹤參數。

## 新增規格

### `dataset-layout`（4 條 Requirement）

| Requirement | 內容摘要 |
|---|---|
| 資料依角色分層存放 | `dataset/` 分為 `raw/videos/<camera>/`、`eval/<dataset-id>/`、`reid/`、`reports/`；根目錄不得有散落影片；每個資料集登記 `role: eval\|source` |
| 評估鏡頭不得作為訓練資料來源 | `eval` 角色鏡頭（目前 C000025）不得出現在任何 `source` 資料集，smoke 測試檢查互斥 |
| 評估 GT 以原始人工標註為準且不可修改 | `cctv_all8` 的 detection／tracking GT 以 `(orign)` 為權威來源（10401 框）；任何 stage 執行不得修改來源檔案 |
| 有疑慮資產以 temp 隔離並標示不可使用 | 搬移（非刪除）到 `temp/` + `DO_NOT_USE.md`；設定檔不得引用 `temp/` 內路徑 |

### `evaluation-roi-filtering`（5 條 Requirement）

| Requirement | 內容摘要 |
|---|---|
| ROI 以鏡頭為鍵定義 | `config/roi/<camera>.yaml`；無 ROI 檔時整張影像為 ROI 並提示；config 頂層 `det_corner` 直接報錯 |
| 追蹤評估以忽略區域限縮範圍 | UA-DETRAC 式 ignore 匹配（見 D6），FP 只來自 ROI 內；已記錄與 MOT17 式相比的代價與風險 |
| 評估中間產物只寫入輸出目錄 | ignore 處理後的 GT／預測不得回寫來源或 stage1 軌跡檔 |
| stage1 軌跡輸出不做 ROI 過濾 | 7 欄軌跡保留全部結果；stage2／stage3／Re-ID 裁切各自套用 ROI |
| detection 評估使用人工 GT 並標示 ROI 狀態 | 不得用 stage1 追蹤標籤充當 GT；報表須標示「未套用 ROI（全畫面）」 |

## 驗證結果摘要（verification.md）

- 環境：conda env `track_eval`、RTX 5090；日期 2026-09-23。
- **新基準穩定性**：`after1`／`after2` 自比對 8 個類別全部 0 差異。
- **與舊基準 `before1` 比對**：
  - **必須相同（0 差異）**：`conflict_events`（stage3 事件）、`reid_crops`（裁切清單）、`homography`、`yolo_labels`（stage1 逐幀追蹤結果，含 `botsort_reid` 暫代權重後仍與 before1 相同）。
  - **預期變動**：`mot_tracks`（stage1 不再過濾，before1 每列都存在於 after1，只多出 ROI 外的列）、`trackeval`（評估範圍改為忽略區域）、`grid_search`（分數隨 stage2 改變，但最佳參數不變）、`other`（少 30 份舊 detection `data.yaml`，多 108 份 `trackers_raw/` 備份）。
- **stage2 指標三方對照**（舊對稱硬過濾／MOT17 式／UA-DETRAC 式，6 段序列）：UA-DETRAC 式相對舊流程 FN 減少（跨界物件不再被誤砍成 FN）、FP 持平或略少、HOTA 普遍略升，此為評估方式修正而非模型進步；MOT17 式的 FP 暴增（例如 bytetrack car FP 60→551）驗證了改採 UA-DETRAC 的必要性。
- **stage3 回歸問題**：第一次執行時 PET 事件因「先分組再過濾」改變軌跡順序而與 before1 不同；修正為逐列過濾後，24 份事件檔全部與 before1 相同。
- **detection 評估改用人工 GT**：6 段序列、共 10401 GT 框，各序列 mAP50 介於 0.961～0.993；改用 hardlink 避免 `.cache` 寫回來源目錄。
- **資料完整性**：搬移前後檔案數一致（1,127,543，只在比對路徑與檔案數，未逐檔讀取內容）；6 份來源 `gt.txt` 雜湊搬移前後相同；`pytest`：128 passed、1 skipped。

## 對後續開發的影響

- Re-ID 資料重建（`rebuild-reid-dataset`）已在此基礎上規劃：`reid_sources` 資料集、`config/roi/` 慣例、`temp/` 隔離慣例、UA-DETRAC ignore 語意都是其前置條件，且該 change 目前**尚未開始執行**（見上方進度總覽）。
- 新的 golden 基準為 `after1`（取代 `before1`），`tests/golden/TOLERANCE.md` 已記錄切換日期與原因；後續任何比對都應以 `after1` 為準。
- `det_corner`（config 頂層欄位）已停用，一律改用 `config/roi/<camera>.yaml`；沿用舊欄位會直接報錯，不會悄悄退回舊行為。
- `botsort_reid.yaml` 的 small_truck／scooter 權重仍是暫代方案，需等 `rebuild-reid-dataset` 產出正式 Re-ID 模型後替換。
- `scripts/reid_data/*` 目前指向已隔離的 `dataset/reid/` 內容，執行會因路徑不存在而報錯，其輸入將由 `rebuild-reid-dataset` 重新定義。

## 附錄：目前專案架構圖

以下架構圖依 repo 現況（2026-09-24）整理，供對照本次變更影響的模組位置。

### 資料流總覽

```
影片 (video/, dataset/raw/videos/<camera>/)
        │
        ▼
┌─────────────────┐   YOLO11 偵測 + class_conf_thresh
│  Stage1 追蹤      │   class-aware BoTSORT / ByteTrack
└─────────────────┘
        │  trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt
        │  (7 欄，無 header，逐類別資料夾)
        ▼
┌─────────────────┐   忽略區域前處理 (ROI, UA-DETRAC 式，見 D6)
│  Stage2 TrackEval │   → HOTA / CLEAR / Identity
└─────────────────┘   → Excel 報表
        │
        ▼
┌─────────────────┐   Homography → 衛星座標
│  Stage3 衝突事件   │   TTC / PET 事件、熱區圖
└─────────────────┘
```

### 目錄結構

```
track_eval.py                      相容入口 wrapper → safetrajai.cli.run_pipeline.main()

scripts/                           CLI 入口
├── run_pipeline.py                主流程（stage1→2→3）
├── run_evaluation.py              只跑 stage2（對既有 stage1 輸出）
├── grid_search_bytetrack.py / grid_search_botsort.py / grid_search_botsort_reid.py
├── validate_homography.py
├── visualize_gt_on_video.py / visualize_gt_on_class_video.py
└── reid_data/                     Re-ID 資料切分與稽核
    ├── build_reid_split.py
    ├── generate_identity_review.py
    ├── apply_review_exclusions.py   （由 quarantine_flagged.py 改名而來）
    └── detect_contamination.py      （新增，尚未 commit）

src/safetrajai/
├── paths.py                       唯一的 PROJECT_ROOT／CONFIG_DIR
├── classes.py                     類別 ID 契約（person/car/scooter/small_truck/bus/truck）
├── roi.py                         以鏡頭為鍵的 ROI 解析（config/roi/<camera>.yaml，見 D4）
├── cli/
│   └── run_pipeline.py            argparse + main()：load_track_config() → load_config() → TrackingEvalPipeline
├── pipeline/                      Stage1（追蹤主流程）
│   ├── tracking_pipeline.py       TrackingEvalPipeline（run_stage1/2/3）
│   ├── sequence_tracker.py        單序列追蹤迴圈
│   ├── reid_crops.py              Re-ID 裁切蒐集（讀軌跡後以 ROI 中心點過濾，見 D5）
│   └── video_io.py
├── detection/
│   └── detector.py                單幀 YOLO 推論 + 逐類別信心門檻
├── tracking/                      class-aware tracker（改動前先讀 HANDOFF.md）
│   ├── class_aware_botsort.py     （override 方法不可更動，見 ADR 0004）
│   ├── class_aware_bytetrack.py   （同上）
│   ├── matching.py                get_dists_class_aware() 等自訂邏輯
│   └── config.py                  load_track_config() → CLASS_TRACKING_PARAMS
├── evaluation/                    Stage2（TrackEval + 報表）
│   ├── config_loader.py           load_config()
│   ├── dataset_layout.py          路徑組裝純函式（唯一權威）
│   ├── mot_converter.py / mot_format.py
│   ├── ignore_region.py           ROI 外 GT 標 ignore、匹配後移除 FP（D6）
│   ├── dataset_prep.py            人工 GT 的 detection 評估資料集（D7）
│   ├── trackeval_runner.py        呼叫 vendored TrackEval
│   ├── excel_export.py
│   └── pipeline.py
├── collision/                     Stage3（衝突事件，非真實碰撞，見 ADR 0003）
│   ├── detection_core.py          TTC/PET 核心運算
│   ├── trajectory_adapter.py      讀 stage1 MOT 軌跡（以 ROI 中心點過濾，見 D5）
│   ├── output.py                  ttc_events / pet_events 輸出
│   └── homography/                轉換／人工點選／驗證
├── reid/                          Re-ID 推論與資料重建（rebuild-reid-dataset 進行中）
│   ├── model.py                   Custom_ReID 實作（utils/model.py 只是 pickle shim）
│   ├── encoders.py                YOLO-as-encoder 推論端（保留，見 D9）
│   ├── manifest.py / dataset_index.py / contamination.py   （新增，重建 dataset/reid/ 用）
├── grid_search/
│   └── core.py                    grid search 共用邏輯（trial → patch params → 還原）
└── visualization/
    └── drawing.py                 ROI 與追蹤框繪製

utils/model.py                     pickle 相容 shim（不可刪除，見 ADR 0001）
third_party/trackeval/             vendored TrackEval（唯讀，見 ADR 0002）

config/                            每檔自包含情境設定
├── bytetrack.yaml / botsort_noreid.yaml / botsort_reid.yaml / botsort_save_reid_crops*.yaml
├── collision_detect.yaml
├── grid_search_bytetrack.yaml / grid_search_botsort.yaml
├── datasets.yaml                  dataset id 對照表（role/layout/cameras，見 D2）
├── reid_split.yaml                （新增）
├── roi/<camera>.yaml              （D4）
└── homography_points/、checkpoints/

openspec/                          規格驅動變更（proposal/design/specs/tasks）
docs/adr/                          關鍵決策紀錄（0001~0006）
tests/                             smoke 測試、golden 比對工具
```

> 說明：`reid/` 下的 `contamination.py`、`dataset_index.py`、`manifest.py`，以及 `collision/` 拆成 `detection_core.py` / `output.py` / `trajectory_adapter.py`，是 `rebuild-reid-dataset` 進行中的產出，尚未完全反映在本文件正文的決策描述中，僅供對照現況參考。

---
*本文件由 Claude Code 依 `openspec/changes/archive/2026-09-23-reorganize-dataset-and-eval-gt/` 下的 `proposal.md`、`design.md`、`tasks.md`、`verification.md` 與 `openspec/specs/dataset-layout/spec.md`、`openspec/specs/evaluation-roi-filtering/spec.md` 整理而成，內容以上述原始檔案為準。*
