# ARCHITECTURE.md

## 專案名稱
safetrajai_eval — CCTV 交通物件偵測與多物件追蹤（MOT）評估流程

## 目標
針對路口 CCTV 影像，用 YOLO11 偵測六類交通參與者（person、car、scooter、small_truck、bus、truck），再用自行修改的 class-aware BoTSORT / ByteTrack 做多物件追蹤，最後用 TrackEval 計算 HOTA / CLEAR / Identity 等指標，輸出 Excel 報表，作為追蹤參數調優與模型選擇的依據。

## 範圍
- **包含**：偵測 → 追蹤 → 評估 的完整 pipeline、追蹤超參數 grid search、GT/追蹤結果疊圖視覺化。
- **不包含**：YOLO 偵測模型本身的訓練（`models/yolo/` 下的權重是外部訓練好的產物）、線上服務化（無 API/後端）。
- `utils/model.py` 是 ReID 模型訓練用的工具函式，目前沒有任何腳本引用，屬於孤兒程式碼，不算現行流程的一部分，但因既有權重檔的 pickle 相容性限制不可搬移（詳見下方架構章節）。

## 技術棧
- Python + PyTorch / torchvision
- Ultralytics（YOLO11 偵測、`BYTETracker` / `BOTSORT` 基底類別）
- OpenCV、NumPy、SciPy
- TrackEval（vendored，於 `trackeval/`）
- openpyxl（Excel 報表輸出）、PyYAML（設定檔）
- 無 `requirements.txt`／`pyproject.toml`、無 CI；已導入 git 版本控制，但實驗版本仍靠檔名區分（如 `config/botsort_reid.yaml`）

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

## 專案架構

Detection / Tracker / Re-ID / grid_search 四個模組統一放在 `module/` 底下、彼此同一層；`evaluation/` 是橫跨 Detection eval 與 Tracker eval 的共用模組，不屬於這四者之一，留在根目錄：

```
safetrajai_eval/
├── track_eval.py                CLI 入口：TrackingEvalPipeline（stage1 tracking → stage2 TrackEval）
├── config/                       6 個情境設定檔（取代根目錄 config.yaml，見下方「設定檔」）
│
├── module/
│   ├── detection/                 【Detection 模組】
│   │   └── detector.py            Detector 類別：YOLO 推論 + 逐類別信心過濾
│   │                               （訓練腳本缺口，見下方「流程缺口」）
│   │
│   ├── tracker/                   【Tracker 模組】（原 trackers/ → tracker/ → module/tracker/）
│   │   ├── class_aware_botsort.py / class_aware_bytetrack.py   class-aware 追蹤核心
│   │   ├── config.py               逐類別參數查表（match_thresh、track_buffer、use_reid…）
│   │   └── matching.py, HANDOFF.md（track_config*.yaml 已移到 config/，見下方）
│   │
│   ├── reid/                       【Re-ID 模組】只保留兩條路徑（SOTA_ReID／reid_models 已移除），
│   │   │                          自行設計的模型架構＋訓練腳本說明見 module/reid/HANDOFF.md
│   │   ├── encoders.py             ReID（YOLO 本身當 encoder）／Custom_ReID（自己訓練的模型，推論期特徵抽取，被 module/tracker/ 引用）
│   │   ├── yolo_reid_trainer.py    YOLO ReID 訓練用 hook/projector（ReIDHookManager、BatchHardTripletLoss）
│   │   └── training/               重新設計的訓練資料/訓練迴圈（取代已移除的 train_utils.py）
│   │       ├── dataset.py           讀取 dataset/reid/crop_dataset/{class}/{id}/*.jpg，PK 抽樣（P 身分 × K 影像／batch）
│   │       └── train.py             CLI 訓練腳本，只走 YOLO-as-encoder 架構，輸出可被 encoders.py::ReID 載入的權重
│   │
│   └── grid_search/                【grid_search 模組】超參數 grid search（botsort、bytetrack、botsort_reid）
│                                     原巢狀在 tracker/grid_search/ 下，後獨立成與其他三者同層的模組
│
├── evaluation/                   【共用】DET 評估 + TrackEval 評估的 pipeline 邏輯，不屬於 module/ 四者之一
│   ├── config_loader.py, dataset_layout.py, dataset_prep.py, mot_converter.py
│   ├── trackeval_runner.py, excel_export.py, mot_format.py, pipeline.py
│
├── utils/                        通用小工具
│   ├── visualization.py           畫面標註（draw_polygon_and_save、draw_track_box 等）
│   └── model.py                    ReIDNet 定義（不可搬移，見下方 pickle 相容性）
│
├── trackeval/                     vendored TrackEval 第三方套件
├── archive/                        歸檔：past_conf/、past_py_conf/、trackers_trackeval_duplicate/ 等舊碼
└── visualize_gt_on_*.py            GT / 追蹤結果疊圖視覺化
```

**pickle 相容性限制**：`utils/model.py` 的模組路徑被既有權重檔（`models/class_reid_weights/**`）以 `torch.save()` 整物件序列化寫死在檔案內，`torch.load(weights_only=False)` 還原時需要能用原路徑 `import`，因此**固定留在原本的位置**，不隨模組重組搬動；詳見 `CLAUDE.md`「架構」章節。

## 資料與輸出目錄（現況全貌）

上方「專案架構」只涵蓋程式碼模組。以下是根目錄下**非程式碼**的資料/輸出/實驗記錄目錄：

```
safetrajai_eval/
├── dataset/                    原始資料（大型，已 gitignore）
│   ├── 中華三路_民生二路(東北角)yolo驗證用_all8/   正式評估資料集（annotations_all8/、mot_annotations_all8/ 等）
│   ├── reid/                     Re-ID 資料集（統一放進這裡，見 dataset/reid/README.md；
│   │   │                          README.md／build_reid_split.py 是 dataset/ 底下唯二不被 gitignore 排除的檔案）
│   │   ├── build_reid_split.py    crop_dataset → reid_category_dataset / reid_dataset 切分腳本
│   │   ├── crop_dataset/           save_reid_crops 蒐集的原始裁切（{class}/{影片}_track{id}/*.jpg），
│   │   │                          目前 1185 身分／33399 張（詳細統計見 dataset/reid/README.md）
│   │   ├── reid_category_dataset/  切分輸出：逐類別各自切 train/val
│   │   └── reid_dataset/            切分輸出：6 類別混合切一次 train/val
│   ├── video_seqs/                config/botsort_save_reid_crops_video.yaml 用的來源結構：
│   │                              annotations_video/annotation_<seq>/<seq>.mp4，來源是 video/（見下方）
│   │                              5 支不同路口原始影片各取前 60 秒
│   └── *.mp4（4 支散落在 dataset/ 根目錄，非上述結構化資料夾內）  來源不明的測試片段，目前無程式碼引用
├── dataset.zip                 dataset/ 的壓縮備份（465MB，內容與 dataset/ 幾乎重複）
│
├── video/                       原始路口影片（480MB，已 gitignore）：5 支不同路口的
│                                原始 mp4（非既有「中華三路_民生二路」資料集），用來補足 dataset/reid/crop_dataset/
│                                裡 bus/person/truck/small_truck 身分數過少的問題，見 config/botsort_save_reid_crops_video.yaml
│
├── models/                     權重檔（大型，已 gitignore），已依類型分子目錄：
│   ├── yolo/                    YOLO 偵測權重（yolo11x_dennis.pt 等）
│   ├── yolo-cls/、yolo-reid/     YOLO 分類 / YOLO-as-ReID-encoder 權重
│   └── class_reid_weights/       Custom_ReID 逐類別權重（person/、scooter/，pickle 相依 utils/model.py，見上方限制）
│
├── runs/                        pipeline 輸出（大型，已 gitignore），目前混雜三種性質不同的內容：
│   ├── track_eval/               目前這次執行的即時輸出（= config/*.yaml 的 OUTPUT_ROOT，會被下次執行覆寫）
│   ├── detect/                   ultralytics `model.val()`/`predict()` 自動產生的暫存輸出（train、val1-9…），非本專案手動管理，重新執行會自動重建
│   └── track_eval(0411_ByteTrack)/、track_eval(0413_BotSort_noreid)/ 等   人工封存的歷史實驗結果（含展開目錄與對應 .zip，命名以實驗日期＋tracker/設定區分）
│
├── wandb/                       Weights & Biases 訓練 log（74MB，已 gitignore），對應舊 ReID 訓練流程，由 wandb 套件依執行時 cwd 自動產生
│
├── .claude/                      Claude Code 本機設定，非專案邏輯的一部分
└── requirement.txt               環境建置腳本（pip install 指令集合，非標準 requirements.txt 格式；未版控）
```

**現況觀察**（依此給出下方「目錄整理建議」）：
- `runs/` 底下混雜「目前即時輸出」「ultralytics 自動暫存」「人工封存的歷史實驗」三種性質，職責不清。
- `dataset.zip`（465MB）與展開後的 `dataset/` 目錄內容重複，用途待確認（備份快照或可清理的舊檔）。
- `dataset/` 根目錄下 4 支散落 mp4 未歸入任何結構化子資料夾。

## 已知限制
- 無自動化測試，改動需手動跑 `track_eval.py` 驗證輸出（見 `GUIDE.md`）。
- 實驗版本與參數調整靠 `config/*.yaml` 情境檔名區分（固定 6 個，見 `CLAUDE.md`「設定檔」），容易混淆的舊變體已歸檔到 `archive/track_config_variants/`。
- ReID 目前預設關閉：實驗顯示開啟後對 person 有幫助，但會損害 scooter 追蹤並增加 IDSW（見 `EXPERIMENTS.md`）。
- **Detection 模組缺訓練腳本**：YOLO 偵測模型訓練本來就不在此專案範圍內（`models/yolo/*.pt` 為外部訓練產物），`module/detection/` 目前只有推論用的 `Detector`。待未來有專門的 YOLO 訓練腳本再一併整理進 `module/detection/`（見下方「流程缺口」）。
- **Re-ID 訓練資料蒐集 + 訓練流程**：`track_eval.py` 的 `save_reid_crops` 功能（`config/botsort_save_reid_crops.yaml`）蒐集 `dataset/reid/crop_dataset/{class}/{影片}_track{id}/*.jpg`，`module/reid/training/` 提供對應的資料讀取（PK 抽樣）與訓練腳本，只支援 YOLO-as-encoder + triplet loss 架構（對應 `encoders.py::ReID`）；`Custom_ReID` 用的 `ReIDNet`（`utils/model.py`）沒有訓練程式碼，不在這次重新設計範圍內。`dataset/reid/build_reid_split.py` 可把 `crop_dataset/` 依身分切成 train/val，輸出到 `dataset/reid/reid_category_dataset/`（逐類別）或 `dataset/reid/reid_dataset/`（6 類別混合），詳見 `dataset/reid/README.md`。

---

## 待辦與已知落差

待辦與未滿足需求清單。以下項目是整理程式碼時發現的落差，非正式規劃，請專案負責人確認優先順序與是否要處理。已完成項目的驗證過程見 git commit history，不重複列在此處。

### Re-ID 訓練資料 / 模型 / 訓練方式（進行中）
- [ ] **舊資料 `dataset/reid/crop_dataset/train/{train,val}/{identity}/` 與新資料 `{class}/{影片}_track{id}/` 是兩套不同結構**：舊資料是扁平的（不分類別，446 個身分資料夾，各自命名為純數字如 `000351`），新的 crop 蒐集功能改成先分類別再分身分。需要決定：(a) 舊資料要不要轉換/歸類到新結構下，(b) 新舊資料是否共用同一套 Dataset 讀取邏輯，(c) 訓練資料的 train/val 切分方式（舊資料是資料夾層級切好的，新蒐集的資料現在可以用 `build_reid_split.py` 切）。
- [ ] **`bus` 身分數過少（目前 5 個，遠低於其他類別）**：現有兩批來源影片（中華三路_民生二路 6 支序列 + `video/` 5 支不同路口影片）路口車流裡都幾乎沒拍到公車。需要另外找有公車經過的路口影片，用 `config/botsort_save_reid_crops.yaml` 或 `botsort_save_reid_crops_video.yaml`（換 `SOURCE_VIDEO_DIR`／`video/` 來源）補蒐集，否則 `bus` 類別的 Re-ID 訓練資料量會嚴重不足。
- [ ] **舊版扁平 Re-ID 資料（`dataset/reid/crop_dataset/train/`，444 身分，資料夾直接命名為 6 碼數字如 `000100`，無類別標籤）如何處理**：與現行 6 類別資料夾是不同產生流程，因無類別標籤，無法自動合併進現行分類別結構，需人工逐一重新分類才能整合（資料標註工作，不在程式碼重構範圍）。**已決定**：搬到 `archive/reid_legacy_flat_dataset/` 做區隔（比照 `archive/track_config_variants/` 等既有命名慣例），讓 `crop_dataset/` 底下只剩現行 6 類別資料夾；搬移是批次檔案操作，待確認後執行，尚未執行。
- [ ] **Re-ID 資料品質過濾機制**（架構已規劃、門檻未定）：現行 `ReidCropCollector` 只做多邊形區域過濾，`training/dataset.py::ReIDCropDataset` 只做身分層級「影像數不足就整個身分捨棄」，都沒有單張影像品質判斷（尺寸過小/模糊/邊界截斷）或跨圖一致性檢查。已規劃兩層架構：Tier 1（`ReidCropCollector` 新增 `quality_filters` 掛載點，蒐集時濾單張影像，介面已定義、預設不啟用）、Tier 2（新增 `dataset/reid/filter_crop_dataset.py`，訓練前處理跨圖/身分層級問題，只標記不自動刪除）。具體過濾條件與門檻值留待實際資料調校。

### 待確認 / 待清理
- [ ] `archive/` 底下的舊碼是否還需要保留，或可進一步刪除，留給專案負責人決定。
- [ ] **`track_buffer` 單位在 BoTSORT 與 ByteTrack 兩個 tracker 不一致**：`class_aware_botsort.py` 把 `track_buffer` 當作「秒」（`cls_max_time_lost = track_buffer × frame_rate`），`class_aware_bytetrack.py` 仍當作「30fps 基準幀數」（`cls_max_time_lost = (frame_rate/30) × track_buffer`）。同一個 `track_buffer: 2.0` 在 30fps 下，BoTSORT 算出 60 幀存活時間，ByteTrack 只有 2 幀——兩者相差 30 倍。**已確認不會修正**：換算公式就在繼承自 `ultralytics.trackers` 的 `update()` 內，該方法程式碼一字不可更動（使用者明確指示，見 `.claude/rules/tracker.md`「不可更動的核心方法」），因此統一決定作廢，維持現狀分歧。

### 流程缺口
- [ ] 沒有自動化測試，也沒有「基準指標」可供迴歸比對（見 `GUIDE.md`「手動驗證流程」待補充）；評估是否要建立至少一組基準 HOTA/AssA 數值。
- [ ] 沒有 `requirements.txt` / `pyproject.toml`，環境重建靠人工記憶依賴套件；建議補上。
- [ ] **Detection 模組缺訓練腳本**：`module/detection/` 目前只有推論用的 `Detector`（`module/detection/detector.py`），沒有 YOLO 訓練腳本——這本來就不在專案原始範圍內（`models/yolo/*.pt` 是外部訓練產物）。待日後有專門的 YOLO 訓練腳本，再一併整理進 `module/detection/`。
- [ ] Tracker / Re-ID 模組同樣沒有「測試腳本」，目前是用 `module/grid_search/*.py`（含 `grid_search_botsort_reid.py`）兼作超參數搜尋與驗證，沒有獨立的單元測試或迴歸測試腳本；是否要另外補上，待評估。

### 待驗證（見 `EXPERIMENTS.md` 對應段落）
- [ ] ReID on/off 的量化數據補齊（目前只有程式碼註解結論，無 Excel/CSV 佐證）。
- [ ] FPS/解析度比較尚未整理出整體結論摘要（原始數據 `fps_compare/` 已移除，如需重新整理需從 git 歷史 commit `7061f97` 取回或重新跑一次比較）。
- [ ] `det_corner` 偵測區域多邊形設定對指標的影響。
- [ ] `fuse_score` 開關（目前全域關閉）對各類別的影響。
- [ ] `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據。

### 目錄整理建議
`git mv` 前一律先確認：

| 項目 | 現況 | 建議目標 | 風險 / 需同步更新 |
|---|---|---|---|
| `runs/track_eval(0411_ByteTrack)/` 等歷史封存實驗＋對應 `.zip` | 混在 `runs/` 底下，與「目前即時輸出」`runs/track_eval/` 放在一起 | 移到 `archive/experiment_runs/` 或 `output/archive/`，讓 `runs/` 只保留 `config/*.yaml` 的 `OUTPUT_ROOT` 即時輸出 | 中。總量約 1.5GB，實體搬移耗時；已 gitignore 不影響版控，但搬移後若有腳本用相對路徑讀取這些歷史結果需另外確認（目前掃描沒有程式碼引用） |
| `runs/detect/`（train、val1-9…） | ultralytics 自動產生的暫存輸出 | 不建議搬，可視需要清空 | 低。重新執行 `evaluation/pipeline.py` 的 `det_model.val()` 會自動重建 |
| `dataset/` 根目錄下 4 支散落 mp4 | 與兩個結構化資料集資料夾平行放置，無程式碼引用 | 若確認非現行流程需要，移到 `dataset/samples/` 或清除；若不確定用途則先不動 | 中。屬於唯讀原始資料範疇，需確認這 4 支影片是否還在使用 |
| `dataset.zip`（465MB） | 根目錄，內容與展開後的 `dataset/` 幾乎重複 | 確認是否為必要備份；若是建議移出工作目錄（如另外的備份位置），不建議留在 repo 工作目錄 | 高，需決定。體積大且無法判斷是否為唯一備份來源，**不會主動刪除** |
| `wandb/`（74MB） | 根目錄，舊 ReID 訓練 run 的 log | 不建議手動搬移 | wandb 套件依執行時 cwd 自動產生位置，手動搬了下次訓練還是會在原處重建；要規範化須在訓練腳本設定 `WANDB_DIR`，不是檔案搬移能解決的 |
| `models/` | 已依 yolo/yolo-cls/yolo-reid/class_reid_weights 分類 | 現況已算整齊，可選配補一份 `registry.yaml` 記錄各權重對應的訓練來源/實驗 | 低，錦上添花，非必要 |
| `requirement.txt`（未版控） | 根目錄，pip install 環境建置腳本 | 位置正確（根目錄慣例），內容其實是 shell 腳本非標準 requirements 格式，可考慮改名如 `setup_env.sh` 更準確；是否納入版控由你決定 | 低 |

## 待補充
> 此檔案是根據現有程式碼與文件整理出的「已知落差」，不包含尚未提出的新功能需求。若有新的功能/優化需求，請直接加在下方，並標註提出人與日期。
> 以上為新建骨架，需要專案負責人補充：專案的實際使用者/利害關係人、交付週期、與上游（偵測模型訓練）/下游（如有應用系統）的介接方式。
