# Tasks

## 1. 來源鏡頭的 ROI（D1）

- [ ] 1.1 新增 `scripts/roi/make_roi_picker.py --camera <id>`：取該鏡頭第一支原始影片的首幀，輸出嵌入影像的單檔 HTML 到 `runs/roi_picker/<camera>.html`，點選頂點後即時顯示並可下載 `config/roi/<camera>.yaml` 內容；以 `--help` 正常、對 C000043 產生的 HTML 可在瀏覽器開啟並輸出與影片解析度一致的 `image_size` 作為驗證
- [ ] 1.2 為 C000043、C000054、C000085、C000294、C000297 產生點選頁並交給使用者繪製；取得使用者提供的 5 份 YAML 後寫入 `config/roi/`；以 `resolve_roi()` 對每支鏡頭的一支影片都能成功解析（尺寸檢查通過）作為驗證

## 2. 程式修改

- [ ] 2.1 `evaluation/pipeline.py` 的 detection 評估改用 `cfg.imgsz`；以 `pytest` 通過、`after1` 情境設定下 imgsz 仍為 640 作為驗證
- [ ] 2.2 `tracking_pipeline` 支援 `frame_rate: auto`（建立追蹤器前讀影片幀率並取整），數值設定維持原行為；以單元測試涵蓋 `auto` 與數值兩種情況作為驗證
- [ ] 2.3 `save_mot_format()` 的影像與 label glob 改為 `*`；以非 C000025 序列名稱的單元測試能產生非空軌跡檔作為驗證
- [ ] 2.4 新增 `src/safetrajai/reid/dataset_index.py`：序列名稱轉日期（解析失敗報錯）、索引欄位定義、索引讀寫；以單元測試涵蓋日期解析成功與失敗、讀寫往返作為驗證
- [ ] 2.5 改寫 `ReidCropCollector`：ROI 過濾、最短邊 32px、每條軌跡 0.5 秒抽樣、索引一次寫出、索引已存在時報錯；移除 `quality_filters` 與 `make_min_size_quality_filter`；`sequence_tracker` 傳入 `CAP_PROP_POS_MSEC`（異常時退回幀號換算）；以單元測試涵蓋 spec 的抽樣、過小框、索引與檔案一致、日期取得情境作為驗證
- [ ] 2.6 `run_stage1()` 在 `save_reid_crops` 開啟時，於追蹤前檢查評估鏡頭與 ROI 檔是否存在，任一不通過就報錯；以單元測試（C000025 序列被拒、缺 ROI 的來源鏡頭被拒，兩者都沒有任何輸出）作為驗證
- [ ] 2.7 情境設定支援選用的 `paths.cameras`（只能是資料集登記鏡頭的子集，否則報錯）；以單元測試涵蓋子集與非法鏡頭作為驗證
- [ ] 2.8 新增 `src/safetrajai/reid/manifest.py`：排除清單套用、最少 4 張、最多 30 張的均勻抽樣、依 `config/reid_split.yaml` 切分、評估鏡頭檢查、依鍵排序；新增 `config/reid_split.yaml`；以單元測試涵蓋 spec 的身分張數、依日期切分、評估鏡頭不出現、可重現情境作為驗證
- [ ] 2.9 改寫 `scripts/reid_data/build_reid_split.py` 為 `--version` CLI，輸出 `manifest.csv`、`summary.json`、`split_rules.yaml`，版本已存在時報錯；以對測試用的小型索引產生 manifest、`--help` 正常作為驗證
- [ ] 2.10 新增 `scripts/reid_data/detect_contamination.py`：讀索引、每張裁切只讀一次，以 DINOv2 特徵同時做類別中心點比對與軌跡內外觀離群值，輸出兩份標記檔到 `dataset/reports/reid/`；刪除 `check_class_consistency.py`；以對測試用小型裁切集執行後裁切目錄檔案清單不變、每張裁切只被開啟一次作為驗證
- [ ] 2.11 改寫 `generate_identity_review.py`：讀索引與兩份標記、涵蓋標記身分與全部 bus、每身分最多 12 張縮圖；將 `quarantine_flagged.py` 改名為 `apply_review_exclusions.py`，改為合併進 `dataset/reid/exclusions.csv`（預設 dry-run）；以測試用資料產生審查頁、套用確認清單後排除清單正確且裁切檔案未移動作為驗證
- [ ] 2.12 跑 `pytest` 確認全部通過

## 3. 工具模型與追蹤參數比較（D2、D3，使用者已同意執行範圍）

- [ ] 3.1 以 `after1` 的 stage1 輸出，跑 dennis@640 與 `yolo11x.pt`@1280 的 detection 評估（6 段序列、432 張人工 GT 影像，輸出在 `runs/`）；依 D2 規則選定工具模型，結果寫入 `verification.md`
- [ ] 3.2 撰寫純度優先的 `class_tracking` 初版，與現行參數各以 `frame_rate: auto` 在 `cctv_all8` 上跑 stage1、stage2；比較 IDSW、IDP、Frag，依 D3 規則選定（最多迭代一次），結果寫入 `verification.md`
- [ ] 3.3 建立 `config/reid_crops.yaml`（`paths.dataset: reid_sources`、`reid_crop_dir: ./dataset/reid/crops`、選定的工具模型與追蹤參數、`frame_rate: auto`）；更新 `botsort_save_reid_crops_video.yaml` 的輸出位置或標註改用新設定；以 config 載入測試通過作為驗證

## 4. golden 裁切情境（D9，使用者已同意執行範圍）

- [ ] 4.1 `tests/golden/configs/botsort_save_reid_crops.yaml` 改用 `reid_sources` 與 `paths.cameras: [C000043]`；跑 `tests/golden/run_all.sh reid1` 與 `reid2`（各約 11 分鐘）；以 `compare.py reid1 reid2` 為 0 差異、`compare.py after1 reid1` 除裁切情境外其他情境皆為 0 差異作為驗證
- [ ] 4.2 在 `tests/golden/TOLERANCE.md` 記錄基準改為 `reid1` 的日期、原因與差異範圍；同步 CLAUDE.md 的 golden 比對指令

## 5. pilot（D10，需使用者同意執行範圍）

- [ ] 5.1 以 `config/reid_crops.yaml` 只對 `C000054 民族一路 天祥二路_20250623065608` 跑裁切蒐集、污染偵測與審查頁，並輸出這支影片前 1 分鐘的疊框影片；檢查索引與檔案一致
- [ ] 5.2 統計 pilot 的裁切數、身分數、各類別分布、各類別因最短邊 < 32px 而略過的框數與該序列目錄的寫入量，外推全量（31 支 C000054＋4 支 60 秒影片），並檢視 DINOv2 距離分布、定案門檻；把估算與門檻寫入 `verification.md`，向使用者回報並請使用者目視抽查疊框影片，取得同意後才進行第 6 節

## 6. 全量產生與審查（需使用者同意）

- [ ] 6.1 對 `reid_sources` 全部 35 支影片執行裁切蒐集；以每支序列的索引都存在、索引列數等於裁切檔案數（只列路徑，不讀內容）作為驗證
- [ ] 6.2 執行污染偵測（DINOv2 單次讀取），產生審查頁；產生 `v1` manifest；以 `summary.json` 顯示評估鏡頭為 0、每身分張數介於 4～30、C000054 的 0629 全在 val 作為驗證
- [ ] 6.3 使用者完成審查並匯出確認清單後，以 `apply_review_exclusions.py --apply` 寫入排除清單，產生 `v2` manifest；以 `v2` 不含排除身分、裁切檔案未移動作為驗證

## 7. 文件與收尾

- [ ] 7.1 更新 `scripts/reid_data/README.md`（新流程：蒐集 → 標記 → 審查 → 排除 → manifest）、CLAUDE.md 的 Re-ID 章節與常用指令、`CONTEXT.md`（視需要補上 manifest、排除清單等術語）；移除 CLAUDE.md 中已解決的 `save_mot_format` 已知限制
- [ ] 7.2 撰寫 `verification.md` 的總結（工具模型、追蹤參數、pilot 估算、全量數量、`v1`／`v2` 摘要）；跑 `pytest` 與 `openspec validate --strict` 確認通過
