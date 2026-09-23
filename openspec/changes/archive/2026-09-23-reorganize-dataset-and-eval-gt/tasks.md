# Tasks

## 1. 搬移前準備

- [x] 1.1 確認 `runs/refactor_golden/before1` 存在且完整；以 `python tests/golden/compare.py before1 before1` 自比無差異作為驗證
- [x] 1.2 以 `find dataset models bus_video -type f -print0 | xargs -0 sha256sum` 產生搬移前清單，存成 `runs/reorganize-dataset-and-eval-gt/inventory_before.txt`（只含路徑與雜湊；清單約 200MB，放在 gitignore 的 runs/ 而非 change 目錄）；以檔案行數等於 `find ... | wc -l` 作為驗證

## 2. 隔離有疑慮的資產（D3）

- [x] 2.1 在 `models/yolo-reid/temp/` 撰寫 `DO_NOT_USE.md`，再把 `models/yolo-reid/*.pt` 全部 `mv` 進去；以 `ls models/yolo-reid` 只剩 `temp/`，且表格列數等於搬入的檔案數作為驗證
- [x] 2.2 在 `models/class_reid_weights/temp/` 撰寫 `DO_NOT_USE.md`，再把 `person/`、`scooter/` 移入；以 `torch.load(..., weights_only=False)` 能從新路徑載入 `ReIDNet` 作為驗證
- [x] 2.3 在 `dataset/reid/temp/` 撰寫 `DO_NOT_USE.md`，再把 `dataset/reid/` 既有的全部項目移入（含 `crop_dataset`、兩種切分、`identity_review`、報表、`.py`、`__pycache__`）；以 `ls dataset/reid` 只剩 `temp/` 作為驗證

## 3. 評估資料集與原始影片歸位（D1、D2）

- [x] 3.1 將 `dataset/中華三路_民生二路(東北角)yolo驗證用_all8/` 改名為 `dataset/eval/cctv_all8/`；先在它底下建立 `temp/DO_NOT_USE.md`，把 `annotations_all8`、`annotations_all8(compasate)`、`mot_annotations_all8`、`mot_annotations_all8(Ver2)` 移入 `temp/`；以 `ls` 確認結果作為驗證
- [x] 3.2 將 `annotations_all8(orign)` 改名為 `annotations_all8`、`mot_annotations_all8(orign)` 改名為 `mot_annotations_all8`；把 `mot_annotation_*` 內非 GT 的產物（`trackers/`、`*.xlsx`、`*.docx`）移到 `temp/` 並記錄；以每個 `mot_annotation_*` 只剩 `gt/`、`img1/`、`seqinfo.ini` 作為驗證
- [x] 3.3 驗證權威 GT：寫一段一次性檢查（放在 change 目錄的 `checks/`，不進 `src/`），逐幀比對 YOLO label 框數與 MOT GT 列數；以 6 段序列全部相同、總數 10401 作為驗證
- [x] 3.4 建立 `dataset/raw/videos/<camera>/`，把 `video_seqs` 的 8 支、`bus_video/` 的 31 支、`dataset/` 根目錄的 C000025 mp4 依鏡頭 `mv` 進去；4 支內容相同的重複影片先記錄 checksum 再只保留一份；以 `dataset/raw/videos` 共 36 支、`bus_video/` 與 `dataset/video_seqs/` 已不存在作為驗證
- [x] 3.5 建立空的 `dataset/reports/`；在 `runs/reorganize-dataset-and-eval-gt/` 產生只含路徑的 `paths_after.txt`（不讀檔案內容，見 CLAUDE.md「禁止大量 I/O」），以「搬移後檔案總數 = 搬移前總數 − 4 支重複影片 + 新增的 `DO_NOT_USE.md`」，且搬移前清單中的每個檔名都能在搬移後找到作為驗證

## 4. 資料集登記（D2）

- [x] 4.1 更新 `config/datasets.yaml`：`cctv_all8` 的 root 改為 `./dataset/eval/cctv_all8/`，並加上 `role: eval`、`layout: annotated`、`cameras: [C000025]`；`video_seqs` 改為 `reid_sources`，設 `role: source`、`layout: raw`、`cameras: [C000043, C000054, C000085, C000294, C000297]`，root 為 `./dataset/raw/videos/`；以 `load_dataset_registry()` 能讀出新欄位作為驗證
- [x] 4.2 在 `dataset_layout.py` 新增序列名稱轉鏡頭編號的解析，以及 `layout: raw` 的影片探索；`discover_source_sequences()` 依 layout 分派；以新增的單元測試（鏡頭解析成功與失敗、raw 探索）通過作為驗證
- [x] 4.3 `config/botsort_save_reid_crops_video.yaml` 改用 `paths.dataset: reid_sources`；以 `python track_eval.py --config config/botsort_save_reid_crops_video.yaml` 的 dry 探索（只列出序列、不跑追蹤）列出 35 支 source 影片作為驗證
- [x] 4.4 在 `tests/test_smoke.py` 新增檢查：每個資料集都有合法 `role`；eval 與 source 的鏡頭互斥；`config/`、`tests/golden/configs/` 內沒有指向 `temp/` 的路徑；以 `pytest` 通過，且故意製造衝突時失敗作為驗證

## 5. 以鏡頭為鍵的 ROI（D4）

- [x] 5.1 新增 `config/roi/C000025.yaml`（`camera`、`image_size: [640, 480]`、`polygon` 由現行 `det_corner` 移植）；以多邊形頂點與現行 config 的 `det_corner` 相同作為驗證
- [x] 5.2 新增 ROI 解析函式：依序列名稱取得鏡頭、讀取 ROI、檢查影像尺寸；沒有檔案時回傳整張影像並提示；尺寸不符時報錯；以單元測試涵蓋這三種情況作為驗證
- [x] 5.3 `config_loader` 遇到頂層 `det_corner` 時報錯並提示新位置；從 `config/*.yaml` 與 `tests/golden/configs/*.yaml` 移除 `det_corner`；以 smoke 測試的 config 載入全部通過，另外加一個含 `det_corner` 的 config 應載入失敗的測試作為驗證

## 6. stage1 與 stage3 的 ROI 套用（D5）

- [x] 6.1 `tracking_pipeline`／`sequence_tracker` 改成逐序列呼叫 ROI 解析函式，供 Demo 繪製與 Re-ID 裁切使用；以 `botsort_save_reid_crops` 在單一序列產生的裁切清單與 `before1` 相同作為驗證
- [x] 6.2 `save_mot_format()` 移除多邊形過濾；以單一序列的 stage1 軌跡列數大於或等於 `before1`，且 `before1` 的每一列都能在新輸出中找到作為驗證
- [x] 6.3 stage3 讀取軌跡時，以序列鏡頭的 ROI 依 bbox 中心點硬過濾；以 `collision_detect` 在單一序列的 `ttc_events`、`pet_events` 與 `before1` 相同作為驗證

## 7. 追蹤評估的 ignore 前處理（D6）

- [x] 7.1 新增 ignore 前處理模組（逐幀、逐類別匈牙利匹配，IoU ≥ 0.5，移除匹配到 ignore GT 的預測，再移除 ignore GT）；以單元測試涵蓋 spec 的情境（ROI 外正確偵測、跨界、ROI 外誤報不計分、ROI 內誤報計 FP、跨界未達 IoU 門檻、優先匹配有效 GT；語意已改為 UA-DETRAC 式）作為驗證
- [x] 7.2 `prepare_mot_gt_shadow()` 改用 ignore 前處理取代 `keep_mot_gt_valid_ids()`，只處理 stage2 輸出目錄中的 GT shadow 與 tracker 副本，並輸出各類別的 ignore 統計；以跑單一序列 stage2 後來源 GT 與 stage1 軌跡雜湊不變、TrackEval 正常產出指標作為驗證

## 8. detection 評估改用人工 GT（D7）

- [x] 8.1 stage2 在 `<輸出>/det_eval/<seq>/` 建立 images hardlink（symlink 會被 ultralytics 解析回來源，導致 .cache 寫進來源）、labels 複本與 `data.yaml`，`det_model.val()` 改指向這份 yaml，並停止對 stage1 標籤呼叫 `fix_det_labels_to_5cols()`；以 val 讀到的 label 數等於 GT 框數、來源目錄沒有新增 `.cache` 作為驗證
- [x] 8.2 detection 的 Excel 報表與終端摘要加註「ROI：未套用（全畫面）」；以開啟單一序列的報表確認註記存在作為驗證

## 9. 權重引用與淘汰訓練端（D8、D9）

- [x] 9.1 `config/botsort_reid.yaml` 與 `tests/golden/configs/botsort_reid.yaml` 的 small_truck／scooter `model` 改為 `models/yolo/yolo11x.pt`；`scripts/grid_search_botsort_reid.py` 的候選清單移除已隔離的路徑；以 `grid_search_botsort_reid.py --dry-run` 正常列出組合、4.4 的 temp 引用檢查通過作為驗證
- [x] 9.2 刪除 `scripts/train_reid.py`、`src/safetrajai/reid/yolo_reid_trainer.py`、`src/safetrajai/reid/training/`，並更新 `tests/test_smoke.py` 的模組清單；以 `grep -rn "yolo_reid_trainer\|reid.training\|train_reid" src scripts tests config` 沒有結果、`pytest` 通過作為驗證
- [x] 9.3 驗證 `utils.model` shim 仍可用：smoke 測試新增一項，從 `models/class_reid_weights/temp/` 載入權重（檔案不存在時 skip）；以 `pytest` 通過作為驗證

## 10. golden 基準重建（D10）

- [x] 10.1 跑 `pytest` 確認全部通過
- [x] 10.2 跑 `tests/golden/run_all.sh after1` 與 `after2`（約 11 分鐘，各別輸出到 `runs/refactor_golden/`，不覆寫 `before1`）；以 `compare.py after1 after2` 自比差異為 0 作為驗證
- [x] 10.3 跑 `compare.py before1 after1`，確認 stage3 事件、Re-ID 裁切清單、homography 驗證三項沒有差異；以比對輸出中這三項為 0 差異作為驗證
- [x] 10.4 撰寫 change 目錄下的 `verification.md`：列出預期變動項目（stage1 軌跡、stage2 指標、grid search、botsort_reid）與原因，附上同一批 stage1 輸出在新舊 GT 下的主要指標對照；在 `tests/golden/TOLERANCE.md` 記錄基準改為 `after1` 的日期與原因

## 11. 文件同步

- [x] 11.1 更新 CLAUDE.md，涵蓋下列內容；以文件中不再出現 `det_corner` 作為 config 欄位的描述、`train_reid.py` 指令已移除作為驗證：
  - 資料路徑與 `datasets.yaml` 的新欄位。
  - 軌跡契約改為 stage1 不過濾、由下游套用 ROI。
  - ROI 與 ignore 語意。
  - detection 評估改用人工 GT。
  - Re-ID 章節移除訓練指令。
  - 隔離慣例。
- [x] 11.2 更新 `CONTEXT.md` 的術語（ROI、忽略區域、權威 GT、隔離），更新 `src/safetrajai/reid/HANDOFF.md`（只保留推論端）、`scripts/reid_data/README.md`（標註資料已隔離、待 `rebuild-reid-dataset` 重建），並修正 `reid_crops.py` docstring 中失效的 openspec 引用；以 `grep -rn "reid-training-data-pipeline" src` 沒有結果作為驗證
