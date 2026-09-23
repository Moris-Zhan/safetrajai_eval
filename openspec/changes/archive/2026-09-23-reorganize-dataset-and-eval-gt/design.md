# Design

## Context

動機見 proposal.md 的 Why，需求見 `specs/`。這裡只補充追查程式碼後確認的現況，這些現況決定了設計方向：

- **tracking 評估目前是對稱的硬過濾**：
  - stage1 的 `save_mot_format()` 以 `det_corner` 過濾預測。
  - stage2 的 `prepare_mot_gt_shadow()` 把 GT 複製到輸出目錄後，以 `keep_mot_gt_valid_ids()` 用同一個多邊形過濾 GT。
  - 兩者都以 bbox 中心點判斷。邊界上的物件會同時產生假的 FN 與 FP。
- **detection 評估沒有用到人工 GT**：
  - `run_eval_pipeline()` 的 `det_root` 是 stage1 輸出的序列目錄。
  - 那裡的 `labels/` 是追蹤器逐幀寫出的 7 欄預測。`fix_det_labels_to_5cols()` 把它們裁成 5 欄後餵給 `det_model.val()`，等於拿模型去比自己的追蹤輸出。
  - 來源目錄那份 7 欄的 `annotations_all8/labels`，以及同目錄下的 `trackers/`，就是某次 stage1 的輸出被放回來源的結果。
- **`det_corner` 是情境 config 的頂層欄位，一個 config 只有一組**。它被用在 stage1 的 Demo 繪製、Re-ID 裁切、MOT 輸出過濾，以及 stage2 的 GT 過濾。這組多邊形只適用於 C000025，解析度是 640×480。
- **Re-ID 推論只有 `botsort_reid.yaml`（和它在 golden 的副本）啟用**，而且 6 類都走 `encoders.ReID`。small_truck／scooter 引用的 `yolo-reid/yolo11x-reid-ultra.pt` 會被隔離。`Custom_ReID` 目前沒有任何 config 使用。
- **`bus_video/` 有 4 支與 `video_seqs` 重複**：`C000054` 的 0623070611、0625074505、0627071742、0629080441，`cmp` 確認內容逐位元組相同。
- **限制**：
  - `third_party/trackeval/` 唯讀（ADR 0002）。它的 `distractor_classes` 寫死為空，`zero_marked` GT 只會被刪除，匹配到它的預測仍算 FP，所以 ignore 語意無法靠 TrackEval 內建機制完成。
  - tracker override 方法本體凍結（ADR 0004）。
  - `dataset/`、`models/`、`runs/` 都被 gitignore。

## Goals / Non-Goals

**Goals:**
- 評估結果以原始人工 GT 為準，並用 ignore 語意限縮範圍。
- 讓來源資料在任何 stage 執行後都保持位元組不變。
- 被隔離的資產都能追溯、能還原，而且引用它們時會明確失敗。
- 讓 stage3 衝突事件與 Re-ID 裁切在改版前後輸出相同，作為這次結構調整的回歸錨點。

**Non-Goals:**
- detection 評估不做 ignore 語意：ultralytics `val()` 內部自行推論與計分，插不進匹配步驟；自寫 mAP 留給 detection change。
- 不修正 `save_mot_format()` 以 `C000025_*` 做 glob 的已知限制。它只影響非 C000025 鏡頭的 MOT 輸出，這類鏡頭在本 change 不跑 stage2，留給 `rebuild-reid-dataset`。
- 不處理 homography 以序列為鍵的落差。

## Decisions

### D1. 評估資料集改名並沿用舊的內部命名

搬移與改名如下：
- `dataset/中華三路_民生二路(東北角)yolo驗證用_all8/` → `dataset/eval/cctv_all8/`。
- `annotations_all8(orign)` → `annotations_all8`。
- `mot_annotations_all8(orign)` → `mot_annotations_all8`。

改名後，`dataset_layout.py` 的 `annotation_<seq>/`、`mot_annotation_<seq>/` 命名與 `datasets.yaml` 的 `run_det_eval_dir`／`run_track_eval_dir` 都不用改，只需要換 `root`。限縮版與重複版移到 `dataset/eval/cctv_all8/temp/`。

- **替代方案**：保留括號目錄名，只改 `datasets.yaml` 指向 `annotations_all8(orign)`。
- **不採用的原因**：括號與空白在 shell、glob 裡容易出錯，而且名稱會繼續暗示它是次要版本。

改名只移動目錄，檔案內容不變，符合「原始 GT 不可修改」。

### D2. 原始影片依鏡頭歸位，新增 raw 版面的探索

影片搬到 `dataset/raw/videos/<camera>/<seq>.mp4`，來源如下：
- `video_seqs/annotations_video/annotation_*/*.mp4`
- `bus_video/*.mp4`：內容相同的 4 支只保留一份，另一份移除前先記錄 checksum。
- `dataset/` 根目錄的 C000025 mp4

`<camera>` 是鏡頭編號（見 CONTEXT.md），取序列名稱第一個空白之前的字串，不限定 `C` 加數字的格式，因為既有鏡頭中也有 `BEV0001`、`037-` 這類編號；序列名稱沒有空白、無法取得編號時報錯。

`datasets.yaml` 新增欄位：
- `role`：`eval` 或 `source`。
- `cameras`：明確列出涵蓋的鏡頭。
- `layout`：`annotated`，也就是現行的 `annotation_<seq>/` 結構；或 `raw`，也就是依鏡頭分目錄的平放影片。

`dataset_layout.discover_source_sequences()` 依 `layout` 分派，`raw` 版面只列出 `cameras` 清單內各鏡頭的 `<root>/<camera>/*.mp4`，所以同一個 raw 目錄裡的 C000025 影片不會被 source 資料集探索到。資料集登記如下：
- `video_seqs` 更名為 `reid_sources`（`role: source`、`layout: raw`，cameras 為 C000043／054／085／294／297）。
- `cctv_all8` 標為 `role: eval`、`cameras: [C000025]`。
- 根目錄那支 C000025 mp4 放進 raw 但不登記為 source。

`cameras` 用明確清單，而不是掃描檔案推導，理由是讓 smoke 測試在沒有資料的環境也能檢查鏡頭角色互斥。

### D3. 隔離一律用 mv，並附 `DO_NOT_USE.md`

每個 `temp/` 內的 `DO_NOT_USE.md` 用表格記錄：原路徑、隔離日期（2026-09-23 起算的實際日期）、原因、還原指令。先寫紀錄再搬移，避免搬到一半中斷後無從追溯。

`config/` 與 `tests/golden/configs/` 裡對 `temp/` 的引用由 smoke 測試攔截。被隔離的權重路徑一旦不存在，現有的載入流程（`YOLO(path)`／`torch.load`）本來就會拋錯，不需要另外加防呆。

### D4. ROI 以鏡頭為鍵的檔案與解析

`config/roi/<camera>.yaml` 的內容：
- `camera`
- `image_size: [w, h]`
- `polygon: [[x, y], ...]`

C000025 由現行 `det_corner` 移植，`image_size` 是 `[640, 480]`。

新增一個單一的 ROI 解析函式：輸入序列名稱與實際影像尺寸，以 D2 的規則取得鏡頭編號，輸出多邊形。ROI 以鏡頭編號為鍵，既有 homography 以完整鏡頭名稱為鍵，兩者並存是 CONTEXT.md 記錄的已知落差，本 change 不統一。行為如下：
- 沒有 ROI 檔：回傳整張影像，並印出一行提示。
- `image_size` 與實際影像尺寸不符：報錯。

不自動縮放，避免解析度不同的同名鏡頭悄悄套用錯誤的區域。所有原本讀 `cfg.det_corner` 的地方都改成逐序列呼叫這個函式。`config_loader` 遇到頂層 `det_corner` 直接報錯，並提示新位置，因為悄悄忽略會改變行為。

### D5. stage1 不再過濾軌跡，下游自行套用 ROI

`save_mot_format()` 移除多邊形過濾，軌跡檔保留全部結果。ignore 匹配需要看到 ROI 外的預測，才能讓跨越邊界的物件配成 TP。

stage3 在讀入軌跡時，以同一個 ROI 解析函式、同樣的中心點判斷做硬過濾，所以衝突事件應該與改版前完全相同，這可以作為回歸錨點。過濾必須在「依 track_id 分組」之前逐列進行：軌跡的排列順序取決於它第一次出現的位置，PET 計算又與軌跡順序有關；若先分組再過濾，ROI 外先出現的軌跡會改變順序，即使輸入的列相同，PET 事件也會不同（golden after1 第一次執行時實際發生過）。Re-ID 裁切蒐集與 Demo 繪製也改用 ROI 解析函式，輸出不變。

- **替代方案**：stage1 同時輸出過濾版與完整版。
- **不採用的原因**：兩份軌跡會讓「哪一份是下游契約」變得模糊。

### D6. ignore 匹配放在 TrackEval 之前的本專案前處理

新增 ignore 前處理模組，放在 `safetrajai/evaluation/`。它接在 `prepare_mot_gt_shadow()` 做完 `fix_mot_gt_unique_ids()` 之後，取代原本的 `keep_mot_gt_valid_ids()` 呼叫。

處理方式：逐類別讀取 stage2 輸出目錄中的 GT shadow 與 tracker 檔副本，逐幀進行下列步驟：
1. GT 中心點在 ROI 外的標為 ignore。
2. 以 IoU 矩陣做 `scipy.optimize.linear_sum_assignment`，IoU < 0.5 的配對作廢。
3. 匹配到 ROI 內 GT 的預測保留；其餘預測中，匹配到 ignore GT 或中心點在 ROI 外者移除（UA-DETRAC 式）。
4. 移除 ignore GT。

最初實作的是 MOT17 式（只移除匹配到 ignore GT 的預測）。golden after1 實測顯示 ROI 外未匹配的預測大量變成 FP（bytetrack car 的 FP 60→551），與「只評估 ROI 內」的目的相違，使用者決定改為 UA-DETRAC 式；代價與風險記錄在 `specs/evaluation-roi-filtering/spec.md`。MOT17 式的兩組 golden 結果保留在 `runs/refactor_golden/after{1,2}_mot17/` 供對照。

寫回的是 stage2 輸出目錄中的副本。stage2 的 tracker 檔本來就是從 stage1 複製的，所以 stage1 原檔不受影響。

預測與有效 GT、ignore GT 一起匹配，而不是只和 ignore GT 匹配，理由是避免貼近有效 GT 的預測被 ignore GT 搶走。這也是 MOT17 的做法。每次處理都輸出一行統計，列出各類別的 ignore GT 數與被移除的預測數。

### D7. detection 評估改在 stage2 輸出目錄組一份唯讀來源的資料集

在 `<stage2 輸出>/det_eval/<seq>/` 建立下列內容：
- `images/<seq>/`：來源人工 GT 影像的 hardlink（跨磁碟時退回複製）。實作時發現不能用 symlink：ultralytics 會解析 symlink，改從來源目錄推出 label 路徑，並把 `.cache` 寫進來源。
- `labels/<seq>/`：來源 5 欄 label 的複本。
- 對應的 `data.yaml`。

`det_model.val()` 指向這份 yaml。ultralytics 會依影像路徑推出 label 路徑，並把 `.cache` 寫在 label 目錄旁；影像用 hardlink 放在輸出目錄，cache 就會落在輸出目錄，不會寫進來源。

影像與 label 使用人工標註當時的同一批檔案，幀號自然對齊，不必和 stage1 的影格做 +1 對齊。`fix_det_labels_to_5cols()` 不再用在 detection 評估上。Excel 與終端摘要加上「ROI：未套用（全畫面）」。

### D8. `botsort_reid.yaml` 暫代權重

small_truck／scooter 的 `model` 改成 `models/yolo/yolo11x.pt`，與 person／truck 相同，都是 detector backbone 特徵，`use_reid` 維持 `true`。這個情境語意仍是「6 類全開 ReID」，但所有 Re-ID 權重都來自工具模型，等到 `rebuild-reid-dataset` 產出 HF 模型再換掉。

`grid_search_botsort_reid.py` 的候選清單刪除已隔離的路徑。

- **替代方案**：改成 `use_reid: false`。
- **不採用的原因**：這樣會讓 `botsort_reid` 與 `botsort_noreid` 變成同一情境，失去比較意義。

### D9. 淘汰 YOLO-as-encoder 訓練端

刪除以下檔案：
- `scripts/train_reid.py`
- `src/safetrajai/reid/yolo_reid_trainer.py`
- `src/safetrajai/reid/training/`

`reid/HANDOFF.md` 改寫成只描述推論端（`ReID`、`Custom_ReID`、shim），並註明訓練改由後續 HF 路線處理。`PKSampler` 與依身分切分的邏輯有需要時從 git 歷史取回，不另外保留。smoke 測試的 `scripts/**` 清單是自動掃描，只需要更新手寫的模組 import 清單。

### D10. golden 基準重建流程

新基準的建立流程：
1. 以新程式碼跑 `tests/golden/run_all.sh after1` 與 `after2`。
2. 用 `compare.py after1 after2` 確認自比差異為 0。
3. 用 `compare.py before1 after1` 產生新舊差異。

差異要逐項歸類，寫進本 change 的 `verification.md`：

| 類型 | 項目 | 原因 |
|---|---|---|
| 必須相同 | stage3 事件 | ROI 過濾移到 stage3，輸出應不變 |
| 必須相同 | Re-ID 裁切清單 | 同一個 ROI，輸出應不變 |
| 必須相同 | homography 驗證 | 不受本 change 影響 |
| 預期變動 | stage1 軌跡 | 不再過濾 |
| 預期變動 | stage2 指標 | 換 GT、改 ignore 語意 |
| 預期變動 | grid search CSV | 依賴 stage2 |
| 預期變動 | `botsort_reid` 的 stage1 | 暫代權重 |

`tests/golden/configs/*` 同步移除 `det_corner`，並更新 `paths`。之後以 `after1` 作為新基準，`TOLERANCE.md` 記錄基準切換的日期與原因。

## Risks / Trade-offs

- **[風險] 大量搬移 gitignore 目錄，git 追蹤不到** → 先寫 `DO_NOT_USE.md` 再 `mv`；同一磁碟內的 `mv` 不改變內容，只比對搬移前後的路徑清單與檔案數，不逐檔計算雜湊（掛載磁碟禁止大量 I/O，見 CLAUDE.md）。搬移前清單已存在 `runs/reorganize-dataset-and-eval-gt/inventory_before.txt`。
- **[風險] stage1 軌跡不再過濾，會改變 stage3 以外讀軌跡的工具行為**（例如疊圖腳本畫出 ROI 外的框）→ 在 CLAUDE.md 的軌跡契約章節明確更新；疊圖腳本只影響視覺，不影響數值。
- **[風險] 新舊指標不可比**：歷次實驗與 grid search 結論都建立在舊 GT 上，可能讓人誤以為模型退步 → `verification.md` 保留同一批 stage1 輸出在新舊 GT 下的指標對照，說明數值變動來自評估基準。
- **[取捨] detection 評估不套用 ROI** → 報表明確標示；ROI 外誤報也會計入，數值偏保守。
- **[風險] 不同解析度的同一鏡頭**：bus_video 與 video_seqs 的 C000054 是否同為 1280×720，目前還沒全面確認 → D4 的尺寸檢查會在執行時報錯，不會悄悄套錯；C000054 的 ROI 本身屬於 `rebuild-reid-dataset`。
- **[風險] `scripts/reid_data/*` 的預設路徑指向已隔離的 `dataset/reid/` 內容** → 本 change 不修改這些腳本，執行時會因路徑不存在而報錯；`rebuild-reid-dataset` 會重新定義它們的輸入。

## Migration Plan

1. 確認 `runs/refactor_golden/before1` 仍在，當作舊基準。
2. 記錄搬移前的清單與雜湊，接著寫 `DO_NOT_USE.md`，再執行隔離搬移（D3）。
3. 搬移評估資料集與原始影片（D1、D2），並更新 `datasets.yaml`。
4. 修改程式：ROI（D4）→ stage1 與 stage3（D5）→ ignore（D6）→ detection 評估（D7）→ config（D8）→ 淘汰訓練端（D9）。
5. 跑 smoke 測試，再依 D10 重建 golden 基準並撰寫 `verification.md`。
6. 同步文件：CLAUDE.md、CONTEXT.md、HANDOFF.md、`scripts/reid_data/README.md`。

**還原方式**：依各 `DO_NOT_USE.md` 的還原指令把檔案 `mv` 回原處；程式變更以 git revert 還原。舊基準 `before1` 不刪除。

## Open Questions

（無。影響規格或任務拆分的事項都已在上方決定。）
