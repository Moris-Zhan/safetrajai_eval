# Design

## Context

動機見 proposal.md，需求見 `specs/reid-dataset/spec.md`。追查程式與資料後確認的現況：

- **裁切蒐集**：`ReidCropCollector.collect()` 對每條軌跡、每一幀都寫一張 `{class}/{seq}_track{id}/{frame:06d}.jpg`，沒有索引。ROI 已改成以鏡頭為鍵（`resolve_roi`）；`quality_filters` 掛載點存在但沒有人用。
- **影片時間**：`VideoSequenceTracker._open_video()` 以 `int(CAP_PROP_FPS)` 取得幀率，逐幀迴圈只傳幀號。C000054 實際為 10.0～10.3 fps（VFR）；新增的來源鏡頭約為 10～15 fps（C000011、C000012、C000056 約 14～15 fps）。
- **追蹤器的 frame_rate**：情境設定固定為 30。BoTSORT 的 `track_buffer` 以秒計，會乘上 `frame_rate`；在 10fps 影片上，實際的保留時間會被拉長 3 倍，對「寧可斷開」的需求不利。
- **來源鏡頭解析度**：C000054 共 31 支，全部是 1280×720；C000043、C000085、C000011、C000056 是 704×480；C000012、C000033、C000048、C000256 是 1280×720。每支鏡頭內的解析度一致，ROI 的 `image_size` 可以唯一決定。
- **來源鏡頭組成**（2026-09-24 使用者調整）：C000043、C000085 各 1 支 60 秒剪輯（`video/` 另有 10 分鐘完整版，但維持使用 60 秒剪輯，golden 的裁切情境也用它）、C000054 共 31 支，加上新增的 C000011、C000012、C000033、C000048、C000056、C000256 各 1 支約 10 分鐘影片，共 9 支鏡頭、39 支影片。新增影片由 `video/` 以 `mv` 歸位到 `dataset/raw/videos/<camera>/`。C000026（中華三路 民生二路 東南角）與評估鏡頭 C000025 位於同一路口，影片歸位但不列入 `reid_sources`，避免 Re-ID 在 C000025 的驗收偏樂觀。
- **golden 的裁切情境**：`botsort_save_reid_crops` 目前在評估資料集 `cctv_all8`（C000025）上蒐集裁切，與新 spec「評估鏡頭一律拒絕」衝突，必須改用來源鏡頭，`reid_crops` 類別的基準也要跟著重建。
- **`save_mot_format()`**：以 `C000025_*` 做 glob，非 C000025 的鏡頭會輸出空的軌跡檔（CLAUDE.md 已知限制）。改用來源鏡頭後，golden 的裁切情境會碰到這個問題。
- **既有工具**：`scripts/reid_data/` 的四支腳本都以「身分資料夾」為單位運作。`generate_identity_review.py` 會為每一張裁切產生縮圖；`quarantine_flagged.py` 會搬移檔案；`build_reid_split.py` 會複製影像。
- **環境**：伺服器沒有 `$DISPLAY`；timm 1.0.15 已內建 `vit_small_patch14_dinov2`，權重需要從 HF Hub 下載；GPU 為 RTX 5090。
- **限制**：
  - tracker override 方法凍結（ADR 0004）。
  - 掛載磁碟禁止大量 I/O（CLAUDE.md）。
  - 評估鏡頭不得進入任何訓練資料（`dataset-layout` spec）。

## Goals / Non-Goals

**Goals:**
- 裁切數量可以預估、可以控制；全量執行前先以 pilot 估算並取得同意。
- 每張裁切都能追溯到序列、幀、時間與框；資料集版本由 manifest 定義，可以重現。
- 標記、審查、排除都不改動裁切檔案；人工審查只看需要看的身分，縮圖數量有上限。

**Non-Goals:**
- 不訓練模型、不接入追蹤器（屬於 `train-reid-timm`）。
- 不合併碎片化的身分；不做跨鏡頭的身分對應。
- 不改 tracker 的 override 方法，也不改 `CLASS_TRACKING_PARAMS` 的讀取順序。

## Decisions

### D1. ROI 點選頁：離線單檔 HTML

新增 `scripts/roi/make_roi_picker.py --camera <id>`：
1. 從 `dataset/raw/videos/<camera>/` 的第一支影片取首幀。
2. 以 base64 JPEG 嵌入單一 HTML 檔，輸出到 `runs/roi_picker/<camera>.html`。
3. 使用者在頁面上依序點出多邊形頂點，可以復原上一點或清除重畫。
4. 頁面即時顯示 `config/roi/<camera>.yaml` 的內容（`camera`、`image_size`、`polygon`），提供複製與下載。

檔案放在 gitignore 的 `runs/`，CCTV 畫面不進 git，也不上傳。

- **替代方案**：沿用 homography 的「格線圖＋手填座標」。
- **不採用的原因**：使用者已選擇點選頁，而且逐點讀座標容易出錯。

### D2. 工具模型比較：沿用 detection 評估

- `evaluation/pipeline.py` 的 `det_model.val(imgsz=...)` 改讀 `cfg.imgsz`，不再固定 640。目前所有情境設定的 `imgsz` 都是 640，現有行為不變。
- 在 `runs/` 放兩份臨時情境設定：dennis@640，以及 `yolo11x.pt`@1280。兩者都只跑 stage2 的 detection 評估，沿用 `after1` 的 stage1 輸出，不重新追蹤。
- **判定方式**：比較 6 類平均 mAP50 與各類 recall。若 `yolo11x.pt` 的平均 mAP50 高出不到 0.01，而且沒有任何類別的 recall 高出 0.03 以上，就選 dennis；否則選 `yolo11x.pt`，並在裁切情境設定中改用 imgsz 1280。
- C000025 與 C000054 的視角、解析度、幀率都不同，而 C000054 沒有人工 GT。因此 pilot（D10）會另外以選定的模型輸出 1 分鐘的 C000054 疊框影片，由使用者目視抽查漏框與類別誤判；量化判斷用 C000025，實際場景用目視把關。
- 結果記錄在本 change 的 `verification.md`。

### D3. 純度優先的追蹤參數與 `frame_rate: auto`

新增裁切專用的情境設定 `config/reid_crops.yaml`：以 `botsort_save_reid_crops_video.yaml` 為基礎，追蹤參數採用 D3 比較後勝出的版本。

**`frame_rate: auto`**：`tracking_pipeline` 在建立 `VideoSequenceTracker` 前先讀影片幀率，把 `round(fps)` 傳給 `create_tracker()`。只有設定值是 `auto` 時才啟用，數值設定維持原行為，golden 不受影響。這不動 tracker override 的方法本體，只改變傳入的參數。

**純度優先參數的初版**：以 `botsort_noreid.yaml` 的 `class_tracking` 為基礎，每個類別依下列方式調整，最多再迭代一次：
- `track_buffer` 改為 1.0 秒。
- `match_thresh`、`second_match_thresh`、`unconfirmed_match_thresh` 各降 0.1（配對更嚴格）。

**比較方式**：兩組參數都以 `frame_rate: auto` 在 `cctv_all8` 上跑 stage1、stage2（ignore 語意）。
- 主要指標：6 類加總的 IDSW，以及 IDF1 的 IDP。
- 同時記錄 Frag 與 GT 軌跡的斷裂數。
- **判定**：純度優先組的 IDSW 較少、IDP 不低於現行組時採用；否則沿用現行參數。
- 結果記錄在 `verification.md`。

### D4. 裁切蒐集的改寫

`ReidCropCollector` 改為在 `run()` 開始時，依序列建立一份：
- 鏡頭編號、序列名稱、日期（取序列名稱結尾 `_YYYYMMDDhhmmss` 的前 8 碼；解析失敗就報錯）。
- 每條軌跡上一次取樣的時間。

逐幀收到軌跡與時間後，依下列步驟處理：
1. 框中心點在 ROI 外 → 略過。
2. 最短邊 < 32px → 略過（不更新取樣時間）。
3. 距離上一次取樣不到 0.5 秒 → 略過。
4. 寫出 `{class}/{seq}_track{id}/{frame:06d}.jpg`，並在記憶體中累積一列索引。

序列結束時，把索引一次寫到 `<crop_root>/_index/<seq>.csv`。

**影片時間**：`sequence_tracker` 讀完每一幀後取 `CAP_PROP_POS_MSEC`；如果回傳 0 或沒有遞增，就退回 `frame_id / fps`。

**評估鏡頭與 ROI 檢查**：`tracking_pipeline.run_stage1()` 在 `save_reid_crops` 開啟時，先對所有要處理的序列做兩項檢查，任一不通過就在開始追蹤前報錯停止，不會寫出任何東西：
- 鏡頭屬於任何 `role: eval` 資料集。
- 鏡頭沒有 `config/roi/<camera>.yaml`。蒐集裁切時不退回整張影像，避免忘了畫 ROI 卻默默蒐集到遠處小目標與畫面邊緣的殘缺框；評估時的整張影像退回行為不變。

**重跑保護**：序列的索引已經存在時直接報錯，避免重複寫入；要重跑必須先由使用者移走舊輸出。

裁切根目錄改為 `dataset/reid/crops/`。舊的 `quality_filters` 參數與 `make_min_size_quality_filter` 移除，由固定的最小尺寸規則取代。

### D5. `save_mot_format()` 的 glob 修正

影像與 label 檔的 glob 從 `C000025_*` 改為 `*`；stage1 的輸出目錄只有該序列的檔案，行為不變。這讓非 C000025 鏡頭也能產生正確的軌跡檔，並解除 CLAUDE.md 記錄的已知限制。golden 的既有情境都是 C000025，輸出應該完全不變，由 golden 比對驗證。

### D6. 污染偵測

單一腳本 `scripts/reid_data/detect_contamination.py`，每張裁切只讀一次（掛載磁碟禁止大量 I/O，見 CLAUDE.md）：
1. 以 timm `vit_small_patch14_dinov2` 對每張裁切抽取特徵：letterbox 到 224×224，L2 正規化，GPU 批次處理。
2. **類別中心點比對**（沿用舊 `check_class_consistency.py` 的中心點做法：每類以身分平均特徵再平均當中心點，取最近的類別為判定類別）：身分中判定類別與所屬類別不一致的裁切比例過高時標記。距離改用 cosine 相似度，不再做逐維變異數正規化：實作時發現舊做法的逐類別變異數會被污染身分撐大、讓其他類別的乾淨身分被誤標；改成共用變異數後，身分之間變化很小的維度又會被放大成雜訊，判定近乎隨機。DINOv2 特徵本來就在 cosine 空間中使用，因此直接比較 cosine。
3. **軌跡內外觀離群值**：對每個身分計算平均特徵，以及每張裁切到平均特徵的 cosine 距離；以同類別所有裁切距離的第 99 百分位當門檻；超過門檻的裁切數 ≥ 2，或佔該身分 20% 以上的身分，標記為疑似污染。
4. 輸出 `dataset/reports/reid/class_consistency.csv` 與 `dataset/reports/reid/appearance_outliers.csv`，同時包含身分層級與裁切層級的紀錄。

**pilot 後的調整（2026-09-24，使用者同意）**：審查的重點是**身分純度**（一個身分只含同一個實體），不是類別是否正確。6 類共用同一個 Re-ID 模型，四輪車子類別之間的誤標（car↔small_truck、truck↔bus）對訓練的傷害遠小於 ID 串錯；pilot 的類別一致性標記有 16% 集中在這類誤標。因此：
5. **類別一致性改為粗分組**：只在人、二輪（scooter）、四輪（car、small_truck、truck、bus）三組之間不一致時標記；pilot 中會進 manifest 的身分，標記數從 49 降為 3。細分類的判定仍保留在裁切層級的輸出，供參考。
6. **靜止身分**（新增，只讀索引，不讀影像）：身分首末兩張的框中心點位移小於框尺寸（面積平方根）的 0.5 倍，而且持續 ≥ 5 秒。目標是被誤判成物件的號誌等靜止物；pilot 標記 10 個。
7. **運動跳動**（新增，只讀索引）：相鄰取樣之間的速度向量變化量 × 時間間隔，除以框尺寸，大於 1.5。目標是 ID 閃跳（一個身分接上另一個物件）；pilot 標記 21 個，準確率未知，要在 pilot 審查時校準。
8. 所有標記只針對**裁切數 ≥ manifest 最少張數（`config/reid_split.yaml` 的 `min_crops_per_identity`）的身分**，不足的身分不會進 manifest，不需要審查。新增的兩種標記輸出為 `dataset/reports/reid/motion_flags.csv`。

- 兩種判斷的門檻在 pilot 時檢視分布後可以調整，定案值寫進 `verification.md`。
- 腳本只讀裁切、只寫報表，不改動 `dataset/reid/crops/`。
- 原本以 dennis 倒數第二層特徵做類別一致性的 `check_class_consistency.py` 併入本腳本後刪除。
- **替代方案**：保留兩支腳本、各自抽特徵。
- **不採用的原因**：全部裁切要完整讀兩次；dennis 是偵測器 backbone，本來就不是為外觀比對設計的。

### D7. 人工審查與排除清單

- `generate_identity_review.py` 改讀索引與標記檔，挑出「任一檢查標記的身分＋bus 的全部身分＋隨機抽查身分」，只包含裁切數 ≥ `min_crops_per_identity` 的身分。
  - **依標記原因分頁**（靜止、運動跳動、外觀離群、類別不一致、bus、隨機抽查）；同一身分有多個原因時放在優先序最高的分頁（靜止 > 運動跳動 > 外觀離群 > 類別不一致 > bus > 抽查），並列出全部原因與移動距離、持續時間。靜止分頁提供整頁全選。
  - **隨機抽查**：每支鏡頭從未被標記的身分中，以固定種子抽出 N 個（預設 20），用來估計漏網率。某支鏡頭的抽查中，需要排除的身分比例超過 10% 時，回報使用者並檢討該鏡頭的門檻或改為全數審查。
  - 每個身分只產生代表縮圖與最多 12 張依時間均勻抽樣的縮圖，不再為每一張裁切產生縮圖。
  - 輸出到 `dataset/reports/reid/review/`。
  - 匯出 CSV 的欄位維持 `class, identity, image, action`。
- `quarantine_flagged.py` 改名為 `apply_review_exclusions.py`（名稱與行為一致：它做的是排除，不是隔離，見 ADR 0006），改為把確認清單合併進 `dataset/reid/exclusions.csv`（去除重複、依鍵排序），不搬移任何檔案。預設 dry-run，需要 `--apply` 才會寫入。
- **替代方案**：沿用搬移檔案的隔離方式。
- **不採用的原因**：搬移會產生額外 I/O，而且之後調整規則時難以還原；排除清單能讓同一批裁切產生多個版本的 manifest。

### D11. 遮擋後的重新配對（C000048、C000012、C000056）

使用者目視發現 C000048（號誌牌遮擋）、C000012 與 C000056（右下角樹木遮擋）的物件被遮擋後常變成新 ID；三支鏡頭都以追蹤參數處理，不重畫 ROI（使用者決定）。碎片化本身不會污染身分，但會減少每個身分的裁切數與視角變化，讓更多身分達不到最少張數。

- **機制**：遮擋期間軌跡轉成 lost，保留 `track_buffer` 秒並參與第一階段配對。配對成本為 `(1 − center_dist_weight) × IoU 距離 ＋ center_dist_weight × min(中心距離 / norm_factor, 1)`，低於 `match_thresh` 才接回。遮擋後 Kalman 預測框漂移，IoU 常降為 0，成本超過門檻而另開新 ID。
- **可調參數**（都在 `class_tracking` 查表，不動凍結的 override 方法）：`center_dist_weight`、`norm_factor`、`match_thresh`、`track_buffer`。
- **不可調的限制**：
  - 遮擋後剛出現、信心較低的偵測只進第二階段配對，第二階段只看 tracked 狀態的軌跡，無法接回 lost 軌跡（`update()` 凍結，見 ADR 0004）。
  - ReID 無法使用：舊權重已隔離，而這份資料集正是用來訓練新的 Re-ID 模型。
- **評估方式**（沒有 GT，改用代理指標）：對 C000048、C000012、C000056 各 1 支完整影片跑裁切蒐集（輸出在 `runs/`），由索引計算：
  - **疑似碎片數**：同類別的兩個身分，A 結束後 `track_buffer ＋ 1` 秒內 B 出現，且 B 的起點落在 A 末段速度外插位置的 2 倍框尺寸以內。
  - **串錯代理**：運動跳動與外觀離群值的標記數（D6）。
  - 裁切數 ≥ `min_crops_per_identity` 的身分數。
- **把關**：候選參數同樣在 `cctv_all8` 上跑 stage1＋stage2，IDSW 不得多於、IDP 不得低於現行組（sparseOptFlow：IDSW 22、IDP 0.9704）。
- **判定**：疑似碎片數明顯下降、串錯代理不增加、C000025 把關通過，才採用；最多迭代 2 次，並輸出遮擋位置的疊框影片給使用者目視確認。
- **套用範圍**：優先找一組對全部鏡頭都適用的參數；若只對這幾支鏡頭有利，就另建情境設定，以 `paths.cameras` 只套用到這幾支鏡頭，其他鏡頭沿用 `config/reid_crops.yaml`。
- **替代方案**：事後以時間、位置與 DINOv2 外觀把碎片合併成同一身分。
- **不採用的原因**：proposal 已把跨軌跡的身分合併列為不做；先嘗試不改流程、只調參數的做法。
- **結果**（見 `verification.md` 第八節）：遮擋後重新出現的位置離外插位置約 2 個框尺寸，只放寬中心距離無法在不串錯的前提下接回（容許 120px、170px 時 C000025 的 IDP 都低於基準）。使用者決定採用 R1（四輪車 `center_dist_weight` 加重，C000025 的 IDSW 22→9、IDP 0.9704→0.9827）並接受碎片；碎片合併另開後續 change。

### D8. manifest 與切分

- 純函式放在 `src/safetrajai/reid/dataset_index.py`（索引讀寫、日期解析）與 `src/safetrajai/reid/manifest.py`（排除、張數規則、切分），並寫單元測試。
- `build_reid_split.py` 改為 `--version <v>` 的 CLI，讀取以下輸入：
  - 全部 `_index/*.csv`。
  - `dataset/reid/exclusions.csv`。
  - 切分規則 `config/reid_split.yaml`，內容為 val 的鏡頭與日期、最少張數 4、最多張數 30。
- 輸出到 `dataset/reid/splits/<version>/`：
  - `manifest.csv`，依 `class, identity, time_s` 排序。
  - `summary.json`：各類別、split、鏡頭的身分數與張數。
  - `split_rules.yaml`：本次使用的規則複本。
- **均勻抽樣**：以 `numpy.linspace(0, n-1, 30)` 四捨五入後取不重複的索引。這個做法是確定性的，不需要亂數種子。
- 評估鏡頭取自 `datasets.yaml` 中 `role: eval` 的資料集，manifest 產生時再檢查一次，發現就報錯。
- 版本目錄已存在時報錯，不覆寫。

### D9. golden 裁切情境改用來源鏡頭

- 情境設定新增選用的 `paths.cameras`，用來縮小資料集登記的鏡頭清單（只能取子集，不能新增）。
- golden 的 `botsort_save_reid_crops` 改用 `paths.dataset: reid_sources`、`paths.cameras: [C000043]`（60 秒、704×480），並需要 C000043 的 ROI。
- 這個情境的 `mot_tracks`、`yolo_labels`、`reid_crops`、`other` 會刻意改變，golden 基準要重建為新的 run（`reid1`、`reid2`）。其他情境必須與 `after1` 相同。

### D10. pilot 與全量執行

1. pilot：以 `config/reid_crops.yaml` 對 C000054 的 `20250623065608`（約 10 分鐘）跑完蒐集、污染偵測、審查頁，並另外輸出這支影片前 1 分鐘的疊框影片（D2），讓使用者目視抽查。
2. 統計該序列的裁切數、身分數、各類別因最短邊 < 32px 而略過的框數，以及只針對這一個序列目錄的寫入量；外推 31 支 C000054 影片、6 支新增鏡頭的 10 分鐘影片與 2 支 60 秒影片（C000043、C000085）的總量。若某類別大多被尺寸門檻濾掉（例如 person），回報時一併提出是否調整。
3. 向使用者回報，取得同意後才全量執行。
4. 全量執行後產生 `v1` manifest，再做人工審查（2026-09-24 調整：全量的審查與 `v2` 移到後續的自動排除 change，本 change 交付 `v1`）。
   - 審查完成後，把排除清單套進 `v2`。
   - `v1` 保留作為審查前的基準。

## Risks / Trade-offs

- **[風險] 全量裁切數量過大** → D10 先 pilot 並取得同意；0.5 秒的抽樣間隔與最小尺寸已大幅降低數量；每身分 30 張的上限在 manifest 階段套用，不刪除檔案。
- **[風險] ROI 由使用者繪製，是阻塞點** → 點選頁先產生；C000054 優先（佔 98% 資料）；蒐集裁切時缺 ROI 會在追蹤前直接報錯（D4），不會默默以整張影像蒐集。
- **[風險] VFR 影片的時間戳** → 以 `CAP_PROP_POS_MSEC` 為主，異常時退回幀號換算，並以單元測試涵蓋兩種路徑。
- **[風險] DINOv2 離群值門檻偏鬆或偏嚴** → 門檻在 pilot 時依分布調整；標記只決定「要不要給人看」，最終排除仍由人工決定。
- **[風險] 同一台公車跨日出現在 train 與 val** → 已知且接受；val 分數會偏樂觀，最終驗收以 C000025 的 tracking 為準（屬於 `train-reid-timm`）。
- **[取捨] 純度優先參數會增加碎片化** → 碎片化以依日期切分規避；同一天的碎片仍可能同時出現在 train，造成 ID 分類損失的假負樣本，訓練端以 label smoothing 等方式處理。
- **[風險] golden 裁切情境改變後，舊的裁切回歸錨點失效** → 新的錨點建立在 C000043 上，`reid1`、`reid2` 自比為 0；其他情境仍需與 `after1` 相同。

## Migration Plan

1. 產生來源鏡頭的 ROI 點選頁，由使用者畫出 ROI，寫入 `config/roi/`。
2. 修改程式：D2、D3（`frame_rate: auto`）、D4、D5、D6、D9，並撰寫單元測試。

執行範圍的同意：第 3 組（工具模型與追蹤參數比較）與第 4 組（golden 重建）範圍固定、只寫 `runs/`，使用者已事先同意；第 5 組（pilot）與第 6 組（全量）執行前各自再取得同意。
3. 跑工具模型與追蹤參數的比較（D2、D3），定案 `config/reid_crops.yaml`。
4. 重建 golden 的裁切情境（D9），確認其他情境與 `after1` 相同。
5. 執行 pilot（D10），回報估算結果並取得同意。
6. 全量蒐集、污染偵測、產生 `v1` manifest 與審查頁；使用者審查後產生 `v2`。
7. 同步 `scripts/reid_data/README.md`、CLAUDE.md、`CONTEXT.md`。

**還原方式**：裁切、索引、報表、manifest 都在 gitignore 的目錄，可以直接整個移除重跑；程式變更以 git revert 還原。

## Open Questions

- 純度優先參數的具體數值允許迭代一次，最終值由 D3 的比較結果決定，不影響 spec 與任務拆分。
- DINOv2 的類別一致性比例門檻、離群值百分位門檻與比例門檻，在 pilot 時依實際分布定案。
