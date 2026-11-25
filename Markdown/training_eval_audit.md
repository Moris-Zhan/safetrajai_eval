# 補強優先順序與未來計畫（訓練／評估流程）

本檔只保留補強內容：補強優先順序（含各項執行說明）與偵測資料集的未來計畫。盤點結果、Grid Search 覆蓋度歷史、資料集盤點與現用權重現況見 [`training_eval_inventory.md`](training_eval_inventory.md)。

## 補強優先順序（成本／效益判斷）

補充調查日期：2026-11-09。判斷依據：**成本**＝要補齊這個缺口需要新建多少東西（資料／程式碼／人力),**效益**＝補齊後實際影響的範圍與重要性。排序由「最划算」到「最不划算」。

| 順序 | 狀態 | 缺口 | 成本 | 效益 | 為什麼是這個排序 |
|---|---|---|---|---|---|
| 1 | ✅ 已完成（2026-11-09） | **隔離孤立模型權重**（`models/yolo-cls/`、`models/yolo-reid/` 除 `temp/`） | 近乎零——只是 `mv` 到 `temp/` 並記錄 `DO_NOT_USE.md`，不用新建任何資料或程式碼 | 小但立即——避免之後誤用零引用的權重、減少混淆 | 成本幾乎是 0，沒有理由不先做，屬於「順手清理」 |
| 2 | ✅ 已完成（2026-11-10） | **補齊 `tracking/` grid search 覆蓋**（person／car／bus／truck／small_truck × 3 個 tracker） | 低～中——`scripts/grid_search/` 三支腳本、`config/tracking/*.yaml`、`dataset/eval/cctv_all8_subsets/{tune,holdout}/` 全部已經存在，只差：(a) 把 `grid_search_botsort.py`／`grid_search_botsort_reid.py` 裡 car/bus/truck/small_truck 的固定單值網格改回真正的候選範圍，(b) 實際執行並留存 `--result-csv` | 高——tracking 是整條管線的核心，目前 5/6 類別的參數是直接沿用 refactor 前的舊設定，從未驗證是否最佳；影響 HOTA/AssA，也間接影響下游 collision／counting 的輸入品質 | 基礎設施（腳本＋資料＋設定）已經 100% 到位，缺的只是執行與一處小改動，是目前投入產出比最高的一項 |
| 3 | ✅ 已完成（2026-11-11，CI run 9761da6 綠燈） | **建立基本 CI 自動化**（至少跑 `pytest tests/test_smoke.py`） | 中——需要新建 CI workflow，但不需要新資料；要注意 golden 比對依賴掛載磁碟上的 `dataset/`／`models/`，不易搬進一般 CI runner，實際能自動化的範圍可能只到 smoke test | 中～高——跨所有模組的回歸防護網，目前完全靠人工記得要跑 pytest/golden | 單次建置成本中等，但長期每次改動都受益，優先度次於「資料和腳本已齊備只缺執行」的 tracking 缺口 |
| 4 | 🟡 spec 骨架已完成，門檻待定（2026-11-11，見下方「#4 執行說明」） | **將 `collision/`／homography 驗證正式化成驗收 spec** | 中～高——`validate_homography.py`／`check_homography_scale.py` 已存在，但要仿照 reid 的 `acceptance_grid.py` 設計正式門檻與報表；而且現有 check points 只覆蓋 `C000025`／`C000297` 兩支鏡頭，範圍本來就窄 | 中——只影響有 homography 的 4 支鏡頭（C000025/C000043/C000054/C000085）的衝突指標可信度 | 影響範圍比 tracking 窄，且要先有更多鏡頭的 check points 才能把 spec 做紮實，成本效益不如前兩項 |
| 5 | 🟡 進行中（pilot 資料已產生並匯入 CVAT，待人工覆核，2026-11-12，見下方「#5 執行說明」） | **`detection/` 建立微調訓練資料與流程** | 高——repo 內完全沒有偵測訓練用標註資料，必須從頭標註或另外取得資料集，還要新建訓練/驗收腳本 | 不確定——目前沒有證據顯示預訓練 YOLO（`yolo11x_dennis.pt`）在 `cctv_all8` 評估 GT 上的表現不足，微調的必要性未被證實 | 成本最高、效益卻還沒被證實有需求，應該等有具體證據顯示偵測品質不足後才投入 |
| 6 | ⬜ 未開始 | **`collision/` 建立衝突事件人工標註 GT** | 最高——需要人工在影片上標註真實的近距離接近/衝突事件，主觀且耗時，且目前只有 4 支鏡頭有 homography 可用公尺單位 | 低～中——CLAUDE.md 已明確定位這些指標「只能當技術訊號，不能當正式安全結論」，代表專案本身對形式化驗證的需求程度還不高 | 成本最高，而且專案自身的定位已經限縮了這份資料能帶來的效益上限，目前最不划算 |

**#3 執行說明（2026-11-11）**：新增 `.github/workflows/smoke-test.yml`，push／PR 時自動跑 `pytest tests/test_smoke.py`；golden 比對因依賴掛載磁碟上的 `dataset/`／`models/`，維持手動執行不變。沿用這次補強計畫的 Markdown 報告慣例、不重建已被 commit `68523d8` 移除的 `openspec/`。詳細做法與範圍見 `Markdown/ci/smoke_test_ci_setup.md`。

**#3 後續（2026-11-11）**：第一次真實 CI run（`42986579661`）跑了，而且**失敗**了——38 個測試全部死在同一個根因：`pyproject.toml` 的 `dependencies` 沒宣告 `timm`，但 `reid/training/model.py` 無條件 `import timm`，而這個檔案透過 `reid/encoders.py` → `tracking/class_aware_botsort.py` → `tracking/__init__.py` 被整個 `safetrajai.tracking` 套件無條件引入，所以只要 import `tracking`（幾乎任何進入點都會）就一定需要 `timm`，不管任何情境設定的 `use_reid` 是不是全部關閉。本機 conda env `track_eval` 手動裝過 `timm`，不在 `pyproject.toml` 宣告範圍內，才沒提早發現。已把 `timm` 加進 `pyproject.toml` 的核心 `dependencies`（不是放進 optional extra——因為它在 import 層級就是必要的，不是只有訓練 Re-ID 才需要），並用「比對專案原始碼實際 import 的套件 vs. `pyproject.toml` 宣告」的方式做了一次全面稽核，確認沒有其他類似的隱性依賴（`pillow`／`wandb` 雖然宣告但沒有被直接 import，屬於無害的既有宣告，不是缺口）。推送修正（commit `9761da6`）後的下一次 CI run 已確認轉綠（`status=completed, conclusion=success`），這項正式打勾。

**#4 執行說明（2026-11-11）**：新增 `scripts/homography/acceptance_check.py`，讀取 `validate_homography.py` 既有輸出的 `summary.csv`，套用誤差門檻判定每支鏡頭是否驗收通過，輸出 CSV＋Markdown 報表（做法仿照 `scripts/reid_train/acceptance_grid.py` 的 `summarize`）。刻意不改動 `validate_homography.py`／`validate_camera()` 本身的 `validated` 欄位語意，避免連動影響 golden 比對。誤差門檻（`ACCEPTANCE_RMSE_M`）刻意留 `None`（待定），符合使用者「先不定閾值」的決定；點數合格但門檻未定的鏡頭會回報 `threshold_tbd`，不會冒充通過或失敗。目前的首次報表：C000025／C000043 因點數不足回報 `too_few_checkpoints`，C000054（2.64m）／C000085（20.22m）回報 `threshold_tbd`。詳細設計與完整判定邏輯見 `Markdown/homography/acceptance_spec.md`。

**#4 執行說明（2026-11-11）**：部署選點器網頁服務（`scripts/picker/start_picker_service.sh`），使用者手動替 `conflict_analysis` 實際依賴的 4 支鏡頭中原本缺 check points 的 3 支（C000043／C000054／C000085）各補了一組驗證點。重跑 `validate_homography.py` 後發現：`validate_camera()` 目前「驗證」只檢查點數 ≥ 4（`MIN_CHECKPOINTS`），完全沒有誤差門檻，這個弱點被這次的真實資料直接印證——`C000054` 雖然點數足夠被標成 `validated=True`，但逐點比對後發現 `cctv_points`／`sat_points` 兩份清單的對應順序記反（前兩組與後兩組對調），官方算出的 RMSE 高達 749px（≈51.6m）；用來源影片首幀與衛星圖個別裁切、不依賴矩陣本身獨立核對地標後確認是順序問題，修正順序後重跑，真實 RMSE 降到 38.35px（≈2.64m），已直接修正 `config/homography_checkpoints/C000054.yaml`。`C000043` 的 4 組點裡，有 1 組（原第 3 組）經獨立核對後確認兩側根本是不同路口、不是同一地標，判定為誤選並移除，剩 3 組（RMSE 144.37px≈9.95m，因點數 < 4 被標記 `too_few_checkpoints`，不影響其餘 3 組仍是目前最準的驗證資料）。`C000085` 的 6 組點用窮舉配對（Hungarian 演算法）排除了「只是順序記反」的可能——最佳排列也只能把總誤差降低 12%，且只有其中 2 組點明顯改善，其餘 4 組不管怎麼配對都是 269～419px（≈18.5～29m）的大誤差，判定為真實的點位精度不足或矩陣在此路口精度較差，誠實記錄現狀，不強行調整。驗收門檻（可接受的誤差量，公尺或像素）本項仍未定案，待之後視需要再決定；詳細調查過程見 `Markdown/homography/checkpoint_validation_2026-11-11.md`。

**#5 執行說明（2026-11-12）**：依「未來計畫」第 1～7、8、9 點執行了 pilot 階段的取樣、偽標籤、人工覆核前的準備工作：

- **取樣與偽標籤**：新增 `scripts/detection_data/sample_and_pseudolabel.py`，對 `PILOT_FILES` 清單（9 支鏡頭各 1 個檔案，`C000054` 取時間跨度最大的頭尾 2 個檔案，共 10 個檔案）每 1.5 秒取樣一張畫面，用目前在用的偵測權重（`botsort_reid_timm.yaml` 的 `det_model_path`／`iou`／`imgsz`）推論產生偽標籤。輸出 3308 幀到 `dataset/detection/images/`／`labels/`（YOLO 格式，無框的背景幀寫 0-byte 空檔）與 `dataset/detection/splits/pilot_v1/manifest.csv`，仿照 `dataset/reid/crops` 的版本化慣例（對應第 8 點）。逐類別框數：person 1754、car 34882、scooter 24464、small_truck 2184、bus 437、truck 3824，全部超過 pilot 門檻（100，對應第 6 點）。
- **資料集登記**：`config/datasets.yaml` 新增獨立的 `detection_source` 項目（對應第 9 點），鏡頭與 `reid_sources` 重疊但不共用同一個 dataset id。
- **CVAT 部署**：依「手動任務做成網頁服務」慣例，在本機用 Docker Compose 自架 CVAT（`ignore/cvat_deploy/cvat/`，clone 自 `cvat-ai/cvat`，gitignored，不計入版控），對應第 7 點（啟動、停止與覆核腳本用法見 `Markdown/detection/cvat-operations.md`）。過程中發現兩個實務問題：
  1. `/mnt/moris` 是 NFS 掛載，docker daemon 看不到同一份內容，無法用 bind mount 的「share 目錄」方式匯入影像；改用 CVAT 官方 HTTP 上傳 API（`cvat-sdk`，裝在獨立的 `ignore/cvat_deploy/cvat_client_venv/`，不動 conda env `track_eval`）。
  2. `cvat-sdk` 的上傳邏輯內部會對每個檔案路徑呼叫 `Path.resolve()`，若用 symlink 省空間，會被解析回原始真實路徑，導致原本刻意取的唯一檔名失效、暴露出 `dataset/detection/images/<seq>/<frame_idx>.jpg` 這種跨影片共用的 `frame_idx` 檔名（例如多支影片都有 `000000.jpg`），造成上傳時的資料庫唯一約束衝突。改用真正複製檔案（非 symlink）、檔名改成 `<seq_idx:02d>_<frame_idx:06d>.jpg`（`seq_idx` 為該 seq 在 `manifest.csv` 中首次出現的序號）才解決。
- **建立專案與匯入**：建立 CVAT 專案 `safetrajai detection pilot v1`，標籤依 `CLASS_NAMES`（`utils/modules/safetrajai/classes.py`）順序（person／car／scooter／small_truck／bus／truck），單一 task `pilot_v1` 依 manifest 順序上傳全部 3308 幀（對應使用者 2026-11-11 決定：單一 task、一次匯入全部）。先匯出該 task（空的）YOLO 1.1 格式標註以確認 CVAT 預期的 zip 結構（`obj.data`／`obj.names`／`train.txt`／`obj_train_data/<name>.txt`），再用既有偽標籤依同樣結構打包匯入。匯入後核對標註框總數為 67545，與逐類別框數加總（1754+34882+24464+2184+437+3824）完全一致，確認匯入無誤。
- **全片重新追蹤並改用 MOT 格式匯入**（2026-11-12 追加）：接著確認「CVAT 的 Automatic Annotation 能不能批次用演算法預先 track `pilot_v1` 的物件、輸出 MOT (Multiple Object Tracking) 格式」。查證 CVAT 原始碼（`cvat/apps/lambda_manager/views.py`）確認：批次自動標註只接受 `detector`／`reid` 兩種 nuclio function kind，`tracker` kind 只能用在互動式單物件追蹤（AI Tools 逐幀點選延伸），天生不支援批次多物件追蹤；而且 `pilot_v1` 的取樣幀之間間隔 21 個原始影格（每 1.5 秒取樣一次），超過目前追蹤設定的 `track_buffer`（10～15 幀），直接對取樣幀跑追蹤也不會有連續 ID。
  改用「對原始連續影片重新跑完整追蹤、再對齊回取樣幀」的做法：新增 `scripts/detection_data/track_pilot_videos.py`，沿用既有的 `VideoSequenceTracker`／`save_mot_format()`（跟主 pipeline stage1 同一套邏輯），對 `PILOT_FILES` 這 10 支原始影片各跑一次完整逐幀追蹤（共 57211 幀，約 35 分鐘），輸出到 `runs/cvat_pilot_v1_tracking/`。用一個小型測試 task 先實測驗證 CVAT「MOT 1.1」匯入格式的語意（`gt.txt` 的 frame 欄位是任務內 1-based 絕對幀序，不是檔名衍生值；track 之間的幀距超過 `frame_step` 時，匯入器會自動補插 `outside=True` 的消失標記），再把 `pilot_v1` manifest 裡每個取樣幀對齊回完整追蹤輸出的對應影格，track ID 依 `(seq, class, 原始 track_id)` 重新配成全域唯一（因為專案的 track_id 契約是「只在單一序列、單一類別內唯一」），清空並以 MOT 1.1 格式重新匯入 `pilot_v1` task，取代掉原本無 track ID 的 67545 個偵測框。
  匯入後核對：13450 條 track（58718 個真實關鍵幀框＋73632 個含自動補的 outside 標記的 shape 總數），抽查單一 track 的框位置逐幀平滑移動，確認 track ID 連續有意義。逐類別真實框數比原本純偵測版本略少（因為 BoTSort 的追蹤確認邏輯比單幀門檻過濾更嚴格）：person 1754→1384（-21%）、car 34882→31000（-11%）、scooter 24464→21068（-14%）、small_truck 2184→1341（-39%，降幅明顯最大）、bus 437→398（-9%）、truck 3824→3527（-8%），全部仍遠高於 pilot 門檻（100）。
  已知但使用者決定先不處理的瑕疵：(1) 36%（4874/13450）的 track 只有 1 個真實關鍵幀，跟舊版無 track ID 的偵測框資訊量相同；(2) 10 支影片在 CVAT 任務裡首尾相接，有 184 筆「track 結束」的 outside 標記因匯入器的自動補幀邏輯落在下一支不相關影片的第一幀上（不影響任何畫面顯示，純粹是資料裡多一個不顯示的收尾標記）；(3) small_truck 降幅特別大的原因還沒查，先讓人工覆核去發現問題再回頭查。舊版「純偵測、無 track ID」的標註已於匯入前用 `task.remove_annotations()` 清空，CVAT 裡不再保留；來源的偽標籤檔案（`dataset/detection/labels/`）本身沒有被動到，需要的話可以重建對照版本。
- **人工覆核前的抽樣檢視**（2026-11-12 追加）：瀏覽器自動化工具在這台機器上無法使用（沒有 Chrome 執行檔，且以 root 執行會被 Chrome 自己擋下），無法直接開 CVAT 網頁；改為直接把匯入 CVAT 的框／track ID 疊回對應的原始畫面（`dataset/detection/images/`）存成圖片，用多模態直接看圖抽查。
  一般抽樣（5 張，涵蓋不同鏡頭、一般情境＋一條 108 個真實關鍵幀的長 track）：框普遍貼合車輛，類別在清楚案例上都正確；長 track（同一輛長時間停著的車）跨 108 幀（約 2.7 分鐘）track ID 維持同一個號碼沒有跳號，確認這批 MOT 匯入是有意義的連續追蹤；seq 交界處前後幀確認「184 筆 outside 標記跨到下一支影片」不會畫出任何看得見的框。順手看到兩個疑似漏檢（一台只露一角的公車、一位穿越斑馬線的騎士都沒被框到）與一個疑似誤判（一台白色廂型車被標成 car，外觀較像 small_truck）。
  **針對 small_truck／bus 這兩個降幅最大的類別，另外抽了 4 支不同鏡頭各一張（共 8 張）**，結果明顯比一般抽樣差：small_truck 4 張裡，1 張框太小看不清楚、1 張把普通轎車框成 small_truck（誤判）、1 張把背景小客車框成 small_truck、同時畫面中央一台大型水泥攪拌車完全沒被偵測到（誤判＋漏檢）、1 張同一台紅色轎車附近疊了 5 個不同 track ID 的 small_truck 框，實際框的是轎車和旁邊機車（誤判且重複配出多條虛假 track）。bus 4 張裡，1 張 manifest 記錄有 bus 但該幀的 track 資料完全沒有 bus 框（疑似被追蹤器確認邏輯濾掉）、1 張正確框住真正的公車、1 張把路邊機車行店面招牌框成 bus（誤判，跟前述漏檢大貨車是同一個路口相近幀）、1 張疑似把行人/機車騎士那一帶框成 bus。8 張裡只有 1 張乾淨正確。
  這跟前面逐類別框數比對「small_truck 降幅最大（-39%）」是吻合的，而且問題比單純漏框更嚴重：相當比例是把轎車／機車／店面招牌誤判成 small_truck 或 bus。初步判斷這比較像 `yolo11x_dennis.pt`（「起始模型」）本身在這兩類的辨識能力不足，不是追蹤或匯入流程造成的；如果之後依「未來計畫」精煉模型，這兩類應優先改善。本項抽樣純粹是人工覆核前的參考，不是系統性排查，完整結論仍待正式人工覆核。
- **small_truck 人工覆核＋Re-ID 輔助自動修正（2026-11-13 追加）**：這台機器上瀏覽器自動化仍不可用，改部署 `scripts/cvat_review/`（`fetch_task_annotations.py` 用 `cvat_client_venv` 抓全量標註、`serve_review.py` 用 `track_eval` env 把指定類別每條 track 的真實關鍵幀疊框畫出來，做成跟 `scripts/picker/` 同一套 token 保護網頁服務，供使用者透過 `kubectl port-forward` 遠端逐 track 瀏覽）。使用者瀏覽 small_truck 前 22 條（依真實關鍵幀數排序）後回報 6 個錯誤：3 條（#3329／#3871／#9104）實際是 car、3 條（#12069／#12147／#12981）是同一支固定紅綠燈（C000256 鏡頭，中心約 (12,317)、框尺寸約 41px，完全靜止），不是任何車輛類別。
  使用者要求「用 reid 自主修正其他 ID 的同類錯誤」後：
  1. 先驗證兩個零 ML 的幾何假設——「與現有 car track 框重疊」（IoU≥0.5）在已知 3 個 car 誤判裡只命中 1 個、且在已驗證正確的 track 上也有 5 個誤報，**驗證失敗，捨棄**；「同鏡頭＋同位置＋幾乎靜止的重複 track」可以精確找到紅綠燈那一簇（含 3 個已確認＋7 個同位置同尺寸的額外 track），但同一招在別處也會命中**已驗證為正確**的真實停放小貨車（#8519／#4880／#4882／#3556 所在的靜止簇），**不能一概當成誤判自動刪除**。
  2. 改用 `scripts/cvat_review/reid_outlier_scan.py`（沿用 `scripts/reid_data/detect_contamination.py`／`safetrajai/reid/dinov2_features.py` 的 DINOv2 外觀嵌入做法）以 16 條已驗證正確的 small_truck track 當參考，算其餘每條 track 的最大外觀相似度。驗證結果：3 個已知 car 誤判的相似度（0.76～0.82）落在已驗證正確 track 的同一區間（leave-one-out 0.63～0.86），**DINOv2 外觀分不出「car 誤判成 small_truck」這種同屬四輪車的混淆**；但 3 個已知紅綠燈的相似度（0.36 上下）明顯偏低，且掃描全部 635 條未覆核 track 後，額外找出 15 條單一真實關鍵幀（因只有 1 幀，幾何靜止判定完全漏掉）、座標跟已知紅綠燈完全一致的 track。
  3. 最終自主套用的修正（`scripts/cvat_review/apply_corrections.py`，套用前備份、套用後重新抓取驗證）：刪除 25 條（同一支 C000256 紅綠燈的全部重複 track：原始幾何法找到的 10 條＋DINOv2 掃描補上的 15 條單幀 track）、改標 5 條為 car（#3329／#3871／#9104 直接依使用者判定；#9095／#9135 跟 #9104 位置尺寸幾乎一致，判定為同一台停放車輛）。small_truck 從 666 條降到 636 條，car 從 5735 增加到 5740，皆經重新抓取驗證。修正紀錄見 `runs/cvat_review/corrections_applied.csv`（未提交版控）。
  4. **C000033 的 19 條靜止 track 候選，經使用者要求先看圖再決定**：把其中 4 條（#2906／#3073／#4225／#4796，橫跨 t=23.5s～572.5s，涵蓋近全片時長）疊框＋放大裁切後直接看圖，確認框住的是畫面左下角一個固定非車輛物件（類似鐵捲門／機箱的結構，貼在攝影機立柱或路側設施上，兩個相隔超過 6 分鐘的影格裡外觀、位置完全一致），不是任何車輛。視覺確認後補刪這 19 條（第三輪修正）。small_truck 最終從 636 條降到 617 條。
  至此 small_truck 的自動修正總計刪除 44 條（25＋19，兩支不同的固定非車輛物件重複配出的 track）、改標 5 條為 car，從原始 666 條降到 617 條。
- **剩餘 601 條 small_truck 的去重＋外觀分群抽樣（2026-11-16 追加）**：使用者要求把剩餘未逐一覆核的約 600 條，先用 Re-ID 去重再縮減到只要人工看 30 條，經 `/grill-me` 確認機制後動工。
  1. 新增 `scripts/cvat_review/dedup_clusters.py`：延續 C000256／C000033 驗證過的規則（同鏡頭＋框中心距離 <25px＋尺寸比 <1.3＋多關鍵幀 track 位移須 <0.5＋群內 DINOv2 外觀相似度均值 ≥0.75，任一條件不符就整群打散回單條），對剩下 601 條做「身分去重」（判定為同一實體物件、結論可套用到整群）。結果：只找到 16 個確認群（共 47 條），其餘 554 條仍各自獨立——**身分去重只能把 601 條壓到 570 條（省不到 10%）**，確認小貨車的重複偵測問題主要就是前面已處理的那兩支固定物件，不是普遍現象。
  2. 使用者選擇改用「外觀分群抽樣」（新增 `scripts/cvat_review/appearance_sample.py`，純 DINOv2 特徵 k-means，不要求同一實體）把 554 條單條壓成 14 群，每群抽 1 條代表（最接近群中心的 track）；代表的結論**不能**套用到整群，只能當作該外觀類型大致的參考。16 個身分群代表＋14 個外觀群代表＝剛好 30 條覆核量。
  3. 把這 30 條整合進 `serve_review.py` 新增的 `/dedup` 頁面（區分「身分去重」／「外觀分群抽樣」兩種保證層級），服務重啟後驗證可用。
  4. 助理先用疊框＋放大裁切對 30 張代表逐一人工目視覆核（因瀏覽器自動化仍不可用），初步判定：16 個身分群裡只有 3 群正確（small_truck），7 群是 car／bus 誤判，2 群根本不是車輛（路邊灌木叢、人行道垃圾桶），4 群不確定；14 個外觀群代表裡只有 3 個正確，8 個誤判成 car 或行人/機車，3 個不確定——**30 條代表裡超過一半有問題，再次確認 small_truck 的精確率問題貫穿整個資料集，不只侷限在前面找到的兩個固定物件**。
  5. 使用者逐一核對後確認／補充判定（含推翻助理原判：#11986 其實是 car，不是無車輛）：car（#2970／#4418／#859／#11986／#2711／#1656／#2609／#8259／#2201／#4373）、bus（#11622）、truck（#4075）、scooter（#9792，機車騎士）、非車輛刪除（#1482，垃圾桶）；訊息中未提及的項目視為助理原判定成立（#7581／#10792／#1892／#2345 維持 car；#7423／#4990／#4993／#8673／#1514 因助理原判定即為「不確定」，沒有可套用的明確結論，維持 small_truck 不動）。
  6. 套用修正（`apply_corrections.py`，套用前備份、套用後重新抓取驗證）：身分去重群（結論套用整群）——刪除 2 條（#1482 群）、改標 17 條為 car（#859／#4418／#11986／#2970／#7581／#10792 共 6 群）、6 條為 bus（#11622 群）、3 條為 truck（#4075 群）、2 條為 scooter（#9792 群）；外觀分群代表（只改代表本身，其餘成員不動）——8 條改標為 car、1 條（#3998）改標為 scooter。全部 39 筆經重新抓取驗證無誤。small_truck 最終從 617 條降到 **578 條**；car 5740→5765、bus 141→147、truck 974→977、scooter 5572→5575。
  7. **尚未解決**：5 個身分去重群（#7423／#4990／#4993，共 7 條，small_truck 還是 truck 不確定）維持原標籤未動；2 個外觀分群代表（#8673／#1514）同樣因不確定維持不動；外觀分群抽樣天生無法把結論套用到未被抽中的其餘約 540 條單條 track，這些仍是完全未知風險，沒有被這輪修正觸及。
- **bus 人工覆核（沿用 small_truck 同一套方法論，2026-11-16 追加）**：bus 共 147 條，依序套用 `dedup_clusters.py`（身分去重）→ `apply_corrections.py` → `appearance_sample.py`（外觀分群抽樣）→ `apply_corrections.py`，分三輪：
  1. **身分去重**：147 條分出 4 群（共 33 條）＋114 條單條。疊框＋放大裁切逐群看代表後確認：2 群是誤判（#6046 代表的 19 條，C000048 一支固定路名標誌桿；#3214 代表的 5 條，C000033 一支固定的 SYM 機車行招牌），刪除共 24 條；另 2 群（#11622 代表 6 條、#6105 代表 3 條，共 9 條）視覺確認是真公車的正常碎片化重複 track，維持不動。
  2. **外觀分群抽樣第一輪**（k=6，對 114 條單條）：這次額外對每群多抽了 2～5 個成員（不只看代表），**直接拿到會推翻「代表結論可套用整群」這個假設的反例**——代表是轎車的群（#3657）裡混了一條真公車（#714）和一位機車騎士（#13046）；代表是 SYM 招牌誤判的群（#3408）裡其他成員（#465／#6337）是完全不同地點、無法判讀的畫面。這具體驗證了 `appearance_sample.py` docstring 原本就寫的警語（外觀分群代表的結論不能延伸到整群），不是理論上的保守假設，而是這次真的抽到了會打臉的案例。
     同一輪也發現一個例外狀況：代表 #4090 所在的 20 條群，額外抽樣的 5 個成員全部落在同一鏡頭（C000033）、幾乎同一個像素區域（x≈955–1280, y≈193–306），跟代表一樣都是同一塊 SYM 招牌——6/20 個樣本的鏡頭＋位置＋外觀三者都高度一致，判斷這其實是身分去重的幾何門檻沒抓到的同一實體物件，破例把結論延伸到整群 20 條一併刪除；其餘群維持「只動代表本身」的保守規則（#3408 刪除、#3657 改標 car、#13046 改標 scooter，其餘成員不動）。套用後刪除 21 條、改標 2 條。
  3. **外觀分群抽樣第二輪**（k=6，對排除已確認身分群後剩下的 91 條單條）：6 個代表裡 5 個是真公車（含遊覽車），只有 1 個（#4158）是黑色休旅車誤判，改標為 car；另 1 個（#6337）畫面太模糊無法判讀，維持不動當未知風險。
  4. bus 從 147 條最終降到 **99 條**（24＋21 刪除、2＋1 改標，共修正 48 條）。跟 small_truck 不同的是這次誤判集中在少數幾支固定物件與外觀分群的少數代表，沒有再出現 small_truck 那種「超過一半樣本都有問題」的大範圍精確率問題。
  5. **尚未解決**：#6337（模糊不明）；外觀分群兩輪代表以外、沒被抽到的其餘成員（第一輪約 90 條、第二輪約 85 條）仍是完全未知風險，沒有被這輪修正觸及。
- **car／scooter／truck 人工覆核（沿用同一套方法論，2026-11-16～11-17 追加）**：car 5767 條、scooter 5576 條、truck 977 條，依序套用 `dedup_clusters.py`（身分去重）→ `apply_corrections.py` → `appearance_sample.py`（外觀分群抽樣，對身分去重後的 singles 池）→ `apply_corrections.py`，分兩輪：
  1. **身分去重**：car 分出 29 群（136 條）＋5631 條單條；scooter 分出 28 群（131 條）＋5445 條單條；truck 分出 30 群（313 條）＋664 條單條。背景 fork 逐群渲染覆核（truck 前 10 大群 239 條另外手動交叉驗證），結果：car 24 真／2 誤判（5 條）／3 不確定；scooter 10 真／4 誤判（22 條）／14 不確定；**truck 7 真／22 誤判（289 條，占 truck 身分去重總量 92%）／1 不確定**——truck 的固定背景物（高架橋、號誌桿、招牌、護欄等）誤判比例遠高於 small_truck／bus。套用修正：car 刪除 5 條（2 群，均為號誌/標誌物）；scooter 刪除 20 條（3 群，路面標線/雜訊）＋改標 2 條為 person（1 群，機車騎士）；truck 刪除 282 條（19 群，固定結構物）＋改標 7 條為 car（3 群，真實小客車）。套用後：car 5767→5769（−5 刪除，+7 truck 改標進來）、scooter 5576→5554（−22）、truck 977→688（−289）。
  2. **外觀分群抽樣**（k=120 對 car／scooter 的 singles、k=20 對 truck 的 singles，只覆核被抽中的代表本身，結論**不延伸到同群其他成員**）：car 5631 條抽 120 群、scooter 5445 條抽 120 群、truck 664 條抽 20 群，共 260 個代表。背景 fork 覆核結果：car 105 真／12 誤判／3 不確定；scooter 108 真／7 誤判／5 不確定；**truck 3 真／16 誤判（80%）／1 不確定**，再次確認 truck 的固定背景物誤判問題貫穿整個資料集、不只侷限在身分去重抓到的大群。套用修正（共 35 條，僅代表本身）：car 刪除 2 條＋改標 10 條（scooter 3／small_truck 3／bus 3／truck 1，誤判成 car 的機車騎士、貨卡、巴士、拖板車）；scooter 刪除 7 條（空草地/灌木叢、號誌燈頭、機箱等誤判）；truck 刪除 8 條（固定招牌/護欄/天橋/號誌桿）＋改標 8 條（car 4／bus 3／person 1）。套用後：car 5769→5761、scooter 5554→5550、truck 688→673、small_truck 578→581、bus 99→105、person 364→365。
  3. 兩輪合計後，總軌跡數從 13052 降到 **13035**；最終各類別：car 5761、scooter 5550、truck 673、small_truck 581、bus 105、person 365。
  4. **尚未解決**：身分去重的 18 個不確定群（car 3／scooter 14／truck 1）未處理；外觀分群抽樣的 9 個不確定代表（car 3／scooter 5／truck 1）未處理；外觀分群抽樣天生無法把結論套用到未被抽中的其餘上萬條單條 track（扣除本輪已覆核的 260 個代表，car／scooter／truck 的 singles 池絕大多數仍完全未知風險），沒有被這輪修正觸及。
  5. **外觀分群抽樣第二、三輪**（2026-11-17 追加；每輪先從上一輪的輸入池扣掉已抽過的代表，再對剩餘池重新分群，仍然只動代表本身）：
     - 第二輪（剩餘池 car 5511／scooter 5325／truck 644，k=110／110／60，共 280 個代表，背景 fork 覆核）：car 95 真／8 誤判／7 不確定；scooter 93 真／12 誤判／5 不確定；truck 19 真／39 誤判（65%）／2 不確定。套用 59 條修正（`corrections_{car,scooter,truck}_round3.csv`）：car 改標 7 條（scooter 3／small_truck 3／truck 1）＋刪除 1 條；scooter 刪除 10 條＋改標 2 條為 person；truck 改標 18 條（car 10／bus 7／scooter 1）＋刪除 21 條。套用後總數 13035→13003。
     - 第三輪（剩餘池 car 5401／scooter 5215／truck 584，k=108／104／55，共 267 個代表；fork 派送被 hook 擋住，改由主對話把代表拼成 14 張拼圖直接覆核）：car 90 真／11 誤判／7 不確定；scooter 83 真／9 誤判／12 不確定；truck 14 真／34 誤判（62%）／7 不確定。套用 54 條修正（`corrections_{car,scooter,truck}_round4.csv`）：car 改標 10 條（scooter 5／bus 3／small_truck 2）＋刪除 1 條；scooter 刪除 7 條＋改標 2 條為 person；truck 改標 21 條（car 11／bus 9／scooter 1）＋刪除 13 條。套用後總數 13003→**12982**；各類別：car 5763、scooter 5539、truck 601、small_truck 586、bus 124、person 369。
     - **這個迴圈的侷限**：代表誤判率三輪都沒有下降（car 12/120→8/110→11/108；scooter 7/120→12/110→9/104；truck 16/20→39/60→34/55），而每輪只修代表本身，car／scooter 每輪各只修約 10 條。以代表誤判率粗估，剩餘池內還有 car 約 480 條、scooter 約 430 條、truck 約 320 條錯誤（代表是每群一票、不是每條 track 一票，所以只是粗估，不能當驗收數字）。照這個做法無法收斂，因此改採下面的覆核計畫。
     - 三輪外觀抽樣累計 49 個不確定代表（第一輪 9、第二輪 14、第三輪 26）未處理；未被抽到的 singles 仍有 car 5293 條、scooter 5111 條、truck 529 條從未看過。
- **pilot_v1 覆核計畫（2026-11-17，經 `/grill-me` 三輪確認後的共識）**：
  1. **範圍與終點**：分兩階段——第一階段清類別與誤檢，第二階段檢查漏檢與框位置。終點是六類都通過第一階段驗收、第二階段量測並處理完；匯出訓練標籤、微調、det_eval 驗收另外再討論。
  2. **第一階段驗收**：每類殘留類別錯誤率 ≤ 3%，以每類隨機 200 條 track 量測（不足 200 條就全看）。由 Claude 判全部並順手修正，使用者複核每類 40 條；兩者不一致超過 5% 時該類數字作廢重量。外觀分群代表的誤判率不是無偏估計，不作為驗收依據。
  3. **類別邊界**：以 `cctv_all8` 人工 GT 的實際標法為準（驗收用的就是這份 GT，訓練標籤標法不一致會被 det_eval 算成錯誤），由 Claude 抽 GT 的邊界案例歸納規則，寫成 `Markdown/detection/labeling-guide.md`。GT 本身不一致處以多數標法為預設，列給使用者決定。規則定案後回頭複查已套用的修正；規則定案前暫停剩餘池的覆核。
  4. **稽核**：使用者抽查已套用修正 40 條＋Claude 判為「真」的代表 40 條；任一邊錯誤率超過 5%，該類別重審。累計的不確定項目由使用者逐條裁決。兩者與邊界規則確認併成使用者的第一次網頁作業（`serve_review.py`），驗收複核 240 條是第二次。
  5. **各類別做法**：truck 剩餘 529 條、person 369 條、bus 124 條全部看完；small_truck 586 條先做隨機 200 條量測，不過關再全看；car／scooter 用已確認標籤在 DINOv2 特徵上訓練簡單分類器，依可疑度排序、每批 100 條覆核，連續 2 批每批錯誤 ≤ 3 條就去驗收，沒過再看 2 批。
  6. **固定背景誤檢的例外**：同一支鏡頭內、框幾乎不動、且與已確認刪除的誤檢框 IoU ≥ 0.7 的 track，看過該位置的拼圖後可以整組判定。這跟「代表結論不延伸到同群成員」不衝突：後者是外觀相似，這裡是同一個實體物件；停著不動的真車也符合「靜止」，所以仍要看過才判，不自動刪除。
  7. **第二階段**：先從 10 個檔案分層隨機抽 100 幀，整幀連同所有框畫出來量測。門檻事先訂定：漏檢物件佔比 ≤ 5% 且框明顯偏移佔比 ≤ 5% 就不做全面逐幀修正；超標時只修超標的鏡頭或類別。
  8. **套用授權**：第一次網頁作業的抽查通過後，依規則判為明確誤判的項目由 Claude 直接寫修正檔並套用，每完成一個區塊回報前後數量與誤判型態；不確定項目一律留給使用者。抽查沒過則維持每批先問。
- **覆核計畫執行進度（2026-11-18）**：
  1. **類別邊界規則**：`Markdown/detection/labeling-guide.md` 的 §4 十項人工 GT 沒有涵蓋的情況已全部確認並移入 §1、§2：遊覽車、接駁中巴、小巴為 bus；車身明顯比轎車寬、高約一倍或有雙後輪軸為 truck，其餘有獨立貨斗為 small_truck；載客皮卡為 small_truck；聯結車、貨櫃車、預拌車為 truck；自行車、重型機車、三輪機車、電動代步車為 scooter；牽車的人與推車的人只框 person；乘客不另外標；停放無人的機車刪除；包住多個道路使用者又標了錯誤類別的框刪除。§3（GT 內部不一致）仍依多數標法，未逐項確認。
  2. **第一次網頁作業（稽核，`scripts/cvat_review/serve_audit.py`，226 題）**：已套用修正抽 40 題，使用者不同意 1 題（2.5%，`applied-9462`：我改成 car，使用者判為刪除）；判為「真」的代表抽 40 題，不同意 0 題（1 題選看不出來，有結論 39 題）。兩邊都在 5% 以內，通過；樣本只有 40，只能說沒有明顯的系統性錯誤，不是錯誤率的精確估計。依覆核計畫第 8 點，預先授權生效。另有 146 題不確定或疑似標錯的代表逐題裁決（2 題沒答）。
  3. **套用結果**：稽核裁決 43 條（刪除 29、改標 14，`corrections_audit_round1.csv`）；靜態誤檢成員 136 條刪除（`corrections_audit_static_round1.csv`）；17 題沒有結論的題目依規則重判後套用 6 條（刪除 4、改標 small_truck 2，`corrections_audit_round2.csv`）；依新規則重查已套用修正後刪除 2 條（`corrections_recheck_round1.csv`：一個框住整排停放機車的框；`corrections_recheck_round2.csv`：使用者裁決 4097）；另有 3 條依使用者裁決還原舊類別（`corrections_recheck_round3.csv`）。之後的剩餘池覆核：bus 87 條看完（刪除 2、改標 9，`corrections_review_bus_round1.csv`；另依群框規則刪除 6996、10701，`corrections_group_delete_round1.csv`）；truck 548 條看完，約七成標錯（改標 239、刪除 152，`corrections_review_truck_round1.csv`；改標以 bus 103、car 85、small_truck 51 為主，刪除多為店面與施工護欄等非道路使用者，群框 43 條記進 `box_issues.csv`）。person 362 條看完，約六成有問題（刪除 185、改標 scooter 39，`corrections_review_person_round1.csv`；刪除的都是號誌燈頭、黃黑斜紋柱子、路面標線、草地等非道路使用者，騎自行車的人依指南改 scooter）；small_truck 573 條全看（先抽 200 條，因約六成標錯，把剩下 373 條也看完）：改 car 316、bus 10、scooter 13、truck 7、刪除 22（群框，記進 `box_issues.csv`），維持約 205 條（`corrections_review_small_truck_round1.csv`、`round2.csv`）；truck 與 small_truck 的界線是肉眼判斷、沒有量尺寸，中型白色箱型車維持 small_truck，邊界項可能有 10～20 條與使用者判斷不同，留到第二次網頁作業確認。car 與 scooter 依可疑度排序（`scripts/cvat_review/rank_suspicion.py`：框內有兩個以上其他車輛框的「疑似群框」優先，其次是面積超過同鏡頭同類別中位數 3 倍的「疑似過大」）每批 100 條，看到命中率低於 10% 就停：car 第 1 批 36%（刪除 19 個群框、改 bus 9、small_truck 6、truck 2）、第 2 批 6%（改 small_truck 4、刪除群框 2）；scooter 第 1 批 25%（刪除 24 個停等機車群框與 1 個綠色箱子）、第 2 批 7%（刪除群框 7）。看過但維持的 track 記在 `runs/cvat_review/reviewed_kept.csv`，排序與縮圖腳本都會跳過；刪除的群框畫面編號記在 `box_issues.csv`（約 120 條，群框判斷偏嚴，遠處的小框可能有少數誤刪，第二階段補框時一併檢查）。總數 12982→**12395**；各類別：car 6097、scooter 5430、truck 198、small_truck 286、bus 239、person 145。
  4. **外觀分群代表的結論不延伸到同群成員，只有靜態誤檢例外**：抽樣 43 個群各 4 個成員，大多不是同一種東西（同群內有公車、機車、貨車），所以不整群套用。例外是同一支鏡頭內、位置與大小（中心 ≤ 6 px、寬高差 ≤ 25%）都與已刪除代表一致且框不動的成員（號誌桿旁的黃色物體、分隔島植栽、樹叢、號誌燈頭等），共 18 群 136 條。`REP 59`、`REP 7946` 兩群（8 條）看不清楚，不擴及。
  5. **重查已套用修正的結果**：依新規則重查 truck／small_truck／car 之間與框住多個目標的 17 條，只有 1 條明確違規已修正。邊界題與其餘複查項由使用者逐條裁決（2026-11-18）：4097 刪除；2136、8989、1512、7625、3396、4105、4075、12575、2081、9860、7283、6082 保留目前（新）類別，其中框位置有問題的留到第二階段；「維持」指維持舊類別：2193、9993 還原為 truck、551 還原為 car（`corrections_recheck_round3.csv`），13 本來就是 car；6996、12120、17 框內物體不夠清楚，不動。
  6. **六類驗收抽樣與殘餘錯誤率（2026-11-19）**：每類隨機抽 200 條（不足則全看），明確錯誤（改類別、群框、非物件）逐條修正後記錄於 `runs/cvat_review/acceptance_results.csv`。首輪：person 5.5%、truck 10.6%（修正後全部看完）、bus 1.0%（過關）、small_truck 5.0%（全部看完）、scooter 10.0%、car 9.0%。scooter 做靜態誤檢傳播（`static_clusters.py`，同鏡頭同位置同大小且框不動的群；只限 scooter，car 的停放車是真的不可傳播）刪 135 條後重抽 7.0%；再掃面積 < 1000 px² 的 400 條（命中率 15%、15%、17%、8%，第 4 批起低於 10% 停手；`rank_suspicion.py --max-area`）後重抽 8.0%（16/200，與 7.0% 在抽樣雜訊內無差異），剩餘錯誤主要是中等大小（約 1500～5000 px²）的多人／多車合框，不集中在小框。car 修正後重抽 6.5%（13/200）。**決定（使用者 2026-11-19「都先照建議」）**：scooter 不再逐條追 3%，如實記錄殘餘約 7～8%、car 約 6.5%，群框幀列在 `box_issues.csv` 供第二階段與訓練排除，並以 phase-2 的 100 幀實測判斷殘餘錯誤對結果的影響；「每類 ≤ 3%」改為「如實記錄的實測殘餘率」。
  7. **新增帳本與腳本**：`reviewed_kept.csv`（已看過且保留的 track，排序與拼圖腳本會略過）、`box_issues.csv`（群框刪除與第二階段框問題，訓練切分要排除這些幀）、`acceptance_results.csv`；`rank_suspicion.py`（群框數與過大比排序，`--max-area` 小框掃描）、`static_clusters.py`（scooter 靜態群）、`build_review_montage.py`（`--include-reviewed`、`--tag`、`--rank-csv`）。目前 CVAT 約 12132 條 track。
  8. **第二階段量測結果（2026-11-19）**：`phase2_frames.py` 每個序列抽 10 幀共 100 幀，整幀連同現有框畫出；漏檢定義為短邊 ≥ 約 16 px、清楚可辨且沒有任何框覆蓋的人／車／機車（遮擋過半或陰影裡看不清的不算）。目前只目測 32／100 幀（C000011 看了 10 幀，其餘各 2～4 幀），粗估每鏡頭漏檢：C000043 約 5%、C000011 約 14%、C000085 約 15%、C000012 約 18%、C000048 約 25%、C000054 與 C000256 約 30%、C000056 約 35%、C000033 約 45%（`runs/cvat_review/phase2/results.csv`）；沒有看到明顯偏移的框。漏檢主要是天橋陰影下的暗色車、遠處車隊、停止線前的機車群，遠超 5% 門檻。用同一權重做整幀加 2×2 切塊推論（`recover_candidates.py`，信心 ≥ 0.10）找到 2479 個未覆蓋候選，信心 ≥ 0.5 的 664 個中隨機抽 40 個目測，精確率約 55%，誤檢多為路面標線、電線桿、招牌（紅色招牌被判成貨車，信心 0.88）、水泥分隔塊，無法自動收進；推到全部 3308 幀要人工確認約 2 萬個，不實際。**決定（使用者 2026-11-19「都先照建議」）**：不補框，改對各鏡頭「標註不可靠區域」做遮罩（訓練影像塗灰、區域內的框移除），區域由候選密度熱圖產生建議多邊形，再由使用者用選點器網頁確認；剩餘 68 幀不再目測，只記錄粗估與抽樣方式。**進度**：已用 `suggest_masks.py`（24×14 格，高信心候選占該格候選＋現有框的 ≥ 40% 且至少 2 個候選）為 9 個鏡頭產生 29 個建議多邊形（標記比例 C000085 6%～C000054 38%，`runs/cvat_review/phase2/masks_suggested/`）；疊圖顯示它會誤納入紅綠燈、標誌、招牌、路面標線（誤檢候選造成），也會蓋到已框好的車，所以只作起點。現有選點器沒有多多邊形載入與編輯功能，新增 `serve_masks.py`（連接埠 8768，權杖沿用 `runs/cvat_review/server_token`），使用者逐鏡頭修正後存到 `phase2/masks_confirmed/<camera>.json`。**已確認（2026-11-19 09:39～09:49）**：使用者用 `serve_masks.py` 逐鏡頭確認，存於 `runs/cvat_review/phase2/masks_confirmed/<camera>.json`，共 9 個鏡頭、29 個多邊形，遮罩面積占畫面比例（確認版，括號為建議版）：C000011 30%（15%）、C000012 8%（7%）、C000033 16%（27%）、C000043 10%（11%）、C000048 12%（27%）、C000054 23%（36%）、C000056 24%（33%）、C000085 7%（6%）、C000256 10%（12%）。**待辦**：訓練匯出時套用遮罩（塗灰、區域內的框移除）——匯出流程本身（輸出位置與格式、類別、train／val 切分、排除 `box_issues.csv` 的幀）尚未設計，確認版遮罩已複製到 `config/detection_masks/<camera>.json`（進版控，格式 `{camera,image_size,polygons,source,confirmed_at}`，多邊形是要遮掉的排除區，與 `config/roi/` 的保留區相反）。
  9. **訓練匯出與第二次網頁作業（2026-11-19）**：匯出腳本 `scripts/detection_data/export_training_set.py` 已完成並通過試跑（`ft_v1_trial`，不拿去訓練）：以 CVAT 快照產生 YOLO 資料，`box_issues.csv` 的群框刪除幀整幀排除（728／3308 幀），確認版遮罩塗灰（框有 50% 以上在遮罩內就丟棄，保留的框不被塗灰），依序列切分（驗證：C000012 與 C000054 的 20250629 序列）；試跑共 2580 幀（train 1801、val 779），丟棄遮罩內框 5517 個，train 框數 car 14239、scooter 8980、bus 480、truck 425、small_truck 358、person 174。第二次網頁作業（驗收複核）改為外觀分層抽樣：每類以 DINOv2 特徵分 20 群，每群隨機 1 題（共 120 題），結果按群大小加權（`build_acceptance_sample.py`、`serve_audit.py --sample`；舊的隨機 240 題抽樣存 `acceptance_sample_random240.json`）。使用者答完後，加權不同意率：person 13.0%（樣本小，另有 4 題看不出來、2 題未答）、car 6.7%、scooter 3.9%、bus 5.9%、small_truck 35.6%、truck 52.0%；使用者明確判給的 24 題已套用（刪除 7、改標 17，`corrections_acceptance_round1.csv`）。**truck／small_truck 界線**：兩個方向都有不一致，嘗試以框相對同位置轎車的大小比例量測，使用者判斷在 0.9～1.7 倍之間大量重疊、沒有可用門檻，而且外觀相近的車使用者自己的判斷也不一致，所以不重審。**決定（使用者 2026-11-19「都先照建議」）**：truck 與 small_truck 的界線標籤不一致約 30～50%，只在匯出記錄註明；Q8 驗收時這兩類與 bus 只當訊號，不納入「每類都不低於基準」的判定，另外多算一個合併 truck 與 small_truck 的 mAP 看這個大類有沒有進步。
  10. **微調訓練與 v3 雜訊底線（2026-11-24）**：
    - 起始權重是 `yolo11x_dennis.pt`，設定為 AdamW、lr0 0.0002、固定 50 epoch，取 last.pt。
    - 評估在 C000012 + C000056 上進行，用 `scripts/detection_data/eval_finetune.py`。
    - 詳細的逐類別表格、雜訊底線與 segfault 說明見 `Markdown/detection/dataset-build-rules.md` 第 7、10 節。

    合併計算的 mAP50-95：

    | 權重 | 整體 | person |
    | --- | --- | --- |
    | dennis（未微調） | 0.428 | 0.476 |
    | v2 | 0.523 | 0.334 |
    | v3，seed 0 | 0.497 | 0.300 |
    | v3，seed 1 | 0.524 | 0.369 |

    結論：
    - **微調整體有效**：整體分數高於 dennis，主要來自 small_truck、bus、truck。
    - **v2 與 v3 的差異在 seed 雜訊內**：只換 seed，整體就差 0.027。
    - **person 有遺忘**：微調後的 person 都明顯低於 dennis。
    - **兩個防遺忘的變體都不採用**：
      - person 幀 ×3：person 0.323，在雜訊內。
      - 小 person 偽標籤：兩個 seed 的 person 是 0.351 和 0.289。C000012 的改善只在 seed 0 出現（0.337），seed 1 只有 0.165。另外它會增加 person 誤報。
    - 驗證集只有 145 個 person，分辨不出小幅改善。要處理小 person 遺忘，需要更多人工標註的小 person，這可以併入 ft_v4 的新取樣。
    - 誤報和漏檢的量測用 `scripts/detection_data/count_detection_errors.py`。

    - **C000056 person 誤報抽查（2026-11-25）**：212 個 dennis 誤報分三層、抽 51 題，答了 48 題，結果依層大小加權。
      - 騎士一律算 scooter（使用者裁決）。依此，GT 真的漏標只有 10.0%，低於 30% 門檻，不需要補標驗證集。
      - 23.2% 是 dennis 把騎士框成 person，屬於慣例差異；62.3% 是真的誤報。
      - 細節見 `dataset-build-rules.md` 第 7 節。
    - **之後的決定（2026-11-25，使用者「都照建議」）**：
      - 以 ft_v3 的兩個 seed 為比較基準。
      - 身分去重的 18 個不確定群，併到 v4 審查。
      - car、scooter 的殘餘錯誤維持如實記錄，不再追。

    尚未做：
    - 訓練 segfault 的根本原因。
    - ft_v4 的新取樣。
    - 用 `cctv_all8` 的 det_eval 做驗收（「未來計畫」第 10 點）。
    - **2026-11-25 起依 [`detection/finetune-plan.md`](detection/finetune-plan.md) 執行**（grilling Q1～Q10 確認）：保守微調、維持基準模型行為；偽標籤 ≥ 0.3 當標籤、0.1～0.3 塗灰；車輛類別改回基準模型判斷；遮罩縮小；驗收同時看 C000025（tune／holdout）與重做 GT 的公平鏡頭 C000012＋C000056。ft_v4 取樣併入該計畫的 E2（v5b）。
- **尚未完成**：剩餘池覆核已完成 bus、truck、person、small_truck 全數，以及 car、scooter 的可疑度排序與小框抽查；其餘 car、scooter 未逐條看，殘餘錯誤率見上；稽核頁面中使用者選「看不出來」的 15 題與 2 題未答，依規則重判後多半維持原狀；第二次網頁作業已完成（見執行進度第 9 點）；「未來計畫」第 10 點（切分版本、訓練、對 `cctv_all8` 跑 det_eval 驗收比較）同樣沒動工。身分去重的 18 個不確定群尚未經人工確認。CVAT 容器與相關 staging／追蹤輸出檔案、`runs/cvat_review/` 底下的修正檔與稽核回覆只存在於本機、未提交版控（`ignore/`／`runs/` 慣例）。

## 未來計畫：以 `yolo11x_dennis.pt` 為起始模型精煉並製作偵測資料集

記錄日期：2026-11-09（使用者提出，尚未開始實作）。

使用者計畫以目前在用的偵測權重 `models/yolo/yolo11x_dennis.pt` 當作起始模型（seed model），用來精煉（refine）並製作偵測訓練資料集。這對應前面「補強優先順序」第 5 項（`detection/` 建立微調訓練資料與流程）——目前 repo 內完全沒有偵測訓練用的標註資料集，這個計畫就是要補上這個資料缺口。

以下是經過 `/grill-me`（grilling skill）逐輪確認後的共識（2026-11-09）：

1. **來源影片**：`config/datasets.yaml` 已登記的 `reid_sources` 9 支鏡頭（C000011、C000012、C000033、C000043、C000048、C000054、C000056、C000085、C000256）。
2. **精煉方式**：用 `yolo11x_dennis.pt` 自動產生偽標籤，再人工修正——初期採**更高比例的人工逐框覆核**（前幾批每張都看，理由：偵測框的位置誤差／漏檢／誤檢會直接變成訓練標籤的雜訊，比 `reid/` 那種寬鬆的二元排除判斷後果更大），穩定後再視情況拉高自動化比例。repo 內目前沒有任何偽標籤/自動標註的先例可參考，需要全新設計。
3. **類別範圍**：全部 6 類（person／car／scooter／small_truck／bus／truck）一起做。
4. **Pilot 規模**：9 支來源鏡頭各取 1～2 個檔案（約 10～15 個檔案），不是一次用滿 `reid_sources` 全部 39 個檔案。
5. **畫面取樣頻率**：每支影片每 N 秒（建議 1～2 秒）取 1 張畫面，比照 `dataset/reid/` 裁切「每條軌跡每 0.5 秒最多 1 張」的慣例精神，而非逐幀或固定抽樣張數。
6. **Pilot 資料量目標**：逐類別門檻——每個類別至少要有一定數量的標註框（例如每類至少 100 個框）；`bus`／`truck` 這類出現頻率較低的類別若不足，之後再加碼取樣頻率或鏡頭補足，不是只看總畫面張數。
7. **標註/修正工具**：引入 **CVAT**。已確認 repo 內沒有任何現成的「畫框＋選類別」標註工具（`scripts/picker/make_picker.py` 四種模式都只是點選座標點），部署方式與具體操作留到真正開始標註時再決定。
8. **資料集存放結構**：新開 `dataset/detection/`，仿照 `dataset/reid/` 的版本化慣例（`images/`／`labels/`／`splits/<version>/manifest.csv`），不直接沿用現有 `dataset/eval/`／`dataset/raw/` 的扁平結構。
9. **資料集登記**：在 `config/datasets.yaml` 新增一個獨立項目（例如 `detection_source`），即使來源鏡頭跟 `reid_sources` 重疊也不直接共用，讓偵測與 Re-ID 兩條資料管線各自獨立演化。
10. **驗收方式**：精煉後的模型先用現有 `dataset/eval/cctv_all8/annotations_all8/` 這份人工 GT 做 det_eval 比較（precision/recall/mAP），不夠再決定是否擴充評估集。

**背景事實**（調查於 2026-11-09，供執行時參考）：
- 目前 `dennis640`（即 `yolo11x_dennis.pt`）在 `cctv_all8` 6 段序列上的逐類別平均 mAP@50:95：scooter 0.78（最弱）、person 0.84、bus 0.88、car 0.89、small_truck 0.90、truck 0.91（來源：`runs/rebuild-reid-dataset/tool_model/dennis640/trackeval_output/*/det_metrics_*_new.xlsx`，與候選模型 `yolo11x1280` 的比較留存結果）。
- `reid_sources` 9 支鏡頭共 39 個檔案，分佈不均：8 支鏡頭各只有 1 個檔案，`C000054` 一支就有 31 個。
