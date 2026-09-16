## Context

`dataset/reid/crop_dataset/` 目前的 Tier 2 品質過濾（`filter_crop_dataset.py`，見 `reid-training-data-pipeline`）只做「身分內」像素統計相對比較，且該工具與其報告都只針對 DSEC bus 資料併入前的資料量產生過一次，已過期。`reid-training-data-pipeline` 的 pilot（tasks.md 2.4）已確認本專案 YOLO 模型在 DSEC 前視車頭視角對 `bus` 類別會系統性誤判（6 次偵測全部誤判成看板/地面標線），這個既有結論直接限制本次設計能對哪些資料使用「重跑偵測器」這個訊號。crop 檔案本身沒有任何 metadata（`ReidCropCollector.collect()` 只 `cv2.imwrite`），所以任何語意層級判斷都只能靠重新推論或人工目視，無法回頭讀取原始偵測信心。詳見 proposal.md 的 Why。

## Goals / Non-Goals

**Goals:**
- 提供身分層級（agreement_rate）與影像層級（單張不一致）兩種粒度的跨類別不一致偵測，作為人工複核時的排序線索。
- 讓人工能在合理時間內審視全部身分，尤其 car／scooter 這種 300+ 身分的類別，而不是像現有報告只抽 6 個。
- 提供可逆的移除機制，避免像直接 `rm` 那樣一旦誤刪就無法還原。

**Non-Goals:**
- 不追求偵測器重新分類的絕對準確率——分數只是排序輔助，最終刪除與否的判定仍由人工確認清單決定。
- 不為 DSEC 來源身分產生任何自動信心分數（原因見下方 Decision 2），這部分純人工審查。
- 不修改 Tier 1（`ReidCropCollector`）／Tier 2（`filter_crop_dataset.py`）既有邏輯或閾值。
- 不在本次變更中重新訓練或調整 YOLO／ReID 模型權重。
- 不重新訓練或微調任何 ReID 模型；只把既有偵測權重 `models/yolo/yolo11x_dennis.pt` 透過 `encoders.py::ReID` 當成現成的特徵抽取器使用（見 Decision 1（修訂版）），不引入 `models/yolo-reid/`／`models/class_reid_weights/` 底下任何逐類別微調過的權重。

## Decisions

**1.（已修訂，見下方修訂版）用重跑既有 YOLO `Detector` 做一致性檢查，而非 ReID embedding。**
~~Rationale：直接複用專案既有依賴與模型，邏輯是單純的自我一致性檢查（同一顆偵測器、同一批 CCTV 視角資料）；`module/reid/encoders.py` 的 `ReID`／`Custom_ReID` 主要為「同類別內辨識身分」設計，沒有驗證過跨類別語意分離能力，拿來做跨類別離群判斷風險較高，且會引入額外的推論管線與依賴。已考慮的替代方案：ReID embedding 聚類（A2）——使用者已選擇不採用。~~

**推翻原因**：Task 1 完成後對全部類別實跑，發現 `person` 類別的訊號嚴重退化（1082 張裁切中 849 張／78% 偵測器完全輸出不了任何框，agreement_rate 91% 集中在 0.0）。根因是偵測器（`Detector`）設計上要在一張完整場景圖裡「定位＋分類」多個物體，套用在已經裁切好、脫離場景脈絡的單一小圖上先天不合——這跟 small_truck 當初的「語意邊界模糊」問題性質不同（small_truck 是偵測得到框但分類邊界模糊，person 是連框都生不出來），且 person 沒有像 car/small_truck/truck 那樣可以歸群的相近類別，`VEHICLE_GROUP` 放寬對 person 無效。使用者判斷應改用不需要「定位」、只需要對整張已裁切小圖抽特徵的方法，經討論後選擇 embedding 方案。

**1.（修訂版）用 `encoders.py::ReID` 包裝既有偵測權重 `models/yolo/yolo11x_dennis.pt` 當通用特徵抽取器，取最近類別中心點決定 `predicted_class`，取代重跑 `Detector`。**
Rationale：
- 排除的替代權重：`models/yolo-reid/`（YOLO-as-encoder ReID）與 `models/class_reid_weights/`（`Custom_ReID`）底下的權重都是**逐類別各自訓練**的（`module/reid/HANDOFF.md`：「一次只練一個類別的模型...要練多個類別就跑多次、換 `--class-name` 和 `--output`」；`models/class_reid_weights/` 目錄下也只有 `person`／`scooter` 兩個類別有權重）。不同類別的模型輸出向量不在同一個特徵空間，物理上不能拿來跨類別比距離，直接排除。
- 選擇 `yolo11x_dennis.pt`（本專案本來就在用的偵測權重，`config/botsort_noreid.yaml` 的 `det_model_path`）當 encoder 輸入：這是唯一橫跨全部 6 類別訓練過的單一模型，backbone 為了做偵測分類，本來就要學會辨別這 6 類別的視覺差異，其中間層特徵理論上天生具備跨類別分離能力，比任何逐類別微調的 metric-learning 模型更適合做「跨類別」比較。
- 介面銜接：`ReID.__call__(img, dets)` 設計上吃「完整畫面＋bbox」（內部呼叫 `save_one_box` 裁切），但這裡只有已經存檔的裁切小圖、沒有原始畫面。做法是把整張裁切圖當成 `img`，`dets` 給一個涵蓋全圖的 bbox（xywh 格式：`[W/2, H/2, W, H]`）——已確認 `save_one_box` 預設的 `gain=1.02, pad=10` 會被裁切邊界限制夾回原圖範圍，等於把整張裁切圖原封不動傳回去，複用既有、已在 tracking pipeline 裡用過的程式路徑，不修改 `encoders.py` 本身。
- 演算法：對每個非 DSEC 身分的每張裁切抽 embedding → 每個類別的「中心點」＝該類別全部身分的平均 embedding（先算每個身分的平均、再對身分平均取平均，避免張數多的身分主導）→ 每張裁切改成「離哪個類別中心最近」當作 `predicted_class`（取代原本的偵測器判斷）。`agree`／身分聚合／DSEC 排除／CSV schema 全部不變，`VEHICLE_GROUP` 特例整個移除（中心點距離判斷不會有偵測器「完全看不到框」那種問題，不需要手動放寬車輛群組）。
- 已考慮的替代方案：維持偵測器＋對小裁切先放大再推論（方案 B/C）——改動幅度最小，但只能處理「裁切太小」這個根因，car/small_truck/truck 那種「裁切夠大依然誤判」的語意邊界問題不會被放大解決；使用者選擇直接換成 embedding 方案（方案 A），一次處理兩種根因。

**2. DSEC 來源身分用資料夾名稱前綴 `dsecdet_` 識別並直接排除評分，而非嘗試建立替代信心指標。**
Rationale：`reid-training-data-pipeline` 的 pilot 已證實本專案偵測器在 DSEC 視角對 `bus` 系統性誤判，任何用同一顆偵測器對 DSEC 裁切算出的信心分數都不可信；與其產生一個看似量化、實際上不可信的數字，不如明確標示「無自動分數，純人工審查」，避免人工複核時被誤導性數字帶偏。

**3. 全量 contact sheet 而非抽樣報告，並用一致性分數排序而非隨機。**
Rationale：現有 `generate_quality_report.py` 的隨機抽 6 個身分設計只適合抓「整體品質觀感」，無法保證每個身分都被審視過；用一致性分數排序可以把人工複核的心力優先放在最可疑的身分上，同時仍保留「看得到全部身分」的機制。Trade-off：報告與縮圖數量顯著增加（car／scooter 各 300+ 身分），需要分頁輸出避免單一報告過大難以開啟。

**4. 搬移到隔離目錄而非直接刪除。**
Rationale：延續 CLAUDE.md「涉及刪除資料...一律先確認」與專案既有 Tier 2「只標記不刪除」的設計慣性，讓誤判的移除決策可回復；目前資料量（README 記錄約 163MB／38000+ 張）不大，直接刪除節省磁碟空間的急迫性不足以抵銷不可逆的風險。

**5. 全量審查報告用本機靜態 HTML，不用 Markdown、也不發佈成 Artifact。**
Rationale：300+ 身分（car／scooter）嵌在單一 Markdown 檔裡瀏覽體驗差；本機 HTML 可做 grid 排版＋依 agreement_rate 排序/篩選，體驗好很多。不採用發佈成 Artifact：這批 crop 是 CCTV 路口監控裁切，發佈到外部託管服務等同把可能含可識別資訊（人、車牌）的監控畫面外流，即使後續能刪除，快取/索引風險仍在；本機 HTML 開在瀏覽器裡就能達到同樣的排版與互動效果，沒有理由承擔這個風險。已與使用者確認選擇本機 HTML。

**6. 審查報告內建勾選＋前端 JS 匯出 CSV，不另外手動填清單。**
Rationale：報告是本機檔案而非透過 Artifact 發佈，不受 sandbox 下載限制，可以用純前端 JS（`Blob` + `<a download>`）在頁面上勾選「整身分移除」／「單張移除」後直接匯出符合 `quarantine_flagged.py` schema 的 CSV。比起「看報告→另開檔案手動填身分名稱」，可以避免人工抄錯身分資料夾名稱（尤其中文路口影片檔名很長，如 `C000043 民族一路 九如一路_20250514071352_clip60s_track115`）。不做跨頁面狀態同步（每頁各自勾選、各自匯出，多頁需人工合併 CSV），避免引入前端框架或本機儲存機制、維持純靜態 HTML/JS 的簡單性。已與使用者確認選擇此方案。

## Risks / Trade-offs

- [風險，已發生並促成 Decision 1 修訂] `Detector` 原本針對 full-frame 推論設計，套用在已裁切、可能佔滿畫面的 crop 上，anchor／scale 假設可能不成立，導致大量偽陽性或偽陰性一致性分數 → 對 small_truck 用 `VEHICLE_GROUP` 放寬緩解，但對 person（無相近類別可歸群、78% 完全無偵測）緩解無效 → Mitigation（修訂後）：改用 `ReID` 包裝既有偵測權重抽 embedding、取最近類別中心點，不再依賴偵測器的定位能力；同樣先在小規模類別（如剛失敗的 person）驗證訊號是否真的不再退化，不假設一定成功再套用到全量。
- [風險，embedding 方案本身] `ReID.__call__` 唯一驗證過的用途是「同一段影片內的同身分追蹤比對」，「不同類別之間視覺上是否分得開」從未實測過；且 `ReID.__call__` 內部依賴 ultralytics `embed=` 的隱性 predictor 狀態（`__init__` 用一次空 forward 設定，後續 `__call__` 不再顯式傳 `embed=`），行為依賴既有、已在生產環境用過的程式路徑，但仍是間接證據非直接證明 → Mitigation：小規模驗證時同時記錄「新分佈是否不再退化」與「目視核對排序最低的幾個身分是否真的可疑」兩項證據，兩者都要有才算訊號可用，不能只看分佈形狀。
- [風險] 全量掃描 38000+ 張 crop 逐張跑一次 YOLO 推論屬耗時批次作業 → Mitigation：腳本比照既有 Tier 2／grid search 腳本慣例支援 `--classes`／範圍縮小旗標，允許分批執行。
- [風險] 人工審查 300+ 身分的 contact sheet 仍是體力活，排序只能減少無效瀏覽、不保證覆蓋完整 → Mitigation：報告產生時明確統計「已產生縮圖／略過」身分數，避免遺漏未被納入報告的身分。
- [風險] 隔離操作的 manifest 若遺失，後續無法回溯還原 → Mitigation：manifest 寫入隔離目錄本身（與被搬移的檔案放在一起），不寫到臨時或容易被清除的路徑。

## Migration Plan

1. 先重跑既有 `filter_crop_dataset.py` + `generate_quality_report.py`，取得涵蓋 DSEC 併入後全量資料的 Tier 2 基準線（純執行既有工具，非本次新增行為）。
2. 開發一致性檢查工具，先對資料量最小的類別做小規模驗證，確認分數與人工判斷相關。
3. 擴充報告產生器，加入全量、排序、DSEC 獨立區塊。
4. 人工依報告審查，產出確認清單。
5. 隔離腳本先以 dry-run 核對清單，確認無誤後才啟用實際搬移。
6. 搬移後更新 `dataset/reid/README.md` 的「目前資料量」統計。

Rollback：整個流程中唯一會異動 `crop_dataset/` 的操作是隔離腳本的啟用搬移；回滾方式是依該次操作寫入的 manifest，把隔離目錄底下的檔案搬回 manifest 記錄的原路徑。

## Component Interfaces（延續 Decision 5/6 的實作細節）

```
check_class_consistency.py          generate_identity_review.py         quarantine_flagged.py
─────────────────────────           ─────────────────────────           ─────────────────────
逐身分/逐張抽 ReID embedding         讀 check_class_consistency 的CSV      讀人工匯出的CSV
（ReID 包裝 yolo11x_dennis.pt）      + crop_dataset 縮圖                  --dry-run（預設）/ --apply
算各類別中心點，取最近中心點           產出「本機 HTML」報告：                身分或單張兩種粒度
當 predicted_class                  - 依 agreement_rate 由低到高排序      搬到 crop_dataset_quarantine/
略過 dsecdet_ 前綴身分                - DSEC 身分獨立區塊（無分數）          寫入 manifest.json（時間、
輸出 CSV（schema 不變）：             - 每個身分縮圖旁勾選框                 來源清單、每筆原路徑→目的路徑）
class,identity,image,                - 頁面「產生確認清單」按鈕
predicted_class,confidence,agree      （前端 JS Blob+download，
                                       匯出符合 quarantine_flagged.py
                                       schema 的 CSV）
```

（Task 2／Task 3 不受這次 Decision 1 修訂影響——CSV schema、報告格式、隔離機制完全不變，只有 Task 1 內部「怎麼決定 predicted_class」的機制從重跑偵測器換成 embedding 最近中心點。）

- `check_class_consistency.py`：CLI 比照現有 Tier2 慣例，支援 `--classes`、`--crop-root`、`--out`；DSEC 身分在輸出裡標 `skipped_reason=dsec_source`，不進 agreement 統計、也不參與各類別中心點的計算（避免用不可信的資料污染中心點）。
- `generate_identity_review.py`（新檔，不擴充現有 `generate_quality_report.py`——兩者輸入 schema、輸出格式（Markdown vs 互動 HTML）都不同，混在同一支腳本會違反單一職責，比照專案既有「一支腳本一個關注點」慣例分開）：每類別一個 HTML 檔（或依 `--page-size` 分頁），每個身分卡片含縮圖、身分名稱、agreement_rate（DSEC 區塊不顯示）、勾選框；「產生確認清單」按鈕把目前頁面勾選狀態序列化成 CSV 下載。
- `quarantine_flagged.py`：CLI 讀確認清單 CSV，`identity`+`image` 皆有值→單張；`image` 空白→整身分；單張搬移後若身分資料夾清空，一併移除空資料夾（對應 `reid/crop-quarantine` spec 的「不留空資料夾」要求）。

## Open Questions

- **（已過期，Decision 1 修訂後不再適用）** 原本記錄的「`agree` 判定改採 vehicle-group 放寬（`car`/`small_truck`/`truck` 三者互判為一致）」——這是重跑偵測器方案下針對 small_truck 的緩解措施，改用 embedding＋最近中心點後，`VEHICLE_GROUP` 特例已整個移除（不需要手動放寬，中心點距離判斷天然涵蓋所有類別，不會有偵測器「完全看不到框」的問題）。保留這段記錄供追溯：small_truck 小規模驗證中，全 class 最大裁切（131×255px，目視明確是小貨卡）仍被偵測器判成 `car`（conf 0.55），嚴格比對會讓 91% 身分 agreement_rate 為 0.0；後續發現 `person` 類別（無相近類別可歸群）用同一套偵測器方案也退化（78% 完全無偵測），兩者共同促成改用 embedding 方案。
- **新的待驗證項目**：embedding＋最近中心點方案尚未做過任何小規模驗證（即將對 `person` 執行，比照 Task 1 原本的 small_truck 驗證方法論：檢視新的 agreement_rate 分佈是否不再退化、目視核對排序最低的幾個身分是否真的可疑）。驗證結果與最終採用的門檻（若有）待驗證完成後補充於此。
