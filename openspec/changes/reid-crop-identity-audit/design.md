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
- 不採用 ReID embedding 做跨類別聚類（即先前討論的 A2 方案）——已與使用者確認排除。

## Decisions

**1. 用重跑既有 YOLO `Detector` 做一致性檢查，而非 ReID embedding。**
Rationale：直接複用專案既有依賴與模型，邏輯是單純的自我一致性檢查（同一顆偵測器、同一批 CCTV 視角資料）；`module/reid/encoders.py` 的 `ReID`／`Custom_ReID` 主要為「同類別內辨識身分」設計，沒有驗證過跨類別語意分離能力，拿來做跨類別離群判斷風險較高，且會引入額外的推論管線與依賴。已考慮的替代方案：ReID embedding 聚類（A2）——使用者已選擇不採用。

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

- [風險] `Detector` 原本針對 full-frame 推論設計，套用在已裁切、可能佔滿畫面的 crop 上，anchor／scale 假設可能不成立，導致大量偽陽性或偽陰性一致性分數 → Mitigation：分數僅作排序線索、不自動決定刪除；tasks 中安排一次小規模抽樣驗證（如單一資料量最小的類別）確認分數與人工判斷的相關性，再套用到全量資料。
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
逐身分/逐張重跑 Detector             讀 check_class_consistency 的CSV      讀人工匯出的CSV
比對 predicted vs assigned           + crop_dataset 縮圖                  --dry-run（預設）/ --apply
略過 dsecdet_ 前綴身分                產出「本機 HTML」報告：                身分或單張兩種粒度
輸出 CSV：                            - 依 agreement_rate 由低到高排序      搬到 crop_dataset_quarantine/
class,identity,image,                - DSEC 身分獨立區塊（無分數）          寫入 manifest.json（時間、
predicted_class,confidence,agree      - 每個身分縮圖旁勾選框                 來源清單、每筆原路徑→目的路徑）
                                       - 頁面「產生確認清單」按鈕
                                       （前端 JS Blob+download，
                                       匯出符合 quarantine_flagged.py
                                       schema 的 CSV）
```

- `check_class_consistency.py`：CLI 比照現有 Tier2 慣例，支援 `--classes`、`--crop-root`、`--out`；DSEC 身分在輸出裡標 `skipped_reason=dsec_source`，不進 agreement 統計。
- `generate_identity_review.py`（新檔，不擴充現有 `generate_quality_report.py`——兩者輸入 schema、輸出格式（Markdown vs 互動 HTML）都不同，混在同一支腳本會違反單一職責，比照專案既有「一支腳本一個關注點」慣例分開）：每類別一個 HTML 檔（或依 `--page-size` 分頁），每個身分卡片含縮圖、身分名稱、agreement_rate（DSEC 區塊不顯示）、勾選框；「產生確認清單」按鈕把目前頁面勾選狀態序列化成 CSV 下載。
- `quarantine_flagged.py`：CLI 讀確認清單 CSV，`identity`+`image` 皆有值→單張；`image` 空白→整身分；單張搬移後若身分資料夾清空，一併移除空資料夾（對應 `reid/crop-quarantine` spec 的「不留空資料夾」要求）。

## Open Questions

- `agree` 判定改採「vehicle-group 放寬」：`car`/`small_truck`/`truck` 三者互判為一致（`VEHICLE_GROUP`），只有跨到 `bus`/`person`/`scooter` 或完全無偵測才算不一致。原因是 car/small_truck/truck 邊界對本專案偵測器而言是能力上限，不是程式錯誤——small_truck 小規模驗證中，全 class 最大裁切（131×255px，目視明確是小貨卡）仍被判成 `car`（conf 0.55），在放寬前導致 91% 身分 agreement_rate 為 0.0、訊號完全無鑑別力；放寬後 small_truck 的 183 個身分分布為 72 個 0.0、57 個 1.0、54 個落在中間（mean 0.45、median 0.33），才變成可用的排序訊號。已知限制：這個訊號**抓不到** car/small_truck/truck 彼此之間的誤判（例如身分整個歸錯在 car 和 small_truck 之間），這類錯誤仍要靠 Task 2 的全量人工 contact-sheet 審查抓；也抓不到「裁切太小完全無偵測」拖低分數但實際上沒有跨類別混入的情況（例如 395 張裁切裡 212 張因解析度太低而無偵測、但有偵測到的全部正確判為同組，agreement_rate 仍只有 0.46）——這種情況的低分不代表身分歸錯類別。
