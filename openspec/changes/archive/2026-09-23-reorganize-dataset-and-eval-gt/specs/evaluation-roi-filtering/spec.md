# Spec Delta

## Purpose

定義以鏡頭為鍵的關注區域（ROI），以及評估時用忽略區域（ignore region）限縮範圍的語意，讓評估範圍可以調整，但不必修改原始 GT，也不必修改 vendored TrackEval。

## ADDED Requirements

### Requirement: ROI 以鏡頭為鍵定義

每支鏡頭的 ROI SHALL 定義在 `config/roi/<camera>.yaml`，內容是影像像素座標的多邊形頂點，並記錄定義時所依據的影像解析度。序列的 ROI MUST 由序列名稱開頭的鏡頭編號決定，不由情境 config 決定。
- 鏡頭沒有 ROI 檔時，流程 SHALL 以整張影像為 ROI，並在 log 中明確提示該鏡頭沒有 ROI。
- 情境 config 若仍帶有頂層 `det_corner`，載入時 MUST 以錯誤停止，並提示改用 `config/roi/<camera>.yaml`。
- 判斷物件是否位於 ROI 內，一律以 bbox 中心點為準，落在邊界上視為在內。

#### Scenario: 依序列鏡頭取得 ROI
- **WHEN** 處理序列 `C000025 中華三路 民生二路(東北角)_20251209145630_15s_1`
- **THEN** 使用 `config/roi/C000025.yaml` 的多邊形

#### Scenario: 鏡頭沒有 ROI
- **WHEN** 處理的序列所屬鏡頭沒有 ROI 檔
- **THEN** 以整張影像為 ROI，並輸出一行提示訊息

#### Scenario: 舊 config 帶有 det_corner
- **WHEN** 載入含有頂層 `det_corner` 的 config
- **THEN** 載入失敗，錯誤訊息指出應改用 `config/roi/<camera>.yaml`

### Requirement: 追蹤評估以忽略區域限縮範圍

stage2 的追蹤評估 SHALL 在送進 TrackEval 之前，對每一幀、每個類別以 UA-DETRAC 式語意執行 ignore 匹配：
1. 該類別 GT 中心點位於 ROI 外者標為 ignore。
2. 預測與該類別全部 GT（含 ignore）以 IoU ≥ 0.5 做一對一最佳匹配。
3. 匹配到 ROI 內（有效）GT 的預測一律保留，無論其中心點位於何處。
4. 其餘預測中，匹配到 ignore GT 者，或中心點位於 ROI 外者，一律移除，不計為 TP 或 FP。
5. 移除全部 ignore GT。

因此 FP 只會來自 ROI 內。TrackEval 本身 MUST NOT 被修改。

語意選擇與代價：本規格採 UA-DETRAC 式，而不是 MOT17 式（MOT17 式只移除匹配到 ignore GT 的預測，ROI 外未匹配的預測仍計為 FP）。以 golden 基準實測，MOT17 式會讓 ROI 外未標註、或定位偏差的遠處物件大量變成 FP（bytetrack 的 car FP 由 60 增為 551，HOTA 由 78.3 降為 69.3），指標下降來自評估方式而非模型。改用 UA-DETRAC 式的代價與風險：
- 模型在 ROI 外產生的誤報完全不受懲罰；ROI 以外的偵測品質不在本評估的量測範圍內。
- 中心點在 ROI 外、與 ROI 內 GT 的 IoU 未達 0.5 的預測會被移除，該 GT 因此計為 FN；這與舊的對稱硬過濾行為相同，不會比過去更差。
- 評估結果的可信度取決於 ROI 的畫法：ROI 畫得過小會同時排除真正的誤報，指標會偏樂觀。
- 與 MOT17 benchmark 的 ignore 定義不同，數值不可與使用 MOT17 式的外部結果直接比較（本專案為自建 CCTV 資料集，原本即無外部可比性）。
- 與舊流程（對稱硬過濾）相比，FP 的計算範圍一致，差異只在跨越 ROI 邊界的物件：舊流程會產生假的 FN，本規格會正確計為 TP。

#### Scenario: ROI 外的物件被正確偵測
- **WHEN** ROI 外有一個 GT，某預測與它的 IoU ≥ 0.5
- **THEN** 該 GT 與該預測都不影響 TP、FP、FN

#### Scenario: 物件跨越 ROI 邊界
- **WHEN** GT 中心點在 ROI 內、與它匹配的預測中心點在 ROI 外
- **THEN** 該預測保留並與 GT 配成 TP，不產生 FN 或 FP

#### Scenario: ROI 外的誤報
- **WHEN** 某預測不與任何 GT 匹配，而且中心點位於 ROI 外
- **THEN** 該預測被移除，不計為 FP

#### Scenario: ROI 內的誤報
- **WHEN** 某預測不與任何 GT 匹配，而且中心點位於 ROI 內
- **THEN** 該預測計為 FP

#### Scenario: 跨界預測未達 IoU 門檻
- **WHEN** GT 中心點在 ROI 內，某預測中心點在 ROI 外、與該 GT 的 IoU 小於 0.5
- **THEN** 該預測被移除，該 GT 計為 FN

#### Scenario: 預測優先匹配有效 GT
- **WHEN** 某預測同時與一個 ROI 內 GT 和一個 ignore GT 重疊
- **THEN** 依最佳匹配結果處理；若它配到 ROI 內的 GT，就計為 TP，不會被移除

### Requirement: 評估中間產物只寫入輸出目錄

ignore 處理後的 GT、預測，以及其他評估中間產物 SHALL 寫在 stage2 的輸出目錄。MUST NOT 寫回評估資料集的來源目錄或 stage1 的軌跡檔。

#### Scenario: 來源與 stage1 軌跡不被改寫
- **WHEN** 跑完 stage2
- **THEN** 來源 GT 與 stage1 的 `trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt` 內容都與 stage2 執行前相同

### Requirement: stage1 軌跡輸出不做 ROI 過濾

stage1 輸出的 7 欄 MOT 軌跡檔 SHALL 包含所有追蹤結果，不依 ROI 過濾。需要限縮範圍的下游 SHALL 自行套用序列所屬鏡頭的 ROI：
- stage3 以 bbox 中心點硬過濾。
- stage2 以前述忽略區域處理。
- Re-ID 裁切蒐集以 bbox 中心點硬過濾。

#### Scenario: stage3 衝突事件範圍不變
- **WHEN** 以 `collision_detect` 情境對 `cctv_all8` 跑 stage1 與 stage3
- **THEN** `ttc_events`、`pet_events` 與改版前的 golden 輸出相同

#### Scenario: Re-ID 裁切範圍不變
- **WHEN** 以 `botsort_save_reid_crops` 情境對 `cctv_all8` 跑 stage1
- **THEN** 產生的裁切檔案路徑清單與改版前的 golden 輸出相同

### Requirement: detection 評估使用人工 GT 並標示 ROI 狀態

stage2 的 detection 評估 SHALL 以評估資料集的人工 detection GT 為基準。MUST NOT 使用 stage1 產生的追蹤標籤充當 GT。GT 與影格的對應必須以幀號對齊。detection 評估目前不套用 ROI，輸出的報表與終端摘要 MUST 明確標示「未套用 ROI（全畫面）」。

#### Scenario: detection 評估不讀 stage1 標籤
- **WHEN** 跑 stage2 的 detection 評估
- **THEN** 評估所用的標籤內容與人工 GT 逐檔相同，不含 track_id 或 conf 欄位

#### Scenario: 報表標示 ROI 狀態
- **WHEN** 開啟 detection 評估的 Excel 報表
- **THEN** 報表中有欄位或註記標示 ROI 未套用
