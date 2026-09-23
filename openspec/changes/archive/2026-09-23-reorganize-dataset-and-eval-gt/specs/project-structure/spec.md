# Spec Delta

## MODIFIED Requirements

### Requirement: 既有 Re-ID 完整模型權重可載入

以完整模型 pickle 儲存的 Re-ID 權重 SHALL 在不修改權重檔的前提下維持可載入，無論它位於 `models/class_reid_weights/*/best_full_model.pt`，還是已被隔離到 `models/class_reid_weights/temp/` 之下。模組路徑 `utils.model` MUST 持續可被解析。權重被隔離後，情境 config MUST NOT 再引用它；可載入性只用來比對與還原。

#### Scenario: 載入既有權重
- **WHEN** 以 `grid_search_botsort_reid.py` 的候選清單或自訂 config 指定一個仍位於 `models/class_reid_weights/<class>/` 的完整模型權重，並以 BoTSORT 開啟該類別的 ReID 執行追蹤
- **THEN** 權重成功載入並產生特徵；若該權重已被隔離而路徑不存在，流程以找不到檔案的錯誤停止

#### Scenario: 載入被隔離的權重
- **WHEN** 以 `torch.load(..., weights_only=False)` 載入 `models/class_reid_weights/temp/` 下的完整模型權重
- **THEN** 成功還原出模型物件，不出現找不到 `utils.model` 類別的錯誤

#### Scenario: shim 保留
- **WHEN** 執行 smoke 測試
- **THEN** `utils.model` 仍能被 import，並 re-export 權重反序列化所需的類別

### Requirement: 重構前後輸出等價

結構重構 MUST NOT 改變任何 stage 的輸出。對固定的黃金比對組合，重構後輸出與重構前輸出的差異 SHALL 不超過「以重構前程式碼連跑兩次」所量測到的自比差異。

例外：若 change 刻意改變評估語意（例如更換評估 GT 或範圍限縮方式），或因權重被隔離而更換情境所用的權重，受影響的 stage SHALL 重建黃金基準。該 change MUST 記錄：
- 新舊基準差異的來源。
- 新基準以新程式碼連跑兩次的自比差異。
- 未受影響的 stage 仍與舊基準等價。

#### Scenario: stage1 軌跡等價
- **WHEN** 以 `bytetrack`、`botsort_noreid`、`botsort_reid` 三種情境對 `cctv_all8` 全部序列跑 stage1
- **THEN** 6 個類別的 MOT 軌跡檔（`frame,track_id,left,top,w,h,score`）與重構前等價

#### Scenario: stage2 指標等價
- **WHEN** 對上述輸出跑 stage2
- **THEN** TrackEval 的 HOTA、AssA、IDSW 等指標與重構前等價

#### Scenario: stage3 事件等價
- **WHEN** 以 `collision_detect` 情境跑 stage1＋stage3
- **THEN** `ttc_events`、`pet_events` 的內容與重構前等價

#### Scenario: Re-ID 裁切蒐集等價
- **WHEN** 以 `botsort_save_reid_crops` 情境跑 stage1
- **THEN** 產生的裁切檔案路徑清單與數量與重構前相同

#### Scenario: grid search 與 homography 驗證等價
- **WHEN** 以單一參數組合、單一影片跑 grid search，以及執行 homography 驗證
- **THEN** 結果 CSV 與驗證輸出與重構前等價

#### Scenario: 刻意變更評估語意時重建基準
- **WHEN** change 更換評估 GT 或範圍限縮方式
- **THEN** 受影響 stage 的新基準連跑兩次自比差異為 0，change 文件列出新舊差異的來源，未受影響 stage 的輸出仍與舊基準等價
