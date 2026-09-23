# project-structure Specification

## Purpose

定義專案結構必須維持的不變式契約（invariants）：可安裝、入口相容、既有權重可載入、類別 ID 一致、重構不改變任何輸出，以及 tracker 核心邏輯不被動到，讓後續任何結構調整都有可驗證的回歸防線。

## Requirements

### Requirement: 可安裝套件且不依賴執行期路徑修補
專案 SHALL 能在 conda env `track_eval` 中以一次 editable install（不解析依賴）安裝，安裝後專案程式碼與 vendored TrackEval 皆可被 import；任何專案程式碼 MUST NOT 在執行期修改模組搜尋路徑（`sys.path`）。

#### Scenario: 安裝後可從任意工作目錄 import
- **WHEN** 在 `track_eval` env 完成 editable install，並於 repo 以外的目錄啟動 Python
- **THEN** 專案套件與 `trackeval` 皆可成功 import，且 `trackeval` 解析到 repo 內的 vendored 副本

#### Scenario: 安裝不變動既有依賴版本
- **WHEN** 執行專案文件記載的安裝指令
- **THEN** env 內 `torch` 與 `ultralytics` 的版本與安裝前相同

#### Scenario: 原始碼不含路徑修補
- **WHEN** 掃描專案自有的 Python 原始碼（不含 vendored TrackEval）
- **THEN** 找不到任何對 `sys.path` 的修改

### Requirement: 專案根目錄的解析只有單一來源
需要定位 repo 內檔案（例如 `config/`、`config/datasets.yaml`）的程式碼 SHALL 透過單一的根目錄定義取得路徑，結果 MUST NOT 依賴呼叫端的目錄深度。

#### Scenario: 預設設定檔路徑正確
- **WHEN** 未指定 `--config` 或 `--track-config` 而使用預設值
- **THEN** 讀到的是 repo 內 `config/bytetrack.yaml`，與重構前相同

#### Scenario: 資料集對照表路徑正確
- **WHEN** config 以 `paths.dataset: <id>` 引用資料集
- **THEN** 讀到的是 repo 內 `config/datasets.yaml`

### Requirement: 主入口指令相容
`python track_eval.py --config <path>` SHALL 在 repo 根目錄維持可用，參數與行為和重構前相同；若專案尚未安裝，MUST 以一行訊息提示安裝指令並以非零狀態結束，而不是拋出未處理的 traceback。

#### Scenario: 舊指令照常執行
- **WHEN** 已安裝的環境執行 `python track_eval.py --config config/bytetrack.yaml`
- **THEN** 依 config 的 `stages.*` 執行對應階段，輸出位置與重構前相同

#### Scenario: 未安裝時的提示
- **WHEN** 在未安裝專案的環境執行 `python track_eval.py`
- **THEN** 印出包含安裝指令的提示，並以非零狀態結束

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

### Requirement: 類別 ID 契約單一來源
類別 ID 對照（`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`）SHALL 只定義一次；給 MOT GT／TrackEval 使用的 1-based 類別 ID MUST 由該定義推導（`id + 1`），不得另行手寫。vendored TrackEval 內部的對照不在此限。

#### Scenario: MOT 類別 ID 推導
- **WHEN** 查詢任一類別給 TrackEval 使用的類別 ID
- **THEN** 值等於該類別的 0-based ID 加 1

#### Scenario: 設定檔載入的類別名稱與契約一致
- **WHEN** 載入 `config/` 下任一含 `class_tracking:` 的設定檔
- **THEN** 載入後的類別名稱對照與單一來源定義完全相同

#### Scenario: Re-ID 資料工具的類別順序不變
- **WHEN** Re-ID 資料稽核與切分工具遍歷類別
- **THEN** 遍歷順序為類別名稱的字母排序，與重構前相同

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

### Requirement: Tracker override 方法本體不變
`ClassAwareBOTSORT` 與 `ClassAwareBYTETracker` 中 override 自 ultralytics 的方法（`__init__`、`update`、`remove_duplicate_stracks`，以及 BoTSORT 的 `init_track`）的本體 SHALL 與重構前完全相同；搬移時只允許變更檔案層級的 import 敘述。

#### Scenario: 方法本體比對
- **WHEN** 以語法樹（AST）比對重構前後上述方法的本體
- **THEN** 兩者完全相同
