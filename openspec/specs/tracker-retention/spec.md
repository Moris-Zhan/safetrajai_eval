# tracker-retention Specification

## Purpose

定義 class-aware BoTSORT 與 ByteTrack 共用的保留期限語意：`track_buffer` 在兩個追蹤器中都是保留期限幀數，並規定載入追蹤設定時必須為每個類別給出有效的保留期限。

## Requirements

### Requirement: track_buffer 在兩個追蹤器中都是保留期限幀數

逐類別設定 `track_buffer` SHALL 在 class-aware BoTSORT 與 class-aware ByteTrack 中都表示保留期限幀數：遺失中的軌跡在「目前幀號減去最後一次更新的幀號」大於 `track_buffer` 時 SHALL 被移除，否則仍可被重新關聯。換算 MUST NOT 依追蹤器種類或影片幀率而不同。

#### Scenario: 兩個追蹤器的保留期限相同
- **WHEN** 同一個類別設定 `track_buffer: 15`，分別以 BoTSORT 與 ByteTrack 在任意幀率的影片上追蹤
- **THEN** 兩者都在軌跡遺失超過 15 幀後才移除它；遺失第 15 幀時仍可重新關聯

#### Scenario: 保留期限不隨影片幀率變化
- **WHEN** `track_buffer: 30`，分別用於 5fps 與 10fps 的影片
- **THEN** 兩者的保留期限都是 30 幀

### Requirement: 每個類別都必須設定有效的 track_buffer

載入追蹤設定時，`class_names` 中的每個類別 SHALL 解析到一個正整數 `track_buffer`（依類別設定，沒有該類別設定時依預設追蹤參數）。缺少、為 0 或負數、不是整數（包含布林值與帶小數的數值）時，SHALL 在載入追蹤設定時以錯誤停止，錯誤訊息指出類別名稱與 `track_buffer`，並 MUST NOT 修改已載入的追蹤參數。

#### Scenario: 缺少 track_buffer
- **WHEN** 某類別的設定沒有 `track_buffer`，預設追蹤參數也沒有
- **THEN** 載入追蹤設定時以錯誤停止，訊息包含該類別名稱與 `track_buffer`

#### Scenario: 非整數的 track_buffer
- **WHEN** 某類別設定 `track_buffer: 2.5`
- **THEN** 載入追蹤設定時以錯誤停止，已載入的追蹤參數不變

#### Scenario: 所有情境設定都通過檢查
- **WHEN** 載入 `config/` 與 golden 比對使用的每一個追蹤情境設定
- **THEN** 每一份都載入成功，且每個類別都有正整數 `track_buffer`

### Requirement: 單位統一不改變既有情境的追蹤結果

情境設定的 `track_buffer` 數值 SHALL 依該設定的 `frame_rate`（固定值，或 `auto` 時該設定實際執行序列的幀率）換算成與修改前相同的保留期限幀數。對 golden 比對的固定組合，修改後的 stage1／stage2／stage3 輸出 SHALL 與修改前的基準差異為 0。

#### Scenario: golden 比對不變
- **WHEN** 以修改後的程式與換算後的 golden 設定跑完整的 golden 比對
- **THEN** 與基準 `reid1` 比對，所有項目差異為 0；唯一例外是 grid search 的 `best_params_<class>.txt` 因設定顯式寫出 `track_buffer` 而多列出該參數，分數與所有數值仍相同

#### Scenario: 固定幀率的 BoTSORT 設定
- **WHEN** 修改前某 BoTSORT 設定 `frame_rate: 30`、某類別 `track_buffer: 3.0`（秒）
- **THEN** 修改後該類別為 `track_buffer: 90`，保留期限同為 90 幀
