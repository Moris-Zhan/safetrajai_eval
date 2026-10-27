# track-config-loading Specification

## Purpose

定義追蹤設定（`class_tracking`）的載入契約：接受哪些格式、哪些內容必須被拒絕、載入後全域參數如何更新，以及查表函式在缺值時的退回規則，讓追蹤器與偵測器在不同情境設定之間切換時行為可預期。

## Requirements

### Requirement: 設定內容缺漏或型別錯誤時載入失敗
載入設定時，根節點 MUST 是 mapping，且 `class_names`、`class_tracking_params`、`class_conf_thresh` MUST 是非空 mapping，`default_tracking_params` MUST 是 mapping。任一項不符時 MUST 拋出 `ValueError`，訊息指出缺少或錯誤的區塊。

#### Scenario: 缺少 class_names
- **WHEN** 載入的 YAML 沒有 `class_names`，或它是空 mapping
- **THEN** 拋出 `ValueError`，訊息包含 `class_names`

#### Scenario: 缺少 class_tracking_params 或 class_conf_thresh
- **WHEN** 載入的 YAML 缺少非空的 `class_tracking_params` 或 `class_conf_thresh`
- **THEN** 拋出 `ValueError`，訊息包含缺少的區塊名稱

#### Scenario: default_tracking_params 不是 mapping
- **WHEN** `default_tracking_params` 是清單或純量
- **THEN** 拋出 `ValueError`，訊息包含 `default_tracking_params`

#### Scenario: 根節點不是 mapping
- **WHEN** YAML 根節點是清單
- **THEN** 拋出 `ValueError`，訊息指出根節點必須為 mapping

### Requirement: 接受章節式與扁平式兩種格式
載入器 SHALL 在 YAML 含 `class_tracking:` 章節時讀取該章節，沒有該章節時把整份根節點當成設定內容，兩種格式載入後的全域參數相同。

#### Scenario: 章節式與扁平式等價
- **WHEN** 同一份內容分別以 `class_tracking:` 章節和扁平結構寫成兩個 YAML 並載入
- **THEN** `CLASS_NAMES`、`CLASS_TRACKING_PARAMS`、`DEFAULT_TRACKING_PARAMS`、`CLASS_CONF_THRESH` 的內容相同

### Requirement: 已知的驗證失敗不改變已載入的全域參數
類別名稱鍵無法轉成整數、融合參數無效、或有效 `track_buffer` 不是正整數時，載入 MUST 在修改任何全域參數之前失敗，失敗後 `CLASS_NAMES`、`CLASS_TRACKING_PARAMS`、`DEFAULT_TRACKING_PARAMS`、`CLASS_CONF_THRESH` 與載入路徑 MUST 與失敗前相同。

#### Scenario: class_names 的鍵不是整數
- **WHEN** 載入的 YAML 的 `class_names` 含有無法轉成整數的鍵
- **THEN** 拋出 `ValueError`，且全域參數與失敗前相同

#### Scenario: track_buffer 無效
- **WHEN** 某類別的有效 `track_buffer` 缺少、是布林或浮點數、或小於 1
- **THEN** 拋出 `ValueError`，訊息包含類別名稱，且全域參數與失敗前相同

### Requirement: 成功載入後以原地更新取代全域參數
成功載入 SHALL 以原地清除再填入的方式更新全域 dict，使先前 `import` 取得的參考仍指向最新內容；`class_conf_thresh` 的值 MUST 轉成 `float`，`default_conf_thresh` 未設定時 MUST 為 0.15。

#### Scenario: 既有參考看到新內容
- **WHEN** 先取得 `CLASS_TRACKING_PARAMS` 的參考，再載入另一份設定
- **THEN** 該參考與模組內的 `CLASS_TRACKING_PARAMS` 是同一個物件，且內容為新設定

#### Scenario: 信心門檻轉型與預設值
- **WHEN** 載入的 YAML 的 `class_conf_thresh` 以字串數字表示，且沒有 `default_conf_thresh`
- **THEN** `CLASS_CONF_THRESH` 的值是 `float`，`DEFAULT_CONF_THRESH` 為 0.15

### Requirement: 設定檔路徑解析
`load_track_config` SHALL 接受絕對路徑或相對於目前工作目錄的路徑，回傳實際載入檔案的絕對路徑，並讓 `get_track_config_path()` 回傳同一路徑。路徑為 `None` 或空字串時 MUST 載入預設的 `config/bytetrack.yaml`；找不到檔案時 MUST 拋出 `FileNotFoundError`。

#### Scenario: 相對路徑
- **WHEN** 工作目錄下有 `track.yaml`，以 `"track.yaml"` 載入
- **THEN** 回傳該檔的絕對路徑，`get_track_config_path()` 回傳相同路徑

#### Scenario: 使用預設設定
- **WHEN** 以 `None` 載入
- **THEN** 載入的是 repo 內的 `config/bytetrack.yaml`

#### Scenario: 找不到檔案
- **WHEN** 載入一個不存在的路徑
- **THEN** 拋出 `FileNotFoundError`，且全域參數不變

### Requirement: 查表函式在缺值時退回預設
`get_params_for_class` SHALL 接受類別 ID 或類別名稱，類別沒有專屬設定時回傳 `DEFAULT_TRACKING_PARAMS`；`get_class_name` 對未知 ID MUST 回傳 `class_<id>`；`get_conf_thresh_for_class` 對沒有專屬門檻的類別 MUST 回傳預設信心門檻；`get_min_conf_thresh` MUST 回傳所有類別門檻的最小值。

#### Scenario: 以 ID 和名稱查同一類別
- **WHEN** 以類別 ID 與類別名稱查詢同一個有專屬設定的類別
- **THEN** 兩者回傳相同的參數

#### Scenario: 未知類別退回預設
- **WHEN** 查詢不在設定中的類別 ID
- **THEN** `get_params_for_class` 回傳 `DEFAULT_TRACKING_PARAMS`，`get_class_name` 回傳 `class_<id>`，`get_conf_thresh_for_class` 回傳預設信心門檻

### Requirement: 命令列只取出 --track-config
`reload_track_config_from_cli` SHALL 只解析 `--track-config`，忽略其他參數並載入指定設定；沒有該參數時 MUST 載入預設設定。

#### Scenario: 混有其他參數
- **WHEN** 傳入 `["--config", "x.yaml", "--track-config", "y.yaml", "--foo"]`
- **THEN** 載入的是 `y.yaml`，不因其他參數報錯

### Requirement: BoTSORT 在建立時固定類別集合
`create_tracker` 建立 BoTSORT 後，之後再呼叫 `load_track_config` MUST NOT 改變該 tracker 已記錄的類別集合；要使用新設定 MUST 重新建立 tracker。

#### Scenario: 建立後才載入另一份設定
- **WHEN** 以一份設定建立 BoTSORT tracker，再載入類別集合不同的另一份設定
- **THEN** 該 tracker 的 `CLASS_KEYS` 仍是建立當下的類別集合
