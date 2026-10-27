# Spec Delta

## MODIFIED Requirements

### Requirement: 已知的驗證失敗不改變已載入的全域參數
載入設定時，所有驗證與型別轉換 MUST 在修改任何全域參數之前完成：類別名稱鍵無法轉成整數、`class_tracking_params` 的任何項目（包含不在 `class_names` 的類別）不是 mapping、融合參數無效、有效 `track_buffer` 不是正整數、`class_conf_thresh` 或 `default_conf_thresh` 的值無法轉成數字時，載入 MUST 拋出 `ValueError`，失敗後 `CLASS_NAMES`、`CLASS_TRACKING_PARAMS`、`DEFAULT_TRACKING_PARAMS`、`CLASS_CONF_THRESH`、預設信心門檻與載入路徑 MUST 與失敗前相同。

#### Scenario: class_names 的鍵不是整數
- **WHEN** 載入的 YAML 的 `class_names` 含有無法轉成整數的鍵
- **THEN** 拋出 `ValueError`，且全域參數與失敗前相同

#### Scenario: track_buffer 無效
- **WHEN** 某類別的有效 `track_buffer` 缺少、是布林或浮點數、或小於 1
- **THEN** 拋出 `ValueError`，訊息包含類別名稱，且全域參數與失敗前相同

#### Scenario: 不在 class_names 的類別參數不是 mapping
- **WHEN** `class_tracking_params` 含有不在 `class_names` 的類別，且其值不是 mapping
- **THEN** 拋出 `ValueError`，訊息包含該類別名稱，且全域參數與失敗前相同

#### Scenario: 信心門檻不是數字
- **WHEN** `class_conf_thresh` 的某個值（或 `default_conf_thresh`）無法轉成數字，例如 `"not a number"` 或 `null`
- **THEN** 拋出 `ValueError`，訊息指出是哪個設定，且全域參數與失敗前相同
