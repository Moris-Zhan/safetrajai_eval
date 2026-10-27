# Spec Delta

## ADDED Requirements

### Requirement: GT 來源目錄解析不得取到其他序列
`_resolve_gt_source_root` SHALL 優先回傳與序列名稱完全相同的目錄。完全相同的目錄不存在時，只在以序列名稱前綴比對得到恰好一個候選目錄時才回傳該目錄；沒有候選或有多個候選時 MUST 回傳 `None`，不得挑選其中之一。

#### Scenario: 優先使用完全相同的目錄
- **WHEN** 同時存在名稱前綴相同的多個序列目錄，且查詢的序列名稱有完全相同的目錄
- **THEN** 回傳該目錄

#### Scenario: 前綴有多個候選
- **WHEN** 沒有完全相同的目錄，且前綴比對到兩個以上的序列目錄
- **THEN** 回傳 `None`，該序列在 trial 中被跳過

#### Scenario: 沒有任何候選
- **WHEN** 沒有完全相同的目錄，也沒有前綴相符的目錄
- **THEN** 回傳 `None`

## RENAMED Requirements

- FROM: `### Requirement: resume 只跳過既有結果中出現過的組合`
- TO: `### Requirement: resume 只跳過已成功完成的組合`

## MODIFIED Requirements

### Requirement: 從 TrackEval 結果萃取 AssA 平均
`extract_assa_mean` SHALL 從 TrackEval 回傳結果取出指定類別的 `HOTA.AssA` 並回傳平均值，接受 `(results, messages)` tuple；序列層優先使用 `COMBINED_SEQ`；結果中沒有指定類別、任何層級缺值或結果不是 dict 時 MUST 回傳 `None`，不拋例外，也不得改用其他項目（例如跨類別的彙總）的分數。

#### Scenario: 陣列取平均
- **WHEN** 結果內的 `AssA` 是多個 alpha 門檻的 `ndarray`
- **THEN** 回傳其平均值

#### Scenario: 優先使用 COMBINED_SEQ
- **WHEN** 結果同時含 `COMBINED_SEQ` 與單一序列
- **THEN** 回傳的是 `COMBINED_SEQ` 的分數

#### Scenario: 缺值回傳 None
- **WHEN** 結果為 `None`、不是 dict，或缺少 `HOTA` 或 `AssA`
- **THEN** 回傳 `None`

#### Scenario: 指定類別不存在
- **WHEN** 序列層只有其他項目（例如跨類別的彙總 `cls_comb_cls_av`），沒有指定類別
- **THEN** 回傳 `None`，不使用彙總項的分數

### Requirement: trial 期間只修補目標類別並在結束後還原
`score_params` SHALL 在 trial 期間把目標類別的追蹤參數與本次參數合併，其他類別 MUST 不變；無論 trial 正常結束、因 GT 缺漏而跳過序列、或過程拋出例外，結束後目標類別的參數 MUST 還原成 trial 前的狀態——trial 前有內容就還原該內容，trial 前不存在該類別的鍵就 MUST 移除該鍵。

#### Scenario: trial 期間看到修補後的參數
- **WHEN** trial 執行中查詢目標類別的追蹤參數
- **THEN** 內容為原參數與本次參數合併後的結果，且保留本次未指定的欄位（例如 `use_reid`）

#### Scenario: 正常結束後還原
- **WHEN** trial 完成並回傳分數
- **THEN** 目標類別的參數與 trial 前相等，其他類別沒有變動

#### Scenario: 例外後還原
- **WHEN** trial 過程中拋出例外
- **THEN** 目標類別的參數與 trial 前相等，函式回傳 `-inf` 而不是把例外往外丟

#### Scenario: GT 缺漏時跳過
- **WHEN** 某序列找不到 GT
- **THEN** 該序列被跳過，目標類別的參數仍被還原

#### Scenario: 原本沒有該類別的設定
- **WHEN** trial 前 `CLASS_TRACKING_PARAMS` 沒有目標類別的鍵
- **THEN** trial 結束後該鍵不存在，查詢該類別的參數時回到預設參數

### Requirement: resume 只跳過已成功完成的組合
`GridSearchRunner` 在 `resume=True` 時 SHALL 讀取既有 CSV，並只跳過 `combo_id` 已有**有效分數**的組合；分數為 `-inf` 的失敗列不算已完成，該組合 MUST 重跑，且該列不得計入結果與最佳值。`resume=False` 時 MUST 全部重跑。每個執行過的 trial（包含失敗的）MUST 追加一列到 CSV，不覆寫舊列，且最佳分數只在嚴格大於目前最佳時更新。

#### Scenario: 跳過已記錄的組合
- **WHEN** CSV 已含 2 組有效結果，以 `resume=True` 執行 4 組的搜尋
- **THEN** 只對另外 2 組呼叫評分，結果包含 4 組，最佳參數取自全部 4 組

#### Scenario: 不 resume 時全部重跑
- **WHEN** CSV 已含結果但 `resume=False`
- **THEN** 4 組全部被評分

#### Scenario: 平手不更新最佳
- **WHEN** 兩組參數分數相同
- **THEN** 最佳參數保留先得到的那一組

#### Scenario: 失敗的 trial 在 resume 時重跑
- **WHEN** CSV 中某組合的分數為 `-inf`，以 `resume=True` 再次執行
- **THEN** 該組合被重新評分，其餘有效組合仍被跳過；舊的失敗列保留在 CSV，新結果追加在後

#### Scenario: 失敗列不影響最佳值
- **WHEN** resume 載入的 CSV 含 `-inf` 失敗列與有效列
- **THEN** 載入的結果與最佳值只來自有效列
