# grid-search-trials Specification

## Purpose

定義 grid search 單一 trial 的契約：參數組合如何展開並唯一識別、trial 期間如何修補並還原追蹤參數、分數如何萃取與彙總，以及 `--resume` 何時可以跳過組合，讓調參結果可重現且不污染其他 trial 或後續流程。

## Requirements

### Requirement: 參數組合展開順序固定且可唯一識別
`ParameterGrid` SHALL 把 `{參數: [值...]}` 展開成所有排列組合，鍵以排序後的順序展開；組合數 MUST 等於各參數值個數的乘積；`combo_id` MUST 與傳入字典的鍵順序無關，同一組參數永遠得到相同字串。

#### Scenario: 展開與數量
- **WHEN** 以 `{"b": [1, 2], "a": [3, 4, 5]}` 建立
- **THEN** 共 6 組，依 `a`、`b` 的排序順序展開，長度為 6

#### Scenario: combo_id 與鍵順序無關
- **WHEN** 以兩種不同鍵順序的相同參數字典計算 `combo_id`
- **THEN** 兩個結果相同；參數值不同時結果不同

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

### Requirement: 多序列分數以第 25 百分位彙總
`score_params` SHALL 先排除分數為 0 的序列，再以剩餘分數的第 25 百分位作為此組參數的分數；沒有任何有效分數時 MUST 回傳 `-inf`。

#### Scenario: 排除零分後取百分位
- **WHEN** 各序列分數為 `[0.0, 0.2, 0.4, 0.6, 0.8]`
- **THEN** 回傳 `[0.2, 0.4, 0.6, 0.8]` 的第 25 百分位

#### Scenario: 沒有有效分數
- **WHEN** 所有序列分數為 `None` 或 0，或沒有序列
- **THEN** 回傳 `-inf`

### Requirement: trial 的 GT 準備不污染原始 GT
`_setup_trial_gt` SHALL 把來源 GT 複製到 trial 目錄，且每次覆寫上一次的內容；`seqinfo.ini` 存在於來源根目錄或其 `gt/` 之下時 MUST 一併複製；來源沒有 `gt/gt.txt` 時 MUST 回傳 `False` 且不建立任何 GT 檔案。

#### Scenario: 覆寫舊的 GT
- **WHEN** trial 目錄已有被修改過的 `gt/gt.txt`
- **THEN** 準備後內容與來源完全相同，來源檔案未被改動

#### Scenario: 來源缺少 gt.txt
- **WHEN** 來源目錄沒有 `gt/gt.txt`
- **THEN** 回傳 `False`

### Requirement: 影片探索遵循覆寫與數量上限
`discover_grid_search_videos` SHALL 在指定 glob 時直接依 glob 結果（排序）取影片，否則掃描 stage1 輸出樹中存在 `<seq>/<seq>.mp4` 的序列；結果 MUST 依 `max_videos` 截斷。

#### Scenario: 預設掃描 stage1 輸出
- **WHEN** stage1 輸出樹有 3 個序列目錄，其中 1 個沒有對應的 mp4
- **THEN** 回傳另外 2 支影片的路徑，排序固定

#### Scenario: glob 覆寫與上限
- **WHEN** 指定 glob 且 `max_videos=1`
- **THEN** 只回傳排序後的第一支

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
