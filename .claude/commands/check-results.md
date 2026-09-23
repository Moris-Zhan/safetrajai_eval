---
description: 檢查 TrackEval、Excel 或 grid search 結果是否合理
argument-hint: [輸出目錄、Excel、CSV 或序列名稱]
---

閱讀 $ARGUMENTS 指定的結果；若未指定，找出最近一次 `runs/track_eval/` 或 grid search 的輸出。依來源判斷資料鏈路：

```text
stage1 trackers MOT txt → stage2 TrackEval → Excel / grid-search CSV
```

檢查並回報：

1. 每個序列與類別的 tracker 輸出是否存在，空檔是否符合無偵測情況
2. HOTA、DetA、AssA、IDSW 是否有 NaN、全零或不合理落差
3. Excel／CSV 記錄的分數是否與 console 或 TrackEval 原始結果一致
4. 這次結果與基準設定相比的差異，以及可能原因

特別注意：grid search 的 scoring 是 `HOTA.AssA_mean`，不能把它和 HOTA 本身混為同一指標。若結果不足以判斷，明確列出缺少的基準、config 或輸出檔。
