---
description: 解釋、比較或調整追蹤與偵測情境設定
argument-hint: [config 路徑或調參目標]
---

閱讀 $ARGUMENTS 指定的設定檔或調參目標，並以目前 `config/*.yaml` 作為唯一有效設定來源；不要直接修改 `archive/` 的舊設定。

輸出內容應包含：

1. 目前 tracker、Re-ID、GMC、偵測信心門檻與各類別覆寫值
2. 與最接近情境檔的差異及可能影響
3. 建議調整的最小參數集合、每個值的預期效果與風險
4. 對應的驗證方式：單序列 smoke test、TrackEval，或 grid search

若要修改 YAML，先確認目標是既有情境檔還是新實驗檔，避免覆蓋可重現的基準設定。需要系統性搜尋時，優先提供 `scripts/grid_search_*.py` 的 `--dry-run` 指令與組合數，再取得確認才執行完整搜尋。
