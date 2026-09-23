---
description: 分析或實作 class-aware BoTSORT／ByteTrack 追蹤器修改
argument-hint: [需求或檔案路徑]
---

先閱讀 $ARGUMENTS，並在修改前閱讀：

- `src/safetrajai/tracking/HANDOFF.md`
- 同目錄下對應的 `class_aware_botsort_flow.md` 或 `class_aware_bytetrack_flow.md`
- 使用中的 `config/*.yaml` 的 `class_tracking` 區塊

整理並說明：

1. 會影響哪個 matching 階段、類別參數或 Re-ID 行為
2. 是否改變 `match_thresh`、`track_buffer`、新軌跡門檻或 ID 指派
3. BoTSORT 與 ByteTrack 的 `track_buffer` 單位差異是否會影響結果
4. 修改後應以哪些序列、類別及 HOTA／AssA／IDSW 比較回歸結果

若需求包含程式修改，先列出要改的檔案與預期行為，取得確認後再實作。完成後至少執行語法檢查，並依 `markdown/GUIDE.md` 建議執行最小可行驗證。
