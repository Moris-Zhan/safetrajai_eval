---
description: 為指定變更建立最小且可追溯的驗證計畫
argument-hint: [變更內容或檔案路徑]
---

閱讀 $ARGUMENTS 與相關程式碼，並先查閱 `markdown/EXPERIMENTS.md` 和 `markdown/GUIDE.md`，避免重複已完成的實驗。

依變更範圍提出分層驗證：

1. 靜態檢查：Python 語法、import 與 config 載入
2. 模組 smoke test：單影格偵測、單影片追蹤，或 Homography／碰撞核心函式
3. 整合驗證：最小資料範圍的 stage1／stage2／stage3
4. 結果回歸：與基準的 HOTA、AssA、IDSW、事件數或輸出格式比較

本專案沒有自動化測試；不要聲稱「全部測試通過」而未實際執行。每一項回報都要包含執行指令、涵蓋範圍、結果與未驗證項目。若驗證會重跑完整影片、大量 grid search 或覆寫輸出，先說明成本與影響並取得確認。
