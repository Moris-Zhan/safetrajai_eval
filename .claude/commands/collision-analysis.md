---
description: 執行、檢查或修改 TTC/PET 碰撞偵測流程
argument-hint: [config 路徑、序列名稱或需求]
---

閱讀 $ARGUMENTS，並以 `config/collision_detect.yaml` 與 `.claude/rules/collision-and-homography.md` 為準。

先確認：

1. stage1 已產生逐類別 MOT 軌跡；stage3 不應誤讀 GT 檔案
2. Homography 是否找到對應矩陣；找不到時會使用原始 CCTV 像素座標
3. `iou_threshold`、`max_ttc_frames`、`lookback_sec`、`overlap_threshold` 的座標尺度與目的相容
4. 啟用 Homography 時，邊界尺寸應使用衛星圖尺寸，而非 CCTV 影像解析度

執行或分析後，依序整理每個序列的 track 數、TTC/PET 事件數、輸出 JSON/CSV、熱區圖，以及可能的誤報來源。不要將目前的碰撞門檻描述成已正式校正；若需要調參，先用小範圍序列驗證。
