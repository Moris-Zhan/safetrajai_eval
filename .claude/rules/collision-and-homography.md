---
paths:
  - 'module/collision/**/*.py'
  - 'config/collision_detect.yaml'
---

# Collision 與 Homography 開發規範

## 軌跡與事件

- Stage 3 讀取 Stage 1 的逐類別 7 欄 MOT 軌跡；不可讀 GT、影像路徑清單或逐幀中間 label。
- Layer 2 track dict 是內部結構：`Positions`、`Bboxs`、`Velocity`、`Direction`、`Size` 皆以 frame 為 key。速度以相鄰 frame 中心差分計算；中斷幀沒有速度，不能參與 TTC，但仍可參與 PET。
- TTC 與 PET 事件輸出採乾淨的 `ttc_events.{json,csv}` 與 `pet_events.{json,csv}`。不要依賴舊 log 的 `colision_occur_sec`，因為 PET 該欄可能是 frame 而非秒數。

## Homography 與座標系

- 查找順序固定為：既有矩陣 → `config/homography_points/<seq>.yaml` 人工點選 → 無矩陣時使用原始 CCTV 像素座標。
- 既有矩陣將 CCTV 像素轉為衛星底圖像素；人工 fallback 是使用者定義校正座標，不能宣稱為真實地理座標。
- 套用 Homography 時，TTC 的 bounds 必須是衛星圖尺寸；原始像素流程才可使用 CCTV 影像尺寸。
- 內部 TTC／PET 運算可使用衛星圖座標，但寫出的事件點、future box 與 arrow 必須以反矩陣投影回 CCTV 像素座標，才能與原始影像及熱區圖一致。
- Homography 後的速度／方向必須由轉換後位置重新差分，不能直接轉換原始速度向量。

## 校正與風險

- Homography 至少需要四組順序一對一的點；四點剛好可解，沒有容錯，不可把訓練點誤差當成校正精度。
- 驗證使用獨立 check points；沒有 check points 只表示未驗證，不代表矩陣必然失效。現有誤差是像素而非公尺，不可宣稱安全距離精度。
- `iou_threshold`、`max_ttc_frames`、`lookback_sec`、`overlap_threshold` 與座標尺度相關。未經實測校正的事件數只能作技術訊號，不能作正式安全結論。
- 事件數為零時清除舊的熱區圖，避免輸出目錄殘留過期結果。
