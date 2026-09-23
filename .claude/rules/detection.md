---
paths:
  - 'module/detection/**/*.py'
---

# Detection 開發規範

## 模組責任

- `Detector` 只做單影格 YOLO 推論與逐類別信心過濾；不負責追蹤、MOT 序列化、評估或模型訓練。
- 輸入為 OpenCV BGR frame；輸出維持 Ultralytics `Boxes`，其 bbox 使用既有 `xyxy` 像素座標與固定類別 ID。不要在此層改成正規化座標或自訂資料結構。

## 設定契約

- 建立 `Detector` 前先載入目標情境的 tracker config；類別信心門檻由 `class_conf_thresh` 決定。
- Detector 先以最低類別門檻取得候選框，再依各類別門檻過濾；任何優化都不能讓低於該類門檻的框進入 tracker。
- `imgsz`、IoU、模型權重、`det_corner` 都屬實驗條件。改動時要記錄對召回率、誤檢與下游 tracking 的預期影響。

## 驗證

- 最小 smoke test：讀取有交通流量的單一影格，確認輸出類別只在 0–5，且每筆 confidence 符合對應門檻。
- DET mAP 是 Stage 2 的選用工作；變更 DET 資料準備或評估時，確認產生有效 `data.yaml`、五欄 YOLO label 與 Excel 指標，而不是只確認推論不拋錯。
