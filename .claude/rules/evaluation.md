---
paths:
  - 'evaluation/**/*.py'
  - 'track_eval.py'
---

# Evaluation 與軌跡輸出規範

## 軌跡格式契約

- 逐幀 YOLO label 是中間產物；穩定的下游輸入是 `trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt`。
- 每行固定 7 欄、無 header：`frame,track_id,left,top,w,h,score`。bbox 為像素座標，原點在影像左上；`left,top` 是左上角。
- 類別不在檔案欄位內，而是由 `YOLO11x__<class>` 資料夾表示。不得假設標準 MOT 的 9/10 欄、3D 欄位或 class 欄存在。
- `frame` 從 1 開始，代表有效幀序號；插補幀被跳過。`track_id` 僅在單支影片內唯一，跨影片需以「序列名＋track_id」做複合鍵。

## 輸出與 GT

- 六個類別均要建立 tracker 檔；沒有追蹤結果時建立 0-byte 空檔，不能省略資料夾或檔案。
- GT 是人工標註，只供 TrackEval 比對；GT 有 class 欄，tracker MOT 沒有。任何下游風險分析都必須讀 tracker 輸出而非 GT。
- `det_corner` 會在 MOT 輸出階段依 bbox 中心點過濾，區域外軌跡不會出現在最終檔案。
- 非預設資料集目前受 `save_mot_format()` 的 `C000025_*` glob 限制，可能靜默輸出全空軌跡。修正時以非 C000025 序列測試，且在修正前不得把空檔解讀為無偵測。

## 評估與報表

- Stage 2 逐類別計算 HOTA、CLEAR、Identity 指標，Excel 值必須可回溯到 TrackEval 結果。
- 缺單一類別檔應記錄為 note；所有類別皆無有效結果時跳過 Excel 並明確失敗，不可產生誤導性的空報表。
- 改動 MOT 轉換、GT 清理、TrackEval 目錄結構或 Excel 欄位時，檢查 frame 對齊、ID 去重、多邊形過濾及 console／Excel 數值一致性。
