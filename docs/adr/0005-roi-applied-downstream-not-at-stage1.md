# ROI 由下游套用，stage1 軌跡不過濾；評估以忽略區域在 TrackEval 之前處理

stage1 輸出的 7 欄 MOT 軌跡保留全部追蹤結果，不依 ROI 過濾。需要限縮範圍的下游各自套用所屬鏡頭的 ROI：
- stage3 與 Re-ID 裁切蒐集以框中心點硬過濾。
- stage2 以忽略區域處理（UA-DETRAC 式）：ROI 外的人工 GT 標為 ignore，預測與全部 GT 做 IoU ≥ 0.5 的一對一匹配；匹配到 ROI 內 GT 的預測保留，其餘預測中匹配到 ignore GT 或中心點在 ROI 外者移除，最後移除 ignore GT。FP 因此只來自 ROI 內。

過去在 stage1 就依中心點過濾預測、stage2 再同樣過濾 GT，跨越 ROI 邊界的物件會同時產生假的 FN 與 FP。忽略區域需要看到 ROI 外的預測才能正確匹配，所以過濾不能提早到 stage1。這一步寫在本專案的評估前處理，而不是 vendored TrackEval，因為 TrackEval 唯讀（ADR 0002）：它的 `distractor_classes` 寫死為空，`zero_marked` GT 只會被刪除，匹配到它們的預測仍算 FP，無法表達這個語意。

忽略語意最初採 MOT17 式（只移除匹配到 ignore GT 的預測），實測後 ROI 外未標註或定位偏差的物件大量變成 FP，與「只評估 ROI 內」的目的相違，因此改為 UA-DETRAC 式；代價是 ROI 外的誤報不受懲罰，評估可信度取決於 ROI 的畫法。曾考慮 stage1 同時輸出過濾版與完整版軌跡，但兩份軌跡會讓下游契約變得模糊，因此不採用。之後不要為了「提早過濾比較省事」把 ROI 過濾移回 stage1。這項決定由 OpenSpec change `reorganize-dataset-and-eval-gt` 實作。
