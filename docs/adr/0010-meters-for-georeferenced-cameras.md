# 有地理對位的鏡頭可使用公尺單位

## 背景

先前的原則是 Homography 誤差與衝突分析一律以像素為單位，因為沒有鏡頭的經緯度，也無法確認衛星底圖的比例尺。衝突指標（MTTC、DRAC、PICUD、Crash Index）與 2 m 的 TTC／PET 門檻只有在公尺單位下才有物理意義，參考實作 `Sub_Module/trajectoryanalyst2` 也是以公尺計算。

## 決策

一台鏡頭只有在同時滿足以下條件時，其俯視平面才換算公尺（`georef.meters_per_pixel()`）：

1. 鏡頭設定 `config/homography/<camera>.yaml` 指定既有衛星底圖的矩陣（`matrix_tag`），而且沒有 `needs_recalibration`；
2. 有衛星圖參數（`satellite` 的 zoom、scale）；
3. 有路口經緯度（`latlon`）；
4. 比例尺檢查通過（`scale_check: passed`）：已知長度地物的量測落在 ±15% 以內，並且在有金鑰時，驗證用底圖與既有底圖的影像比對縮放落在 1±2% 以內。

換算公式是 Web Mercator 地面解析度：

```
m_per_pixel = 156543.03392 × cos(緯度) / 2^zoom / scale
```

任一條件不滿足時，換算結果為「不可用」，不得退回赤道值或預設值；依賴公尺的下游指標不計算，其他輸出維持像素。人工自訂矩形的俯視平面一律不換算。

## 理由與代價

- 參考實作的 `simple_zoom_scale()` 是 `40075017 / (scale·256·2^zoom)`，漏乘 cos(緯度)，在高雄（北緯約 22.6°）會讓距離高估約 8%。
- 公式只修正緯度與縮放等級，不修正 Homography 本身的誤差，也不處理平面近似（坡道、高架）；公尺值的可信度仍受 check points 誤差限制，驗證報表會並列像素與公尺誤差。
- 比例尺檢查是手動步驟，未檢查的鏡頭預設為 `unchecked`，因此導入初期只有少數鏡頭可用公尺。
- 衝突事件數仍只是技術訊號，不能當成正式的安全結論。
