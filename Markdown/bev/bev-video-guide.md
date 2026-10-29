# 鳥瞰影片（`conflict_analysis.bev_video`）

衝突指標分析的選用輸出：把軌跡畫在衛星圖上，產生 `collision_output/<seq>/conflict/bev.mp4`。**預設關閉**；關閉時既有輸出（含 `summary.json` 的 `params`）逐位元不變。

## 1. 開啟

在已有 `conflict_analysis:` 的情境 config（例如 `config/collision/conflict_analysis.yaml`）加入：

```yaml
conflict_analysis:
  enabled: true
  bev_video: true            # 簡寫；或用下面的區塊寫法調整
  # bev_video:
  #   enabled: true
  #   max_side_px: 960       # 影片長邊上限（像素）；只縮小，不放大
  #   tail_s: 3.0            # 尾跡長度（秒）；實際幀數 = round(tail_s × fps)
```

未知欄位會報錯。需 `stages.run_stage3: true`。

## 2. 適用鏡頭

與衝突指標相同：需要 Homography 與公尺換算（`meters_per_pixel()` 可用）以及衛星底圖，目前是 C000025、C000043、C000054、C000085。鏡頭不適用時整個衝突分析被略過（`summary.json` 的 `skipped_reason` 為 `no_homography`／`no_meters_per_pixel`），不產生影片，也不退回像素；找不到衛星底圖時 `summary.json` 的 `bev_video` 記錄 `written: false` 與 `skipped_reason: no_satellite_image`。

## 3. 畫面內容

- 底圖：衛星圖，依 `max_side_px` 縮小，長寬取偶數（H.264 需要）。
- 軌跡：依類別上色的尾跡、軌跡 ID 與速度（km/h）；畫面左上為類別圖例（深色底）。
- 衝突事件：事件開始到結束的幀，在配對的兩個物體之間畫紅色連線。
- 計數線（同時開啟 `counting:` 且該鏡頭有計數線檔時）：黃色線段；右上角面板顯示各線各方向的累計數。`counting:` 關閉時畫面不含這些元素。
- 時間範圍：從第一個到最後一個有軌跡的幀。

## 4. 限制

- 影片幀率固定為 5 fps（沿用既有 `VideoWriterWithFfmpeg` 行為），已記在 `summary.json` 的 `limitations`（`bev_video_fps`）；原始 fps 較高的序列會被視為慢動作。
- OpenCV 內建字型不支援中文，面板與圖例只用英文（標籤為 `northbound` 等程式自動判斷的名稱）。
- 速度標籤密集時會互相重疊，沒有做避讓。
- 投影誤差在畫面遠端最大，軌跡位置只當技術訊號。
