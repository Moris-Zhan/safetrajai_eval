# config/homography_points/

`module/collision` 用來啟用 Homography 座標轉換的人工對應點檔案，一個序列一份，
檔名對應序列名（跟 `annotation_<seq>/` 的 `<seq>` 部分相同）：

```
config/homography_points/<seq>.yaml
```

`TrackingEvalPipeline.run_stage3()` 會依序列名自動找這份檔案：找得到就算出
Homography 矩陣、該序列改用轉換後座標系跑 TTC/PET；找不到就照舊用原始像素座標，
不會報錯。目前這個目錄預設是空的（沒有任何 `.yaml`），代表所有序列都用原始像素座標。

## 怎麼填

1. 打開對應序列的 `annotation_<seq>/first_frame_with_box.jpg`（stage1 已自動存好），
   自己量測/估計 4 個以上對應點的像素座標（例如路口的四個角）。
2. 決定這些點要對應到「校正後」座標系的哪 4 個點——本專案沒有真實地圖底圖，
   `map_points` 是自訂的目標座標（例如把透視變形的路口攤平成一個矩形），
   不是地理座標。
3. 建立 `config/homography_points/<seq>.yaml`：

```yaml
cctv_points:
  - [x1, y1]
  - [x2, y2]
  - [x3, y3]
  - [x4, y4]
map_points:
  - [x1, y1]
  - [x2, y2]
  - [x3, y3]
  - [x4, y4]
```

`cctv_points` / `map_points` 必須一一對應、順序相同，至少 4 組。

4. 填完建議先用 `module/collision/homography_points.py::save_verification_image()`
   把 `cctv_points` 標在 `first_frame_with_box.jpg` 上另存一張圖，目視確認座標有沒有標錯位置，
   再讓 `run_stage3()` 真的拿去算 Homography。

## 已知限制

只有 4 組點時 `cv2.findHomography()` 是「恰好可解」，沒有容錯機制，點位精準度會
直接反映在轉換結果上；若提供 4 組以上的點，可以在
`module/collision/homography.py::compute_homography()` 自行加上
`method=cv2.RANSAC` 提升容錯度（目前保留原始行為，預設不啟用）。
