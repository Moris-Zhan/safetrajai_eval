# 盲區分析（`conflict_analysis.blind_area`）

衝突指標分析的選用輸出：估計「從相機看過去，被大型車輛擋住的地面」佔關注區域（ROI）的比例，逐幀輸出。**預設關閉**；關閉時既有輸出（含 `summary.json` 的 `params`）逐位元不變。

## 1. 取得相機位置（`camera_view_point`）

盲區需要相機在衛星圖上的位置：

```bash
python scripts/picker/make_picker.py --camera C000025 --mode camera-position
```

在網頁的衛星圖上點出相機所在位置（只需一個點，再點一次會取代），複製產生的一行，貼進 `config/homography/<camera>.yaml`：

```yaml
camera_view_point: [412, 530]     # 衛星圖像素座標（與 Homography 轉換後的座標同一個空間）
```

操作步驟見 [`Markdown/picker/picker-guide.md`](../picker/picker-guide.md)。

## 2. 開啟

```yaml
conflict_analysis:
  enabled: true
  blind_area: true                 # 簡寫；或用區塊調整遮蔽類別
  # blind_area:
  #   enabled: true
  #   occluder_classes: [car, small_truck, bus, truck]   # 預設；不含 person、scooter
  # object_width_m: {person: 0.5, car: 1.8, scooter: 0.8, small_truck: 2.0, bus: 2.5, truck: 2.5}
```

需 `stages.run_stage3: true`。與鳥瞰影片同時開啟（`bev_video`）時，影片會疊上盲區（見 [`Markdown/bev/bev-video-guide.md`](../bev/bev-video-guide.md)）。

## 3. 適用鏡頭與略過原因

和衝突指標相同，需要 Homography 與公尺換算（目前 C000025、C000043、C000054、C000085）；再加上 `camera_view_point` 與衛星底圖。不符合時不輸出 `blind_area.csv`，原因寫在 `summary.json` 的 `blind_area.skipped_reason`：

| `skipped_reason` | 意思 |
|---|---|
| `no_camera_view_point` | 鏡頭設定沒有 `camera_view_point` |
| `no_satellite_image` | 找不到衛星底圖 |

鏡頭沒有 Homography 或 m/px 時，整個衝突分析被略過（`summary.json` 頂層的 `skipped_reason`），也就沒有盲區。

## 4. 幾何假設

- **遮蔽物**：車輛視為矩形。中心是框底邊中點投影到衛星圖座標；長度取 `object_length_m`，寬度取 `object_width_m`；朝向是平滑後的速度方向。
- **朝向**：速度低於 `min_speed_mps` 時沿用前一個有效朝向；軌跡最前面還沒有朝向的幀，該車該幀不計；整條軌跡都沒動過的車不計。數量見 `summary.json` 的 `occluders_without_heading`、`frames_without_heading`。
- **遮蔽物來源**：全部軌跡（不依 ROI 過濾），因為 ROI 外的大車也會把陰影投進 ROI；面積只算 ROI 內。
- **陰影**：從視點看過去，矩形後方、兩條輪廓射線之間的區域，不含車體本身；多個陰影重疊的部分只算一次，站在別人陰影裡的車仍算盲區。
- **視點在車體內或邊上**：該車該幀不產生陰影，次數記在 `viewpoint_inside_skips`。
- **面積計算**：不用多邊形運算套件，在衛星圖像素網格上光柵化後數像素，每條邊約有半個像素的誤差（面積偏大幾個百分點）。

## 5. 輸出

`collision_output/<seq>/conflict/blind_area.csv`，每個影格一列（從第一個到最後一個有軌跡的幀）：

| 欄位 | 意思 |
|---|---|
| `frame`、`time_s` | 幀號、`frame / fps` |
| `num_occluders` | 當幀有產生陰影的遮蔽物數 |
| `blind_area_m2` | ROI 內的盲區面積（平方公尺） |
| `roi_area_m2` | ROI 面積（平方公尺） |
| `blind_ratio` | `blind_area_m2 / roi_area_m2` |

`summary.json` 的 `blind_area` 區塊記錄視點、遮蔽類別、影格數、平均與最大比例及上述略過次數。

## 6. 限制

- **視為 2D**：車輛沒有高度，所以會高估盲區（真實車輛只遮住視線高度以下的範圍；相機越低、車越高，陰影越接近真實）。已寫入 `limitations`（`blind_area_2d`）。
- 結果受視點位置與朝向誤差影響；Homography 在畫面遠端誤差最大。
- 比例只當技術訊號，不是安全結論。與衝突事件的整合（事件註記盲區比例）不在目前範圍，等逐幀數據檢視後再決定。
