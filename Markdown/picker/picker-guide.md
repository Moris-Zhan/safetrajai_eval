# 網頁選點器（`make_picker.py`）

伺服器沒有圖形介面，所以工具輸出**單檔離線 HTML**（影像以 base64 內嵌），在自己電腦的瀏覽器開啟後點選，頁面會即時產生 YAML，可複製或下載。工具**不會修改任何 config**，產生的內容要自己放回對應位置。輸出在被 git 忽略的 `runs/picker/`，CCTV 與衛星圖不進 git、不上傳。

```bash
python scripts/picker/make_picker.py --camera C000054 --mode homography
```

## 模式

| `--mode` | 要點什麼 | 產生的 YAML 放哪裡 | 需要的影像 |
|---|---|---|---|
| `homography` | 左（CCTV 首幀）右（衛星圖）**成對**點選，至少 4 組 | `cctv_points`／`map_points` 貼進 `config/homography/<camera>.yaml` | CCTV＋衛星圖 |
| `checkpoints` | 成對點選驗證點（至少 1 組，少於 4 組頁面會警告） | 存成 `config/homography_checkpoints/<camera>.yaml` | CCTV＋衛星圖 |
| `counting-lines` | 在 CCTV 首幀依序點兩點成一條線，可填名稱與方向標籤 | 存成 `config/counting_lines/<camera>.yaml`（讀取端於後續功能加入） | CCTV |
| `camera-position` | 在衛星圖上點相機位置 | `camera_view_point` 貼進 `config/homography/<camera>.yaml` | 衛星圖 |

## 操作

- 成對模式必須先點左圖、再點右圖，點錯邊會被忽略。
- `z` 或「復原上一點」移除最後一個點；計數線模式按 `Esc` 取消尚未配對的第一點。
- 名稱不可空白或重複；準備好之前頁面會顯示原因。

## 衛星圖與 `matrix_tag`

- 衛星圖依 `config/homography/<camera>.yaml` 決定檔名，從 `models/homography/satellite_images/` 讀取；找不到時錯誤訊息會提示 `scripts/homography/download_satellite.py`。
- 若該鏡頭已指定 `matrix_tag`，`homography` 頁面會提示「這組點要移除 `matrix_tag` 才會生效」（查找順序是先 `matrix_tag`、再人工對應點）。
