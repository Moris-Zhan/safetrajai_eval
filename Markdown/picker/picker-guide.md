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

## 4. 部署成網頁服務（遠端開啟）

```bash
python scripts/picker/serve_picker.py                  # 預設 0.0.0.0:8765
python scripts/picker/serve_picker.py --host 127.0.0.1 # 只開本機，配合 SSH 通道或 kubectl port-forward
```

- 啟動時會印出網址 `http://<主機>:<連接埠>/?token=<權杖>`；開啟一次後瀏覽器以 cookie 記住，之後直接用首頁選鏡頭與模式。頁面在點選時才產生，所以影像永遠是最新的。
- 權杖預設存在 `runs/picker/server_token`（首次啟動隨機產生、權限 600、被 git 忽略）；也可用 `--token` 或環境變數 `PICKER_TOKEN`。伺服器日誌會把網址中的權杖遮蔽成 `***`。沒有權杖或權杖錯誤一律回 401，且不會產生任何頁面。
- **只有純 HTTP，沒有加密**：只在信任的網路（內網、VPN）使用；不信任的網路請改走 SSH 通道或 `kubectl port-forward` 並用 `--host 127.0.0.1`。不要直接暴露到公開網路。
- 路由只認列出的鏡頭與四種模式，請求路徑不會碰到檔案系統。
- 選點後仍是在頁面上複製或下載 YAML，再自行放進 `config/`；服務不會寫入任何設定檔。

### 背景啟動

```bash
bash scripts/picker/start_picker_service.sh     # 背景啟動；該連接埠已有服務回應時什麼都不做
```

環境變數 `PICKER_PORT`（預設 8765）、`PICKER_LOG`（預設 `runs/picker/server.log`，權限 600，被 git 忽略）。日誌第一行含帶權杖的網址。服務是普通的背景行程，pod 重啟後要再執行一次。
