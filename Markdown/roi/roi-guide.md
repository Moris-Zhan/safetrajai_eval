# ROI（關注區域）

## 1. 檔案格式：`config/roi/<camera>.yaml`

```yaml
camera: C000025
image_size: [640, 480]        # 必須與影片解析度一致，否則報錯（不會自動縮放）
polygon:                      # 像素座標，順序為左上→右上→右下→左下
  - [210, 90]
  - [580, 100]
  - [640, 320]
  - [0,   280]
```

- 鏡頭編號取自序列名稱第一個空白之前的字串（例如 `C000025 中華三路 民生二路(東北角)...` → `C000025`）。
- 沒有對應檔案時，以整張影像當作 ROI，並印出提示。
- 解析函式在 `safetrajai/roi.py`，stage1 繪製、stage2、stage3 與 Re-ID 裁切都用同一份。

## 2. 手動畫 ROI（完整步驟）

伺服器沒有圖形介面，所以改成輸出單檔 HTML，在自己電腦的瀏覽器上點選多邊形。整個流程是：**產生點選頁 → 在本機開啟 → 點出頂點 → 下載 YAML → 放回 `config/roi/` → 驗證**。

### 步驟 1：產生點選頁

```bash
python scripts/picker/make_roi_picker.py --camera C000054
python scripts/picker/make_roi_picker.py --camera C000054 --out-dir runs/roi_picker
```

- 程式從 `dataset/raw/videos/<camera>/` 第一支影片取首幀，以 base64 內嵌成單一 HTML 檔。
- 輸出到 `runs/roi_picker/<camera>.html`（gitignore，CCTV 畫面不會進 git）。
- 頁面不需要網路，也不會上傳任何東西。

### 步驟 2：在本機開啟網頁（三選一）

**方式 A：VS Code（最簡單）**
在 VS Code 的檔案總管找到 `runs/roi_picker/`，對 HTML 檔按右鍵選 **Download...**，下載到本機後用瀏覽器打開。

**方式 B：scp 複製到本機**

```bash
scp "<user>@<server>:/mnt/moris/Dcoker_Project/Project/safetrajai_eval/runs/roi_picker/*.html" .
```

**方式 C：在伺服器開臨時 HTTP 服務，再透過轉發的連接埠開啟**

```bash
cd runs/roi_picker && python -m http.server 8765 --bind 127.0.0.1
```

- 用 VS Code Remote-SSH 時，它會自動轉發連接埠；否則在本機執行 `ssh -L 8765:127.0.0.1:8765 <server>`。
- 在瀏覽器開 `http://localhost:8765/C000054.html`。
- 用完按 Ctrl+C 停掉服務。

### 步驟 3：在頁面上畫 ROI

1. 在畫面上**依序點出多邊形頂點**，建議順時針。座標會自動換算成原始影像的像素。
2. 點錯了可以按 **「復原上一點」**；想整個重畫就按 **「清除重畫」**。
3. 頁面下方會即時顯示 YAML（`camera`、`image_size`、`polygon`）。
4. 畫好後按 **「複製 YAML」** 或 **「下載 <camera>.yaml」**。

範圍的建議：框出路面上想分析或蒐集的區域，排除畫面邊緣、遠處的小目標，以及浮水印或時間戳的位置。框中心點落在 ROI 外的偵測，會被 stage3 與 Re-ID 裁切略過，在 stage2 則當作忽略區域處理。

### 步驟 4：把 YAML 放回專案

放在 **`config/roi/`**，檔名維持 `<camera>.yaml`（例如 `C000054.yaml`）。程式以鏡頭編號找檔案，檔名不對就會找不到。

- VS Code：把檔案拖到檔案總管的 `config/roi/` 上就會上傳。
- scp：

  ```bash
  scp C000043.yaml C000054.yaml "<user>@<server>:/mnt/moris/Dcoker_Project/Project/safetrajai_eval/config/roi/"
  ```

- 不要覆蓋既有的 `C000025.yaml`（評估鏡頭）。

### 步驟 5：驗證尺寸

對該鏡頭的一支影片解析 ROI；`image_size` 與影片解析度不符時會報錯：

```bash
python - <<'EOF'
from pathlib import Path
import cv2
from safetrajai.roi import resolve_roi
for cam in ["C000011", "C000012", "C000033", "C000043", "C000048", "C000054", "C000056", "C000085", "C000256"]:
    video = sorted(Path(f"dataset/raw/videos/{cam}").iterdir())[0]
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    print(cam, size, resolve_roi(video.stem, size).shape)
EOF
```

### 目前狀態（2026-09-24）

| 鏡頭 | ROI | 備註 |
|---|---|---|
| C000025 | ✅ | 評估鏡頭 |
| C000043 | ✅ 704×480 | 來源鏡頭 |
| C000054 | ✅ 1280×720 | 來源鏡頭（31 支影片，佔來源影片總時數約 83%） |
| C000085 | ✅ 704×480 | 來源鏡頭 |
| C000011 | ✅ 704×480 | 來源鏡頭（2026-09-24 新增，約 10 分鐘） |
| C000012 | ✅ 1280×720 | 來源鏡頭（新增） |
| C000033 | ✅ 1280×720 | 來源鏡頭（新增） |
| C000048 | ✅ 1280×720 | 來源鏡頭（新增） |
| C000056 | ✅ 704×480 | 來源鏡頭（新增） |
| C000256 | ✅ 1280×720 | 來源鏡頭（新增） |
| C000026 | 不畫 | 與評估鏡頭 C000025 同一路口，影片歸位但不列入 `reid_sources` |
