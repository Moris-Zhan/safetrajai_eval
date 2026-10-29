# 計數線流量統計（`counting:`）

stage3 的選用功能：統計車輛、機車、行人等穿越指定線段的數量與方向。**預設關閉**；沒有 `counting:` 區塊或沒有該鏡頭的計數線檔時，既有輸出完全不變。

## 1. 畫計數線

```bash
python scripts/picker/make_picker.py --camera C000025 --mode counting-lines
```

在瀏覽器的 CCTV 首幀每兩點畫一條線，可命名（可省略，預設 `line_1`、`line_2`…），頁面用黃色箭頭標示 forward、白色箭頭標示 backward 的穿越方向（畫反了兩個箭頭會互換位置）；複製或下載 YAML，存成 `config/counting_lines/<camera>.yaml`（步驟見 [`Markdown/picker/picker-guide.md`](../picker/picker-guide.md)）：

```yaml
camera: C000025
image_size: [640, 480]       # 必須與影片解析度一致，否則報錯（不縮放）
lines:
  - name: north_gate         # 名稱不可空白或重複
    points: [[120, 300], [480, 310]]   # 畫線順序 p1 → p2
    forward_label: southbound          # 可省略：預設由程式依線的方向自動判斷（見下）
    backward_label: northbound         # 可省略
```

## 2. 開啟

在情境 config 加上頂層區塊（例如複製 `config/collision/collision_detect.yaml` 另存新檔後加入）：

```yaml
counting:
  enabled: true
```

執行 `python track_eval.py --config <你的 config>`（需 `stages.run_stage3: true`）。

## 3. 規則

- **座標空間**：CCTV 像素座標，不依賴 Homography，所有鏡頭都可用。
- **判定點**：框底邊中點（接地點）；stage1 軌跡不依 ROI 過濾，所以 ROI 外的穿越也會計入。
- **方向**：依畫線順序。沿 p1→p2 看（影像座標，y 向下），從左側穿越到右側為 forward，反向為 backward。例：`(120,300)→(480,310)` 由上往下穿越是 forward（southbound）。
- **穿越條件**：同一軌跡相鄰兩個取樣點分在線的兩側，且路徑與線段本身相交（只經過延長線不算）。幀號缺口不特別處理。
- **去重**：軌跡鍵為 `(class, track_id)`；同一軌跡對同一條線同一方向只計一次，來回各算一次，所以在線上抖動不會重複計數。

### 畫線順序與四個方向

方向由**畫線順序**決定：從第一點 p1 畫到第二點 p2，面向 p2 站在線上看畫面，從你的**左側穿越到右側**算 forward，反過來算 backward。`forward_label`／`backward_label` 就是你替這兩個方向取的名字。

| 畫法 | forward | backward |
|---|---|---|
| 水平線，由左畫到右 | 由上往下（畫面往下） | 由下往上 |
| 水平線，由右畫到左 | 由下往上 | 由上往下 |
| 垂直線，由上畫到下 | 由右往左 | 由左往右 |
| 垂直線，由下畫到上 | 由左往右 | 由右往左 |

斜線同一個規則。要讓某個方向剛好是 forward，就先決定好行進方向再決定線要從哪一端畫起；畫反了不會出錯，只是 forward 與 backward 的數字（與標籤）互換。

### 方向標籤自動判斷

頁面只負責畫線；`forward`／`backward` 的標籤由程式在執行時依線的方向決定，不寫進 YAML：

- **有 Homography 的鏡頭**：把方向換到衛星圖（上方為北），標籤為 8 個羅盤方向：`northbound`、`northeastbound`、`eastbound`、`southeastbound`、`southbound`、`southwestbound`、`westbound`、`northwestbound`。
- **沒有 Homography 的鏡頭**：退回畫面方向：`up`、`up_right`、`right`、`down_right`、`down`、`down_left`、`left`、`up_left`。之後補上 Homography，標籤會自動升級成羅盤方向，不必重畫。
- 標籤描述的是**車輛在該方向穿越時的行進方向**（水平線由左畫到右、畫面上方為北時，forward 是 `southbound`）。
- 在 YAML 手寫 `forward_label`／`backward_label` 仍然有效並優先；兩個方向各自判斷，只寫一個時另一個仍自動推得。
- `summary.json` 每條線記錄 `label_source`（`manual`／`compass`／`screen`），看得出標籤怎麼來的。
- 鳥瞰影片的計數面板使用 OpenCV 內建字型，**畫不出中文**，所以自動標籤一律是英文；手寫中文標籤只會出現在 `counts.csv`／`summary.json`，影片面板會變成問號。

## 4. 輸出：`collision_output/<seq>/counting/`

| 檔案 | 內容 |
|---|---|
| `counts.csv` | 每次穿越一列：`frame,time_s,line,direction,label,class,track_id,cumulative`；`cumulative` 是該線、該方向、該類別到此為止的累計數（累計曲線即時間序列），`time_s = frame / fps` |
| `summary.json` | 各線各方向各類別的總數（含 `all`）、使用的線、`limitations`；沒有計數線檔的序列只有 `skipped_reason: no_counting_lines` |

## 5. 限制

- ID 切換會讓同一物體重複計數，漏追蹤則少計；數字只當技術訊號，不是正式流量。
- 計數線在 ROI 外的穿越也會計入。
