# 實作計畫：Sub_Module 功能整合（F1–F4）

> 依據：`SPEC-features.md`（功能規格與決策 Q36–Q52）。任務與勾選清單見 `tasks/todo-features.md`。
> 本計畫在 F1、F2a、F2b 已完成後補寫（使用者要求補做 plan／build 流程）；已完成項目依實際 commit 標記。
> 與重構計畫分開存放：`tasks/plan.md`、`tasks/todo.md` 是已完成的重構紀錄，不覆寫。

## 概觀

把 `Sub_Module/` 中主流程沒有的功能，整合成**預設關閉**的選用功能：網頁選點器（F1）、計數線流量統計（F2）、衛星圖鳥瞰影片（F3）、盲區分析（F4）。所有功能關閉或缺少 config 時，既有輸出逐位元不變。

基準線：重構合併後 `main` 上 `pytest` = 790 passed, 11 skipped。進度（2026-10-29）：F1 完成並已合併進 `main`（`078af50`）；F2a、F2b 在 `feature/counting-lines`（`cb9c77b`、`80f70d5`），pytest 888 passed, 11 skipped。

## 架構決策

- **每個功能一條分支、垂直切片**：每個任務都交付一條可測試的完整路徑（設定 → 計算 → 輸出），不先做全部設定再做全部計算。分支：`feature/web-picker`（已合併）、`feature/counting-lines`、之後 `feature/bev-video`、`feature/blind-area`。
- **模組沿用 `collision/conflict/` 的拆檔**：config／純計算／輸出分開；純計算函式不碰檔案，用合成資料測試。
- **F1 方案 A**：離線單檔 HTML，頁面 JS 自行產生 YAML；以 Node 實際執行頁面 JS，要求輸出與 Python `yaml_output` 逐字相同（避免兩邊格式漂移）。
- **座標空間**：F2 用 CCTV 像素（不依賴 Homography，所有鏡頭可用）；F3、F4 用衛星圖像素／公尺，只在 `meters_per_pixel()` 可用的鏡頭（C000025、C000043、C000054、C000085）。
- **不跑 golden**（預設關閉，既有輸出不變）；唯一例外是 `conflict_analysis` 既有輸出出現任何差異時，停下來問使用者。
- **新依賴先問**：盲區多邊形運算預設用 `cv2`／`numpy`；要用 `shapely` 須先經使用者同意。
- **提交方式**：一個功能一個 commit 系列（測試＋實作同一 commit，文件另一 commit）；英文 conventional commits；精確 `git add`；commit 前跑完整 `pytest` 並出示 diff 摘要。

## 依賴圖

```
F1a camera_view_point 欄位 ─┐
F1b yaml_output ─ F1c 頁面 ─ F1d make_picker.py ─ F1e 文件            （完成）
                                      │
        ┌─────────────────────────────┘（F1 提供 counting-lines／camera-position 的資料來源）
        ▼
F2a 載入 ─ F2b counting 區塊 ─ F2c 穿越判定 ─ F2d runner＋stage3 ─ F2e 文件
                                                    │
                       ┌────────────────────────────┴───────────────┐
                       ▼                                            ▼
F3a bev_video 設定 ─ F3b 逐幀繪製 ─ F3c runner＋輸出 ─ F3d 疊加計數線／事件連線       F4a 設定與寬度 ─ F4b 陰影幾何 ─ F4c runner＋csv
                                                                  │                                              │
                                                                  └──────────── F4d 盲區疊進鳥瞰影片 ◄────────────┘
                                                                                       │
                                                                                  F3/F4 文件
```

F3 在 F4 之前，F4 完成後才補盲區疊圖（F4d），所以兩者在 F4d 匯合。

## 階段與檢查點

### 階段 1：F1 網頁選點器（完成）
- F1a–F1e：`camera_view_point` 欄位、`yaml_output`、頁面產生器、`make_picker.py`、文件。

**檢查點 F1（已通過）**：四種模式的 YAML 可被讀回（計數線模式的讀回測試於 F2a 補上）；真實 Chromium 實測四種頁面（點擊座標、`z` 退回、無 console 錯誤）；pytest 863 passed。

### 階段 2：F2 計數線流量統計
- F2a 載入與驗證（完成）、F2b `counting:` 區塊（完成）、F2c 穿越判定、F2d runner 與 `run_stage3()` 接線、F2e 文件。

**檢查點 F2**：合成軌跡涵蓋方向、去重、類別分組、`image_size` 不符、無設定檔的 `skipped_reason`；關閉時 stage3 既有輸出逐位元不變；與使用者審閱後才進 F3。

### 階段 3：F3 鳥瞰影片
- F3a 設定、F3b 逐幀繪製、F3c runner 與輸出、F3d 疊加計數線與事件連線。

**檢查點 F3**：關閉不產生檔案；開啟時對可用鏡頭產生非空 `bev.mp4`，其餘記錄跳過原因；以一支真實序列產生一次並實際檢視影片畫面（單支序列，量小）；審閱。

### 階段 4：F4 盲區分析
- F4a 設定與寬度、F4b 陰影幾何、F4c runner 與 `blind_area.csv`、F4d 疊進鳥瞰影片。

**檢查點 F4（最終）**：合成矩形與已知視點的面積比例與手算值一致；`limitations` 含 2D 假設；SPEC 第 9 節成功標準 1–7 逐項對照；「絕對不做」清單零違反；交付摘要。

## 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| 方向定義（左右、y 向下）搞反 | 高：流量方向全反 | F2c 先用手算案例寫測試，並對照 SPEC 範例（`(120,300)→(480,310)` 的 forward 是由北往南） |
| 軌跡抖動或跨線重複計數 | 中 | 同軌跡同線同方向只計一次，來回各一次；測試涵蓋抖動 |
| ID 切換造成流量高估／低估、stage1 不依 ROI 過濾 | 中 | 寫進 `summary.json` 的 `limitations`，教學說明；不在程式內「修正」 |
| 盲區 2D 假設高估 | 中 | `limitations` 明載；數值只當技術訊號 |
| 盲區多邊形運算需要 `shapely` | 中 | 先以 `cv2`／`numpy` 設計（逐幀以光柵化面積或多邊形裁切）；要新增依賴先問使用者 |
| 既有 `conflict_analysis` 輸出被改動 | 高 | 新欄位預設關閉；每個含 `conflict` 的變更後驗證既有測試與（必要時詢問後）golden |
| 影片編碼拖慢或檔案過大 | 低 | `bev.mp4` 選用輸出；先沿用 `VideoWriterWithFfmpeg`（fps 5 為既有行為，不修正） |
| 瀏覽器 UI 只有語法檢查 | 低（F1 已實測） | 之後若改頁面，重做一次真實瀏覽器點擊驗證 |
| GateGuard 阻擋首次 Write／Edit | 低 | 先在文字中說明事實再重試 |
| 工作樹有使用者的未提交修改（`skills-lock.json` 等） | 中 | 只 `git add` 自己改的檔案；不使用 `git add -A` |

## 待確認事項

1. 盲區多邊形運算是否使用 `shapely`（新依賴，須先問；預設以 `cv2`／`numpy` 實作）。
2. F2d 已提議、待使用者確認：`counts.csv` 以「每次穿越一列、附該線該方向該類別的累計值」表示時間序列（需要逐幀或逐秒取樣時另議）；`summary.json` 放各線各方向各類別的總數。
3. F3 鳥瞰影片的解析度（預設用衛星圖尺寸）與軌跡尾跡長度的預設值（F3a 時提出）。
4. F2c 規則（已向使用者提出）：`side = dx·(py−y1) − dy·(px−x1)`，`side < 0` 為畫線方向左側；左→右為 forward；相鄰取樣點 `side` 符號改變且交點在線段內才算穿越；幀號缺口不特別處理。
