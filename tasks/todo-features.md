# 任務清單：Sub_Module 功能整合（F1–F4）

> 依據：`SPEC-features.md`、`tasks/plan-features.md`。每個任務一個 commit（英文 conventional commits，結尾附 Co-Authored-By 與 Claude-Session 兩行）；測試與實作同一 commit、文件另一 commit；commit 前跑完整 `pytest` 並出示 diff 摘要；只 `git add` 自己改的檔案。
> 規模：XS 1 檔／S 1–2 檔／M 3–5 檔。

通用驗證（每個任務都做）：
- [ ] 完整 `pytest` 通過（只增不減；目前 947 passed, 11 skipped）
- [ ] `git diff --stat` 未觸及 `Sub_Module/`、tracker override、`utils/modules/_vendor/trackeval/`、`utils/model.py`、`tests/golden/`
- [ ] 新功能關閉（預設）時既有輸出不變

---

## 階段 1：F1 網頁選點器（分支 `feature/web-picker`，已合併 `078af50`）

- [x] **F1a**（S）`camera_view_point: [x, y]` 選用欄位｜`7c3c2f0`｜驗收：可讀、可就地寫入、錯誤格式報錯並指出欄位
- [x] **F1b**（S）`picker/yaml_output.py`：點 → YAML 與驗證｜`f061f24`｜驗收：四種輸出皆可被既有載入函式讀回
- [x] **F1c**（M）`picker/page.py` 頁面產生器｜`d5cee4e`｜驗收：Node 執行頁面 JS，輸出與 Python 逐字相同；成對點擊、退回、就緒規則
- [x] **F1d**（M）`scripts/picker/make_picker.py` 與 `picker/build.py`｜`75504ec`｜驗收：缺衛星圖提示 `download_satellite.py`；`matrix_tag` 提示；輸出在 `runs/picker/`；真實 Chromium 實測四種模式
- [x] **F1e**（S）文件｜`5b117a3`｜`README.md`（覆蓋測試）、`Markdown/picker/picker-guide.md`、`CLAUDE.md`

### 檢查點 F1
- [x] 四種模式 YAML 可讀回（計數線讀回於 F2a 補上）；pytest 863 passed；真實瀏覽器驗證

---

## 階段 2：F2 計數線流量統計（分支 `feature/counting-lines`）

- [x] **F2a**（S）`collision/counting/lines.py`：`load_counting_lines`、`resolve_counting_lines`｜`cb9c77b`
  - 驗收：未知欄位、`camera` 與檔名不符、點數、零長度、名稱重複皆報錯；`image_size` 不符報錯；無檔回傳 `None`；選點器 YAML 讀回一致
- [x] **F2b**（XS）`counting:` 區塊（預設關閉）接進 `config_loader`｜`80f70d5`
  - 驗收：缺區塊或空區塊為關閉；未知欄位與非布林報錯；`config/collision/*.yaml` 載入後皆為關閉
- [x] **F2c**（S）`counting/crossing.py`：穿越判定、方向、去重（純函式）
  - 驗收：方向（左→右 forward，與 SPEC 範例一致）、交點在線段外不算、同軌跡同線同方向只計一次、來回各一次、類別分組、抖動不重複計數
  - 驗證：`pytest tests/test_counting_crossing.py`（手算案例；變異檢查：方向對調、移除線段範圍檢查皆使測試失敗）
  - 依賴：F2a｜檔案：`crossing.py`、測試
- [x] **F2d**（M）`counting/runner.py` 與 `run_stage3()` 接線，輸出 `counts.csv`、`summary.json`
  - 驗收：無設定檔 → 只寫 `summary.json` 並 `skipped_reason: no_counting_lines`；`limitations` 含 ID 切換與 stage1 不依 ROI 過濾；`enabled: false` 時 stage3 既有輸出與呼叫不變
  - 驗證：合成序列端對端測試；`tests/test_stage3_sequences.py` 延伸（開／關兩種）
  - 依賴：F2b、F2c｜檔案：`runner.py`、`tracking_pipeline.py`、測試
  - 待確認：`counts.csv` 欄位與取樣（見計畫待確認事項 2）
- [x] **F2e**（S）文件｜`Markdown/` 教學（座標空間、去重規則、輸出欄位）、`README.md`、`CLAUDE.md`、`SPEC-features.md` 補決策

### 檢查點 F2
- [ ] 全部測試通過；關閉時既有輸出不變；與使用者審閱後合併 `feature/counting-lines` 並開 F3

---

## 階段 3：F3 鳥瞰影片（分支 `feature/bev-video`）

- [x] **F3a**（S）`conflict_analysis.bev_video`（預設 false）、尾跡長度等欄位，進 `conflict/config.py`
  - 驗收：缺欄位維持現況；未知值報錯；既有 `ConflictConfig` 預設不變
  - 已決定：`max_side_px` 預設 960、`tail_s` 預設 3 秒
- [x] **F3b**（M）`conflict/bev.py`：逐幀繪製函式（衛星圖底圖、軌跡尾跡與 ID、速度 km/h）
  - 驗收：輸入輸出形狀與尺寸；已知軌跡在已知像素有繪製；不測影片像素
- [x] **F3c**（M）runner 輸出 `conflict/bev.mp4`，沿用 `VideoWriterWithFfmpeg`
  - 驗收：關閉不產生檔案；開啟時對可用鏡頭產生非空 mp4；m/px 不可用或無衛星圖 → `summary.json` 記錄原因
- [x] **F3d**（S）疊加衝突事件配對連線，及計數線與累計數（F2 開啟時）
  - 驗收：F2 關閉時畫面不含計數線元素
- [x] **F3e**（S）文件｜`Markdown/bev/bev-video-guide.md`、`README.md`

### 檢查點 F3
- [x] 測試通過；用一支真實序列產生一次 `bev.mp4` 並實際檢視（同一次實跑一併驗證計數線；需先用選點器畫一條線；先告知範圍再執行）；既有 `conflict` 輸出不變；審閱

---

## 階段 4：F4 盲區分析（分支 `feature/blind-area`）

- [ ] **F4a**（S）`blind_area` 設定（`enabled`、`occluder_classes`）與 `object_width_m`（預設 car 1.8、small_truck 2.0、bus 2.5、truck 2.5）
  - 驗收：預設關閉；類別必須在 6 類內；寬度為 6 類都給值且為正數
- [ ] **F4b**（M）`conflict/blind_area.py`：矩形＋視點 → 陰影多邊形；與 ROI 相交；面積比例
  - 驗收：合成矩形與已知視點，面積比例與手算值一致（容許誤差寫進測試）；視點在矩形內等退化情況有明確行為
  - 已決定：不用 `shapely`，以 `cv2` 光柵化
- [ ] **F4c**（M）朝向（平滑速度方向，速度太小沿用前一個有效朝向，沒有就跳過）、runner、`blind_area.csv`、`summary.json`
  - 驗收：無 `camera_view_point` 或 m/px 不可用 → `skipped_reason`；`limitations` 含 2D 假設
- [ ] **F4d**（S）盲區疊進鳥瞰影片
  - 驗收：F4 關閉時畫面不含盲區
- [ ] **F4e**（S）文件（幾何假設、`camera_view_point` 取得方式）

### 檢查點 F4（最終）
- [ ] SPEC 第 9 節成功標準 1–7 逐項對照；「絕對不做」清單零違反
- [ ] 交付摘要：新增行數、各功能預設值與開啟方式、已知限制
- [ ] 使用者核准後合併各分支
