# module/collision/ — 設計文件

- 日期:2026-09-03
- 狀態:待審查
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第六個子專案**。

## 1. 背景與結論

`module/collision/`(2026-08 新增,11 個檔案、約 2000 行)逐檔審查後,結構本身**不需要調整**:

- 職責分工清楚:`homography.py`(核心運算)、`homography_points.py`(矩陣查表/人工點選載入)、`homography_pick_aid.py`/`homography_validation.py`/`homography_heatmap.py`/`homography_compare_video.py`(校正輔助工具,各自獨立檔案)、`detection_core.py`(TTC/PET 數學)、`trajectory_adapter.py`(讀 Stage1 軌跡轉內部結構)、`output.py`(寫檔)、`__init__.py`(唯一對外入口 `run_collision_detection_for_sequence()`,比照 `module/tracker` 只曝露 `create_tracker()` 的慣例)。
- `__init__.py` 本體(196 行)雖然是整條流程的 orchestrator,但每一步都有清楚的文件字串與座標系警告註記(如 `_resolve_ttc_bounds()` 明確標註衛星圖 vs CCTV 解析度混用的風險),不像 `track_eval.py` 那樣把多種職責硬擠進一個大迴圈。
- `markdown/NEED.md` 沒有這個模組的已知待辦項目。
- 曾評估是否要跟 `module/tracker/matching.py` 共用 IoU 計算(`detection_core.py::rects_iou()` vs `matching.py::iou_distance()`),但兩者用途不同(一個算 TTC/PET 的少量 pair-wise 判斷,一個算 MOT 匹配用的 N×M cost matrix)且刻意不讓 `module/collision` 依賴 `module/tracker` 內部函式——這是合理的模組邊界,不建議合併。

在審查過程中發現一個屬於**前一輪 dataset-path spec 範圍**、但當時清單漏列的項目:`module/collision/__init__.py` 與 `trajectory_adapter.py` 各有一處 `seq_output_dir.stem.replace("annotation_", "")`,是同一類字串前綴轉換問題。已回頭修訂 `2026-09-03-dataset-path-refactor-design.md` 的影響檔案清單補上這兩處,以及三個檔案裡提到 `annotation_<seq>/` 命名慣例的文件字串(純文件更新)。

## 2. 目標 / 非目標

**目標**
- 記錄本輪審查結論:`module/collision/` 不需要結構性重構。
- 把審查中發現的路徑相關殘留項目正確歸戶到 `2026-09-03-dataset-path-refactor-design.md`(已完成)。

**非目標**
- 不處理 homography 真實座標轉換的新需求(使用者已表示這是獨立議題,本輪選擇不處理)。
- 不改變 `run_collision_detection_for_sequence()` 對外介面或任何 TTC/PET 計算邏輯。
- 不合併 `module/collision` 與 `module/tracker` 的幾何計算函式(已評估,結論是維持模組邊界分離)。

## 3. 影響檔案清單

本輪沒有獨立的程式改動——發現的項目已併入 `2026-09-03-dataset-path-refactor-design.md`(見該文件第 3.7 節新增的三列)。

## 4. 驗證計畫

隨 `2026-09-03-dataset-path-refactor-design.md` 實作時一併驗證(見該文件第 5 節),額外針對本輪發現項目:
1. Stage3(碰撞偵測)在 dataset-path 重構落地後,對至少一支序列跑 `config/collision_detect.yaml`,確認 `seq_name` 解析正確(等於 `seq_output_dir.stem`,不再需要 `.replace()`),TTC/PET 事件輸出檔名與內容跟重構前一致。

## 5. 風險總結

- **極低風險**:本輪未提出任何獨立的程式碼變更,只是修正另一份 spec 的遺漏項目。
