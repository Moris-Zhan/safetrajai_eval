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

## 附錄:原 `markdown/` 參考文件(移除前原封不動保留)

以下內容原封不動保留自 `markdown/COLLISION_DETECTION.md`、`markdown/HOMOGRAPHY.md`、`markdown/PET_TTC_OUTPUT_FORMAT.md`、`markdown/reference_code/homography_core.py`、`markdown/reference_code/collision_detection_core.py`、`markdown/reference_code/draw_utils.py`(移除操作見使用者指示,原檔案已刪除)。文中相互參照的檔案連結(如 `[TRAJECTORY_FORMAT.md]`)指向原 `markdown/` 目錄下的相對位置,現已不存在對應檔案,保留原文不做改寫。

### 附錄 A:原 `markdown/COLLISION_DETECTION.md`

# SafeTrajAI 碰撞偵測核心演算法說明

本文件說明 SafeTrajAI 用來偵測交通衝突的兩種代理安全指標(Surrogate Safety
Measures)演算法原理,並提供已萃取為**純運算、無影像/檔案 I/O 依賴**的參考程式碼,方便提供給其他專案介接或改寫。

參考程式碼:[`reference_code/collision_detection_core.py`](./reference_code/collision_detection_core.py)(已可獨立執行,內附範例)
資料格式定義:請先參閱 [TRAJECTORY_FORMAT.md](./TRAJECTORY_FORMAT.md)

## 1. 兩種指標的差異

| | TTC(Time-To-Collision) | PET(Post-Encroachment Time) |
|---|---|---|
| 中文 | 碰撞時間 | 先後通過時間 |
| 核心假設 | 兩物體維持**目前速度**等速前進,是否會在未來**同一時刻**於同一位置重疊 | 兩物體是否曾經(或即將)經過**同一空間位置**,即使**不是同一時刻** |
| 適合偵測的情境 | 兩車正朝彼此逼近、若不煞車/閃避即將發生碰撞 | 兩車路徑交叉但時間點錯開的「有驚無險」情境(例如一車剛通過路口,另一車隨後才到) |
| 需要的資料 | 位置、速度、方向、尺寸(bbox) | 位置、時間(frame) |
| 原始程式 | `utils/coll_pointer/ttc_analyzer.py` (`TTCAnalyzer`) | `utils/coll_pointer/pet_analyzer.py` (`PETAnalyzer`) |

兩者互補:TTC 抓「正在逼近、快要撞上」的動態風險,PET 抓「路徑重疊、時序錯開」的靜態衝突熱點,SafeTrajAI 主流程會對同一批軌跡分別跑多組參數的 TTC 與 PET 分析(見 [Project.md](./Project.md) 第 4.1 節)。

## 2. TTC 演算法邏輯

**核心概念**:等速度線性外推模擬(constant-velocity extrapolation)。

```
對每一對物體 (A, B) 、每個共同出現的 frame:
    取得該 frame 下 A、B 的位置(pos)、速度(vel)、尺寸(size)、方向(dir)
    以時間步進 dt(預設 0.1 秒),模擬 t = 0, dt, 2dt, ... 直到 max_sim_time:
        future_posA = posA + velA * t
        future_posB = posB + velB * t
        計算 future_posA、future_posB 兩個(以中心點表示的)矩形的 IoU
        若 IoU >= iou_threshold:
            視為「預測會在 t 秒後於此處碰撞」
            記錄 碰撞時間、碰撞位置、碰撞類型,停止模擬
    max_sim_time = min(max_time, max_ttc_frames / fps)
```

**碰撞類型判斷**(依兩物體方向向量夾角):

| 夾角範圍 | 類型 |
|---|---|
| 70°~110° | 側撞(Right-angle collision) |
| cos(夾角) > 0.9(接近同向) | 追撞(Rear-end collision) |
| cos(夾角) < -0.9(接近反向) | 對撞(Head-on collision) |
| 其他 | Other |

**關鍵參數**:

| 參數 | 意涵 | SafeTrajAI 主流程預設 |
|---|---|---|
| `max_ttc_frames` | 模擬時間上限(frame 數),即「往未來看多久」 | 迴圈跑 15、30、45、60(見 `Cust_Model_compare_tracking.py`) |
| `fps` | 影片幀率,用於秒數 ⇄ frame 換算 | 依來源影片而定 |
| `iou_threshold` | 判定為「碰撞」所需的邊界框重疊比例 | 0.8(`TTCAnalyzer.compute()` 內呼叫 `_find_ttc` 時指定) |
| `dt` | 模擬時間步進(秒) | 0.1 |
| `bounds` | 有效碰撞區域範圍(height, width),避免模擬結果落在畫面外 | 影片實際解析度 |

**輸入資料需求**:每個 frame 需有位置、速度(`Velocity`)、方向(`Direction`)、尺寸(`Size`)四項欄位,這些在 SafeTrajAI 中由 `utils/csv_utils.py` 的 `get_dir_velocity()` / `get_size()` 從平滑後的座標序列計算而來(中心差分法)。若其他專案的軌跡資料只有座標,需自行先計算這四項欄位(參考中心差分公式:`v = (pos[t+1] - pos[t-1]) / (2/fps)`)。

## 3. PET 演算法邏輯

**核心概念**:在允許的時間窗內,窮舉搜尋兩條軌跡「空間距離最近」的一對時空點。

```
對每一對物體 (A, B):
    取得 A 的完整 (位置, frame) 序列、B 的完整 (位置, frame) 序列
    窮舉所有 (i, j) 組合(i 屬於 A 的序列, j 屬於 B 的序列):
        若 |timeA[i] - timeB[j]| > max_time_diff:  跳過(時間差過大,不視為同一衝突事件)
        計算 distance(posA[i], posB[j])
        若 distance <= overlap_threshold 且為目前找到的最小值:
            記錄為候選最近點
    若找到候選點:
        PET = |timeA[i] - timeB[j]| / fps   # 兩物體通過該最近位置的時間差(秒)
        cross_point = (posA[i] + posB[j]) / 2
```

**關鍵參數**:

| 參數 | 意涵 | SafeTrajAI 主流程預設 |
|---|---|---|
| `max_time_diff`(frame) | 時間窗上限,對應 `lookback_sec * fps` | `lookback_sec` 迴圈跑 2.0~4.5 秒(間隔 0.5),見 `Cust_Model_compare_tracking.py` |
| `overlap_threshold` | 視為「同一空間位置」的距離門檻(座標單位,如像素) | 0.2(`PETAnalyzer._find_conflicts_new()` 內設定;注意此為地圖座標系下的門檻值,依實際座標尺度需自行調整) |
| `fps` | 影片幀率 | 依來源影片而定 |

**與原始程式的差異(重要)**:SafeTrajAI 原始的 `PETAnalyzer` 是為了**即時疊圖顯示**設計的,採用逐 frame 維護 lookback 滑動視窗的方式運算(`_update_track_histories()`),每個 frame 都重新掃描目前視窗內的軌跡。這是針對「邊讀影片邊畫疊圖」場景的效能最佳化寫法。

參考程式碼中的 `compute_pet_for_pair()` 是**數學等價的批次簡化版**:直接對兩條軌跡各自的完整 `(位置, frame)` 陣列做一次全域最近點搜尋(`compute_min_distance_pair()`,與原程式使用同一套核心邏輯,含 numba 加速),省略滑動視窗與即時繪圖的部分。**若目標專案是離線批次處理(已有完整軌跡資料,不需要逐幀即時疊圖),建議直接使用批次版本**,結果與原始逐幀版本一致,程式更簡單。

## 4. 參考程式碼使用方式

```python
from collision_detection_core import (
    compute_ttc_for_pair, compute_ttc_for_all_pairs,
    compute_pet_for_pair, compute_pet_for_all_pairs,
)

# track1, track2 需符合 TRAJECTORY_FORMAT.md 第 4 節「層次 2」格式
# (至少含 Track ID / Positions or Homo_Positions / Bboxs / Velocity / Size / Direction)

ttc_events = compute_ttc_for_pair(
    track1, track2,
    fps=30, max_ttc_frames=60, bounds=(frame_height, frame_width),
    use_homography=True,   # True: 用地圖座標系(Homo_*); False: 用原始影像座標系
    iou_threshold=0.8,
)

pet_event = compute_pet_for_pair(
    track1, track2,
    fps=30, max_time_diff_frames=90, overlap_threshold=0.2,
    use_homography=True,
)

# 整批軌跡兩兩配對(對應原始 Cust_Model_compare_tracking.py 的用法)
all_ttc = compute_ttc_for_all_pairs(tracks, fps=30, max_ttc_frames=60, bounds=(h, w))
all_pet = compute_pet_for_all_pairs(tracks, fps=30, max_time_diff_frames=90, overlap_threshold=0.2)
```

檔案內建 `if __name__ == "__main__":` 範例(兩組合成軌跡,一組示範 PET 情境、一組示範 TTC 情境),可直接執行驗證:

```bash
python3 collision_detection_core.py
```

## 5. 已知限制(與原始演算法相同,非本次萃取造成)

- **TTC 為等速度假設**:未考慮加速度、轉向、煞車等真實駕駛行為,實際碰撞風險可能被低估或高估。
- **座標單位需自行對齊**:`overlap_threshold`(PET)、`iou_threshold`(TTC)等門檻值是在特定座標尺度(像素)下調校的經驗值,套用到不同解析度的影像或不同比例尺的地圖座標時,需重新調整門檻。
- **未考慮物體實際碰撞外形**:以矩形邊界框近似物體形狀,對於行人、機車等狹長/不規則外形物體精確度有限。
- **PET 的 `overlap_threshold` 與 TTC 的 `iou_threshold` 是不同性質的門檻**,前者是距離門檻、後者是重疊比例門檻,介接時不要混用參數。

### 附錄 B:原 `markdown/HOMOGRAPHY.md`

# SafeTrajAI Homography 座標轉換說明

本文件說明 SafeTrajAI 如何計算並套用 Homography(單應性變換)矩陣,將 CCTV 影像座標轉換為地圖底圖(鳥瞰視角)座標,並提供已萃取為**純運算、無 GUI 依賴**的參考程式碼,方便提供給其他專案介接或改寫。

參考程式碼:[`reference_code/homography_core.py`](./reference_code/homography_core.py)(已可獨立執行,內附範例,已用實際 opencv 環境驗證)
下游用途說明:座標轉換後的資料格式請見 [TRAJECTORY_FORMAT.md](./TRAJECTORY_FORMAT.md);PET/TTC 計算如何使用轉換後座標見 [COLLISION_DETECTION.md](./COLLISION_DETECTION.md)

## 1. 重要前提:專案內其實有兩套工具,只有一套實際生效

盤點 `utils/map_utils.py` 後發現有兩個功能相似的類別,**容易混淆**,先釐清:

| 類別 | 輸出路徑 | 是否被主流程使用 |
|---|---|---|
| `HomographyMapping` | `homography_data/homography_matrix.npy` | **✅ 是**——所有 `utils/tracker/**/*_analyzer.py` 的 `VideoAnalyzer.__init__()` 皆寫死讀取此路徑(`np.load('homography_data/homography_matrix.npy')`) |
| `HomographyEditor` | `homography_result/matrix.npy` | ❌ 否——專案內除了它自己,沒有任何程式碼讀取這個路徑,是獨立於主流程外、未被接入的工具(疑似單張影像「校正成俯視矩形」的早期實驗,非「CCTV↔地圖對齊」用途) |

本文件與參考程式碼萃取的是 **`HomographyMapping`** 這一套的核心邏輯。

## 2. 計算流程

### 步驟一:人工點選對應點

對 CCTV 影像與地圖衛星底圖(`utils/map_utils.py` 的 `get_map()` 下載,見 [Project.md](./Project.md#45-座標與地圖) )**分別**開一個 OpenCV 滑鼠互動視窗,人工依序點選 **4 個對應點**(例如路口的四個角、或四個明顯地標),兩邊點的**順序必須一一對應**。點選時已依顯示縮放比例還原回原始解析度座標(`x_real = x / scale`),分別存成 JSON:

```
homography_data/cctv_points.json   # CCTV 影像上的 4 個點
homography_data/map_points.json    # 地圖底圖上對應的 4 個點(順序需與 CCTV 點一一對應)
```

原始程式:`HomographyMapping.select_points()` → `_select_on_image()`(`utils/map_utils.py`)。此步驟為互動 GUI 操作,無法自動化,也是整條轉換鏈中**唯一需要人工介入**的環節。

### 步驟二:計算 Homography 矩陣

```python
src_pts = np.array(self.cctv_pts, dtype=np.float32)  # CCTV 4 點
dst_pts = np.array(self.map_pts, dtype=np.float32)   # 地圖 4 點
self.H, _ = cv2.findHomography(src_pts, dst_pts)
np.save("homography_data/homography_matrix.npy", self.H)
```

原始程式:`HomographyMapping.compute_homography()`。

**⚠️ 關鍵限制**:原始程式固定只用 **4 組點**,`cv2.findHomography()` 在只有 4 組點時是「恰好可解」的 DLT(Direct Linear Transform)——8 個自由度、4 個點提供剛好 8 個方程式,**沒有 RANSAC 或最小平方容錯機制**。因此點選精準度會直接、線性地反映在所有下游座標(PET/TTC 衝突點位置)的精準度上,沒有多餘點可以互相校正誤差。

### 步驟三:套用矩陣轉換座標

矩陣算出後,在 `utils/csv_utils.py` 用於轉換軌跡資料:

```python
def transform_point(x, y, H):
    point = np.array([x, y, 1.0])   # 轉齊次座標
    transformed = H @ point         # 3x3 矩陣乘法
    transformed /= transformed[2]   # 除以 w,回到 2D 座標
    return transformed[0], transformed[1]
```

- **單點**(如物體中心點):用上述 `transform_point()`,矩陣乘法 + 齊次座標歸一化。
- **邊界框(bbox)**:不能只轉換左上、右下兩點再連直線——透視變形下矩形會變成不規則四邊形,必須用 `cv2.perspectiveTransform()` 對四個角點分別轉換(`transform_box()`),才能得到正確形狀。SafeTrajAI 對 bbox 保留兩種轉換結果:`Homo_Bboxs`(簡化版,僅轉換左上右下兩點)與 `Homo_Bboxs_2`(精確版,四角個別轉換的四邊形),詳見 [TRAJECTORY_FORMAT.md 第 4 節](./TRAJECTORY_FORMAT.md#4-層次-2後處理單軌跡格式procTrack_rowpkl)。

### 步驟四(選用):反向轉換

PET/TTC 分析在地圖座標系下算出衝突點後,若要疊圖標註回原始 CCTV 影像,需要反向矩陣:

```python
H_inv = np.linalg.inv(H)
cctv_point = transform_point(*map_point, H_inv)  # 或用 cv2.perspectiveTransform
```

原始程式:`PETAnalyzer.__init__` 的 `self.H_inv = np.linalg.inv(loaded_H_matrix)`,以及 `TTCAnalyzer.draw_collision_overlay()` 用 `H_inv` 把地圖座標系算出的碰撞點轉換回 CCTV 影像座標系疊圖標註。

### 步驟五(選用):目視驗證

```python
warped = cv2.warpPerspective(cctv_img, H, (map_w, map_h))       # 將 CCTV 影像整張攤平貼到地圖座標系
overlay = cv2.addWeighted(map_img, 0.5, warped, 0.5, 0)          # 與地圖底圖疊合
```

原始程式:`HomographyMapping.load_and_test_homography()` / `warp_and_show()`,用 `cv2.imshow()` 顯示疊圖供人工目視確認對齊精準度,輸出 `homography_data/overlay_orign.jpg`。

## 3. 資料流向總覽

```
CCTV 影像截圖              地圖衛星底圖(Google Maps Static API)
     │                            │
     └──────────┬─────────────────┘
                 │ 人工各點選 4 個對應點(GUI)
                 ▼
     cctv_points.json      map_points.json
                 │                │
                 └───────┬────────┘
                          │ cv2.findHomography(cctv_pts, map_pts)
                          ▼
              homography_data/homography_matrix.npy
                          │
                          │ VideoAnalyzer.__init__() 讀取(np.load)
                          ▼
          套用於整支影片的所有軌跡座標轉換
     (Positions -> Homo_Positions, Bboxs -> Homo_Bboxs/_2)
                          │
                          ▼
              PET / TTC 分析(於地圖座標系下運算)
                          │
                          │ 需要疊圖標註時,用 H_inv 反投影
                          ▼
                  回 CCTV 影像座標系
```

## 4. 參考程式碼使用方式

[`reference_code/homography_core.py`](./reference_code/homography_core.py) 已將計算/轉換邏輯抽離成不依賴 GUI 的純函式(互動選點的 OpenCV 滑鼠事件與目標專案的標註工具高度相關,不易通用,故未納入,由使用端自行以任何方式取得 4 組以上對應點座標即可):

```python
from homography_core import (
    compute_homography, save_homography, load_homography,
    transform_point, transform_points_batch, transform_box,
    transform_positions_and_bboxes, invert_homography,
    warp_and_overlay,
)

# 1. 計算矩陣(cctv_pts / map_pts 為對應點座標 list,至少 4 組、順序一一對應)
H = compute_homography(cctv_pts, map_pts)
save_homography(H, "homography_data/homography_matrix.npy")

# 2. 載入矩陣、轉換座標
H = load_homography("homography_data/homography_matrix.npy")
map_x, map_y = transform_point(cctv_x, cctv_y, H)

# 3. 反向轉換(地圖座標 -> CCTV 座標)
H_inv = invert_homography(H)
cctv_x, cctv_y = transform_point(map_x, map_y, H_inv)

# 4. 批次轉換一條軌跡的 Positions/Bboxs(對齊 TRAJECTORY_FORMAT.md 層次2格式)
new_positions, new_bboxes, new_bboxes_polygon = transform_positions_and_bboxes(positions, bboxes, H)

# 5. 目視驗證對齊精準度(回傳影像陣列,不強制開視窗,適合無 GUI 環境)
warped, overlay = warp_and_overlay(cctv_img, map_img, H)
cv2.imwrite("overlay_check.jpg", overlay)
```

檔案內建 `if __name__ == "__main__":` 範例,已用實際 opencv 環境執行驗證:計算矩陣 → 單點轉換 → 反向轉換還原(誤差 < 1e-10,確認矩陣運算正確)→ bbox 四邊形轉換 → 批次軌跡轉換,皆正確運作:

```bash
python3 homography_core.py
```

## 5. 已知限制(與原始演算法相同,非本次萃取造成)

- **4 點 DLT 無容錯**:如第 2 節所述,沒有多餘點做最小平方或 RANSAC,人工點選誤差無法被平均掉。若其他專案需要更穩健的結果,可在 `compute_homography()` 呼叫 `cv2.findHomography()` 時提供 4 組以上的點並加上 `method=cv2.RANSAC`(參考程式碼中已在函式註解說明,但為保留原始行為預設未啟用)。
- **矩陣是全域單一檔案,非依影片/路口區分**:`homography_data/homography_matrix.npy` 是寫死的單一路徑,換路口分析前必須重新跑一次點選流程覆蓋此檔案,否則會沿用上一個路口的矩陣(這也代表:**若要批次處理多個不同路口的影片,現有程式碼架構無法直接支援,需自行擴充為依影片/路口分別存取對應矩陣檔**)。
- **平面近似假設**:Homography 是 2D 到 2D 的透視對應關係,假設地面近似平面。對高度變化大的場景(高架橋、坡道、明顯起伏路面)精準度會下降,物體離地面越高(如卡車車頂)、其影像座標經轉換後對應到地圖座標的誤差也會越大(因為實際取的是物體中心點/bbox 座標,並非真正貼地的點)。
- **並非真實世界座標**:轉換後仍是地圖底圖的**像素座標**,不是 GPS 經緯度。若需要經緯度或公尺制距離,需另外知道地圖底圖的比例尺(依 Google Maps Static API 的 `zoom` 等級與 `scale` 參數換算)才能進一步推算,SafeTrajAI 目前未提供此換算。

### 附錄 C:原 `markdown/PET_TTC_OUTPUT_FORMAT.md`

# SafeTrajAI PET / TTC 輸出結果資料格式說明

本文件說明 PET(Post-Encroachment Time)與 TTC(Time-To-Collision)分析完成後,**輸出的衝突事件資料格式**,包含程式執行時記憶體中的資料結構,以及落地儲存的檔案格式。

- 輸入軌跡資料格式請見 [TRAJECTORY_FORMAT.md](./TRAJECTORY_FORMAT.md)
- 演算法邏輯與純運算參考程式碼請見 [COLLISION_DETECTION.md](./COLLISION_DETECTION.md)

> ⚠️ **先說結論**:PET 與 TTC 兩者的原始輸出格式**並不對稱**——TTC 有結構化的逐筆事件 CSV,PET 沒有;且兩者部分欄位命名相同但語意不同(詳見第 5 節)。若要提供給另一個專案做**機器可讀**的介接,建議採用 [COLLISION_DETECTION.md](./COLLISION_DETECTION.md) 參考程式碼 `compute_ttc_for_pair()` / `compute_pet_for_pair()` 的回傳格式作為統一標準,而非直接解析 SafeTrajAI 現有的 log/CSV 檔案。

## 1. 輸出目錄結構

每次呼叫 `run_ttc_analysis(max_ttc_frames, ...)` / `run_pet_analysis(lookback_sec, ...)`(見 `utils/tracker/custom/deepsort_analyzer.py`),會產生一組獨立的輸出目錄:

```
analysis_{model_name}/exp_{video_name}/{DeepSORT|BoT-SORT}/trace/
├── TTC_analysis/TTC_{max_ttc_frames}/
│   ├── ttc_collision_heatmap_{video_name}.png   # 碰撞熱區圖
│   ├── ttc.log                                   # 人工可讀文字紀錄(每筆事件一行)
│   ├── frame_processing_stats.csv                # ⚠️ 效能統計(逐幀運算耗時),非碰撞事件資料
│   ├── collision_ttc_log.csv  或                  # ★ 結構化逐筆 TTC 事件紀錄(詳見第 4 節)
│   │   collision_ttc_homo_log.csv                #   （檔名依是否有 Homography 反轉矩陣而定）
│   ├── out_trace.mp4                              # CCTV 視角完整軌跡疊圖影片
│   ├── out_gle_trace.mp4                          # 地圖視角完整軌跡疊圖影片
│   ├── coll_ttc_{video_name}/frame_{frame_id}.jpg      # 發生 TTC 事件當幀的 CCTV 視角截圖
│   └── coll_ttc_gle_{video_name}/frame_{frame_id}.jpg  # 發生 TTC 事件當幀的地圖視角截圖
│
└── PET_analysis/PET_{lookback_sec}/
    ├── pet_collision_heatmap_{video_name}.png    # 碰撞熱區圖
    ├── pet.log                                    # 人工可讀文字紀錄(每筆事件一行)
    ├── frame_processing_stats.csv                 # ⚠️ 效能統計,非碰撞事件資料
    ├── out_trace.mp4                              # CCTV 視角完整軌跡疊圖影片
    ├── out_gle_trace.mp4                          # 地圖視角完整軌跡疊圖影片
    ├── conflict_area/history_{frame_id}.jpg       # 發生 PET 衝突當幀的 CCTV 視角截圖
    └── conflict_area_gle/history_{frame_id}.jpg   # 發生 PET 衝突當幀的地圖視角截圖
    # 注意：PET 沒有對應 collision_ttc_log.csv 的結構化逐筆事件 CSV
```

## 2. TTC 事件資料結構

### 2.1 記憶體中的原始結構(`TTCAnalyzer.ttc_collision_point`)

型別為 `defaultdict(list)`,**key 是偵測當下的「現在幀」`curr_frame`**(不是預估碰撞發生的幀,務必區分),value 是該幀偵測到的所有事件:

```python
ttc_collision_point: Dict[int, List[dict]]

# 每筆事件:
{
    "ID1": track_id,                  # 車輛1 Track ID
    "ID2": track_id,                  # 車輛2 Track ID
    "colision_occur": int,            # 【預估】碰撞發生的絕對 frame = curr_frame + ttc_frames
    "colision_occur_sec": float,      # 【預估】碰撞發生的絕對秒數 = curr_frame/fps + ttc_time
    "dur_times": float,               # 從 curr_frame 到碰撞所需秒數(即 TTC 值本身)
    "coll_type": str,                 # "Right-angle collision" / "Rear-end collision" / "Head-on collision" / "Other"
    "colision_point": (x, y),         # 碰撞位置(兩車外推後邊界框中心點的中點)
    "future_box1": ((x1,y1),(x2,y2)), # 車輛1 碰撞當下的外推邊界框
    "future_box2": ((x1,y1),(x2,y2)), # 車輛2 碰撞當下的外推邊界框
    "arrow1": ((x,y),(x,y)),          # 車輛1 從目前位置到碰撞位置的位移(起點, 終點)
    "arrow2": ((x,y),(x,y)),          # 車輛2 同上
    "text": str,                      # 顯示用標籤，格式為 "ttc@{ttc_time:.2f}"
}
```

**同一對車輛(依 `sorted((ID1, ID2))` 判重)只會記錄一次**,找到第一筆滿足碰撞條件的事件後即停止該配對的後續比對(`checked_pairs` 機制)。

### 2.2 座標系統

`colision_point`、`future_box*`、`arrow*` 所使用的座標系統,取決於呼叫 `TTCAnalyzer` 時的 `with_homo` 參數:
- `with_homo=True`(SafeTrajAI 主流程預設):使用 `Homo_Positions` 等欄位,座標為**地圖底圖像素座標系**
- `with_homo=False`:使用原始影像像素座標系

與 [TRAJECTORY_FORMAT.md 第 2 節](./TRAJECTORY_FORMAT.md#2-座標系統務必留意) 的座標系統定義一致,**皆非 GPS 經緯度或公尺制**。

## 3. PET 事件資料結構

### 3.1 記憶體中的原始結構(`PETAnalyzer.collision_points`)

型別為 `defaultdict(list)`,**key 是 `min(t1, t2)`**(兩車通過最近點的兩個 frame 中較早的那個):

```python
collision_points: Dict[int, List[dict]]

# 每筆事件:
{
    "track_id1": track_id,
    "track_id2": track_id,
    "cross_pt": (x, y),        # 兩車最近點的中點座標(int)
    "pet_time": float,         # PET 值(秒) = |t2 - t1| / fps
    "t1": int,                 # 車輛1 通過最近點的 frame
    "t2": int,                 # 車輛2 通過最近點的 frame
    "distance": float,         # 兩車最近點之間的實際距離(四捨五入到小數點後 2 位)
}
```

### 3.2 格式化後結構(`PETAnalyzer.pet_collision_point`,由 `_format_pet_collision_points()` 產生)

為了與 TTC 事件共用同一套熱區圖繪製函式(`draw_collision_heatmap()`),PET 事件會被轉換成與 TTC 相似的欄位名稱:

```python
pet_collision_point: Dict[int, List[dict]]   # key 與 collision_points 相同(min(t1, t2))

# 每筆事件:
{
    "ID1": track_id1,
    "ID2": track_id2,
    "colision_point": cross_pt,       # 同 collision_points 的 cross_pt
    "dur_times": pet_time,            # 同 collision_points 的 pet_time(秒)
    "colision_occur_sec": t2,         # ⚠️ 見下方「欄位命名不一致」說明
    "text": str,                      # 顯示用標籤，格式為 "pet@{pet_time:.2f}"
}
```

### 3.3 座標系統

同 TTC,取決於 `PETAnalyzer` 建構時的 `with_homo` 參數(SafeTrajAI 主流程預設 `True`,使用地圖底圖像素座標系)。

## 4. TTC 結構化 CSV(`collision_ttc_log.csv` / `collision_ttc_homo_log.csv`)

僅 TTC 有此結構化逐筆事件 CSV(由 `TTCAnalyzer.draw_collision_overlay()` 在 `process_video()` 逐幀輸出影片時同步寫入,PET 沒有對應機制)。欄位為中文表頭,每次執行**以附加(append)模式寫入**,若重跑分析且未清空輸出目錄,舊資料不會被覆蓋而是持續累加:

| CSV 欄位(中文表頭) | 對應資料 | 說明 |
|---|---|---|
| `FrameID` | `frame_id` | 觸發此筆事件時,`process_video()` 播放到的 frame(即偵測當下的「現在幀」,對應第 2.1 節的 `curr_frame`) |
| `Video` | `self.video_name` | 影片名稱 |
| `車輛1` | `pt["ID1"]` | 車輛1 Track ID |
| `車輛2` | `pt["ID2"]` | 車輛2 Track ID |
| `預估碰撞時間點` | `pt["colision_occur_sec"]` | 預估碰撞發生的**絕對秒數**(相對影片起點) |
| `TTC碰撞時長` | `round(pt["dur_times"], 2)` | 從觸發當下到預估碰撞所需秒數(TTC 值本身) |
| `碰撞點` | `pt["colision_point"]` | 碰撞位置座標 `(x, y)` |
| `碰撞類型` | `pt["coll_type"]` | 碰撞類型分類 |
| `TTC標籤[TTC@dur_times]` | `pt["text"]` | 顯示用標籤字串 |
| `指標` | 固定字串 `"TTC"` | 用於區分指標類型(此檔案恆為 `"TTC"`) |

檔名 `collision_ttc_log.csv` 或 `collision_ttc_homo_log.csv` 取決於是否有提供 Homography 反轉矩陣(`H_matrix`);SafeTrajAI 主流程一律提供,故實務上產生的檔名固定為 **`collision_ttc_homo_log.csv`**。

## 5. ⚠️ 欄位命名不一致(跨專案介接前務必留意)

整理原始程式碼時發現以下命名可能造成混淆的地方,提供給對方專案參考,避免依欄位名稱誤判語意:

1. **`colision_occur_sec` 在 TTC 與 PET 中語意不同**:
   - TTC(`TTCAnalyzer.compute()`):`colision_occur_sec = curr_frame / fps + ttc_time`,**確實是秒數**(絕對時間)。
   - PET(`PETAnalyzer._format_pet_collision_points()`):`colision_occur_sec = coll_data['t2']`,**直接指派為 frame index(整數幀數),並未除以 fps**,實際上不是秒數。這是原始程式碼的命名遺留問題(可能是沿用 TTC 的欄位命名格式但漏了單位轉換),使用 PET 輸出時請改用 `t1`/`t2`(原始結構)或自行以 `t2 / fps` 換算為真正的秒數,不要直接信任 `colision_occur_sec` 這個欄位名稱在 PET 資料中的字面意義。
2. **`dur_times` 語意在 TTC/PET 中一致**(皆為秒數:TTC 是「多久後會撞」、PET 是「兩車通過同一點的時間差」),可放心視為同義欄位。
3. **`ID1`/`ID2` 命名一致**,可放心對齊。
4. **`colision_point` 拼字為 "colision"(少一個 l)**,非標準英文拼法,是原始程式碼既有寫法,並非本文件筆誤,跨專案定義欄位時建議自行改用正確拼字(`collision_point`)。

## 6. 跨專案介接建議

- **若對方需要逐筆事件的結構化資料(CSV/JSON)**:
  - TTC 可直接使用 `collision_ttc_homo_log.csv`,但建議先依第 5 節修正/確認欄位語意,且需自行處理「附加寫入、可能包含多次執行結果」的狀況(建議先確認 `FrameID` + `車輛1` + `車輛2` 組合是否重複,或直接以乾淨輸出目錄重跑一次)。
  - PET 目前**沒有**對應的結構化 CSV,建議改用 [COLLISION_DETECTION.md](./COLLISION_DETECTION.md) 的 `compute_pet_for_pair()` / `compute_pet_for_all_pairs()`,直接取得乾淨的 dict/list 結果自行序列化為 CSV 或 JSON,欄位命名也已在參考程式碼中統一修正(如 `pet_seconds` 確實是秒數、`cross_point`/`collision_type` 等皆為正確拼字)。
- **若對方只需要熱區分布,不需逐筆事件明細**:可直接使用 `*_collision_heatmap_{video_name}.png`,但此為視覺化圖片,非結構化資料。
- **若對方需要人工複查佐證畫面**:可使用 `coll_ttc_{video_name}/` / `conflict_area/` 等逐幀截圖目錄,檔名 `frame_{frame_id}.jpg` 或 `history_{frame_id}.jpg` 可對應回結構化資料中的 frame 欄位。
- **建議統一採用 [COLLISION_DETECTION.md](./COLLISION_DETECTION.md) 參考程式碼的輸出格式作為跨專案交換標準**,原因:(1) TTC/PET 欄位命名已統一且修正拼字與單位問題,(2) 兩者輸出結構對稱一致,(3) 不含只有 SafeTrajAI 內部繪圖流程才需要的欄位(如 `future_box`、`arrows`),交換資料更精簡。

### 附錄 D:原 `markdown/reference_code/homography_core.py`

```python
"""
SafeTrajAI Homography 座標轉換核心運算（純運算版，供跨專案參考）

本檔案從 SafeTrajAI 原始程式碼中萃取出 Homography（單應性變換）矩陣的計算、
儲存/載入與座標轉換邏輯。原始程式碼中，點位挑選是透過 OpenCV 滑鼠互動視窗
完成的（見 utils/map_utils.py 的 HomographyMapping._select_on_image()），
此互動 UI 部分與目標專案的標註工具高度相關、不易通用，因此本檔案**不包含**
互動選點的 GUI 程式碼，只保留「輸入 4 組以上對應點座標 -> 計算矩陣 -> 轉換
座標」這段與 UI 無關的核心運算，方便其他專案改寫成自己的標註流程。

原始程式位置：
- utils/map_utils.py 的 class HomographyMapping（實際被主流程使用者）
- utils/csv_utils.py 的 transform_point() / transform_box() / transform_pos_bbox()
- utils/coll_pointer/ttc_analyzer.py 內對 H_inv 的使用（碰撞點反投影回 CCTV 視角）

## 重要提醒（來自原始碼盤點）

SafeTrajAI 的 utils/map_utils.py 內其實有兩個類似功能的類別：
- `HomographyMapping`：**主流程實際使用**，計算結果存於
  `homography_data/homography_matrix.npy`，所有 `*_analyzer.py` 皆從此
  路徑讀取矩陣。
- `HomographyEditor`：獨立於主流程之外，輸出到 `homography_result/matrix.npy`，
  但專案內沒有任何地方讀取這個路徑，屬於未接入主流程的工具。

本檔案萃取的是 `HomographyMapping` 這一套的核心運算邏輯。

## 相依套件

- numpy（必要）
- opencv-python（cv2，必要，用於 findHomography / perspectiveTransform / warpPerspective）
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


# =============================================================================
# 1. 計算 Homography 矩陣
# =============================================================================

def compute_homography(src_pts: Sequence[Point], dst_pts: Sequence[Point]) -> np.ndarray:
    """由來源影像（如 CCTV）與目標影像（如地圖底圖）上的對應點，計算 3x3 Homography 矩陣。

    對應原始程式 HomographyMapping.compute_homography()。

    Args:
        src_pts: 來源影像上的對應點座標，至少 4 組，如 [(x1,y1), (x2,y2), ...]
        dst_pts: 目標影像上的對應點座標，需與 src_pts 一一對應、順序相同

    Returns:
        3x3 numpy array，將 src 座標系轉換到 dst 座標系的 Homography 矩陣

    Note:
        - 原始程式固定使用 4 組點,`cv2.findHomography()` 在僅有 4 組點時
          會做「恰好可解」的 DLT 運算,沒有 RANSAC 容錯機制,點位精準度會
          直接反映在轉換結果上。
        - 若提供 4 組以上的點,`cv2.findHomography()` 預設會改用最小平方法,
          建議搭配 `method=cv2.RANSAC` 以提升對雜訊點的容錯度(原始程式未使用,
          此處保留原始行為為預設,如需更穩健的結果可自行加上
          `method=cv2.RANSAC, ransacReprojThreshold=3.0` 等參數)。
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    if len(src) < 4 or len(dst) < 4:
        raise ValueError("Homography 計算至少需要 4 組對應點")
    if len(src) != len(dst):
        raise ValueError("src_pts 與 dst_pts 的點數必須相同且一一對應")

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Homography 計算失敗，請確認提供的對應點是否共線或重複")
    return H


def save_homography(H: np.ndarray, path: str) -> None:
    """儲存 Homography 矩陣為 .npy 檔案（對應原始 homography_data/homography_matrix.npy）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, H)


def load_homography(path: str) -> np.ndarray:
    """載入 Homography 矩陣（對應原始 VideoAnalyzer 內
    `np.load('homography_data/homography_matrix.npy')` 的讀取方式）。
    """
    return np.load(path, allow_pickle=True)


# =============================================================================
# 2. 套用 Homography 轉換座標
# =============================================================================

def transform_point(x: float, y: float, H: np.ndarray) -> Point:
    """使用 Homography 矩陣轉換單一點座標。

    對應原始程式 utils/csv_utils.py 的 transform_point()。
    """
    point = np.array([x, y, 1.0])
    transformed = H @ point
    transformed /= transformed[2]  # 齊次座標歸一化
    return float(transformed[0]), float(transformed[1])


def transform_points_batch(points: Sequence[Point], H: np.ndarray) -> List[Point]:
    """向量化版本，一次轉換多個點（比逐點呼叫 transform_point() 更有效率）。

    內部使用 cv2.perspectiveTransform，數學上與 transform_point() 逐點運算
    結果相同。
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return [(float(p[0][0]), float(p[0][1])) for p in transformed]


def transform_box(top_left: Point, bottom_right: Point, H: np.ndarray) -> List[List[List[float]]]:
    """將 bounding box 的四個角點分別轉換，回傳透視變形後的四邊形（非矩形）。

    對應原始程式 utils/csv_utils.py 的 transform_box()。由於透視變換不保持
    矩形形狀，僅轉換左上/右下兩點再連直線會失真，因此需將四個角點分別轉換。
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    box_pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    transformed_pts = cv2.perspectiveTransform(box_pts, H)
    return transformed_pts.tolist()


def transform_positions_and_bboxes(
    positions: Dict[int, Tuple[float, float, float]],
    bboxes: Dict[int, Tuple[Point, Point]],
    H: np.ndarray,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[Point, Point]], Dict[int, list]]:
    """批次轉換一條軌跡的 Positions / Bboxs（對應 TRAJECTORY_FORMAT.md 層次 2 格式）。

    對應原始程式 utils/csv_utils.py 的 transform_pos_bbox()。

    Args:
        positions: {frame: (x, y, z)}，z 不會被轉換，原值保留
        bboxes: {frame: ((x1,y1), (x2,y2))}
        H: 3x3 Homography 矩陣

    Returns:
        (new_positions, new_bboxes, new_bboxes_polygon)
            new_positions: {frame: (x', y', z)}
            new_bboxes: {frame: ((x1',y1'), (x2',y2'))}  # 僅轉換左上右下兩點(簡化版，不考慮透視變形)
            new_bboxes_polygon: {frame: [[x,y],[x,y],[x,y],[x,y]]}  # 四角分別轉換後的四邊形(精確版)
    """
    new_positions = {}
    for frame, (x, y, z) in positions.items():
        new_x, new_y = transform_point(x, y, H)
        new_positions[frame] = (new_x, new_y, z)

    new_bboxes = {}
    new_bboxes_polygon = {}
    for frame, (pt1, pt2) in bboxes.items():
        new_x1, new_y1 = transform_point(pt1[0], pt1[1], H)
        new_x2, new_y2 = transform_point(pt2[0], pt2[1], H)
        new_bboxes[frame] = ((new_x1, new_y1), (new_x2, new_y2))
        new_bboxes_polygon[frame] = transform_box(pt1, pt2, H)[0]

    return new_positions, new_bboxes, new_bboxes_polygon


def invert_homography(H: np.ndarray) -> np.ndarray:
    """回傳反向矩陣，用於將目標座標系（如地圖）的點反投影回來源座標系（如 CCTV 影像）。

    對應原始程式 PETAnalyzer.__init__ 內的 `self.H_inv = np.linalg.inv(loaded_H_matrix)`，
    以及 TTCAnalyzer.draw_collision_overlay() 內用 H_inv 將地圖座標系下算出的碰撞點
    轉換回 CCTV 影像座標系以便疊圖標註。
    """
    return np.linalg.inv(H)


# =============================================================================
# 3. 驗證工具（無 GUI 版本）
# =============================================================================

def warp_and_overlay(
    src_img: np.ndarray, dst_img: np.ndarray, H: np.ndarray, alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """將來源影像依 H 轉換到目標影像座標系，並與目標影像疊合，用於人工目視檢查對齊精準度。

    對應原始程式 HomographyMapping.load_and_test_homography() / warp_and_show()，
    差異是本函式**不呼叫 cv2.imshow()**（不需要顯示視窗，可在無 GUI 的伺服器/批次
    環境下執行),改為直接回傳影像陣列,由呼叫端自行決定要顯示、儲存或進一步處理。

    Args:
        src_img: 來源影像（如 CCTV 畫面），numpy array (H, W, 3)
        dst_img: 目標影像（如地圖底圖），numpy array (H, W, 3)
        H: 3x3 Homography 矩陣（src -> dst 方向）
        alpha: 疊合透明度，0.5 表示來源/目標影像各佔一半權重

    Returns:
        (warped, overlay)：
            warped: src_img 依 H 轉換到 dst_img 座標系與尺寸後的結果
            overlay: warped 與 dst_img 依 alpha 疊合後的結果，可用於目視比對是否對齊
    """
    h, w = dst_img.shape[:2]
    warped = cv2.warpPerspective(src_img, H, (w, h))
    overlay = cv2.addWeighted(dst_img, 1 - alpha, warped, alpha, 0)
    return warped, overlay


# =============================================================================
# 使用範例
# =============================================================================
if __name__ == "__main__":
    # 建立一組已知關係的示意對應點：CCTV 影像上的四邊形 -> 地圖底圖上的矩形
    # （相當於使用者在兩張影像上分別點選 4 個對應點後得到的座標）
    cctv_pts = [(338, 156), (602, 244), (315, 449), (35, 230)]   # CCTV 上一個透視變形的四邊形
    map_pts = [(0, 0), (300, 0), (300, 300), (0, 300)]           # 地圖上對應的矩形四角

    H = compute_homography(cctv_pts, map_pts)
    print("Homography 矩陣:\n", H)

    # 單點轉換範例：CCTV 上的一個追蹤點座標，轉換到地圖座標系
    cctv_point = (400, 250)
    map_point = transform_point(*cctv_point, H)
    print(f"CCTV 座標 {cctv_point} -> 地圖座標 {map_point}")

    # 反向轉換（地圖座標 -> CCTV 座標），驗證是否可還原
    H_inv = invert_homography(H)
    back_point = transform_point(*map_point, H_inv)
    print(f"地圖座標 {map_point} -> 反投影回 CCTV 座標 {back_point} (應接近原始 {cctv_point})")

    # bbox 轉換範例
    bbox_top_left, bbox_bottom_right = (380, 200), (420, 280)
    polygon = transform_box(bbox_top_left, bbox_bottom_right, H)
    print(f"bbox {bbox_top_left, bbox_bottom_right} -> 地圖座標系四邊形 {polygon}")

    # 批次軌跡轉換範例
    positions = {0: (400, 250, 0.0), 1: (405, 248, 0.0)}
    bboxes = {0: ((380, 200), (420, 280)), 1: ((385, 198), (425, 278))}
    new_pos, new_bbox, new_bbox_poly = transform_positions_and_bboxes(positions, bboxes, H)
    print("轉換後 Positions:", new_pos)
    print("轉換後 Bboxs(簡化版):", new_bbox)
```

### 附錄 E:原 `markdown/reference_code/collision_detection_core.py`

```python
"""
SafeTrajAI 碰撞偵測核心演算法（純運算版，供跨專案參考）

本檔案是從 SafeTrajAI 專案原始程式碼中萃取出的核心運算邏輯，已移除所有
cv2 繪圖、影片讀寫、檔案輸出等與影像流程綁定的部分，只保留純數學/資料
運算，方便其他專案直接參考或改寫。

包含兩種交通衝突代理安全指標（Surrogate Safety Measures):

1. TTC（Time-To-Collision，碰撞時間）
   對兩個物體目前的位置與速度做「等速度線性外推」，模擬未來一段時間內
   是否會發生邊界框重疊（IoU 超過門檻），若會，回傳预估碰撞所需時間、
   碰撞位置與碰撞類型（追撞／對撞／側撞）。

   原始程式位置：utils/coll_pointer/ttc_analyzer.py（class TTCAnalyzer）

2. PET（Post-Encroachment Time，先後通過時間）
   在允許的最大時間差（時間窗）內，尋找兩條軌跡「空間距離最近」的一對
   時空點；若該最近距離小於重疊門檻，代表兩物體先後通過了幾乎同一個
   空間位置，兩者的時間差即為 PET。

   原始程式位置：utils/coll_pointer/pet_analyzer.py（class PETAnalyzer，
   核心比對函式 find_conflict_area_by_center_distance_v2 /
   compute_min_distance_pair）

## 資料格式假設

track1 / track2 為單一物件的軌跡 dict，欄位語意與 SafeTrajAI
`markdown/TRAJECTORY_FORMAT.md` 第 4 節「層次 2：後處理單軌跡格式」一致，
至少需包含：

    {
        "Track ID": int | str,
        "Positions" 或 "Homo_Positions": {frame_id(int): (x, y, z), ...},
        "Bboxs"     或 "Homo_Bboxs":     {frame_id(int): ((x1,y1),(x2,y2)), ...},
        "Velocity"  或 "Homo_Velocity":  {frame_id(int): (vx, vy), ...},
        "Direction" 或 "Homo_Direction": {frame_id(int): (hx, hy), ...},
        "Size"      或 "Homo_Size":      {frame_id(int): (w, h), ...},
    }

座標可以是原始影像像素座標，也可以是經 Homography 轉換後的地圖座標
（use_homography=True 時使用 "Homo_*" 系列欄位），兩者運算邏輯完全相同，
差別只在輸入的是哪一組座標。

## 相依套件

- numpy（必要）
- numba（選用，用於加速 PET 的最近距離搜尋；未安裝時會自動退回純
  Python 迴圈，結果相同、速度較慢）
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba 為選用加速套件
    _HAS_NUMBA = False

    def njit(func=None, **kwargs):
        """numba 不存在時的空裝飾器，讓程式仍可正常運作（僅較慢）。"""
        if func is None:
            return lambda f: f
        return func


# =============================================================================
# 1. TTC（Time-To-Collision）
# =============================================================================

def rects_iou(
    pos1: Sequence[float], size1: Sequence[float],
    pos2: Sequence[float], size2: Sequence[float],
) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
    """計算兩個「以中心點表示」的矩形之間的 IoU（交集面積 / 聯集面積）。

    Args:
        pos1, pos2: 矩形中心座標 (x, y)
        size1, size2: 矩形尺寸 (width, height)

    Returns:
        (iou, box1, box2)：box1/box2 為對應矩形的 ((x1,y1),(x2,y2)) 表示法。
    """
    x1_min = pos1[0] - size1[0] / 2
    x1_max = pos1[0] + size1[0] / 2
    y1_min = pos1[1] - size1[1] / 2
    y1_max = pos1[1] + size1[1] / 2

    x2_min = pos2[0] - size2[0] / 2
    x2_max = pos2[0] + size2[0] / 2
    y2_min = pos2[1] - size2[1] / 2
    y2_max = pos2[1] + size2[1] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = size1[0] * size1[1]
    area2 = size2[0] * size2[1]
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    box1 = ((int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)))
    box2 = ((int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)))
    return iou, box1, box2


def get_collision_type(dir1: Sequence[float], dir2: Sequence[float]) -> str:
    """依兩物體的行進方向向量夾角，判斷碰撞類型。

    Returns:
        "Right-angle collision"（側撞，約略垂直，70~110 度）
        "Rear-end collision"（追撞，方向幾乎相同）
        "Head-on collision"（對撞，方向幾乎相反）
        "Other"（其他角度）
    """
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v

    d1 = normalize(np.asarray(dir1, dtype=np.float64))
    d2 = normalize(np.asarray(dir2, dtype=np.float64))

    cos_theta = float(np.dot(d1, d2))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))

    if 70 <= angle_deg <= 110:
        return "Right-angle collision"
    elif cos_theta > 0.9:
        return "Rear-end collision"
    elif cos_theta < -0.9:
        return "Head-on collision"
    else:
        return "Other"


def find_ttc(
    pos1: np.ndarray, vel1: np.ndarray, size1: np.ndarray,
    pos2: np.ndarray, vel2: np.ndarray, size2: np.ndarray,
    fps: float, max_ttc_frames: int,
    bounds: Tuple[int, int],
    max_time: float = 10.0, dt: float = 0.1, iou_threshold: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """對兩物體目前的位置/速度做等速度線性外推模擬，尋找是否會發生碰撞。

    以 dt 為步進間隔，逐步將兩物體沿各自速度方向推進，只要外推後的邊界框
    IoU 超過 iou_threshold，即視為「預估會碰撞」，回傳第一個滿足條件的時間點。

    Args:
        pos1, pos2: 目前中心座標 (x, y)（numpy array）
        vel1, vel2: 目前速度 (vx, vy)（單位：座標單位/秒）
        size1, size2: 邊界框尺寸 (w, h)
        fps: 影片幀率，用於將秒數換算為 frame 數
        max_ttc_frames: 最大模擬 frame 數上限（模擬時間上限 = max_ttc_frames / fps）
        bounds: (height, width)，碰撞點需落在此範圍內才視為有效
        max_time: 模擬時間上限（秒），與 max_ttc_frames 換算後取較小值
        dt: 模擬時間步進（秒）
        iou_threshold: 判定為碰撞的 IoU 門檻

    Returns:
        None，或包含以下欄位的 dict：
            ttc_frames: 預估碰撞所需 frame 數
            ttc_time: 預估碰撞所需秒數
            collision_point: 碰撞位置 (x, y)（兩物體外推位置中點）
            iou: 碰撞當下的 IoU 值
            arrows: ((起點,終點), (起點,終點)) 兩物體從目前位置到碰撞位置的位移
            future_box: (box1, box2) 碰撞當下兩物體的外推邊界框
    """
    height, width = bounds
    max_sim_time = min(max_time, max_ttc_frames / fps)

    for t in np.arange(0, max_sim_time, dt):
        future_pos1 = pos1 + vel1 * t
        future_pos2 = pos2 + vel2 * t

        iou, future_box1, future_box2 = rects_iou(future_pos1, size1, future_pos2, size2)
        if iou >= iou_threshold:
            ttc_frames = int(round(t * fps))
            collision_point = (future_pos1 + future_pos2) / 2
            x, y = collision_point
            if (0 <= int(x) < height) and (0 <= int(y) < width) and (ttc_frames != 0):
                return {
                    "ttc_frames": ttc_frames,
                    "ttc_time": float(t),
                    "collision_point": collision_point,
                    "iou": iou,
                    "arrows": (
                        (tuple(map(int, pos1.tolist())), tuple(map(int, future_pos1.tolist()))),
                        (tuple(map(int, pos2.tolist())), tuple(map(int, future_pos2.tolist()))),
                    ),
                    "future_box": (future_box1, future_box2),
                }
    return None


def compute_ttc_for_pair(
    track1: Dict[str, Any], track2: Dict[str, Any],
    fps: float, max_ttc_frames: int, bounds: Tuple[int, int],
    use_homography: bool = True, iou_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """對一組物體軌跡配對，逐一比對其共同出現的 frame，計算 TTC 事件。

    對應原始程式 TTCAnalyzer.compute()。找到第一個發生碰撞的 frame 後即
    停止比對該配對（與原始行為一致：同一配對只回報一次）。

    Args:
        track1, track2: 軌跡 dict（見檔案頂部說明）
        fps: 影片幀率
        max_ttc_frames: 最大模擬 frame 數
        bounds: (height, width) 有效碰撞區域範圍
        use_homography: True 時使用 "Homo_*" 欄位（地圖座標系），
                         False 時使用原始影像座標系欄位
        iou_threshold: 判定為碰撞的 IoU 門檻（原程式預設 0.8）

    Returns:
        list[dict]，每筆為一次 TTC 碰撞事件：
            {
                "ID1", "ID2": 兩物體的 Track ID,
                "collision_frame": 預估碰撞發生的 frame,
                "collision_occur_sec": 預估碰撞發生的秒數（相對影片起點）,
                "collision_point": (x, y),
                "collision_type": "Right-angle collision" 等,
                "ttc_seconds": 從目前 frame 到碰撞所需秒數,
                "iou": 碰撞當下 IoU,
                "future_box1", "future_box2": 兩物體碰撞當下外推邊界框,
                "arrow1", "arrow2": 兩物體位移向量（起點, 終點）,
            }
        通常最多 1 筆（找到即停止），無碰撞則為空 list。
    """
    prefix = "Homo_" if use_homography else ""
    keys = [f"{prefix}{k}" for k in ("Positions", "Bboxs", "Velocity", "Size", "Direction")]

    shared_frames = sorted(set(track1.get(f"{prefix}Bboxs", {}).keys())
                            & set(track2.get(f"{prefix}Bboxs", {}).keys()))
    if not shared_frames:
        return []

    events: List[Dict[str, Any]] = []

    # 與原程式一致，從第二個共同 frame 開始比對（需要至少一個先前點才有意義）
    for i in range(1, len(shared_frames)):
        curr_frame = shared_frames[i]

        if not all(k in track1 and curr_frame in track1[k] and
                   k in track2 and curr_frame in track2[k] for k in keys):
            continue

        pos1 = np.array(track1[f"{prefix}Positions"][curr_frame][:2], dtype=np.float64)
        pos2 = np.array(track2[f"{prefix}Positions"][curr_frame][:2], dtype=np.float64)
        vel1 = np.array(track1[f"{prefix}Velocity"][curr_frame], dtype=np.float64)
        vel2 = np.array(track2[f"{prefix}Velocity"][curr_frame], dtype=np.float64)
        dir1 = np.array(track1[f"{prefix}Direction"][curr_frame], dtype=np.float64)
        dir2 = np.array(track2[f"{prefix}Direction"][curr_frame], dtype=np.float64)
        size1 = np.array(track1[f"{prefix}Size"][curr_frame], dtype=np.float64)
        size2 = np.array(track2[f"{prefix}Size"][curr_frame], dtype=np.float64)

        result = find_ttc(pos1, vel1, size1, pos2, vel2, size2,
                           fps=fps, max_ttc_frames=max_ttc_frames, bounds=bounds,
                           iou_threshold=iou_threshold)

        if result is not None:
            events.append({
                "ID1": track1["Track ID"],
                "ID2": track2["Track ID"],
                "collision_frame": curr_frame + result["ttc_frames"],
                "collision_occur_sec": curr_frame / fps + result["ttc_time"],
                "collision_point": tuple(result["collision_point"][:2]),
                "collision_type": get_collision_type(dir1, dir2),
                "ttc_seconds": result["ttc_time"],
                "iou": result["iou"],
                "future_box1": result["future_box"][0],
                "future_box2": result["future_box"][1],
                "arrow1": result["arrows"][0],
                "arrow2": result["arrows"][1],
            })
            break  # 同一配對找到一次碰撞事件即停止（與原程式行為一致）

    return events


def compute_ttc_for_all_pairs(
    tracks: List[Dict[str, Any]],
    fps: float, max_ttc_frames: int, bounds: Tuple[int, int],
    use_homography: bool = True, iou_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """對整批軌跡做兩兩配對（itertools.combinations），彙整所有 TTC 碰撞事件。

    對應原始 Cust_Model_compare_tracking.py 的用法：
        for track1, track2 in combinations(tracks, 2):
            ttc_computer.compute(track1, track2)
    """
    all_events: List[Dict[str, Any]] = []
    for track1, track2 in combinations(tracks, 2):
        all_events.extend(
            compute_ttc_for_pair(track1, track2, fps, max_ttc_frames, bounds,
                                  use_homography=use_homography, iou_threshold=iou_threshold)
        )
    return all_events


# =============================================================================
# 2. PET（Post-Encroachment Time）
# =============================================================================

@njit
def compute_min_distance_pair(
    path1_arr: np.ndarray, path2_arr: np.ndarray,
    time1_arr: np.ndarray, time2_arr: np.ndarray,
    max_time_diff: int, overlap_threshold: float,
):
    """在時間差限制內，窮舉搜尋兩條軌跡座標點中「空間距離最小」的一對。

    對應原始 PETAnalyzer.compute_min_distance_pair（numba 加速版，邏輯完全
    相同）。若未安裝 numba，本函式會自動以純 Python 執行（見檔案頂部的
    njit 空裝飾器 fallback），結果相同、速度較慢。

    Args:
        path1_arr, path2_arr: shape=(N, 2) 的座標陣列 (x, y)
        time1_arr, time2_arr: 對應座標的 frame index 陣列
        max_time_diff: 允許比對的最大時間差（frame 數），超過則略過該組合
        overlap_threshold: 視為「同一空間位置」的距離門檻（座標單位，如像素）

    Returns:
        (t1, t2, idx1, idx2, dist)：
            t1, t2: 兩軌跡各自最近點的 frame index
            idx1, idx2: 對應在輸入陣列中的索引
            dist: 該對點之間的歐氏距離
        若找不到符合門檻的配對，回傳 (-1, -1, -1, -1, -1.0)
    """
    min_dist = 1e10
    best_pair = (-1, -1, -1, -1, -1.0)
    for i in range(path1_arr.shape[0]):
        for j in range(path2_arr.shape[0]):
            dt = abs(time1_arr[i] - time2_arr[j])
            if dt > max_time_diff:
                continue
            dx = path1_arr[i, 0] - path2_arr[j, 0]
            dy = path1_arr[i, 1] - path2_arr[j, 1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= overlap_threshold and dist < min_dist:
                min_dist = dist
                best_pair = (time1_arr[i], time2_arr[j], i, j, dist)
    return best_pair


def compute_pet_for_pair(
    track1: Dict[str, Any], track2: Dict[str, Any],
    fps: float, max_time_diff_frames: int, overlap_threshold: float,
    use_homography: bool = True,
) -> Optional[Dict[str, Any]]:
    """計算一組軌跡配對的 PET（先後通過時間）事件。

    這是 PETAnalyzer 逐幀滑動視窗演算法的「批次／全軌跡」等效簡化版：
    原程式為了即時疊圖顯示，逐 frame 維護 lookback 視窗（見
    PETAnalyzer._update_track_histories），但核心的「找最近時空距離點」
    運算（compute_min_distance_pair）本身與逐幀處理與否無關——只要提供
    兩條軌跡各自完整的 (位置, 時間) 陣列，一次呼叫即可得到全域最近點，
    數學結果等價。若目標專案是離線批次處理（已有完整軌跡，非即時逐幀
    處理影片），建議直接使用本函式，不需重建原本的滑動視窗機制。

    Args:
        track1, track2: 軌跡 dict（見檔案頂部說明）
        fps: 影片幀率，用於將 frame 差換算為秒數
        max_time_diff_frames: 允許比對的最大時間差（frame 數），
                               對應原程式 lookback_sec * fps
        overlap_threshold: 視為「同一空間位置」的距離門檻（座標單位）
        use_homography: True 時使用 "Homo_Positions"（地圖座標系）

    Returns:
        None，或包含以下欄位的 dict：
            {
                "ID1", "ID2": 兩物體的 Track ID,
                "cross_point": (x, y)  兩物體最近點的中點座標,
                "pet_seconds": PET 值（秒）,
                "frame1", "frame2": 兩物體各自通過該最近點的 frame,
                "distance": 最近距離,
            }
    """
    prefix = "Homo_" if use_homography else ""
    pos_key = f"{prefix}Positions"

    pos1_dict = track1.get(pos_key, {})
    pos2_dict = track2.get(pos_key, {})
    if not pos1_dict or not pos2_dict:
        return None

    frames1 = sorted(pos1_dict.keys())
    frames2 = sorted(pos2_dict.keys())

    pos1_arr = np.array([pos1_dict[f][:2] for f in frames1], dtype=np.float32)
    pos2_arr = np.array([pos2_dict[f][:2] for f in frames2], dtype=np.float32)
    time1_arr = np.array(frames1, dtype=np.int32)
    time2_arr = np.array(frames2, dtype=np.int32)

    t1, t2, idx1, idx2, dist = compute_min_distance_pair(
        pos1_arr, pos2_arr, time1_arr, time2_arr, max_time_diff_frames, overlap_threshold
    )

    if idx1 == -1 or idx2 == -1:
        return None

    p1 = pos1_arr[idx1]
    p2 = pos2_arr[idx2]
    cross_point = (float((p1[0] + p2[0]) / 2), float((p1[1] + p2[1]) / 2))
    pet_seconds = abs(t2 - t1) / fps

    return {
        "ID1": track1["Track ID"],
        "ID2": track2["Track ID"],
        "cross_point": cross_point,
        "pet_seconds": float(pet_seconds),
        "frame1": int(t1),
        "frame2": int(t2),
        "distance": round(float(dist), 2),
    }


def compute_pet_for_all_pairs(
    tracks: List[Dict[str, Any]],
    fps: float, max_time_diff_frames: int, overlap_threshold: float,
    use_homography: bool = True,
) -> List[Dict[str, Any]]:
    """對整批軌跡做兩兩配對，彙整所有 PET 事件（distance 未超過門檻的才會有結果）。"""
    events: List[Dict[str, Any]] = []
    for track1, track2 in combinations(tracks, 2):
        event = compute_pet_for_pair(track1, track2, fps, max_time_diff_frames,
                                      overlap_threshold, use_homography=use_homography)
        if event is not None:
            events.append(event)
    return events


# =============================================================================
# 使用範例
# =============================================================================
if __name__ == "__main__":
    fps = 30

    # --- 範例一：兩物體路徑「空間上」會交叉，但「時間上」錯開（近似真實的先後通過情境）
    # A 沿 y=200 往右行駛，於 f=40 時到達 x=300；B 沿 x=300 往上行駛，於 f=20 時到達 y=200
    # 兩者不會同時抵達交叉點 -> 不構成 TTC 意義下的「預測碰撞」，但構成 PET 意義下的「先後通過」
    track_a = {
        "Track ID": "A",
        "Positions": {f: (100 + f * 5, 200, 0) for f in range(0, 40)},
        "Bboxs": {f: ((90 + f * 5, 190), (110 + f * 5, 210)) for f in range(0, 40)},
        "Velocity": {f: (5 * fps, 0) for f in range(0, 40)},
        "Direction": {f: (5, 0) for f in range(0, 40)},
        "Size": {f: (20, 20) for f in range(0, 40)},
    }
    track_b = {
        "Track ID": "B",
        "Positions": {f: (300, 100 + f * 5, 0) for f in range(0, 40)},
        "Bboxs": {f: ((290, 90 + f * 5), (310, 110 + f * 5)) for f in range(0, 40)},
        "Velocity": {f: (0, 5 * fps) for f in range(0, 40)},
        "Direction": {f: (0, 5) for f in range(0, 40)},
        "Size": {f: (20, 20) for f in range(0, 40)},
    }

    print("=== 範例一：A/B 空間交叉但時間錯開 ===")
    print("TTC events:", compute_ttc_for_pair(
        track_a, track_b, fps=fps, max_ttc_frames=90, bounds=(1080, 1920),
        use_homography=False, iou_threshold=0.3,
    ))
    print("PET event :", compute_pet_for_pair(
        track_a, track_b, fps=fps, max_time_diff_frames=90, overlap_threshold=15,
        use_homography=False,
    ))

    # --- 範例二：兩物體對向行駛，會在同一時間於同一地點相遇 -> 構成 TTC 意義下的預測碰撞
    track_c = {
        "Track ID": "C",
        "Positions": {f: (100 + f * 5, 200, 0) for f in range(0, 40)},
        "Bboxs": {f: ((90 + f * 5, 190), (110 + f * 5, 210)) for f in range(0, 40)},
        "Velocity": {f: (5 * fps, 0) for f in range(0, 40)},
        "Direction": {f: (5, 0) for f in range(0, 40)},
        "Size": {f: (20, 20) for f in range(0, 40)},
    }
    track_d = {
        "Track ID": "D",
        "Positions": {f: (300 - f * 5, 205, 0) for f in range(0, 40)},
        "Bboxs": {f: ((290 - f * 5, 195), (310 - f * 5, 215)) for f in range(0, 40)},
        "Velocity": {f: (-5 * fps, 0) for f in range(0, 40)},
        "Direction": {f: (-5, 0) for f in range(0, 40)},
        "Size": {f: (20, 20) for f in range(0, 40)},
    }

    print("\n=== 範例二：C/D 對向行駛，預測會相撞 ===")
    print("TTC events:", compute_ttc_for_pair(
        track_c, track_d, fps=fps, max_ttc_frames=90, bounds=(1080, 1920),
        use_homography=False, iou_threshold=0.3,
    ))
    print("PET event :", compute_pet_for_pair(
        track_c, track_d, fps=fps, max_time_diff_frames=90, overlap_threshold=15,
        use_homography=False,
    ))
```

### 附錄 F:原 `markdown/reference_code/draw_utils.py`

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import random
from collections import defaultdict
from utils.utils import transform_video, prepare_directory
from copy import deepcopy
import csv
plt.ion()

################################################################
def draw_trace(img, coords, color, thickness=1, radius=2, H=None):
    keys = list(coords.keys())
    if len(keys) < 2:
        return
    for i in range(1, len(keys)):
        pt1 = coords[keys[i - 1]][:2]
        pt2 = coords[keys[i]][:2]

        if H is not None:
            # 轉換為齊次座標並使用 warpPerspective
            pt1 = cv2.perspectiveTransform(np.array([[pt1]], dtype=np.float32), H)[0][0]
            pt2 = cv2.perspectiveTransform(np.array([[pt2]], dtype=np.float32), H)[0][0]

        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.circle(img, pt2, radius, color, thickness + 1)

#################################################################
def draw_collision_heatmap(collision_point, color_heatmap_trace, background, radius=5, alpha=0.5, dpi=100, withbar=True):
    """
    在底圖 color_heatmap_ttc 上根據 ttc_collision_point 的點繪製碰撞熱點圖，並與 background 加權混合。
    
    :param ttc_collision_point: List 或 numpy array，格式為 [(x1, y1), (x2, y2), ...]
    :param color_heatmap_ttc: 底圖（用於畫 colorbar 的比例與畫布大小）
    :param background: 原始背景圖（要與熱度圖混合），BGR 或 RGB 皆可，會自動轉換
    :param radius: 每個點的影響範圍半徑
    :param alpha: 熱度圖的透明度，用來控制加權強度
    :return: matplotlib.figure.Figure 物件
    """
    height, width = color_heatmap_trace.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    # 標記每個碰撞點
    for point_data_list in collision_point.values():
        for point in point_data_list:
            x, y = point["colision_point"]
            if 0 <= int(y) < height and 0 <= int(x) < width:
                cv2.circle(heatmap, (int(x), int(y)), radius, 1, -1)
                # cv2.putText(heatmap, text, (int(x) + 5, int(y) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 模糊與正規化熱度圖
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=radius)
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

    # 將 heatmap 映射為彩色圖（jet colormap）
    heatmap_color = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 確保背景為 RGB 格式
    if background.shape[2] == 3:
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB) if background[..., 0].max() > 1 else background
    else:
        raise ValueError("背景圖必須為彩色影像 (3 channels)")

    # 混合背景與熱度圖
    overlay = cv2.addWeighted(background_rgb, 1 - alpha, heatmap_color, alpha, 0)

    figsize = (width / dpi, height / dpi)
    fig = Figure(figsize=figsize, dpi=dpi)
    fig.set_facecolor('black')
    canvas = FigureCanvas(fig)

    # colorbar 貼齊最右邊，並且保留高度
    ax = fig.add_axes([0, 0, 1, 1])

    # 標記每個碰撞點
    # for text, point in ttc_collision_point.items():
    #     x, y = point
    #     if 0 <= int(y) < height and 0 <= int(x) < width:
    #         cv2.putText(overlay, text, (int(x) + 5, int(y) - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    im = ax.imshow(overlay)
    ax.axis("off")

    if withbar:        
        cax = fig.add_axes([0.95, 0.05, 0.03, 0.9])  # 貼齊最右側

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Collision Count (color intensity)", fontsize=10, color='white', labelpad=10)
        cbar.ax.tick_params(labelsize=10, colors='white')

        # 將刻度移到左側
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        canvas.draw()

    ax.set_title('TTC Collision Heatmap')
    ax.axis('off')
    return fig

def draw_depth(map, width, height, dpi=100, show_cbar=False):
    figsize = (width / dpi, height / dpi)
    fig = Figure(figsize=figsize, dpi=dpi)
    fig.set_facecolor('black')
    canvas = FigureCanvas(fig)

    # 主圖像佔滿整個 figure（不保留任何空間）
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(map, cmap="turbo")
    ax.axis("off")

    if show_cbar:
        # colorbar 貼齊最右邊，並且保留高度
        # [left, bottom, width, height]
        cax = fig.add_axes([0.95, 0.05, 0.03, 0.9])  # 貼齊最右側

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Depth (Normalized)", fontsize=10, color='white', labelpad=10)
        cbar.ax.tick_params(labelsize=10, colors='white')
        
        # 將刻度移到左側
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    canvas.draw()
    image_with_colorbar = np.asarray(canvas.buffer_rgba())[..., :3]

    if show_cbar:
        image_with_colorbar = cv2.resize(image_with_colorbar, (width, height))

    return np.ascontiguousarray(image_with_colorbar)


def to_bright(color, brightness_increase=250):
    # 假設 color 是 (b, g, r)
    color_np = np.uint8([[color]])  # 變成 (1,1,3) 形狀
    hsv = cv2.cvtColor(color_np, cv2.COLOR_BGR2HSV)
    hsv = cv2.split(hsv[0,0])[0]
    h, s, v = hsv[0][0], hsv[1][0], hsv[2][0]
    v = np.clip(v + brightness_increase, 0, 255)
    hsv_bright = np.uint8([[[h, s, v]]])
    bright_color = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)[0,0]
    # 確保轉成純 Python int tuple
    return tuple(int(c) for c in bright_color)

def draw_text_with_bg(img, text, org, font, font_scale, text_color, bg_color, thickness=1, pad=2, alpha=0.6):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    top_left = (x - pad, y - text_h - pad)
    bottom_right = (x + text_w + pad, y + baseline + pad)

    # 確保座標在影像範圍內
    top_left = (max(0, top_left[0]), max(0, top_left[1]))
    bottom_right = (
        min(img.shape[1] - 1, bottom_right[0]),
        min(img.shape[0] - 1, bottom_right[1])
    )

    # 依 alpha 值選擇是否套用透明度
    if alpha < 1.0:
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        # 無需透明，直接畫純色背景
        cv2.rectangle(img, top_left, bottom_right, bg_color, -1)

    # 畫文字
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_arrow_with_fixed_tip(frame, start_pt, end_pt, color, thickness=2, min_length=5, tip_size=10):
    """
    繪製箭頭，箭頭主體長度至少為 min_length，且箭頭頭部大小為固定像素 tip_size。
    """
    vec = np.array(end_pt) - np.array(start_pt)
    length = np.linalg.norm(vec)

    if length == 0:
        return  # 起點終點相同，不畫

    # 若箭頭過短，則放大到最小長度
    if length < min_length:
        vec = vec * (min_length / length)
        end_pt = (int(start_pt[0] + vec[0]), int(start_pt[1] + vec[1]))
        length = min_length

    # 正規化方向向量
    direction = vec / length

    # 箭頭主幹終點（減去 tip_size 像素）
    line_end = np.array(end_pt) - direction * tip_size

    # 畫箭頭主體
    cv2.line(frame, tuple(start_pt), tuple(line_end.astype(int)), color, thickness)

    # 畫箭頭頭部（兩個副線形成箭頭尖端）
    # 垂直向量（用於兩邊分岔）
    perp = np.array([-direction[1], direction[0]])

    # 箭頭角度與寬度調整（可以修改這個比例讓箭頭更尖或更鈍）
    side_length = tip_size * 0.5
    tip1 = np.array(end_pt) - direction * tip_size + perp * side_length
    tip2 = np.array(end_pt) - direction * tip_size - perp * side_length

    # 繪製箭頭尖端兩條邊
    cv2.line(frame, tuple(tip1.astype(int)), tuple(end_pt), color, thickness)
    cv2.line(frame, tuple(tip2.astype(int)), tuple(end_pt), color, thickness)    

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    """
    Draws a dashed rectangle on an image.

    Parameters:
        img (numpy.ndarray): The image on which to draw the rectangle.
        pt1 (tuple): Top-left corner of the rectangle (x1, y1).
        pt2 (tuple): Bottom-right corner of the rectangle (x2, y2).
        color (tuple): Color of the rectangle in (B, G, R).
        thickness (int): Thickness of the dashed lines.
        dash_length (int): Length of each dash in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Define the four sides of the rectangle as line segments
    lines = [
        ((x1, y1), (x2, y1)),  # Top side
        ((x2, y1), (x2, y2)),  # Right side
        ((x2, y2), (x1, y2)),  # Bottom side
        ((x1, y2), (x1, y1))   # Left side
    ]

    for line in lines:
        start_point, end_point = line
        # 底層白色虛線（加厚）
        draw_dashed_line(img, start_point, end_point, (255, 255, 255), thickness + 2, dash_length)
        # 上層指定顏色虛線
        draw_dashed_line(img, start_point, end_point, color, thickness, dash_length)

def draw_dashed_line(img, start_point, end_point, color, thickness=1, dash_length=10):
    """
    Draws a dashed line between two points.

    Parameters:
        img (numpy.ndarray): The image on which to draw the line.
        start_point (tuple): Starting point of the line (x1, y1).
        end_point (tuple): Ending point of the line (x2, y2).
        color (tuple): Color of the line in (B, G, R).
        thickness (int): Thickness of the line.
        dash_length (int): Length of each dash in pixels.
    """
    x1, y1 = start_point
    x2, y2 = end_point

    # Calculate the total distance and direction vector
    length = int(np.hypot(x2 - x1, y2 - y1))
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    # Draw dashes along the line
    for i in range(0, length, dash_length * 2):
        start_x = int(x1 + dx * i)
        start_y = int(y1 + dy * i)
        end_x = int(x1 + dx * min(i + dash_length, length))
        end_y = int(y1 + dy * min(i + dash_length, length))
        
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


def draw_collision_overlay(
    trace_dir,
    frame,
    coll_frame,
    frame_id,
    collision_data_dict,
    label_prefix,
    save_path_prefix,
    video_name,
    track_colors,
    is_homography=False
):
    if frame_id not in collision_data_dict:
        return
    
    # 設定 CSV 儲存路徑

    csv_path = os.path.join(trace_dir, f"collision_{label_prefix}_homo_log.csv") if is_homography \
        else os.path.join(trace_dir, f"collision_{label_prefix}_log.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            if "ttc" in label_prefix.lower():
                writer.writerow([
                    "FrameID", "Video", "車輛1", "車輛2", "預估碰撞時間點",
                    "TTC碰撞時長", "碰撞點", "碰撞類型", f"{label_prefix}標籤[{label_prefix}@dur_times]", "指標"
                ])
            else:
                writer.writerow([
                    "FrameID", "Video", "車輛1", "車輛2", "預估碰撞時間點",
                    "PET碰撞時長", "碰撞點", f"{label_prefix}標籤[{label_prefix}@dur_times]", "指標"
                ])

        point_list = collision_data_dict[frame_id]
        for point in point_list:
            prefix = "homo_frame" if is_homography else "frame"
            prefix_ = prefix.replace("homo_frame", "homo")
          
            if "ttc" in label_prefix.lower():
                # 寫入 CSV
                writer.writerow([
                    frame_id,
                    video_name,
                    point["ID1"],
                    point["ID2"],
                    point["colision_occur_sec"],
                    round(point["dur_times"], 2),
                    point["colision_point"],
                    point.get("coll_type", "") if "ttc" in label_prefix.lower() else "",
                    point.get("text", ""),
                    label_prefix
                ])
            else:
                # 寫入 CSV
                writer.writerow([
                    frame_id,
                    video_name,
                    point["ID1"],
                    point["ID2"],
                    point["colision_occur_sec"],
                    round(point["dur_times"], 2),
                    point["colision_point"],
                    point.get("text", ""),
                    label_prefix
                ])

            x, y = map(int, point['colision_point'])

            # 準備要顯示的文字
            if "ttc" in label_prefix.lower():
                text1 = f"{point['coll_type']}-({point['ID1']}-{point['ID2']})"
            else:
                text1 = f"PET-({point['ID1']}-{point['ID2']})"
            text2 = f"{point['text']}"

            # 設定字型與大小
            font = cv2.LINE_AA 
            font_scale = 0.3
            thickness = 1
            pad = 2            

            if "ttc" in label_prefix.lower():
                color1 = track_colors[point["ID1"]]
                draw_dashed_rectangle(coll_frame, point['future_box1'][0], point['future_box1'][1], color=color1, thickness=2)
                color2 = track_colors[point["ID2"]]
                draw_dashed_rectangle(coll_frame, point['future_box2'][0], point['future_box2'][1], color=color2, thickness=2)

                draw_arrow_with_fixed_tip(coll_frame, point['arrow1'][0], point['arrow1'][1], color1)
                draw_arrow_with_fixed_tip(coll_frame, point['arrow2'][0], point['arrow2'][1], color2)
            
            # 畫背景矩形（先畫再放字）       
            draw_text_with_bg(coll_frame, text1, (x + 20, y - 20), font, font_scale, (40, 40, 255), (255, 255, 255), thickness, pad, alpha=1.0)
            draw_text_with_bg(coll_frame, text2, (x + 20 , y - 5), font, font_scale, (40, 40, 255), (255, 255, 255), thickness, pad, alpha=1.0)
            cv2.circle(coll_frame, (x, y), 3, (0, 0, 255), -1)

    # 遮罩疊合原始畫面
    mask = cv2.cvtColor(coll_frame, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
   
    coll_fg = cv2.bitwise_and(coll_frame, coll_frame, mask=binary_mask)
    inv_mask = cv2.bitwise_not(binary_mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    merged = cv2.add(frame_bg, coll_fg)

    save_path = f"{trace_dir}/{save_path_prefix}_{video_name}/{prefix}_{frame_id}.jpg"
    cv2.imwrite(save_path, merged)

def valid_ttc_trace_video(video_path, trajectories, H_matrix, smooth_trace_graph, track_colors, 
                       ttc_collision_point, trace_ttc_dir, fps, fst_frame, ttc_times, alpha=0.4):
    
    video_name = os.path.basename(video_path).replace(".mp4", "")
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    active_objects = set()
    draw_trajectories = defaultdict(list)

    prepare_directory(f"{trace_ttc_dir}/coll_ttc_{video_name}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = fst_frame.shape
    out_trace = cv2.VideoWriter(f"{trace_ttc_dir}/out_trace.mp4", fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        coll_ttc_frame = deepcopy(frame)
        if not ret:
            break
        frame_id += 1

        ttc_coll_IDs = set()
        if frame_id in ttc_collision_point:
            for item in ttc_collision_point[frame_id]:
                ttc_coll_IDs.add(item["ID1"])
                ttc_coll_IDs.add(item["ID2"])

        frame_tracks = trajectories.get(frame_id)
        if not frame_tracks:
            continue

        for track_id, data in frame_tracks.items():
            if not isinstance(track_id, int):
                continue

            # 判斷是否仍在追蹤期間
            trace_st, trace_en = data["trace_st"], data["trace_en"]
            is_active = trace_st <= frame_id <= trace_en

            if is_active:
                active_objects.add(track_id)
            else:
                active_objects.discard(track_id)

            cx, cy, _ = map(int, data["Position"])
            p1 = tuple(map(int, data["Bbox"][0]))
            p2 = tuple(map(int, data["Bbox"][1]))
            color = to_bright(track_colors[track_id])

            cv2.circle(frame, (cx, cy), 3, color, -1)
            cv2.rectangle(frame, p1[:2], p2[:2], color, 2)

            draw_text_with_bg(
                frame, f'{data["text"]}', (p1[0], p1[1] - 5),
                font=cv2.LINE_AA, font_scale=0.3,
                bg_color=(20, 20, 20), text_color=color,
                thickness=1, pad=2
            )

            if is_active:
                traj = draw_trajectories[track_id]
                traj.append((cx, cy))
                if len(traj) >= 2:
                    for i in range(1, len(traj)):
                        cv2.line(frame, traj[i - 1], traj[i], color, 2)

            ##################################################################################
            if frame_id in ttc_collision_point.keys(): 
                if track_id in ttc_coll_IDs:
                    cv2.circle(coll_ttc_frame, (cx, cy), 3, color, -1)
                    cv2.rectangle(coll_ttc_frame, p1[:2], p2[:2], color, 2)

                    draw_text_with_bg(
                        coll_ttc_frame, f'{data["text"]}', (p1[0], p1[1] - 5),
                        font=cv2.LINE_AA, font_scale=0.3,
                        bg_color=(20, 20, 20), text_color=color,
                        thickness=1, pad=2
                    )

                    if is_active:   
                        traj = draw_trajectories[track_id]                     
                        if len(traj) >= 2:
                            for i in range(1, len(traj)):
                                cv2.line(coll_ttc_frame, traj[i - 1], traj[i], color, 2)

            ##################################################################################

        # 初始化熱區圖（不使用 deepcopy）
        height, width = frame.shape[:2]
        ttc_coll_frame = np.zeros((height, width, 3), dtype=np.uint8)

        draw_collision_overlay(
            trace_ttc_dir,
            frame=coll_ttc_frame,
            coll_frame=ttc_coll_frame,
            frame_id=frame_id,
            collision_data_dict=ttc_collision_point,
            label_prefix=f"TTC",
            save_path_prefix="coll_ttc",
            video_name=video_name,
            track_colors=track_colors
        )       

        # if frame_id % 1000 == 0:
        #     print(frame_id)

        out_trace.write(frame)    
    out_trace.release()

    csv_filename = f"{trace_ttc_dir}/frame_processing_stats.csv"
    # 統計輸出（中文）
    avg_time = np.mean(ttc_times)
    max_time = np.max(ttc_times)
    min_time = np.min(ttc_times)
    peak_threshold = avg_time * 1.5
    peak_count = sum(t > peak_threshold for t in ttc_times)
    offpeak_count = len(ttc_times) - peak_count

    with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["統計項目", "數值（毫秒）"])
        writer.writerow(["平均處理時間", f"{avg_time:.3f}"])
        writer.writerow(["最大處理時間", f"{max_time:.3f}"])
        writer.writerow(["最小處理時間", f"{min_time:.3f}"])
        writer.writerow(["高峰處理次數", peak_count])
        writer.writerow(["非高峰處理次數", offpeak_count])

    print(f"\n處理時間統計已輸出到檔案：{csv_filename}")


def valid_pet_trace_video(video_path, trajectories, H_matrix, smooth_trace_graph, track_colors, 
                       pet_collision_point, trace_pet_dir, alpha=0.4):
    
    video_name = os.path.basename(video_path).replace(".mp4", "")
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    active_objects = set()
    draw_trajectories = defaultdict(list)

    all_coll_track_ids =  set()
    for fid, data_list in pet_collision_point.items():
        for data in data_list:
            all_coll_track_ids.add(data["ID1"])
            all_coll_track_ids.add(data["ID2"])

    prepare_directory(f"{trace_pet_dir}/coll_pet_{video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        frame_tracks = trajectories.get(frame_id)
        if not frame_tracks:
            continue

        for track_id, data in frame_tracks.items():
            if not isinstance(track_id, int):
                continue

            if track_id not in all_coll_track_ids: continue

            # 判斷是否仍在追蹤期間
            trace_st, trace_en = data["trace_st"], data["trace_en"]
            is_active = trace_st <= frame_id <= trace_en

            if is_active:
                active_objects.add(track_id)
            else:
                active_objects.discard(track_id)

            cx, cy, _ = map(int, data["Position"])
            p1 = tuple(map(int, data["Bbox"][0]))
            p2 = tuple(map(int, data["Bbox"][1]))
            color = to_bright(track_colors[track_id])

            cv2.circle(frame, (cx, cy), 3, color, -1)
            cv2.rectangle(frame, p1[:2], p2[:2], color, 2)

            draw_text_with_bg(
                frame, f'{data["text"]}', (p1[0], p1[1] - 5),
                font=cv2.LINE_AA, font_scale=0.3,
                bg_color=(20, 20, 20), text_color=color,
                thickness=1, pad=2
            )

            if is_active:
                traj = draw_trajectories[track_id]
                traj.append((cx, cy))
                if len(traj) >= 2:
                    for i in range(1, len(traj)):
                        cv2.line(frame, traj[i - 1], traj[i], color, 2)

        # 初始化熱區圖（不使用 deepcopy）
        height, width = frame.shape[:2]
        pet_coll_frame = np.zeros((height, width, 3), dtype=np.uint8)

        draw_collision_overlay(
            trace_pet_dir,
            frame=frame,
            coll_frame=pet_coll_frame,
            frame_id=frame_id,
            collision_data_dict=pet_collision_point,
            label_prefix="PET",
            save_path_prefix="coll_pet",
            video_name=video_name,
            track_colors=track_colors
        )

        # if frame_id % 1000 == 0:
        #     print(frame_id)

########################################################################
def draw_track(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w, h = track.to_ltrb()
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
        cv2.putText(frame, f"ID {track.track_id}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame
```
