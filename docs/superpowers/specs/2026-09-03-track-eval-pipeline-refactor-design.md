# track_eval.py 主流程重構 — 設計文件

- 日期:2026-09-03
- 狀態:待審查
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第二個子專案**,接續 `2026-09-03-dataset-path-refactor-design.md`。

## 0. 前提依賴

本設計假設 **dataset 路徑重構**(`2026-09-03-dataset-path-refactor-design.md`,已核准)已經實作完成:
- `VideoSequenceTracker.run()` 已不含補幀(frame-interpolation compensation)讀取/跳幀邏輯。
- stage1/stage2/stage3 的路徑組裝一律呼叫 `evaluation/dataset_layout.py` 提供的函式(`stage1_seq_dir()`、`stage2_seq_dir()`、`source_gt_dir()`、`list_stage1_sequences()` 等),不再有 `.replace('annotation_', '')` 這類字串轉換。

若實作順序上此計畫先於 dataset 路徑重構執行,則本計畫涉及路徑組裝的部分需暫時沿用現行邏輯,待路徑重構完成後再替換呼叫對象。

## 1. 背景與問題

1. `VideoSequenceTracker.run()` 是單一 method(~350 行),混雜:影片讀寫設定、偵測+追蹤主迴圈、label/image 存檔、Re-ID 訓練裁切蒐集、三段幾乎重複的標註繪圖邏輯(matched/unmatched/removed 各自重寫一次「多邊形過濾 → 畫框 → 文字標註」)、`cv2.imshow`、影片寫檔與 ffmpeg H.264 轉檔。任何局部修改(例如只想調整標註樣式)都要在整個大 method 裡定位。
2. 標註繪圖三段邏輯結構相同但各自獨立撰寫,任何樣式調整(顏色、文字位置、多邊形過濾條件)要改三處。
3. `TrackingEvalPipeline.run_stage1()` 混雜:序列探索、逐影片建立/執行 tracker、MOT 格式輸出、`data.yaml` 產生、stage1→stage2 目錄複製、SOURCE GT 複製——多種職責擠在同一個迴圈本體裡。
4. `run_stage3()` 依賴 `run_stage1()` 先跑過,目前只靠註解說明,程式面是「找不到就印警告後 `return`」的隱性 soft-skip,沒有獨立、可測試的檢查函式。

## 2. 目標 / 非目標

**目標**
- 拆分 `VideoSequenceTracker.run()` 的職責,主迴圈只做協調(orchestration)。
- 消除標註繪圖三段重複邏輯,抽成 `utils/visualization.py` 的共用函式。
- 拆分 `run_stage1()` 的職責,迴圈本體只做「探索 → 逐影片處理」。
- 把 stage3 對 stage1 輸出的存在性檢查抽成獨立、明確、可測試的函式,行為維持現行語意(soft-skip,不是硬性報錯)。
- 對外 CLI 介面(`python track_eval.py --config <path>`)與輸出檔案格式/內容**不變**——這是純結構重構,不改變任何 stage 的產出。

**非目標**
- 不動 `module/tracker`/`module/detection`/`module/reid` 的模組邊界或對外介面(除了呼叫已核准的 `dataset_layout.py`)。
- 不合併 `visualize/visualize_gt_on_video.py` / `visualize_gt_on_class_video.py` 的獨立疊圖邏輯——那是讀 GT/tracker txt 檔案的獨立工具,資料來源與使用情境跟 `VideoSequenceTracker` 的即時追蹤疊圖不同,合併是額外大工程,不在本次範圍。
- 不處理 homography 真實座標轉換相關需求——使用者已表示這是另一個獨立需求,留待後續另開 brainstorm。
- 不改變 `module/grid_search/*.py` 的呼叫方式(它們目前不直接呼叫 `VideoSequenceTracker`/`TrackingEvalPipeline`,而是各自實作相近但獨立的迴圈;是否讓它們改呼叫本次拆分出的元件,留給 grid_search 子專案的 brainstorm 決定)。

## 3. 新設計

採用先前討論的**方案 C**:抽出協作物件(方案 B),並把標註繪圖邏輯移到 `utils/visualization.py`(方案 C)。

### 3.1 `utils/visualization.py` 新增:`draw_track_box()`

```python
def draw_track_box(
    annotator: Annotator,
    box: tuple[float, float, float, float],
    det_corner: np.ndarray | None,
    color: tuple[int, int, int],
    *,
    embed_label: str | None = None,
    overlay_label: str | None = None,
    overlay_anchor: tuple[int, int] | None = None,
) -> None:
    """統一的「多邊形過濾 → 畫框 → 標註」邏輯，取代 VideoSequenceTracker.run() 內
    matched/unmatched/removed 三段重複程式碼。

    - det_corner 為 None 時不做多邊形過濾（全部畫）；不為 None 時，只在框中心點落在
      多邊形內才畫（沿用現行 cv2.pointPolygonTest 判斷）。
    - embed_label：文字直接嵌入 box_label()（matched track 目前的畫法：框內嵌 track_id）。
    - overlay_label + overlay_anchor：額外用 annotator.text() 疊加文字（unmatched/removed
      目前的畫法：紅框/黑框 + 框外文字標籤）。两种模式互斥，由呼叫方依三種情境各自決定。
    """
```

呼叫方(matched/unmatched/removed 三處)只傳入不同的 `color`/`embed_label`/`overlay_label`/`overlay_anchor`,不再各自重寫過濾與畫框邏輯。

### 3.2 `track_eval.py` 新增協作物件

```python
class ReidCropCollector:
    """蒐集 Re-ID 訓練用裁切圖（save_reid_crops=True 時啟用）。"""
    def __init__(self, crop_dir: Path, det_corner: np.ndarray | None): ...
    def collect(self, tracks, clean_frame, seq_stem: str, frame_id: int, width: int, height: int) -> None:
        """對每個 track：多邊形過濾 → 裁切 → 存到
        crop_dir/{class}/{seq_stem}_track{id}/{frame_id:06d}.jpg（邏輯與現行完全一致，只是搬出方法）。"""


class VideoWriterWithFfmpeg:
    """包裝 cv2.VideoWriter，close() 時額外做 ffmpeg H.264 轉檔（失敗則保留原始 mp4v 檔並印警告，行為與現行一致）。"""
    def __init__(self, output_path: Path, fps: int, width: int, height: int): ...
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...
```

### 3.3 `VideoSequenceTracker` 拆分後結構

```python
class VideoSequenceTracker:
    def __init__(self, ...):
        self.detector = Detector(model_path)
        self.tracker = create_tracker(...)
        self._reid_collector = ReidCropCollector(reid_crop_dir, det_corner) if save_reid_crops else None
        ...

    def run(self, video_path: str, output_dir: Path) -> tuple[Path, Path, Path]:
        cap, fps, width, height, total_frames = self._open_video(video_path)
        output_path, images_path, labels_path = self._prepare_output_dirs(output_dir, video_path)
        video_writer = VideoWriterWithFfmpeg(...) if self.save else None
        stats = self._run_frame_loop(cap, output_path, images_path, labels_path, video_writer, width, height, total_frames)
        cap.release()
        if video_writer:
            video_writer.close()
        if self.show:
            cv2.destroyAllWindows()
        self._print_summary(stats)
        return output_path, images_path, labels_path

    def _open_video(self, video_path: str): ...
    def _prepare_output_dirs(self, output_dir: Path, video_path: str): ...
    def _run_frame_loop(self, cap, output_path, images_path, labels_path, video_writer, width, height, total_frames): ...
    def _process_frame(self, frame, clean_frame, frame_id, images_path, labels_path, width, height) -> np.ndarray | None:
        """偵測 → 追蹤 → 存 label/image → reid 裁切蒐集 → 回傳（若需要）已標註的 frame。"""
    def _annotate_frame(self, frame, tracks, unmatched, removed, frame_id, total_frames) -> np.ndarray:
        """呼叫 utils.visualization.draw_track_box() 三次（matched/unmatched/removed），
        取代現行內嵌的三段重複迴圈。"""
    def _print_summary(self, stats) -> None: ...
```

`run()` 對外簽章、回傳值(`output_path, images_path, labels_path`)與行為不變,只是內部從一個大迴圈變成協調呼叫上述私有方法與協作物件。

### 3.4 `TrackingEvalPipeline.run_stage1()` 拆分

```python
def run_stage1(self) -> Path:
    cfg = self.cfg
    det_output_dir, track_output_dir = ...  # 建立輸出根目錄
    mp4s = dataset_layout.discover_source_sequences(cfg.SOURCE_VIDEO_DIR, cfg.run_det_eval_dir)
    for video_path in mp4s:
        self._process_single_video(video_path, det_output_dir, track_output_dir)
    return det_output_dir

def _process_single_video(self, video_path: str, det_output_dir: Path, track_output_dir: Path) -> None:
    """建立 VideoSequenceTracker → run() → save_mot_format() → make_det_eval_dataset()
    → 複製 trackers/ 到 stage2 → 複製 SOURCE GT 到 stage2（路徑一律用 dataset_layout.py）。"""
```

### 3.5 stage3 依賴 stage1 的結構化檢查

```python
def _ensure_stage1_output_exists(self, det_output_dir: Path) -> bool:
    """檢查 det_output_dir 下是否有任何序列輸出。沒有則印警告並回傳 False——
    這是合法的 soft-skip 情境（見 config/collision_detect.yaml 的「只重跑 stage3、
    沿用之前 stage1 舊輸出」用法：run_stage1: false + run_stage3: true）。
    行為與現行完全一致，只是抽成獨立、可單獨測試的函式。"""

def run_stage3(self):
    det_output_dir = self.cfg.OUTPUT_ROOT / self.cfg.run_det_eval_dir
    if not self._ensure_stage1_output_exists(det_output_dir):
        return
    ...
```

不採用「找不到就 raise 中止整個 pipeline」的嚴格方案——因為「只重跑 stage3、沿用舊 stage1 輸出」是既有合法使用情境,不該被當成錯誤。

### 3.6 影響檔案清單

| 檔案 | 改動 |
|---|---|
| `track_eval.py` | `VideoSequenceTracker`/`TrackingEvalPipeline` 拆分,新增 `ReidCropCollector`、`VideoWriterWithFfmpeg` 兩個協作 class;對外 CLI 與各 stage 輸出格式不變 |
| `utils/visualization.py` | 新增 `draw_track_box()` |

不動其餘模組(`module/*`、`evaluation/*`),除了呼叫已核准的 `evaluation/dataset_layout.py`。

## 4. 錯誤處理與邊界情況

- `_ensure_stage1_output_exists()` 回傳 `False` 時的印警告文字與現行 `run_stage3()` 開頭的訊息一致,不新增/移除任何錯誤情境。
- `VideoWriterWithFfmpeg.close()` 的 ffmpeg 轉檔失敗處理(保留原始 mp4v 檔 + 印警告)與現行 `VideoSequenceTracker.run()` 尾端邏輯完全一致,只是搬移位置。
- `draw_track_box()` 的 `embed_label`/`overlay_label` 兩種模式互斥;呼叫方(matched 用 `embed_label`,unmatched/removed 用 `overlay_label`)在拆分時需逐一核對現行三段邏輯的視覺輸出(顏色、文字位置、anchor)完全對應,避免拆分過程中不小心改變畫面呈現。

## 5. 驗證計畫

1. 挑 1-2 支序列,重構前後各跑一次 `python track_eval.py --config config/bytetrack.yaml`(限制序列數),比對:
   - `labels/`、`images/`、`trackers/` 輸出檔案數量與內容(MOT txt 逐行比對)一致。
   - 開啟 `save: true` 產生 `Demo.mp4`,肉眼比對標註畫面(matched 框內文字、unmatched 紅框、removed 黑框)與重構前一致。
2. `save_reid_crops: true`(如 `config/botsort_save_reid_crops.yaml`)重構前後裁切圖數量與檔名一致。
3. stage3-only rerun(`run_stage1: false` + `run_stage3: true`)在「有 / 沒有」舊 stage1 輸出兩種情況下分別驗證 `_ensure_stage1_output_exists()` 行為與現行一致。
4. Excel(HOTA/AssA/CLEAR)數值與重構前一致(純結構重構不應影響追蹤/評估數值)。

## 6. 風險總結

- **低風險**:純結構重構,對外 CLI 與輸出格式契約不變,`module/tracker` 核心 matching 邏輯完全不動。
- **中風險**:`draw_track_box()` 要同時涵蓋 matched(嵌入式文字)與 unmatched/removed(疊加式文字)兩種現行畫法,拆分時需仔細核對三段現行程式碼的視覺輸出,避免行為漂移(behavior drift)。
- **需注意的依賴**:本計畫的路徑組裝部分依賴 dataset 路徑重構已落地;若執行順序調換,需要先確認 `dataset_layout.py` 的介面存在。

## 附錄:原 `markdown/TRAJECTORY_FORMAT.md`(移除前原封不動保留)

以下內容原封不動保留自 `markdown/TRAJECTORY_FORMAT.md`(移除操作見使用者指示,原檔案已刪除)。這是 stage1(`track_eval.py`)輸出格式的完整規格,`module/collision/` 是下游消費方。文中相互參照的檔案連結(如 `[COLLISION_DETECTION.md]`)指向原 `markdown/` 目錄下的相對位置,現已不存在對應檔案,保留原文不做改寫。內文提到的行號出處(如 `track_eval.py:245`)是移除當下的行號,本次(dataset-path/track-eval 重構)後續改動可能使行號偏移,不代表本文件已同步更新。

# TRAJECTORY_FORMAT.md

## 用途
本文件是本專案追蹤軌跡輸出格式的完整規格，供跨專案介接（例如碰撞偵測模組）參考。若程式碼異動到輸出格式（欄位、路徑、座標系統等），需同步更新本文件。

## 總覽
本文件分四節，前兩節是落地檔案格式，後兩節是碰撞偵測（`module/collision/`）內部用的記憶體格式：

| | 用途 | 產生位置 | 誰該讀 |
|---|---|---|---|
| **追蹤器輸出**（見第 1 節） | tracker 逐幀輸出的實際追蹤結果，CSV 落地檔案 | `evaluation/mot_format.py::save_mot_format()` | TrackEval 評估、**下游模組（如碰撞偵測）應接這份** |
| GT 標註格式（見第 2 節） | 人工標註的 ground truth，CSV 落地檔案，僅供評估比對 | 外部標註工具產出，`evaluation/mot_converter.py` 清理 | TrackEval 評估（stage2），非即時流程，一般不用接 |
| Layer 2 後處理格式（見第 3 節） | 第 1 節轉成單一物件軌跡 dict，記憶體內結構，非落地檔案 | `module/collision/trajectory_adapter.py` | `module/collision/detection_core.py`（TTC/PET 運算） |
| Homography 座標轉換（見第 4 節） | 選用，把 Layer 2 座標轉換到有地理意義的衛星圖座標系 | `module/collision/homography*.py` | 同上，`use_homography=True` 時使用 |

---

## 1. 追蹤器輸出格式（下游對接用）

### 檔案位置
```
annotation_<seq>/trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt
```
- `<seq>`：序列名稱（已去掉 `annotation_` 前綴），例：`C000025 中華三路 民生二路(東北角)_20251209152639_15s_1`
- `<class>`：6 類固定其中之一：`person`／`car`／`scooter`／`small_truck`／`bus`／`truck`
- 每個序列 × 每個類別各一個檔案（固定六個檔案；即使該類別完全沒有追蹤結果，也會建立 0 bytes 的空檔）
- 這份最終會被複製到 `runs/track_eval/<run_track_eval_dir>/mot_annotation_<seq>/trackers/...`（同結構）供 TrackEval 讀取，是 pipeline 裡最終、格式最穩定的軌跡輸出

### 欄位定義
CSV（逗號分隔），**無 header**，每行代表一個 bbox：

| # | 欄位 | 型別/精度 | 說明 |
|---|---|---|---|
| 1 | `frame` | int，從 1 開始 | 該序列的有效幀序號（見下方「frame 語意」） |
| 2 | `track_id` | int | tracker 指派的 ID，**只在同一支影片內唯一**（見下方「track_id 語意」） |
| 3 | `left` | float，小數 2 位，像素 | bbox 左上角 x |
| 4 | `top` | float，小數 2 位，像素 | bbox 左上角 y |
| 5 | `w` | float，小數 2 位，像素 | bbox 寬 |
| 6 | `h` | float，小數 2 位，像素 | bbox 高 |
| 7 | `score` | float，小數 4 位 | 追蹤/偵測信心分數（沿用 YOLO detector 輸出，非 IoU 等追蹤品質分數） |

範例：
```
1,3,512.30,220.15,64.20,88.40,0.8123
1,7,10.00,400.00,120.50,90.00,0.6543
2,3,515.10,221.00,64.00,88.10,0.8050
```

### frame 語意
- 從 1 開始遞增，代表「有效幀」序號，等同上游 `VideoSequenceTracker.run()` 內的 `show_frame_id`（`track_eval.py:245`）。
- 「有效幀」＝已扣掉 `<seq>_frame_comparison.json` 標記為插補（interpolated）的幀（`track_eval.py:140-153`）；若序列沒有這份 json，所有幀都算有效幀，`frame` 就等於原始影片 frame index。
- 換算實際時間需搭配影片原始 fps；若序列有插補幀被跳過，不能單純用 `frame / fps` 反推，需另外對照 `frame_comparison.json` 的 `orig_frames` / `interpolated_frame_ids`。
- 實作上 `save_mot_format()` 不是解析檔名裡的幀號，而是把該序列排序後的逐幀 label 檔用 `enumerate(..., start=1)` 直接編號（`evaluation/mot_format.py:45`）。這依賴兩個前提：label 檔名用 6 位數零補齊（字串排序＝數值排序）、且每個有效幀都會產生一個 label 檔（即使該幀沒有追蹤結果，檔案仍會建立、只是內容空，`track_eval.py:247-268`）。只要這兩個前提沒被打破，frame 編號就會對齊；之後若改了命名規則或中間漏幀，會靜默錯位、不會報錯。

### track_id 語意
- 由 tracker（class-aware BoTSORT／ByteTrack）在單支影片內部指派，**每支影片各自從 tracker 初始化開始重新計數**（`VideoSequenceTracker` 每支影片各自建立一個 tracker 實例，`track_eval.py:529-545`）。
- **跨影片 track_id 會重複**，不能拿 track_id 當全局唯一身分；下游若需要跨影片/跨序列的穩定身分，需另外用「序列名 + track_id」複合鍵。
- 同一 track_id 在同一類別檔案內代表同一條追蹤軌跡；class-aware 追蹤逐類別分組，理論上同一 track_id 不會同時出現在兩個不同類別的檔案裡。

### 類別編碼
不在欄位裡，用**資料夾名稱** `YOLO11x__<class>` 表示：

| cls_id（內部） | class 名稱 |
|---|---|
| 0 | person |
| 1 | car |
| 2 | scooter |
| 3 | small_truck |
| 4 | bus |
| 5 | truck |

### 座標系統
- `left, top, w, h` 皆為**像素座標**，非正規化（0–1）；由 tracker 輸出的 `[x1,y1,x2,y2]` 換算成 `left=cx-w/2, top=cy-h/2`（`evaluation/mot_format.py:58-59`）。
- 原點為畫面左上角 `(0,0)`，x 向右、y 向下（OpenCV/影像座標慣例）。
- 只有 bbox **中心點**落在 `det_corner` 偵測多邊形內（該序列未設定多邊形時視為整張畫面）的框才會寫入；多邊形外的偵測會被丟棄，不會出現在檔案中（`evaluation/mot_format.py:62-65`）。**下游若需要偵測區域外的軌跡，目前這份輸出拿不到。**

### 與標準 MOT Challenge 格式的差異
標準 MOT Challenge tracker 輸出通常是 `frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`（9～10 欄，末 3 欄是 3D 世界座標佔位，2D 任務固定 -1）。本專案格式：
- 拿掉了 3D 佔位欄（`x,y,z`）
- 沒有 class 欄——改用「一個類別一個資料夾＋檔案」表示，對應 TrackEval 逐類別評估時預期的目錄結構

這是刻意簡化，不是漏欄位；跨專案交流時建議明確告知對方這兩點差異，避免對方直接套用標準 MOT parser 解析失敗或欄位對錯位。

---

## 2. GT 標註格式（僅供評估比對，非即時追蹤輸出）

由外部標註工具產出，經 `evaluation/mot_converter.py` 的 `fix_mot_gt_unique_ids()` / `keep_mot_gt_valid_ids()` 清理後，供 TrackEval stage2 評估使用：

- 位置：`mot_annotation_<seq>/gt/gt.txt`
- 欄位（依 `mot_converter.py` 內 docstring，最多 9 欄）：`frame,id,x,y,w,h,conf,class,vis`
  - `x,y` 為左上角像素座標，語意與第 1 節相同
  - 有 `class` 欄（第 1 節格式沒有，靠資料夾名稱表示類別）
  - `vis`（visibility，遮擋比例）欄位是否存在依標註工具而定；程式對 ≥6 欄的行就會嘗試解析前 6 欄
- 前處理內容：
  - `fix_mot_gt_unique_ids()`：同一 frame 內同一 id 出現多次時，依 `largest_area` 或 `first` 策略保留一筆，其餘重新指派新 id
  - `keep_mot_gt_valid_ids()`：只保留中心點落在偵測多邊形內的框（與第 1 節相同的區域過濾邏輯）

**這份不是追蹤器的即時輸出，是人工標註的正確答案，只在算 HOTA/CLEAR/Identity 等指標時使用。碰撞偵測若要接「系統實際追蹤到的軌跡」，應該用第 1 節的格式，不是這份。**

---

## 3. Layer 2：後處理單軌跡格式（碰撞偵測用，記憶體內結構，非落地檔案）

`module/collision/trajectory_adapter.py::load_sequence_tracks()` 把第 1 節的扁平 CSV
轉成 `module/collision/detection_core.py`（TTC/PET 核心運算，移植自
`markdown/reference_code/collision_detection_core.py`）需要的單一物件軌跡 dict，一個
track 一個 dict：

```python
{
    "Track ID": int,
    "class_id": int, "class_name": str,          # 本專案擴充欄位，非原始格式必要
    "Positions": {frame: (x, y, z)},              # bbox 中心點，z 固定 0.0
    "Bboxs":     {frame: ((x1,y1),(x2,y2))},      # bbox 左上/右下
    "Velocity":  {frame: (vx, vy)},                # 中心差分法，見下方「速度/方向怎麼算」
    "Direction": {frame: (vx, vy)},                # 與 Velocity 同值，見下方說明
    "Size":      {frame: (w, h)},
    # 有 Homography 矩陣時才有（見第 4 節）：
    "Homo_Positions": {...}, "Homo_Bboxs": {...},
    "Homo_Velocity": {...}, "Homo_Direction": {...}, "Homo_Size": {...},
}
```

### 跨類別合併的前提
`load_sequence_tracks()` 把同一序列六個類別的軌跡檔都讀進來、合併成一份清單。這能成立
是因為本專案的 class-aware tracker 對每支影片只建立**一個** tracker 實例，`track_id`
計數器是該實例共用的單一計數器（class-aware 只影響配對邏輯，不影響 ID 配發，底層是
Ultralytics `BaseTrack` 的類別層級計數器），所以同一支影片內、不同類別的軌跡檔之間
`track_id` 不會重複。這個前提**只在單一序列內成立**，跨序列/跨影片仍會重複（見第 1 節
「track_id 語意」），因此 `load_sequence_tracks()` 一次只處理一個序列。

### 速度/方向怎麼算
Layer 1 只有逐幀座標，沒有速度/方向欄位，`trajectory_adapter._center_diff_velocity()`
用中心差分法算出來：
- 中間幀：前一幀、後一幀在 frame 編號上都恰好相鄰時，`v = (pos[f+1] - pos[f-1]) * fps / 2`
- 軌跡全域首/尾幀：只有單側相鄰幀存在時，用單邊差分（前向/後向）
- 其餘情況（相鄰幀因追蹤中斷缺幀、或孤立的單幀偵測）：該幀不計算速度，**不會出現在
  `Velocity`/`Direction` dict 裡**，該幀因此無法參與 TTC 比對（`compute_ttc_for_pair()`
  要求 `Velocity`/`Direction` 等五個欄位在該 frame 都要有值才會計算，見
  `detection_core.py` 的 `keys` 檢查），但仍可用於 PET（PET 只需要 `Positions`）。
- `Direction` 直接等於 `Velocity`（同一個 dict）：`detection_core.get_collision_type()`
  只取兩個方向向量的夾角，內部會先正規化，兩者方向相同即可，不需要分開維護。

### 空類別/空序列
若某序列某類別完全沒有追蹤結果，該類別不會產生 track（Layer 1 的空檔案讀出空 dict，
迴圈不會新增任何項目），不會有殘留的空 track 混進清單。

---

## 4. Homography 座標轉換（碰撞偵測用，選用）

TTC/PET 可以選擇在原始像素座標系或「轉換後座標系」下計算（`use_homography` 參數，
見 `markdown/COLLISION_DETECTION.md`）。轉換後座標系是否等於真實地理座標，取決於
Homography 矩陣的來源，見下方。

### 矩陣來源與優先順序
`module/collision/homography_points.py::resolve_homography(seq_name, ...)` 依序嘗試：

1. **既有已算好的矩陣（優先）**：`models/homography/matrices/H_<tag>.npy`，`<tag>` 命名
   規則是 `{路口名}-{SCALE}-{ZOOM}-{寬}-{高}`（例如
   `C000025 中華三路 民生二路(東北角)-2-20-1280-1280`）。這批資料是從
   `markdown/reference_results/homongraphy_val/`（本專案自製的 Homography 批次驗證工具，
   跟 `markdown/HOMOGRAPHY.md` 描述的單一路口最小範例 `homography_core.py` 不是同一份
   程式碼）搬過來的既有成果，涵蓋 **26 個路口**，`find_tag_for_sequence()` 用「tag 去掉
   數字後綴＝路口名，序列名是否以路口名開頭」比對出對應 tag。這個矩陣是把 CCTV 像素
   座標轉換到**真實衛星底圖**（Esri World Imagery tiles，`models/homography/
   satellite_images/<tag>.png`）座標系，是有地理意義的轉換，不是本專案自己編的座標系。
2. **人工點選（fallback）**：路口不在上述 26 個之列時，退回
   `config/homography_points/<seq>.yaml`（見該檔案模組 docstring 第 2 節），此時
   `map_points` 是使用者自訂的校正座標（例如把透視變形的路口攤平成矩形），不是地理座標。
3. 兩者都沒有：回傳 `None`，該序列的 TTC/PET 退回原始像素座標（`use_homography=False`）。

`module/collision/trajectory_adapter.py::add_homography_fields()` 把矩陣套用到
Layer 2 track dict，附加 `Homo_*` 系列欄位：`Homo_Positions`/`Homo_Bboxs` 是座標點直接
套用矩陣轉換（`module/collision/homography.py::transform_positions_and_bboxes()`），
`Homo_Velocity`/`Homo_Direction` 是對**轉換後座標**重新跑一次中心差分法（不是把原始
速度向量套用矩陣——Homography 是非線性透視變換，速度向量不能像座標點一樣直接乘矩陣），
`Homo_Size` 取 `Homo_Bboxs`（簡化版兩點）的寬高。

**⚠️ 兩個座標系混用的坑（2026-08 實測 `config/collision_detect.yaml` 時發現並修掉）**：
1. `detection_core.find_ttc()` 判斷碰撞點是否落在畫面內用的 `bounds` 參數，必須跟實際
   計算座標系一致——`use_homography=True` 且用既有矩陣時，座標是衛星圖座標系（通常
   1280x1280），不是原始 CCTV 影片解析度（通常 640x480）。若誤用 CCTV 解析度當 `bounds`，
   幾乎所有合法的衛星圖座標點都會被誤判成「落在畫面外」而靜默丟棄，TTC 事件會被大量
   漏檢（實測從 0 筆到放寬 IoU 門檻後變 49 筆才發現）。`run_collision_detection_for_sequence()`
   內的 `_resolve_ttc_bounds()` 已修正：`use_homography=True` 且有 tag 時改讀衛星圖實際尺寸。
2. `compute_ttc_for_pair()`/`compute_pet_for_pair()` 回傳的 `collision_point`/`cross_point`/
   `future_box*`/`arrow*` 座標，語意上就是**運算當下用的那個座標系**——`use_homography=True`
   時就是衛星圖座標，不會自動轉回 CCTV 像素座標。若直接把這些事件寫進輸出檔案，
   數值會跟熱區圖（疊在 CCTV 第一幀上）對不起來，也跟本文件第 1 節的「CCTV 像素座標」
   慣例不一致。`run_collision_detection_for_sequence()` 已修正：`_reproject_ttc_events()`/
   `_reproject_pet_events()` 在寫檔前一律把座標反投影回 CCTV 像素座標（`H_inv`），
   所以 `ttc_events.json`/`pet_events.csv`/`collision_*_log.csv` 裡看到的座標永遠是 CCTV
   像素座標，不受 `use_homography` 影響；只有記憶體內部運算會用到衛星圖座標系。

### 資料佈局（`models/homography/`，大型檔案已 gitignore，比照 `models/` 慣例）
```
models/homography/
├── matrices/            H_<tag>.npy（26 個路口的矩陣）、pts_cctv_<tag>.npy／
│                         pts_sate_<tag>.npy（算矩陣用的 4 個訓練點，只供對照，不要重新拿來驗證）
├── satellite_images/    <tag>.png（26 張，1280x1280，Esri 衛星底圖）
├── checkpoints/         原始搬過來的既有 check points（目前只有 C000297 一個路口）
├── pick_aid/            <tag>_pick_aid.png（CCTV 第一幀＋衛星圖並排＋座標格線，
│                         人工點選 check points 用的輔助圖，見下方「驗證」）
└── validation_output/   module/collision/run_homography_validation.py 的輸出（見下方「驗證」）
```

### 驗證（reprojection error）
`module/collision/homography_validation.py::validate_tag()` 用**獨立於訓練點之外**的
check points 驗證矩陣準確度（用訓練點驗證會循環論證，誤差趨近於 0，不是有意義的驗證）：
把 check points 的 CCTV 座標套用矩陣投影到衛星座標系，跟 check points 的衛星座標（GT）
比較，算出 `dist_mean/std/median/rmse/p95/max_px`、`dx/dy_mean/std_px`，並列出誤差最大
的幾個點。`output_dir` 有提供時額外輸出（皆存到 `models/homography/validation_output/`）：

| 檔案 | 內容 | 產生條件 |
|---|---|---|
| `<tag>_overlay.png` | CCTV warp 疊到衛星圖，綠點=GT、紅點=預測（單張靜態圖） | 有 check points |
| `<tag>_stats.xlsx` | `summary`（彙總統計）+ `per_point`（逐點 gt/pred/dx/dy/dist/rank）兩個 sheet | 有 check points |
| `<tag>_heatmap_{idw,linear,rbf}.png` | 誤差空間內插熱力圖，plain 風格（底圖調暗、無色階圖例、單行標題） | check points ≥ 4 |
| `<tag>_pretty_heatmap_{idw,linear,rbf}.png` | 同上，pretty 風格（底圖全亮度、右側色階圖例、深色標題底） | check points ≥ 4 |
| `<tag>_compare.mp4` | 逐幀 warp 比對影片：左＝原始 CCTV＋check point 標記、右＝逐幀 warp 疊底圖＋GT/預測點＋統計文字（見 `module/collision/homography_compare_video.py`） | 有 check points 且提供來源影片路徑 |

以上皆改寫自 `markdown/reference_results/C000025 中華三路 民生熱力圖/` 的既有產出風格
（含檔名慣例：plain 用 `heatmap_`、pretty 用 `pretty_heatmap_` 前綴），但原始產生程式碼
已經找不到，是重新設計、獨立實作的版本，數學細節（配色、遮罩方式、字型排版）不保證
完全一致。`<tag>_compare.mp4` 需要重新逐幀讀整支來源影片做 `cv2.warpPerspective()`，
相對耗時，`run_homography_validation.py::run_all(generate_videos=False)` 可以跳過只重算
數值/圖片。

**⚠️ 只做像素誤差，沒有公尺換算**：原始 `homography_validator.py` 有公尺換算
（Web Mercator meters-per-pixel 公式，需要路口緯度），但來源的 GPS mapping Excel
（`交通局擬定RSI檢查之40個路口清單_20250430_gps_mapping.xlsx`）沒有包在
`reference_results/` 這份快照裡，目前拿不到，所以本專案版本只有像素誤差。

**⚠️ 26 個路口裡只有 6 個能真的驗證**：check points 需要同時看 CCTV 影格與衛星圖才能
標，而本專案（含 `markdown/reference_results/homongraphy_val/` 自己）都沒有其餘 20 個
路口的 CCTV 影片，物理上選不出 check points。能驗證的只有本專案實際有來源影片的 6 個：
`C000025`、`C000043`、`C000054`、`C000085`、`C000294`、`C000297`（`pick_aid/` 已為這 6
個路口產好並排點選輔助圖）。其餘 20 個矩陣仍照樣搬過來可用（`resolve_homography()`
第 1 步的查表不看有沒有驗證資料），只是 `validate_tag()` 對它們永遠回傳
`{"validated": False, "reason": "no_checkpoints"}`——**這不代表矩陣有問題，只是缺乏
驗證資料**，套用前請自行評估風險。check points 填寫方式見
`config/homography_checkpoints/README.md`；批次驗證所有 26 個路口用
`python -m module.collision.run_homography_validation`。

---

## 已知限制／踩雷點

1. **`save_mot_format()` 的檔名搜尋 glob 寫死 `"C000025_*"`**（`evaluation/mot_format.py:29,39`）：只對序列名以 `C000025_` 開頭的預設 all8 資料集有效。`config/botsort_save_reid_crops_video.yaml`（`video/` 底下其他路口影片）等非此命名的序列，會導致這份格式的軌跡檔（合併檔＋六個逐類別檔）**全部靜默輸出成空檔案，不會報錯**。串接非預設資料集的軌跡前必須先修這個 bug。
2. **`annotation_<seq>/<seq>.txt`（不在 `trackers/` 底下）不是軌跡檔**：`save_mot_format()` 額外寫的一份影像路徑清單（一行一個檔案路徑），檔名跟第 1 節格式的檔名規則很像但目錄不同、內容不是 MOT 格式，也沒有任何下游程式碼讀取，串接時應忽略，避免抓錯檔案。
3. **`mot_class = cls_id + 1`**（`evaluation/mot_format.py:60`）算出來但沒有寫進任何輸出欄位，是死程式碼；目前類別辨識完全依賴資料夾名稱，不要誤以為輸出裡有隱藏的 class 欄。
4. **track_id 不是全局唯一**：見上方「track_id 語意」。

---

## 相關程式碼位置

| 功能 | 檔案:行號 |
|---|---|
| 逐幀追蹤主迴圈，寫出中間產物（逐幀 YOLO 格式 label） | `track_eval.py:125-475`（label 寫出在 `:247-268`） |
| 呼叫 `save_mot_format()` 產生軌跡檔 | `track_eval.py:554-562` |
| 軌跡檔格式實作 | `evaluation/mot_format.py::save_mot_format()`（`:16-84`） |
| `trackers/` 複製到 stage2 輸入目錄 | `track_eval.py:567-573` |
| GT 清理（去重複 ID、多邊形過濾） | `evaluation/mot_converter.py::fix_mot_gt_unique_ids()` / `keep_mot_gt_valid_ids()` |
| stage3 入口（碰撞偵測） | `track_eval.py::TrackingEvalPipeline.run_stage3()` |
| Layer 1 → Layer 2 轉換、跨類別合併、速度/方向計算 | `module/collision/trajectory_adapter.py` |
| TTC/PET 純運算 | `module/collision/detection_core.py` |
| Homography 矩陣計算/轉換函式 | `module/collision/homography.py` |
| Homography 矩陣查找（既有矩陣優先，人工點選 fallback） | `module/collision/homography_points.py::resolve_homography()` |
| Homography 驗證（reprojection error 統計、overlay） | `module/collision/homography_validation.py` |
| Homography 誤差空間內插熱力圖（IDW/Linear/RBF，plain+pretty 兩種風格） | `module/collision/homography_heatmap.py` |
| Homography 統計數字輸出 Excel | `module/collision/homography_validation.py::export_stats_excel()` |
| Homography 逐幀 warp 比對影片 | `module/collision/homography_compare_video.py` |
| Homography 人工點選輔助圖（CCTV+衛星圖並排＋座標格線） | `module/collision/homography_pick_aid.py` |
| 批次驗證所有既有路口矩陣 | `module/collision/run_homography_validation.py` |
| TTC/PET 事件輸出（乾淨 CSV/JSON、比照原始欄位 log、熱區圖） | `module/collision/output.py` |

---

## 待確認（跨專案交流前建議跟對方對齊）
- 對方模組期望的座標系統/單位：是否也接受「像素、左上角原點」，或需要正規化座標／中心點座標
- 是否需要跨影片全局唯一的物件身分（目前 track_id 只在單影片內唯一）
- 是否需要偵測多邊形以外的軌跡（目前會被過濾掉，拿不到）
- frame 編號是否要換算成時間戳（需搭配 fps 與 `frame_comparison.json`，且需先確認目標序列沒有踩到上方「已知限制」第 1 點的 bug）
