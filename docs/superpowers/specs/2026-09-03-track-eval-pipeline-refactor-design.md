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
