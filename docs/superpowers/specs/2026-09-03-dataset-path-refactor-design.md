# Dataset 路徑結構重構 — 設計文件

- 日期:2026-09-03
- 狀態:待審查
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第一個子專案**。其餘子專案(`track_eval.py` 主流程拆分、`module/grid_search/` 去重複、各模組個別調整)各自另開 brainstorm,不在本文件範圍內。

## 1. 背景與問題

目前 `SOURCE_VIDEO_DIR` 與 `OUTPUT_ROOT` 之間的路徑命名靠散落在多個檔案的字串拼接/替換維持一致,具體痛點:

1. **`annotation_`/`mot_annotation_` 字串拼接重複**:`track_eval.py`(stage1→stage2 複製、stage3 seq 名稱還原)、三支 `module/grid_search/*.py`、`module/collision/homography_pick_aid.py` 都各自用 f-string 組出 `annotation_<seq>`、`mot_annotation_<seq>`,並用 `.replace('annotation_', '')` 在兩種前綴間轉換。任何命名規則變動要改四處以上。
2. **SOURCE 端巢狀結構要求人工對齊**:`SOURCE_VIDEO_DIR` 底下同時要有 `annotations_all8/annotation_<seq>/<seq>.mp4`(來源影片)與 `mot_annotations_all8/mot_annotation_<seq>/gt/`(人工標註 GT),兩棵樹的序列名要手動保持一致,沒有結構化保證。
3. **8 個 `config/*.yaml` 各自重複宣告 `paths` 區塊**:其中 6 個的 `SOURCE_VIDEO_DIR`/`run_det_eval_dir`/`run_track_eval_dir` 完全相同,新增情境檔容易複製貼上出錯或漏改。
4. **中文長路徑**(如「中華三路_民生二路(東北角)yolo驗證用_all8」)散落在多個腳本的預設值/硬編碼字串裡(`evaluation/config_loader.py`、`module/collision/homography_pick_aid.py`)。
5. **補幀(frame-interpolation compensation)邏輯**:`VideoSequenceTracker.run()` 讀取 `<seq>_frame_comparison.json`、`interpolated_frame_ids` 來跳過補幀,對應 SOURCE 端 `annotations_all8(compasate)` 變體目錄與 `_org.mp4`/`_interpolated_frames.txt`/`_new_frames.json`/`_orig_frames.json` 等衍生檔案。**已查證目前沒有任何 `config/*.yaml` 的 `run_det_eval_dir` 指向 `(compasate)` 變體**,此邏輯與其對應的目錄慣例是死碼,可安全整段移除。

### 已查證的關鍵事實

- TrackEval 本身只要求 `gt/mot_challenge/<benchmark>-<split>/<seq>/...` 與 `trackers/mot_challenge/<benchmark>-<split>/<tracker>/data/<seq>.txt` 存在(`evaluation/trackeval_runner.py`);**不要求**外層資料夾名稱是 `annotation_<seq>` 或 `mot_annotation_<seq>`。這兩個前綴純粹是本專案自訂慣例。
- `annotations_all8`(base,無後綴)下有 30 支序列,`(compasate)` 變體只有其中 6 支且無任何 config 引用,確認為死碼路徑。

## 2. 目標 / 非目標

**目標**
- 消除 `annotation_`/`mot_annotation_` 字串拼接與 `.replace()` 轉換,改為單一共用模組組裝路徑。
- 新增 dataset id 對照表,讓 config 用短 id 取代中文長路徑字面值,同時縮減 `paths` 區塊重複。
- 整段移除補幀相關邏輯與其死碼路徑依賴。
- 不更動 SOURCE 端既有磁碟資料(影片、人工標註 GT)的實體路徑與命名。

**非目標(明確排除,列入後續子專案或不處理)**
- `track_eval.py` 的 `VideoSequenceTracker.run()` 整體流程拆分(偵測/標註/reid裁切/影片編碼職責分離)——另開子專案。
- `module/grid_search/` 三支腳本的 trial runner / CSV 邏輯去重複——另開子專案。
- `module/tracker`、`module/reid`、`module/detection` 個別模組調整——另開子專案(視需要)。
- 整併 stage1/stage2 兩棵輸出樹為一棵(方案 B)——評估後風險較高,不在本次範圍。
- 實際重新命名磁碟上的中文資料夾/影片檔——使用者已明確選擇不動實體資料夾。
- 刪除磁碟上未被引用的 `(compasate)`/`(orign)`/`(Ver2)` 變體目錄——那是刪除既有資料的破壞性操作,需另外經使用者確認,不包含在本次程式碼重構範圍。

## 3. 新設計

### 3.1 目錄命名慣例(改前 vs 改後)

| 樹 | 改前 | 改後 | 是否為實體既有資料 |
|---|---|---|---|
| SOURCE 影片 | `SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_<seq>/<seq>.mp4` | **不變** | 是,唯讀,不動 |
| SOURCE GT | `SOURCE_VIDEO_DIR/mot_annotations_all8/mot_annotation_<seq>/gt/...` | **不變** | 是,唯讀,不動 |
| OUTPUT stage1(本專案產生) | `OUTPUT_ROOT/<run_det_eval_dir>/annotation_<seq>/...` | `OUTPUT_ROOT/<run_det_eval_dir>/<seq>/...` | 否,可重跑產物 |
| OUTPUT stage2(本專案產生) | `OUTPUT_ROOT/<run_track_eval_dir>/mot_annotation_<seq>/...` | `OUTPUT_ROOT/<run_track_eval_dir>/<seq>/...` | 否,可重跑產物 |

只改**我們自己產生的輸出樹**命名慣例(拿掉前綴),SOURCE 端既有資料完全不動。因為兩棵輸出樹改用同一個 `<seq>` 命名,原本用來在兩者間互轉的 `.replace('annotation_', '')` 自然消失——不是集中封裝轉換邏輯,而是直接讓轉換不再需要存在。

### 3.2 `evaluation/dataset_layout.py`(新檔)

集中所有路徑組裝為 pure function,供 `track_eval.py`、三支 `grid_search_*.py`、`module/collision/` 共用:

```python
def discover_source_sequences(source_video_dir: Path, run_det_eval_dir: str) -> list[Path]:
    """掃描 SOURCE_VIDEO_DIR/<run_det_eval_dir>/annotation_*/**/*.mp4，
    跳過 Demo.mp4。取代 track_eval.py 現有 glob + 補幀相關檔名過濾（一併移除）。"""

def stage1_seq_dir(det_output_dir: Path, seq_name: str) -> Path:
    """OUTPUT_ROOT/<run_det_eval_dir>/<seq_name>/"""

def stage2_seq_dir(track_output_dir: Path, seq_name: str) -> Path:
    """OUTPUT_ROOT/<run_track_eval_dir>/<seq_name>/"""

def source_gt_dir(source_video_dir: Path, seq_name: str) -> Path:
    """SOURCE_VIDEO_DIR/mot_annotations_all8/mot_annotation_<seq_name>/gt/
    （SOURCE 端既有命名不變，此函式只是讀取，不產生新命名）"""

def seq_name_from_video_path(video_path: str | Path) -> str:
    """Path(video_path).stem —— 集中定義「序列名」的單一來源。"""

def list_stage1_sequences(det_output_dir: Path) -> list[Path]:
    """列舉 OUTPUT_ROOT/<run_det_eval_dir>/ 下所有已產生的序列目錄。
    取代 track_eval.py::run_stage3() 現有的 det_output_dir.glob("annotation_*")
    （stage1 輸出已無前綴，改為列舉全部子目錄）。"""
```

呼叫方一律用這些函式取得路徑,不再自行組字串。`module/collision/run_collision_detection_for_sequence()` 讀 stage1 輸出時,也改用 `stage1_seq_dir()` 定位序列目錄(現況已是 `seq_dir.glob("annotation_*")` + `.replace("annotation_", "")`,改為直接用 `seq_name`)。

### 3.3 `config/datasets.yaml`(新檔)—— dataset id 對照表

```yaml
datasets:
  cctv_all8:
    root: "./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/"
    run_det_eval_dir: annotations_all8
    run_track_eval_dir: mot_annotations_all8
  video_seqs:
    root: "./dataset/video_seqs/"
    run_det_eval_dir: annotations_video
    run_track_eval_dir: mot_annotations_video
```

### 3.4 `evaluation/config_loader.py` 變更

`paths:` 區塊新增可選欄位 `dataset:`(dataset id)。解析順序:

1. 讀 `config/datasets.yaml`,若 `paths.dataset` 有值,取該 id 的 `root`/`run_det_eval_dir`/`run_track_eval_dir` 作為起始值。
2. 若 yaml 的 `paths` 區塊另外顯式寫了 `SOURCE_VIDEO_DIR`/`run_det_eval_dir`/`run_track_eval_dir`,以顯式值覆蓋 registry 值(逐欄位覆蓋,不是整區塊二選一)——保留 escape hatch,未登錄的臨時資料集仍可直接寫字面路徑。
3. 都沒有 `dataset` 也沒有顯式路徑時,回退現有 `_DEFAULTS`(行為不變)。
4. `dataset` 指向 `config/datasets.yaml` 中不存在的 id 時,`load_config()` 直接拋 `ValueError`(明確失敗,不可靜默回退到預設值,避免誤用錯的資料集卻無感知)。

`AppConfig` 對外屬性(`cfg.SOURCE_VIDEO_DIR`、`cfg.run_det_eval_dir` 等)維持不變,呼叫端(`track_eval.py`、grid_search、collision)不需要因為這個新機制改介面。

### 3.5 情境 yaml 精簡範例

`config/bytetrack.yaml` 的 `paths:` 區塊,改前:
```yaml
paths:
  SOURCE_VIDEO_DIR: ./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/
  TRACKEVAL_OUT_DIR: trackeval_output
  OUTPUT_ROOT: ./runs/track_eval
  run_det_eval_dir: annotations_all8
  run_track_eval_dir: mot_annotations_all8
```
改後:
```yaml
paths:
  dataset: cctv_all8
  TRACKEVAL_OUT_DIR: trackeval_output
  OUTPUT_ROOT: ./runs/track_eval
```
`config/botsort_save_reid_crops_video.yaml` 只需把 `dataset` 改成 `video_seqs`,不必重複整段路徑。

### 3.6 補幀邏輯移除範圍(全部移除)

`track_eval.py::VideoSequenceTracker.run()` 刪除:
- 讀取 `<seq>_frame_comparison.json`、`interpolated_frame_ids`、`orig_frames` 的整段邏輯。
- `should_skip_frame()` 函式與呼叫處;主迴圈不再判斷是否跳過補幀,改為對每個 frame 正常處理(等同現在「非補幀序列」的既有行為,6 個實際使用中的情境檔本來就是這個路徑)。
- `discover_source_sequences()` 不再過濾 `_org.mp4`(該過濾條件本來就是補幀變體的副產物,拿掉掃描來源後這個條件本身也是死碼)。

### 3.7 影響檔案清單

| 檔案 | 改動 |
|---|---|
| `evaluation/dataset_layout.py`(新) | 集中路徑組裝與來源序列掃描函式 |
| `config/datasets.yaml`(新) | dataset id 對照表 |
| `evaluation/config_loader.py` | 解析 `paths.dataset`、讀 registry、逐欄位覆蓋邏輯、`dataset` 找不到時報錯 |
| `track_eval.py` | 移除補幀邏輯;stage1/stage2 路徑組裝改呼叫 `dataset_layout.py`;`run_stage3()` 改用 `list_stage1_sequences()` 列舉序列目錄,`seq_dir.stem.replace("annotation_", "")` 改為直接用 `seq_dir.stem`(stage1 目錄已不帶前綴);移除所有 `.replace('annotation_', '')` |
| `config/*.yaml`(8 個) | `paths:` 區塊精簡為 `dataset:` + 差異欄位 |
| `module/grid_search/grid_search_bytetrack.py` / `grid_search_botsort.py` / `grid_search_botsort_reid.py` | 路徑組裝改呼叫 `dataset_layout.py`,移除自行組的 `mot_annotation_{seq_stem}` |
| `module/collision/homography_pick_aid.py` | 硬編碼中文路徑改用 `config/datasets.yaml` 的 `cctv_all8` |

## 4. 錯誤處理與邊界情況

- `dataset` 欄位打錯字/不存在 registry → `load_config()` 拋 `ValueError`,訊息列出可用的 dataset id,避免誤跑到預設值。
- `config/datasets.yaml` 檔案不存在 → 視為沒有任何 dataset 可用;若某 config 有 `paths.dataset` 但 registry 檔案缺失,同樣拋錯而非靜默忽略。
- 舊的 `runs/track_eval/` 輸出(改版前產生,目錄仍帶 `annotation_`/`mot_annotation_` 前綴)與新程式碼**不相容**(新程式碼找 `<seq>/` 找不到會視為沒有該序列的既有輸出)。這是預期行為,不做新舊格式相容讀取;建議重構落地後,對受影響情境重新跑 stage1(是可重生產物,非破壞性,但仍屬於「大範圍重跑」,實際執行前依規則需先確認範圍與預估時間)。

## 5. 驗證計畫

依 `markdown/TEST.md` 流程,針對本次改動:
1. `python track_eval.py --config config/bytetrack.yaml`,限制在 1-2 支序列(改 `SOURCE_VIDEO_DIR` 或程式內切片)跑完整 stage1+stage2。
2. 確認輸出目錄變成 `runs/track_eval/annotations_all8/<seq>/`、`runs/track_eval/mot_annotations_all8/<seq>/`(無前綴)。
3. 比對 Excel 的 HOTA/AssA/CLEAR 數值與改動前一次已知結果一致(見 `markdown/VERIFY.md` 是否有可比對的既有基準;若無,以改動前 commit 跑一次同序列作為 baseline)。
4. `config/botsort_save_reid_crops_video.yaml`(`video_seqs` dataset)跑 stage1,確認 `dataset:` 別名解析與 escape hatch(顯式覆蓋)都正確。
5. 任一支 `module/grid_search/*.py` 用 `--dry-run` 確認路徑組裝不報錯,再用 `--max-videos 1` 跑一組 trial 驗證 CSV 輸出正常。
6. 確認補幀相關程式碼路徑已整段移除後,既有 6 支主要序列(非補幀資料)行為不受影響(這些序列本來就沒有 `_frame_comparison.json`,現況已是跑「不跳幀」路徑)。

## 6. 風險總結

- **低風險**:SOURCE 端資料完全不動;輸出目錄命名改變只影響可重生的 `runs/` 產物。
- **中風險**:四類呼叫方(track_eval.py、3 支 grid_search、collision、homography_pick_aid)都要同步改完,任一處遺漏字串拼接會造成路徑對不上;需要在驗證階段逐一跑過。
- **需使用者額外確認才執行**:清理磁碟上未被引用的 `(compasate)`/`(orign)`/`(Ver2)` 目錄(不在本次程式碼變更範圍內,只是文件中提及其死碼狀態)。
