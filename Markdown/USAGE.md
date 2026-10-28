# safetrajai 使用說明（USAGE）

> 整理日期：2026-09-24
> 依據：repo 現況（各腳本 `--help`、`config/`、`utils/modules/safetrajai/`）。
> 術語見 `CONTEXT.md`。

---

## 0. 兩次變更的重點（總結）

### 0.1 `restructure-project-layout`（已歸檔，32/32）

把原本靠 `sys.path.insert` 和 `Path(__file__).parents[N]` 拼湊起來的專案，整理成可安裝的套件 `safetrajai`（`src/` 佈局）：

| 項目 | 重構後的規則 |
|---|---|
| 安裝 | conda env `track_eval` + `pip install -e . --no-deps`，任意工作目錄都能 `import safetrajai` |
| 專案根目錄 | 只能透過 `safetrajai.paths`（`PROJECT_ROOT`／`CONFIG_DIR`）取得 |
| 類別 ID 契約 | 只在 `safetrajai/classes.py` 定義一次；MOT 用的 1-based ID 由 `id + 1` 推導 |
| 主入口 | `track_eval.py` 成為相容 wrapper，實作拆到 `safetrajai/pipeline/*`、`safetrajai/cli/run_pipeline.py` |
| CLI | 全部集中到 `scripts/`；舊的 `module/...`、`visualize/...` 路徑已失效（Breaking） |
| Tracker | 整檔搬到 `safetrajai/tracking/`，override 方法本體一字不改（smoke 測試以 AST 比對） |
| Re-ID 權重 | `utils/model.py` 改為 pickle 相容 shim，實作移到 `safetrajai/reid/model.py` |
| 回歸防線 | `tests/test_smoke.py` + `tests/golden/`（5 個情境、grid search、homography，容忍度 0） |

### 0.2 `reorganize-dataset-and-eval-gt`（已歸檔，33/33）

修正「評估 GT 其實不是原始人工 GT」這個問題，同時把資料依角色分層、決定 ROI 怎麼套用：

| 決策 | 內容 |
|---|---|
| D1 評估資料集改名 | `dataset/eval/cctv_all8/`，以原始人工標註（`(orign)`，10401 框）為準，內容不可修改 |
| D2 原始影片歸位 | `dataset/raw/videos/<camera>/<seq>.mp4`；`datasets.yaml` 新增 `role`／`layout`／`cameras` |
| D3 隔離慣例 | 有疑慮的資產一律 `mv` 到同目錄的 `temp/`，並寫 `DO_NOT_USE.md`，不刪除 |
| D4 ROI 以鏡頭為鍵 | `config/roi/<camera>.yaml`；config 頂層的 `det_corner` 停用，出現就報錯 |
| D5 stage1 不過濾 | 7 欄軌跡保留全部結果；stage3／Re-ID 裁切逐列以中心點過濾（**要在依 track_id 分組之前過濾**） |
| D6 忽略區域評估 | stage2 在 TrackEval 前做 UA-DETRAC 式 ignore 匹配，FP 只來自 ROI 內 |
| D7 detection 評估 | 在 `OUTPUT_ROOT/det_eval/<seq>/` 以 hardlink 組出人工 GT 資料集；報表標示「ROI：未套用（全畫面）」 |
| D8 暫代權重 | `botsort_reid.yaml` 的 small_truck／scooter 暫用 `models/yolo/yolo11x.pt` |
| D9 移除舊訓練端 | 刪除 YOLO-as-encoder 訓練程式；推論端 `ReID`／`Custom_ReID` 保留 |
| D10 golden 基準 | 基準由 `before1` 改為 `after1` |

### 0.3 目前進行中：`rebuild-reid-dataset`

> 注意：`openspec-reorganize-dataset-and-eval-gt-總結.md` 寫的是「0/約 26，尚未開始」，但這份文件撰寫時 `tasks.md` 已有 **14 項完成、12 項未完成**，以下以目前狀態為準。

| 階段 | 狀態 |
|---|---|
| 1.1 ROI 點選頁工具（`scripts/picker/make_roi_picker.py`） | ✅ 完成 |
| 1.2～1.3 來源鏡頭的 ROI 檔 | ✅ 9 支來源鏡頭全部完成（見 5.2） |
| 2.1～2.12 程式修改（`frame_rate: auto`、`paths.cameras`、裁切索引、manifest、污染偵測、審查頁、排除清單等） | ✅ 完成 |
| 3.1～3.3 工具模型、追蹤參數比較與 `config/reid/reid_crops.yaml` | ✅ 完成 |
| 4～7（golden 裁切情境、pilot、全量產生與審查、文件收尾） | ⏳ 未完成 |

---

## 1. 安裝

只能用 conda env `track_eval`，**不要用 uv 或另外建 venv**；安裝時一律加 `--no-deps`，避免 pip 把 CUDA 版 torch 換成 PyPI 版。

```bash
conda activate track_eval
pip install -e . --no-deps
pip install pytest          # dev 依賴
```

- 依賴以 `pyproject.toml` 為準；`requirement.txt` 是舊的 pip 腳本，不是權威來源。
- 還沒安裝就執行 `python track_eval.py` 時，程式會印出上面的安裝指令並以 exit 1 結束。
- 所有指令都**從 repo 根目錄執行**（`models/`、`dataset/`、`config/` 都是相對路徑）。
- 執行時可能會出現 `Error importing BURST due to missing underlying dependency: No module named 'tabulate'`，這是 vendored TrackEval 的選用資料集模組，本專案不會用到，可以忽略。

---

## 2. 資料與目錄慣例

### 2.1 `dataset/` 分層

```
dataset/
├── raw/videos/<camera>/<seq>.mp4      原始影片（依鏡頭歸位）→ datasets.yaml: reid_sources（role: source）
├── eval/cctv_all8/                    評估資料集（人工 GT，唯讀）→ datasets.yaml: cctv_all8（role: eval）
│   ├── annotations_all8/annotation_<seq>/          detection GT + 序列影片
│   └── mot_annotations_all8/mot_annotation_<seq>/  MOT GT
├── reid/                              Re-ID 裁切、排除清單、資料集版本（舊內容已隔離到 temp/）
└── reports/                           稽核報表（例如 reports/reid/）
```

### 2.2 `config/datasets.yaml`

情境 config 用 `paths.dataset: <id>` 引用，loader 會自動補上 `root`、`run_det_eval_dir`、`run_track_eval_dir`：

| id | role | layout | cameras |
|---|---|---|---|
| `cctv_all8` | `eval`（只用來評估） | `annotated`（`<root>/annotations_all8/annotation_<seq>/<seq>.mp4`） | `C000025` |
| `reid_sources` | `source`（用來產生訓練資料） | `raw`（`<root>/<camera>/<seq>.mp4`） | `C000011`、`C000012`、`C000033`、`C000043`、`C000048`、`C000054`、`C000056`、`C000085`、`C000256`（`C000026` 與評估鏡頭同路口，影片歸位但不列入） |

- eval 與 source 的鏡頭不能重疊（smoke 測試會檢查）；**評估鏡頭 C000025 不能進任何訓練資料**。
- 情境 config 可以另外加 `paths.cameras: [...]`，只跑資料集登記鏡頭中的一部分；列出未登記的鏡頭會直接報錯。
- 想縮小範圍時，也可以在 `paths:` 直接寫 `SOURCE_VIDEO_DIR` 指向單一序列所在目錄。

### 2.3 隔離慣例（`temp/`）

- 有疑慮、不可使用的權重或資料，用 `mv` 移到**所在目錄**下的 `temp/`，不刪除。
- **先寫紀錄再搬移**：在 `temp/DO_NOT_USE.md` 記錄原路徑、日期、原因與還原指令。
- `config/` 與 `tests/golden/configs/` 不能引用 `temp/` 內的路徑（smoke 測試會攔截）。
- 目前已隔離：`dataset/reid/` 全部舊內容、`models/yolo-reid/*`、`models/class_reid_weights/{person,scooter}`、`all8` 的限縮版與重複版標註。

### 2.4 禁止大量 I/O

`dataset/`、`models/`、`runs/` 在掛載磁碟上，檔案數以百萬計。**不要**做全量 `sha256sum`／`md5sum`、全量 `du`、`find … -exec cat`、`rsync --checksum`。需要核對時，只比對路徑清單與檔案數（`find -type f | wc -l`）。

---

## 3. 主流程：`track_eval.py`

```bash
python track_eval.py --config <config>          # 等同 python scripts/pipeline/run_pipeline.py --config <config>
python track_eval.py --config <config> --track-config <tracker_yaml>
```

| 參數 | 說明 |
|---|---|
| `--config` | 情境設定檔；不帶時預設為 `config/tracking/bytetrack.yaml` |
| `--track-config` | 另外指定 class-aware tracker 參數 YAML；不帶時直接讀 `--config` 檔內的 `class_tracking:` |

執行順序：`load_track_config()` → `load_config()` → `TrackingEvalPipeline`，依 config 的 `stages.run_stage1/2/3` 執行各階段。

### 3.1 內建情境

| config | stage | 用途 |
|---|---|---|
| `config/tracking/bytetrack.yaml` | 1 + 2 | ByteTrack 完整流程（預設） |
| `config/tracking/botsort_noreid.yaml` | 1 + 2 | BoTSORT，不開 ReID |
| `config/tracking/botsort_reid.yaml` | 1 + 2 | BoTSORT + 逐類別 ReID（small_truck／scooter 權重是暫代的 `yolo11x.pt`） |
| `config/reid/botsort_save_reid_crops.yaml` | 1 | 追蹤並蒐集 Re-ID 裁切（`paths.dataset: cctv_all8`，見下方注意事項） |
| `config/reid/botsort_save_reid_crops_video.yaml` | 1 | 同上，改對 `reid_sources` 蒐集 |
| `config/collision/collision_detect.yaml` | 1 + 3 | 追蹤後做衝突事件分析（TTC／PET） |
| `config/grid_search/grid_search_*.yaml` | — | grid search 腳本使用的基準設定 |

> ⚠️ `rebuild-reid-dataset` 的 2.6 已加上裁切前檢查：開啟 `save_reid_crops` 時，**評估鏡頭一律拒絕**，而且每支鏡頭都必須有 `config/roi/<camera>.yaml`。因此：
> - `botsort_save_reid_crops.yaml`（`cctv_all8`，C000025）目前會直接報錯。
> - 正式的裁切設定是 `config/reid/reid_crops.yaml`（`reid_sources`、`frame_rate: auto`，輸出到 `dataset/reid/crops`）。
> - `botsort_save_reid_crops_video.yaml` 保留作為 `frame_rate: 30` 的對照情境，裁切輸出改到 `runs/reid_crops_video/crops`。

### 3.2 情境設定檔結構

每個情境 `config/<功能>/*.yaml` 都是**完整、自包含**的設定。下列區塊名稱是穩定介面，不能改名。新實驗請另外建檔，不要覆寫既有的基準情境。

```yaml
paths:
  dataset: cctv_all8               # 對照 config/datasets.yaml
  # cameras: [C000043]             # 選用：只跑登記鏡頭中的一部分
  # SOURCE_VIDEO_DIR: ...          # 選用：顯式覆寫來源目錄
  TRACKEVAL_OUT_DIR: trackeval_output
  COLLISION_OUT_DIR: collision_output   # stage3 使用
  OUTPUT_ROOT: ./runs/track_eval
models:
  det_model_path: ./models/yolo/yolo11x_dennis.pt
trackeval: { benchmark_name: CCTV, split_name: val, tracker_name: YOLO11x }
stages:        { run_stage1: true, run_stage2: true, run_stage3: false }
eval_flags:    { run_det_eval: false, run_track_eval: true }   # 只控制 stage2 內的子工作
classes_to_run: { person: 0, car: 1, scooter: 2, small_truck: 3, bus: 4, truck: 5 }
tracking_params:
  tracker: bytetrack               # bytetrack | botsort
  conf: 0.15
  iou: 0.50
  imgsz: 640
  frame_rate: 30                   # 數值，或 auto（讀影片實際幀率並四捨五入）
  gmc_method: ecc                  # 只對 BoTSORT 有作用
save params:
  save: true                       # 輸出標註影片 Demo.mp4
  save_txt: true
  save_reid_crops: false
  reid_crop_dir: ./dataset/reid/crops
class_tracking:                    # 由 load_track_config() 讀取
  class_names: { 0: person, 1: car, ... }
  class_tracking_params:
    car: { match_thresh: 0.7, second_match_thresh: 0.6, unconfirmed_match_thresh: 0.9,
           new_track_thresh: 0.25, track_buffer: 3.0, use_reid: false, ... }
```

重點：

- `stages.*` 決定要跑哪幾個階段；`eval_flags.*` 只控制 stage2 內部要做 detection 評估還是 tracking 評估，兩者意義不同。
- 頂層 `det_corner` 已停用，出現就報錯；ROI 請改寫在 `config/roi/<camera>.yaml`。
- `track_buffer` 在 BoTSORT 版被當成「秒」（會乘上 frame rate），在 ByteTrack 版則是以 30fps 為基準的幀數。這是已知落差，不會修正。
- ReID 預設關閉：實驗結論是它對 person 有幫助，但會讓 scooter 變差、IDSW 增加。

### 3.3 輸出目錄

```
OUTPUT_ROOT/                                   （預設 ./runs/track_eval）
├── annotations_all8/<seq>/                    stage1
│   ├── images/<seq>/*.jpg、labels/<seq>/*.txt
│   ├── trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt   ← 下游的穩定輸入
│   └── Demo.mp4（save: true 時）
├── mot_annotations_all8/<seq>/                stage2 用的 tracker 副本（原始副本備份在 trackers_raw/）
├── det_eval/<seq>/                            stage2 detection 評估資料集（人工 GT 的 hardlink）
├── trackeval_output/<seq>/                    stage2 報表（det_metrics_*.xlsx、trackeval_metrics_*_MULTI_new.xlsx）
└── collision_output/<seq>/                    stage3（ttc_events.*、pet_events.*、熱區圖）
```

輸出端的目錄直接用 `<seq>/`；來源端沿用舊命名 `annotation_<seq>/`、`mot_annotation_<seq>/`。程式內的路徑一律用 `safetrajai/evaluation/dataset_layout.py` 的函式組出來，不要自己拼字串。

### 3.4 軌跡格式契約（stage1 → stage2／stage3）

- 檔案：`trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt`
- 每行 7 欄、沒有 header：`frame,track_id,left,top,w,h,score`（像素座標、以左上角為原點，`frame` 從 1 開始）。
- 檔案內沒有 class 欄，類別由資料夾名稱決定；6 個類別都一定會有檔案，沒有結果時是 0-byte 空檔。
- **不依 ROI 過濾**，保留全部追蹤結果。
- `track_id` 只在「單一序列 × 單一類別」內唯一；跨序列要用「序列名＋track_id」當鍵。

---

## 4. 各階段說明

### 4.1 Stage1：偵測 + class-aware 追蹤

- `safetrajai/detection/`：`Detector` 做單幀 YOLO11 推論，再依逐類別信心門檻（`class_conf_thresh`）過濾。
- `safetrajai/tracking/`：唯一入口是 `create_tracker()`。偵測結果**先依類別分組，再做三階段匈牙利匹配**，每個類別的參數由 `get_params_for_class()` 查表取得。
- `load_track_config()` 必須在 `create_tracker()`／`Detector` 建立之前呼叫。順序錯了不會報錯，只會悄悄用到空的或舊的參數。
- 改動 tracker 前，先讀 `utils/modules/safetrajai/tracking/HANDOFF.md` 與 `class_aware_*_flow.md`。`ClassAwareBOTSORT`／`ClassAwareBYTETracker` 的 override 方法（`__init__`、`update()`、`remove_duplicate_stracks()`、`init_track()`）**不可更動**（使用者明確指示，詳見 `CLAUDE.md` 的 Tracker 一節）。

### 4.2 Stage2：TrackEval 評估

只對既有的 stage1 輸出重跑評估：

```bash
python scripts/pipeline/run_evaluation.py --config <config>
```

處理順序：

1. 準備 GT shadow（輸出目錄中的副本）。
2. **忽略區域前處理**（`evaluation/ignore_region.py`，UA-DETRAC 式）：
   1. GT 中心點在 ROI 外的標為 ignore。
   2. 預測與全部 GT 以 IoU ≥ 0.5 做一對一匹配。
   3. 匹配到 ROI 內 GT 的預測保留。
   4. 其餘預測中，匹配到 ignore GT 或中心點在 ROI 外的移除。
   5. 移除 ignore GT。
3. 呼叫 vendored TrackEval（HOTA／CLEAR／Identity），輸出 Excel 報表。
4. `eval_flags.run_det_eval: true` 時，以人工 GT 組 `det_eval/<seq>/` 資料集，交給 `det_model.val()`（imgsz 取 `tracking_params.imgsz`）。報表會標示「ROI：未套用（全畫面）」。

所有中間產物只寫入輸出目錄，不會改動來源 GT 或 stage1 軌跡。

### 4.3 Stage3：衝突事件（TTC／PET）

```bash
python track_eval.py --config config/collision/collision_detect.yaml
```

- 程式碼中的 `collision` 指的是**衝突事件**（TTC／PET 低於門檻的接近事件），不是實際碰撞。
- 輸入只讀 stage1 的 7 欄軌跡，不讀 GT；讀入後逐列以 ROI 中心點過濾。
- 參數在 config 的 `collision:` 區塊：`homography_points_dir`、`max_ttc_frames`、`iou_threshold`、`lookback_sec`、`overlap_threshold`。
- 事件結果請讀 `ttc_events.{json,csv}`、`pet_events.{json,csv}`，不要依賴舊 log 的 `colision_occur_sec`（PET 的這個欄位可能記的是 frame）。
- Homography 查找順序：`models/homography/` 既有鏡頭矩陣 → `config/homography_points/<seq>.yaml` 人工點選 → 都沒有時直接用 CCTV 像素座標。
- 誤差單位是像素，不是公尺；未經實測校正的事件數只能當作技術訊號，不能當成正式的安全結論。

---

## 5. ROI（關注區域）

### 5.1 檔案格式：`config/roi/<camera>.yaml`

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

### 5.2 手動畫 ROI（完整步驟）

伺服器沒有圖形介面，所以改成輸出單檔 HTML，在自己電腦的瀏覽器上點選多邊形。整個流程是：**產生點選頁 → 在本機開啟 → 點出頂點 → 下載 YAML → 放回 `config/roi/` → 驗證**。

#### 步驟 1：產生點選頁

```bash
python scripts/picker/make_roi_picker.py --camera C000054
python scripts/picker/make_roi_picker.py --camera C000054 --out-dir runs/roi_picker
```

- 程式從 `dataset/raw/videos/<camera>/` 第一支影片取首幀，以 base64 內嵌成單一 HTML 檔。
- 輸出到 `runs/roi_picker/<camera>.html`（gitignore，CCTV 畫面不會進 git）。
- 頁面不需要網路，也不會上傳任何東西。

#### 步驟 2：在本機開啟網頁（三選一）

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

#### 步驟 3：在頁面上畫 ROI

1. 在畫面上**依序點出多邊形頂點**，建議順時針。座標會自動換算成原始影像的像素。
2. 點錯了可以按 **「復原上一點」**；想整個重畫就按 **「清除重畫」**。
3. 頁面下方會即時顯示 YAML（`camera`、`image_size`、`polygon`）。
4. 畫好後按 **「複製 YAML」** 或 **「下載 <camera>.yaml」**。

範圍的建議：框出路面上想分析或蒐集的區域，排除畫面邊緣、遠處的小目標，以及浮水印或時間戳的位置。框中心點落在 ROI 外的偵測，會被 stage3 與 Re-ID 裁切略過，在 stage2 則當作忽略區域處理。

#### 步驟 4：把 YAML 放回專案

放在 **`config/roi/`**，檔名維持 `<camera>.yaml`（例如 `C000054.yaml`）。程式以鏡頭編號找檔案，檔名不對就會找不到。

- VS Code：把檔案拖到檔案總管的 `config/roi/` 上就會上傳。
- scp：

  ```bash
  scp C000043.yaml C000054.yaml "<user>@<server>:/mnt/moris/Dcoker_Project/Project/safetrajai_eval/config/roi/"
  ```

- 不要覆蓋既有的 `C000025.yaml`（評估鏡頭）。

#### 步驟 5：驗證尺寸

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

#### 目前狀態（2026-09-24）

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

---

## 6. 追蹤參數 Grid Search

```bash
python scripts/grid_search/grid_search_bytetrack.py   --target-class scooter
python scripts/grid_search/grid_search_botsort.py     --target-class scooter --tracker botsort
python scripts/grid_search/grid_search_botsort_reid.py --target-class person
```

| 參數 | 說明 |
|---|---|
| `--target-class` | 要調優的類別：`person`／`car`／`scooter`／`small_truck`／`bus`／`truck` |
| `--config`／`--track-config` | 基準設定（`--track-config` 不帶時讀 `--config` 的 `class_tracking:`） |
| `--tracker` | `botsort`／`bytetrack` |
| `--model` | YOLO 權重，預設取 config 的 `det_model_path` |
| `--video-glob`／`--max-videos` | 指定影片範圍；`--max-videos` 預設 2 |
| `--params-subset` | 只搜尋指定參數，例如 `match_thresh,new_track_thresh` |
| `--dry-run` | 只計算組合數，不實際執行 |
| `--resume`／`--result-csv` | 從既有 CSV 接續（只跳過完全相同且已完成的組合） |
| `--output-root`／`--verbose` | 暫存輸出位置／印出逐幀進度 |
| （僅 botsort）`--conf`、`--iou`、`--imgsz`、`--frame_rate`、`--gmc_method` | 覆寫推論與 GMC 參數 |

- 分數是 `HOTA.AssA_mean`，屬於關聯指標，**不等於 HOTA**。
- 每個 trial 的流程固定：深拷貝參數 → patch `CLASS_TRACKING_PARAMS` → tracking → MOT 輸出 → TrackEval → 取分數 → 寫 CSV → 還原參數。無論成功、跳過或拋出例外，都會還原參數。
- 建議先用 `--dry-run` 看組合數，再用 `--max-videos 1` 做最小驗證。

---

## 7. Re-ID 資料工具（`rebuild-reid-dataset` 新流程）

> 這組工具已經寫好並通過單元測試，但**實際裁切蒐集還沒執行**（要等 ROI 1.2、裁切設定 3.3 與 pilot 完成）。舊的 `dataset/reid/` 內容與舊權重都已隔離，不要引用或拿來訓練。`scripts/reid_data/README.md` 目前仍是舊流程的說明（預計在 7.1 更新）。

新流程：**蒐集 → 標記 → 審查 → 排除 → manifest**

```
stage1（save_reid_crops: true，只限 source 鏡頭且需要 ROI 檔）
   │  裁切 + <crop-root>/_index/*.csv（每張裁切一列）
   ▼
detect_contamination.py        DINOv2 特徵 → 類別一致性／軌跡內外觀離群值標記
   ▼
generate_identity_review.py    產生本機 HTML 審查頁 → 匯出確認清單 CSV
   ▼
apply_review_exclusions.py     合併進 dataset/reid/exclusions.csv（預設 dry-run）
   ▼
build_reid_split.py            產生 dataset/reid/splits/<version>/manifest.csv
```

### 7.1 裁切蒐集（stage1 的 `save_reid_crops`）

- 以 ROI 中心點過濾，最短邊 < 32px 的框略過，每條軌跡每 0.5 秒取一張。
- 索引一次寫出；索引已存在時報錯（不會覆寫）。
- 追蹤前會先檢查：評估鏡頭一律拒絕，每支鏡頭都必須有 ROI 檔，任一不通過就不會產生任何輸出。
- 序列名稱必須能解析出日期，否則報錯。

### 7.2 污染偵測

```bash
python scripts/reid_data/detect_contamination.py
python scripts/reid_data/detect_contamination.py --percentile 98 --min-agreement 0.6
```

- 每張裁切只讀一次，以 DINOv2 特徵同時做類別中心點比對與軌跡內外觀離群值偵測。
- 輸出到 `--out-dir`（預設 `dataset/reports/reid/`）：`class_consistency.csv`、`appearance_outliers.csv`、`contamination_crops.csv`。
- 只寫標記檔，不搬移、改名或刪除裁切；DINOv2 權重第一次使用時會從 HF Hub 下載。
- 其他參數：`--crop-root`、`--batch-size`、`--min-outliers`、`--min-fraction`。

### 7.3 人工審查頁

```bash
python scripts/reid_data/generate_identity_review.py
python scripts/reid_data/generate_identity_review.py --out-dir dataset/reports/reid/review --page-size 100
```

- 審查範圍是「任一檢查標記的身分＋bus 的全部身分」，依類別分頁；每個身分最多 12 張依時間均勻抽樣的縮圖。
- 頁面上的「產生確認清單」按鈕會匯出 CSV（`class,identity,image,action`；`image` 空白代表整個身分）。

### 7.4 套用排除清單

```bash
python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv          # dry-run
python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv --apply  # 實際寫入
```

- 合併進 `dataset/reid/exclusions.csv`（可用 `--exclusions` 指定），去除重複並依鍵排序。
- 只寫排除清單，不動任何裁切；排除是在產生資料集版本時由 `build_reid_split.py` 套用。

### 7.5 產生資料集版本（manifest）

```bash
python scripts/reid_data/build_reid_split.py --version v1
python scripts/reid_data/build_reid_split.py --version v2 --exclusions dataset/reid/exclusions.csv
```

- 輸入：`<crop-root>/_index/*.csv`、排除清單（可以不存在）、`config/reid/reid_split.yaml`（可用 `--rules` 指定）。
- 輸出到 `dataset/reid/splits/<version>/`（可用 `--splits-root` 指定），版本已存在時報錯：
  - `manifest.csv`：`crop_path,identity,class,camera,date,split`
  - `summary.json`：各類別／split／鏡頭的身分數與張數
  - `split_rules.yaml`：本次使用的規則複本
- 不複製任何影像。
- 切分規則（`config/reid/reid_split.yaml`）：列在 `val` 的「鏡頭＋日期」組合為 val（目前是 C000054 的 20250629），其他來源鏡頭全部是 train；每個身分最少 4 張、最多 30 張（均勻抽樣）；評估鏡頭永遠不會出現在任何版本。

> CLAUDE.md 的常用指令仍寫著 `build_reid_split.py --mode category|flat`，那是舊版介面；現在的 CLI 是 `--version`。

### 7.6 Re-ID 推論端

- `safetrajai/reid/encoders.py`：`ReID`（YOLO 當 encoder）與 `Custom_ReID`（`models/class_reid_weights/`）兩條推論路徑，透過 BoTSORT 的逐類別 `use_reid` 啟用。
- `utils/model.py` 是 pickle 相容 shim，**不可刪除或改名**（舊格式權重靠它反序列化）。
- 不要再引入已經移除的 SOTA_ReID、FastReID、TransReID。

---

## 8. 驗證與視覺化工具

### 8.1 Homography 驗證

```bash
python scripts/homography/validate_homography.py
```

- 沒有任何參數（也不支援 `--help`，**直接執行就會開始驗證**）。它會掃描全部序列，對有 checkpoint 的序列輸出比對影片與誤差統計，其餘印出 `skipped (no_checkpoints)`。
- 輸出到 `models/homography/validation_output/`（`<seq>_compare.mp4`、`summary.csv`）。
- 目前只有 C000025 有 checkpoint，而且少於 4 點，所以會跳過誤差熱力圖。

### 8.2 在序列影片上疊 GT／追蹤結果

```bash
python scripts/visualization/visualize_gt_on_video.py \
  --trackeval_root <mot_annotation_<seq> 根目錄> \
  --video_path <來源影片，可自動偵測> \
  --output_video <輸出.mp4>
```

常用選項：`--show_class_filter person car`、`--color_by {class,id}`、`--no_id`、`--no_class`、`--show_conf`、`--no_legend`、`--bbox_alpha 0~1`、`--max_frames N`（除錯用）。`scripts/visualization/visualize_gt_on_class_video.py` 的介面相同。

---

## 9. 測試與回歸驗證

| 工具 | 指令 | 用途 |
|---|---|---|
| smoke／單元測試 | `pytest` | 模組 import、全部 config 載入、類別 ID 契約、tracker AST 不變性、`sys.path` 掃描、`scripts/**/*.py --help`、ROI／忽略區域、Re-ID 資料流程。幾秒內跑完，任何結構調整後都要跑 |
| golden 執行 | `tests/golden/run_all.sh <run>` | 5 個情境＋grid search＋homography 驗證，平行執行約 16 分鐘，輸出到 `runs/refactor_golden/<run>/` |
| golden 比對 | `python tests/golden/compare.py reid1 <run>` | 與基準 `reid1` 比對，容忍度 0，任何非零差異都代表行為改變 |

- 基準在 2026-09-23 由 `before1` 改為 `after1`，2026-09-24 再改為 `reid1`（裁切情境改在來源鏡頭 C000043 上蒐集），原因見 `tests/golden/TOLERANCE.md`。
- 改動演算法時，要實際跑受影響的 stage 再比對輸出；比較追蹤效果時，看同一基準下的 HOTA、AssA、IDSW，再用疊圖影片檢查 ID switch 和跨類別誤配。
- 全量序列、grid search 和重跑 stage1 都很耗時，也可能覆寫既有輸出，執行前要先確認範圍。

---

## 10. 常見情境速查

| 我想要… | 做法 |
|---|---|
| 跑基準評估 | `python track_eval.py --config config/tracking/bytetrack.yaml` |
| 只重算 stage2 | `python scripts/pipeline/run_evaluation.py --config <config>` |
| 只跑某支鏡頭 | 另建 config，加上 `paths.cameras: [<camera>]`（必須是資料集登記鏡頭的子集） |
| 開 detection 評估 | `eval_flags.run_det_eval: true`（使用人工 GT，全畫面、不套 ROI） |
| 跑衝突事件 | `python track_eval.py --config config/collision/collision_detect.yaml` |
| 新鏡頭畫 ROI | `python scripts/picker/make_roi_picker.py --camera <id>` → 本機瀏覽器點選 → 存成 `config/roi/<id>.yaml`（完整步驟見 5.2） |
| 調某類別的追蹤參數 | `grid_search_*.py --target-class <cls> --dry-run` → `--max-videos 1` → 正式跑 |
| 隔離一份有問題的權重 | 先寫 `temp/DO_NOT_USE.md`，再 `mv` 到同目錄的 `temp/` |
| 確認沒有改壞 | `pytest`；行為可能有變時再跑 golden 並與 `reid1` 比對 |

---

## 11. 已知限制

- `track_buffer` 在 BoTSORT 版與 ByteTrack 版的單位不同（秒 vs. 30fps 幀數），兩者實際存活時間差 30 倍。
- `botsort_reid.yaml` 的 small_truck／scooter Re-ID 權重是暫代方案，要等 `rebuild-reid-dataset` 產出新模型後替換。
- Homography 目前以序列為鍵（`config/homography_points/<seq>.yaml`），與「Homography 屬於鏡頭」的設計有落差（見 `CONTEXT.md`）。
- detection 評估目前不套用 ROI（全畫面）。
- `save_mot_format()` 只 glob `C000025_*` 的問題已在 `rebuild-reid-dataset` 2.3 修正（改為 `*`），但 CLAUDE.md 的「已知限制」段落要到 7.1 才會移除。
