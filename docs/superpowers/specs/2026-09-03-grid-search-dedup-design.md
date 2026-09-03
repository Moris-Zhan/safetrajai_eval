# module/grid_search/ 去重複 — 設計文件

- 日期:2026-09-03
- 狀態:待審查
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第三個子專案**,接續 `2026-09-03-dataset-path-refactor-design.md`。

## 0. 前提依賴

`_resolve_gt_source_root()` 與各 `main()` 的影片蒐集 glob 目前都依賴 `mot_annotation_<seq>`/`annotation_*` 前綴慣例,與 **dataset 路徑重構**(`2026-09-03-dataset-path-refactor-design.md`,已核准)處理的是同一組路徑問題。本設計假設該重構已落地,共用核心模組直接呼叫 `evaluation/dataset_layout.py` 提供的函式;若尚未落地,先沿用現行字串邏輯,待路徑重構完成後替換。

與 `2026-09-03-track-eval-pipeline-refactor-design.md` 無直接依賴,兩者可平行或任一順序實作。

## 1. 背景與問題

`module/grid_search/grid_search_bytetrack.py`(992 行)、`grid_search_botsort.py`(1186 行)、`grid_search_botsort_reid.py`(1041 行)三支腳本逐段比對後,確認以下函式/類別在三支腳本間**完全逐字相同**:

- `ParameterGrid` class
- `extract_assa_mean()`
- `score_params()`
- `generate_config_patch()`(botsort 與 botsort_reid 之間也完全相同——ReID 的差異全部只在 `PARAM_GRIDS` 資料內容,不在程式邏輯)
- `_resolve_gt_source_root()` / `_setup_trial_gt()`
- `GridSearchRunner` 的 `_load_existing_results()`/`_append_csv()`/`run()`/`print_summary()` 方法本體
- `main()` 內解析引數之後的執行邏輯(蒐集影片、建立 runner、印出/存 patch 檔)

唯一有實質差異的部分:
- `_run_tracking_for_video()`:botsort 版本把 `conf`/`iou`/`imgsz`/`frame_rate`/`gmc_method` 開放成參數,`gmc_method` 只在 `tracker_type == "botsort"` 時加入 `tracker_kwargs`;bytetrack 版本把這些值寫死(`conf=get_min_conf_thresh()`、`iou=0.5`、`imgsz=1280`、`frame_rate=30`,無 `gmc_method`)。
- `GridSearchRunner.__init__()`:botsort 版本的建構參數比 bytetrack 多帶上述 5 個欄位。
- `PARAM_GRIDS`(合理應該不同的搜尋空間資料)與 `main()` 的 `argparse` 定義(合理應該不同的 CLI 預設值,如 `--tracker`/`--gmc_method`/`--config` 預設路徑)。

每支腳本大約 90% 以上是逐字複製貼上;新增一個共用邏輯的調整(例如 CSV 欄位、resume 判斷條件)目前要同步改三處。

## 2. 目標 / 非目標

**目標**
- 把上述「完全逐字相同」與「僅建構參數差異」的邏輯整併成單一共用模組,三支腳本呼叫同一份邏輯。
- 三支腳本瘦身為:`PARAM_GRIDS` 常數 + `main()`(各自 argparse 定義與預設值)+ 呼叫共用模組。
- **CLI 使用方式不變**:三個獨立指令入口(`python module/grid_search/grid_search_bytetrack.py ...` 等)、既有旗標(`--dry-run`/`--resume`/`--max-videos`/`--params-subset` 等)與輸出(CSV 欄位、patch 檔格式)保持不變。

**非目標**
- 不合併三支腳本的 CLI 入口(方案 C,使用者已明確排除)。
- 不幫 bytetrack 腳本新增 `--conf`/`--iou`/`--imgsz`/`--frame_rate`/`--gmc_method` 等目前沒有的 CLI flag——合併後 `GridSearchRunner` 雖然統一開放這些建構參數,但 bytetrack 的 `main()` 一樣把現有寫死的值當常數傳入,對外行為與可用旗標不變。這是消除重複的分寸,不是順手擴充功能。
- 不改變 `PARAM_GRIDS` 的搜尋空間內容或任何腳本目前的預設值。
- 不處理 `module/tracker`/`module/reid`/`module/detection`/`module/collision` 的內部邏輯。

## 3. 新設計

### 3.1 `module/grid_search/core.py`(新檔)

搬移(邏輯不變,只是換位置)以下項目:

```python
class ParameterGrid: ...
def extract_assa_mean(...): ...
def score_params(...): ...
def generate_config_patch(...): ...
def _resolve_gt_source_root(cfg, seq_stem: str) -> str | None: ...
def _setup_trial_gt(gt_src_root: str, trial_dir: Path) -> bool: ...
```

統一(現行差異部分整併為單一版本):

```python
def _run_tracking_for_video(
    detector: Detector,
    video_path: str,
    tracker_type: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_rate: int,
    gmc_method: str,
    ...,
) -> ...:
    """統一簽章，一律接受 conf/iou/imgsz/frame_rate/gmc_method；
    gmc_method 只在 tracker_type == "botsort" 時加入 tracker_kwargs
    （沿用現行 botsort 版本的判斷邏輯，行為對兩種 tracker 都不變）。

    【2026-09-03 module/detection/ round 修訂】原本各腳本手刻的
    `model = YOLO(model_path)` + `model.predict(...)` + 逐偵測套用
    get_conf_thresh_for_class() 過濾，改成呼叫 module/detection/detector.py
    的 Detector.detect(frame, iou=iou, imgsz=imgsz)（與 track_eval.py 共用同一份
    偵測+信心過濾邏輯，不再各自重寫）。呼叫端（score_params()）改成每個 trial
    開始時建立一次 detector = Detector(model_path)（取代 model = YOLO(model_path)），
    效能特性不變（trial 內的影片迴圈仍重用同一個 detector 實例）。"""


class GridSearchRunner:
    def __init__(
        self,
        param_grid: dict[str, list],
        target_class: str,
        video_paths: list[str],
        model_path: str,
        cfg,
        tracker_type: str = "botsort",
        conf: float = 0.15,
        iou: float = 0.50,
        imgsz: int = 640,
        frame_rate: int = 30,
        gmc_method: str = "ecc",
        output_root: Path = Path("./runs/grid_search"),
        result_csv: Path | None = None,
        resume: bool = False,
        verbose: bool = False,
    ):
        """統一建構參數（沿用現行 botsort 版本的完整參數集）。
        bytetrack 呼叫端傳入現行寫死的值（conf=get_min_conf_thresh()、iou=0.5、
        imgsz=1280、frame_rate=30），不透過 CLI 開放。"""
    def _load_existing_results(self): ...
    def _append_csv(self, combo_id, score, params, elapsed): ...
    def run(self) -> dict: ...
    def print_summary(self): ...
```

新增(把 `main()` 裡重複的影片蒐集邏輯抽出):

```python
def discover_grid_search_videos(cfg, video_glob_override: str | None, max_videos: int) -> list[str]:
    """沿用現行 main() 內的邏輯：video_glob_override 有值就直接 glob；
    否則用 cfg.OUTPUT_ROOT / cfg.run_det_eval_dir 下的序列（呼叫
    evaluation.dataset_layout 的路徑函式，取代現行手刻的 annotation_* glob），
    跳過 Demo.mp4，取前 max_videos 支。"""
```

### 3.2 三支腳本瘦身後結構(以 `grid_search_bytetrack.py` 為例)

```python
from module.grid_search.core import (
    ParameterGrid, GridSearchRunner, generate_config_patch,
    discover_grid_search_videos,
)

PARAM_GRIDS: dict[str, dict[str, list]] = {
    # ...（內容不變，此腳本特有的搜尋空間）
}

def main():
    parser = argparse.ArgumentParser(...)
    # ...（此腳本特有的 argparse 定義與預設值，不變）
    args = parser.parse_args()

    # ...（load_track_config / load_config，不變）

    video_paths = discover_grid_search_videos(cfg, args.video_glob, args.max_videos)

    runner = GridSearchRunner(
        ...,
        tracker_type = "bytetrack",
        conf         = get_min_conf_thresh(),   # 現行寫死值，不開放 CLI
        iou          = 0.5,
        imgsz        = 1280,
        frame_rate   = 30,
        gmc_method   = "none",                  # bytetrack 不使用 GMC，占位值
    )
    outcome = runner.run()
    runner.print_summary()
    # ...（產生/存 patch 檔，呼叫 generate_config_patch，不變）

if __name__ == "__main__":
    main()
```

`grid_search_botsort.py`/`grid_search_botsort_reid.py` 結構相同,差別只在各自 `PARAM_GRIDS` 內容與 `main()` 的 argparse 預設值(`--gmc_method` 預設值不同等),以及把 `conf`/`iou`/`imgsz`/`frame_rate`/`gmc_method` 從 `args.*` 傳入(這是它們現有的行為,不變)。

### 3.3 影響檔案清單

| 檔案 | 改動 |
|---|---|
| `module/grid_search/core.py`(新) | 承接三支腳本共用邏輯 |
| `module/grid_search/grid_search_bytetrack.py` | 瘦身為 `PARAM_GRIDS` + `main()`,從 992 行降到約 150-250 行 |
| `module/grid_search/grid_search_botsort.py` | 同上,從 1186 行降到約 150-250 行 |
| `module/grid_search/grid_search_botsort_reid.py` | 同上,從 1041 行降到約 150-250 行 |

不動 `module/tracker`、`module/reid`、`module/detection`、`module/collision`、`evaluation/*`(除了呼叫已核准的 `dataset_layout.py`)。

## 4. 錯誤處理與邊界情況

- `--resume`/`--dry-run`/`--params-subset`/`--max-videos` 等旗標的判斷邏輯全部隨對應函式搬進 `core.py`,行為(包含 CSV 部分寫入後中斷續跑的判斷條件)不變。
- `_run_tracking_for_video()` 統一簽章後,bytetrack 呼叫端必須明確傳入現行寫死的 5 個值(不可用 `core.py` 的預設值,因為預設值是 botsort 的數值);審查時需逐一核對這幾個值與現行 `grid_search_bytetrack.py` 寫死的常數完全一致,避免搬移過程中誤用 botsort 的預設值。
- `GridSearchRunner.output_root` 現行邏輯是 `Path(output_root) / tracker_type`(自動依 tracker 分子目錄),搬移時保留此行為。

## 5. 驗證計畫

1. 對三支腳本分別跑 `--dry-run`,確認參數組合數與現行輸出一致(表示 `PARAM_GRIDS`/`ParameterGrid` 搬移無誤)。
2. 對三支腳本分別用 `--max-videos 1` + 縮小過的 `--params-subset` 各跑一組 trial,確認 CSV 輸出欄位、`HOTA.AssA_mean` 數值與重構前一致。
3. `--resume` 驗證:先跑一組產生部分 CSV,中斷後用 `--resume` 續跑,確認正確跳過已完成組合。
4. `grid_search_bytetrack.py` 跑完一組 trial 後,確認 `_run_tracking_for_video()` 實際使用的 `conf`/`iou`/`imgsz`/`frame_rate` 與重構前的寫死值相同(可從 `--verbose` 逐幀 log 或追蹤呼叫參數確認)。

## 6. 風險總結

- **低風險**:三支腳本 90% 以上是逐字重複,搬移風險主要是「複製貼上時漏改」而非邏輯設計風險。
- **中風險**:`_run_tracking_for_video()`/`GridSearchRunner` 統一簽章後,務必確認 bytetrack 呼叫端傳入的常數值與現行完全一致,否則會悄悄改變 bytetrack grid search 的實際搜尋行為(而不是報錯,難以察覺)。
- **需注意的依賴**:`_resolve_gt_source_root()` 與 `discover_grid_search_videos()` 的路徑組裝依賴 dataset 路徑重構落地。
