# USAGE.md

四大模組（Detection / Tracker / Re-ID / Evaluation）與主要腳本的使用範例。架構總覽見 `Project.md`，逐檔案職責見 `CLAUDE.md`。

所有指令預設在專案根目錄（`safetrajai_eval/`）下執行。標了 `python -m` 的模組**不能**用 `python evaluation/pipeline.py` 這種直接執行檔案的方式跑，因為它們用絕對 import（`from evaluation.xxx import ...`），只有 `-m` 形式會把根目錄加進 `sys.path`；根目錄的 `track_eval.py` 與 `module/grid_search/*.py` 則自己在檔案開頭做了 `sys.path` 處理，兩種方式都可以直接執行。

## 1. Detection 模組（`module/detection/`）

只有一個 `Detector` 類別：包 YOLO 模型 + 逐類別信心過濾（門檻定義在 `module/tracker/config.py` 的 `CLASS_CONF_THRESH`，隨 `load_track_config()` 載入的 YAML 而定）。`module/tracker/config.py` 的 `DEFAULT_TRACK_CONFIG_PATH` 預設指向 `config/bytetrack.yaml`，所以不呼叫 `load_track_config()` 也能跑；要套用其他情境（如含 ReID 欄位的 `botsort_reid.yaml`）就要顯式呼叫（2026-07-24 起：`config.yaml` + `module/tracker/track_config_*.yaml` 已整併成 `config/` 底下 6 個自包含情境檔，見 `CLAUDE.md`「設定檔」）：

```python
from module.tracker.config import load_track_config
from module.detection.detector import Detector
import cv2

load_track_config("config/botsort_reid.yaml")  # 不呼叫則沿用 config/bytetrack.yaml 的 class_tracking 章節

detector = Detector("models/yolo/yolo11x_dennis.pt")

frame = cv2.imread("some_frame.jpg")
boxes = detector.detect(frame, iou=0.50, imgsz=640)   # 回傳 ultralytics Boxes，已依類別過濾信心分數
print(boxes.cls, boxes.conf, boxes.xyxy)
```

> 目前沒有訓練腳本（YOLO 偵測模型訓練不在本專案範圍內，見 `NEED.md`）。`models/yolo/*.pt` 都是外部訓練好的產物，直接餵路徑即可。

## 2. Tracker 模組（`module/tracker/`）

對外只有一個工廠函式 `create_tracker()`。同上，`module/tracker/config.py` 預設載入 `config/bytetrack.yaml` 的 `class_tracking:` 章節；若要用其他情境檔，一樣要顯式呼叫 `load_track_config()`，而且**必須在 `create_tracker()` 之前呼叫**——順序反了不會報錯，只是會靜靜地套用錯的類別參數表，最容易漏測的坑（見 `TEST.md` 第 5 項）：

```python
from module.tracker.config import load_track_config
from module.tracker import create_tracker

load_track_config("config/botsort_reid.yaml")   # 套用含 ReID 欄位的情境

tracker = create_tracker(
    tracker_type="bytetrack",     # 或 "botsort"
    frame_rate=30,
    gmc_method="ecc",             # 僅 botsort 有效
    track_high_thresh=0.15,
    track_low_thresh=0.15 * 0.3,
)

# 每一幀呼叫一次，dets 是 (N, 6) numpy array: x1,y1,x2,y2,conf,cls
tracks, unmatched, removed = tracker.update(dets, frame)
```

實務上很少需要手動這樣逐幀呼叫——`track_eval.py` 的 `VideoSequenceTracker`（見第 6 節）已經包好整個「讀影片 → 偵測 → 追蹤 → 存檔」流程，一般改動只需要調整 `config/*.yaml` 對應情境檔。

### 逐類別 ReID 開關（YAML 設定，不是 Python API）
`config/*.yaml` 的 `class_tracking:` 章節裡每個類別可以獨立設定 `use_reid` + `model`，`module/tracker/class_aware_botsort.py` 會依 `model` 路徑字串自動選擇對應的 Re-ID encoder（見 `module/reid/encoders.py`）：

```yaml
class_tracking_params:
  person:
    use_reid: true
    model: models/yolo-reid/yolo11x-reid.pt              # 路徑不含 class_reid_weights → 用 ReID（YOLO 當 encoder）
  scooter:
    use_reid: true
    model: models/class_reid_weights/scooter/best_full_model.pt   # 路徑含 class_reid_weights → 用 Custom_ReID
```

目前預設全部 `use_reid: false`（見 `VERIFY.md`「ReID 開關」段落：對 person 有幫助但會損害 scooter 並增加 IDSW）。

> 2026-07-24：原本還有第三條路徑 `SOTA_ReID`（路徑含 `reid_model_weights` → 載入 vendored FastReID/TransReID 權重）已移除，只保留 `ReID` 與 `Custom_ReID` 兩條路徑。詳見 `VERIFY.md`「移除 reid_models」段落。

## 3. Re-ID 模組（`module/reid/`）

`module/reid/encoders.py` 的兩個 encoder 類別很少需要手動 new，通常透過上面的 YAML 設定自動建立。若要單獨測試某個 encoder：

```python
from module.reid.encoders import ReID, Custom_ReID
import numpy as np

encoder = Custom_ReID("models/class_reid_weights/scooter/best_full_model.pt")
# img: 原始 frame (H,W,3)；dets: (N,4) xywh 格式的偵測框
embeddings = encoder(img, dets)   # -> list[np.ndarray]，每個偵測框一個特徵向量
```

`module/reid/yolo_reid_trainer.py` 是訓練用的（見第 8 節），`module/reid/training/dataset.py` + `train.py` 是消費它的資料讀取與訓練迴圈（見第 9 節）。

## 4. Evaluation 模組（`evaluation/`）

通常由 `track_eval.py`（第 6 節）呼叫，也可以獨立跑（假設 `runs/track_eval/annotations_all8/`、`mot_annotations_all8/` 底下已經有第一階段 tracking 輸出）：

```bash
python -m evaluation.pipeline --config config/bytetrack.yaml
```

或在程式裡呼叫：

```python
from evaluation.config_loader import load_config
from evaluation.pipeline import run_eval_pipeline

cfg = load_config("config/bytetrack.yaml")
run_eval_pipeline(cfg)   # 讀 cfg.base_dir 下的序列，逐類別跑 TrackEval，輸出 Excel 到 cfg.trackeval_output_dir
```

`evaluation/` 底下其他檔案（`mot_converter.py`、`trackeval_runner.py`、`excel_export.py`、`mot_format.py`）都是 `pipeline.py` 的內部依賴，一般不需要單獨呼叫。

## 5. `visualize/visualize_gt_on_*.py` — 疊圖視覺化

兩支都是獨立腳本（不 import `tracker`/`reid`/`evaluation`），可以直接執行：

```bash
# 畫 GT（trackeval_root/gt/gt.txt）
python visualize/visualize_gt_on_video.py \
    --trackeval_root ./runs/track_eval/mot_annotations_all8/mot_annotation_C000025_xxx \
    --output_video   ./runs/gt_viz/C000025_gt.mp4

# 手動指定影片、類別名稱對照表，並只畫特定類別
python visualize/visualize_gt_on_video.py \
    --trackeval_root ./runs/track_eval/mot_annotations_all8/mot_annotation_C000025_xxx \
    --video_path     ./dataset/C000025.mp4 \
    --class_names    0:person 1:car 2:scooter 3:bus 4:truck 5:small_truck \
    --output_video   ./runs/gt_viz/C000025_gt.mp4 \
    --show_class_filter person car

# 畫「追蹤結果」而非 GT：讀 trackers/mot_challenge/<benchmark>-<split>/YOLO11x__<class>/data/<seq>.txt
python visualize/visualize_gt_on_class_video.py \
    --trackeval_root ./runs/track_eval/mot_annotations_all8/mot_annotation_C000025_xxx \
    --video_path     ./dataset/C000025.mp4 \
    --output_video   ./runs/track_viz/C000025_track.mp4 \
    --show_class_filter person car
```

`visualize_gt_on_video.py` 認的是 `gt/gt.txt`（含 class_id 欄）；`visualize_gt_on_class_video.py` 認的是逐類別分開存的 `trackers/mot_challenge/.../data/<seq>.txt`（無 class_id 欄，class 從資料夾名推斷）——兩者目錄假設不同，不要混用。其餘旗標（`--color_by`、`--no_id`、`--no_class`、`--show_conf`、`--no_legend`、`--bbox_alpha`、`--max_frames`）兩支腳本共用同一組定義。

## 6. `TrackingEvalPipeline`（`track_eval.py`）— 完整流程

**CLI（最常用）：**

```bash
# --config 預設就是 config/bytetrack.yaml，內建 class_tracking 章節不需另外指定
python track_eval.py --config config/bytetrack.yaml

# 換其他情境檔即可套用不同 tracker / ReID 設定，例如含 ReID 欄位的 botsort_reid
python track_eval.py --config config/botsort_reid.yaml
```

依序執行 `run_stage1()`（逐影片 class-aware tracking，輸出 MOT 格式結果）與 `run_stage2()`（TrackEval 評估，輸出 Excel），實際跑哪幾階段由該情境檔的 `stages.run_stage1`/`run_stage2` 決定（預設都 `true`；`config/botsort_save_reid_crops.yaml` 把 `run_stage2` 設 `false`，只做資料蒐集）。輸出目錄結構見 `track_eval.py` 檔頭 docstring。

**程式化呼叫**（例如只想跑 stage1，或包進自己的腳本）：

```python
from module.tracker.config import load_track_config
from evaluation.config_loader import load_config
from track_eval import TrackingEvalPipeline

load_track_config("config/botsort_reid.yaml")   # 務必在 create_tracker 之前呼叫；可指定 config/ 底下任一情境檔
cfg = load_config("config/botsort_reid.yaml")

pipeline = TrackingEvalPipeline(cfg)
pipeline.run_stage1()   # 只跑 tracking
# pipeline.run_stage2()  # 只跑 TrackEval 評估（需要 stage1 的輸出已存在）
# pipeline.run()         # 依 cfg 的 stages.run_stage1/run_stage2 決定要跑哪幾階段
```

**單一影片只想跑 tracking**（不經過整個 stage1 的影片掃描邏輯），可以直接用 `VideoSequenceTracker`：

```python
from pathlib import Path
from track_eval import VideoSequenceTracker

seq_tracker = VideoSequenceTracker(
    model_path="models/yolo/yolo11x_dennis.pt",
    tracker_type="bytetrack",
    conf=0.15, iou=0.50, imgsz=640, frame_rate=30, gmc_method="ecc",
    save=True, save_txt=True,
)
output_path, images_path, labels_path = seq_tracker.run(
    video_path="./dataset/C000025_clip_10s.mp4",
    output_dir=Path("./runs/adhoc_test"),
)
```

### Re-ID 訓練資料蒐集（`save_reid_crops`，2026-07-24 新增）

情境檔的 `save params.save_reid_crops`（預設 `false`）打開後，`VideoSequenceTracker.run()` 每一幀會把每個追蹤到的物件裁切存檔到：

```
{reid_crop_dir}/{class}/{影片檔名}_track{track_id}/{frame:06d}.jpg
```

`reid_crop_dir` 預設 `./dataset/reid/crop_dataset`。身分資料夾用「影片檔名_track_id」而不是單純 `track_id`——因為 tracker 是每支影片重新建立一個，`track_id` 只在同一支影片內唯一，不同影片可能都有 `track_id=3` 但代表不同物體，直接用 `track_id` 當資料夾會把不同身分的照片混在一起。有設定 `det_corner` 時，裁切也會套用同樣的多邊形過濾（只存落在偵測區域內的物件）。

```bash
# config/botsort_save_reid_crops.yaml 已內建 save_reid_crops: true 且 stages.run_stage2: false（只蒐集資料，跳過評估）
python track_eval.py --config config/botsort_save_reid_crops.yaml

# 情境 3 的變體：來源改成 video/ 底下不同路口的原始影片（非既有 all8 資料集），
# 用來補足特定類別（如 bus/person/truck/small_truck）身分數過少的問題
python track_eval.py --config config/botsort_save_reid_crops_video.yaml
```

`config/botsort_save_reid_crops_video.yaml` 的來源目錄 `dataset/video_seqs/annotations_video/annotation_<seq>/<seq>.mp4` 要符合 stage1 掃描的結構（跟 `dataset/中華三路_民生二路(東北角)yolo驗證用_all8/annotations_all8/` 同一套模式），不能直接指向 `video/` 底下的原始 mp4——新增來源影片時要先按這個結構整理好。這個變體也把 `det_corner` 關閉（`[]`），因為不同路口鏡頭的偵測區域多邊形不通用；換新路口來源時記得一併確認 `det_corner` 是否要沿用。

或程式化只對單一影片蒐集：

```python
seq_tracker = VideoSequenceTracker(
    model_path="models/yolo/yolo11x_dennis.pt",
    tracker_type="bytetrack",
    conf=0.15, iou=0.50, imgsz=640, frame_rate=30, gmc_method="ecc",
    save=False, save_txt=False,
    save_reid_crops=True,
    reid_crop_dir="./dataset/reid/crop_dataset",
)
seq_tracker.run(video_path="./dataset/C000025_clip_10s.mp4", output_dir=Path("./runs/adhoc_test"))
```

這個功能只是「蒐集資料」，跟 `dataset/reid/crop_dataset/train/{train,val}/{identity}/` 這個扁平（不分類別）的舊資料結構不是同一套配置——舊資料是否要轉成新的 `{class}/{影片}_track{id}/` 結構、以及後續訓練怎麼讀取，正在重新設計，見 `NEED.md`。

### `dataset/reid/build_reid_split.py` — 把蒐集到的裁切切成 train/val

蒐集到 `dataset/reid/crop_dataset/{class}/{identity}/*.jpg` 後，用這支腳本依身分（不是依影像）切 train/val：

```bash
# 逐類別各自切一次，輸出到 dataset/reid/reid_category_dataset/{class}/{train,val}/{identity}/
python dataset/reid/build_reid_split.py --mode category

# 6 類別身分全部混在一起切一次（身分資料夾加類別前綴避免 track_id 撞名），
# 輸出到 dataset/reid/reid_dataset/{train,val}/{class}_{identity}/
python dataset/reid/build_reid_split.py --mode flat

# 輸出目錄已存在且非空時預設報錯中止，要清空重建才加 --overwrite
python dataset/reid/build_reid_split.py --mode category --overwrite --val-ratio 0.2 --seed 42
```

兩種模式跑完都會在輸出目錄下產生 `numbers.txt`（每身分一行 "張數 相對路徑"）與 `res.txt`（`tree -h` 目錄結構），完整架構說明、目前資料量統計見 `dataset/reid/README.md`。

## 7. Grid Search（`module/grid_search/`）— 追蹤參數超參數搜尋

三支腳本，介面幾乎相同（`grid_search_botsort_reid.py` 額外把 ReID 相關參數固定在搜尋空間裡，見 `PARAM_GRIDS` 的 `use_reid`）：

```bash
# 先看有多少組合，不要一次跑全部
python module/grid_search/grid_search_bytetrack.py --target-class scooter --dry-run

# 小規模先跑一次確認流程沒問題（限制 1 支影片）
python module/grid_search/grid_search_bytetrack.py --target-class scooter --max-videos 1

# 正式搜尋，可用 --resume 續跑（會跳過 CSV 裡已有的組合）
python module/grid_search/grid_search_bytetrack.py --target-class scooter --config config/grid_search_bytetrack.yaml --resume

# 只搜尋指定參數，其餘用目前情境檔 class_tracking 的值
python module/grid_search/grid_search_botsort.py --target-class car --tracker botsort \
    --params-subset match_thresh,new_track_thresh

# 含 ReID 開關的版本（針對 person / scooter 的 ReID on/off 對照）
python module/grid_search/grid_search_botsort_reid.py --target-class person --dry-run
```

常用旗標：`--config`、`--track-config`、`--tracker {botsort,bytetrack}`、`--model`（覆寫 YOLO 權重路徑）、`--conf`/`--iou`/`--imgsz`/`--frame_rate`（僅 botsort 腳本有，用於推論參數）、`--gmc_method`（僅 botsort）、`--video-glob`（覆寫預設影片掃描 pattern）、`--max-videos`、`--output-root`、`--result-csv`、`--resume`、`--verbose`、`--dry-run`、`--params-subset`。結果以 `HOTA.AssA_mean` 當 scoring，寫入 CSV。

> ⚠️ 大規模 grid search 前務必先 `--dry-run` 確認組合數，再抓 1-2 組實際跑一次估算單組耗時，避免整批跑下去才發現要花好幾天（見 `CLAUDE.md`「Grid Search / 大範圍參數掃描前，先確認搜尋空間大小與預估執行時間」）。

## 8. `yolo_reid_trainer`（`module/reid/yolo_reid_trainer.py`）— ReID 訓練工具

核心是 `ReIDHookManager`：直接在 ultralytics YOLO 原生模型倒數第二層掛 forward hook 取特徵，接一個 `ReIDProjector` 做 metric embedding，不另外包 wrapper nn.Module。訓練出來的權重可以直接被 `module/reid/encoders.py::ReID` 載入（推論期用同一份 `ReIDHookManager`/`load_reid_model` 邏輯重建）。

**低階 API（手動組訓練迴圈時使用）：**

```python
from ultralytics import YOLO
from module.reid.yolo_reid_trainer import ReIDHookManager, BatchHardTripletLoss, save_yolo_reid, load_reid_model

yolo = YOLO("models/yolo/yolo11x_dennis.pt")
yolo.model.to("cuda")

manager = ReIDHookManager(yolo, device="cuda")   # 建構時自動 probe feat_dim 並建立/附掛 projector
manager.freeze_backbone(True)                    # 只訓練 projector

criterion = BatchHardTripletLoss(margin=0.3)

manager.train_mode()
embeddings = manager.embed(imgs)          # imgs: (B,3,H,W)，回傳 L2-normalized (B,D)
loss = criterion(embeddings, labels)      # labels: (B,) 每張圖的身分 ID
loss.backward()
...

save_yolo_reid(yolo, "models/yolo-reid/my_new_reid.pt", train_args={"epochs": 50})

# 之後推論或繼續訓練時載回：
yolo2, manager2 = load_reid_model("models/yolo-reid/my_new_reid.pt", device="cuda")
manager2.eval_mode()
embed = manager2.embed(imgs)
```

⚠️ `save_yolo_reid()` 存檔前會暫時清空每個 submodule 的 `_forward_hooks`/`_forward_pre_hooks`/`_backward_hooks` 以避開無法 pickle 的 hook，2026-07-24 之前這裡誤用 `{}` 清空（型別變成一般 `dict`），會導致存檔內容裡的 hook 容器型別跟著變成 `dict`，之後 `load_reid_model()` 重新 `register_forward_hook()` 時丟 `TypeError: cannot create weak reference to 'dict' object`。已修正為 `OrderedDict()`；此修正只影響用 `save_yolo_reid()` 存檔之後的檔案，舊權重不受影響。

## 9. Re-ID 訓練資料蒐集 + 訓練（`module/reid/training/`，2026-07-24 新增）

**資料蒐集**：先用 `track_eval.py` 的 `save_reid_crops` 功能蒐集資料（見第 6 節），或直接跑一次完整 pipeline 累積 `dataset/reid/crop_dataset/{class}/{影片}_track{id}/*.jpg`。

**資料讀取**（`module/reid/training/dataset.py`）：

```python
from module.reid.training.dataset import discover_identities, split_identities, ReIDCropDataset, PKSampler
from torch.utils.data import DataLoader

identity_dirs = discover_identities("dataset/reid/crop_dataset", "scooter")
train_dirs, val_dirs = split_identities(identity_dirs, val_ratio=0.2, seed=42)   # 依身分切分，不會資料外洩

train_dataset = ReIDCropDataset(train_dirs, transform=my_transform, min_images_per_identity=4)
train_loader = DataLoader(train_dataset, batch_sampler=PKSampler(train_dataset.labels, P=8, K=4))
# 每個 batch 固定 8 個身分 × 4 張影像，符合 BatchHardTripletLoss 需要同 batch 內有 positive pair 的假設
```

**訓練**（`module/reid/training/train.py`，取代已移除的 `train_utils.py`，只走 YOLO-as-encoder 這條架構，不使用 `utils/model.py` 的 `ReIDNet`）：

```bash
python module/reid/training/train.py \
    --class-name scooter \
    --data-root dataset/reid/crop_dataset \
    --yolo-weights models/yolo/yolo11x_dennis.pt \
    --output models/yolo-reid/scooter_reid.pt \
    --epochs 50 --P 8 --K 4 --lr 1e-4 --val-ratio 0.2 --patience 10
```

常用旗標：`--freeze-backbone`（只訓練 projector）、`--img-size`、`--margin`、`--min-images`（身分資料夾影像數低於此值整個捨棄）、`--wandb`（預設關閉，開啟才會呼叫 `wandb.init()`；要用的話先 `wandb login` 或設 `WANDB_MODE=offline`，對應根目錄的 `wandb/` log，見 `Project.md`「資料與輸出目錄」）。一次只練一個類別的模型（呼應 `config/*.yaml` 的 `class_tracking:` 逐類別指定 `model:` 路徑的慣例），要練多個類別就跑多次、換 `--class-name` 和 `--output`。

已用實際蒐集的 crop 資料驗證完整流程：蒐集 → 訓練（loss 確實下降）→ 存檔 → `load_reid_model()` 與 `ReID` class 兩條載入路徑皆能正確還原並產生 embedding。

## 待補充
> 以上範例對應目前程式碼實際簽章（2026-07-24 確認，含 `dataset/reid/build_reid_split.py`、`config/botsort_save_reid_crops_video.yaml`、`python -m evaluation.pipeline` 獨立執行路徑）。若函式簽章之後有變動，此檔案要一併更新；若發現範例與實際行為不符，以程式碼為準並回報修正。
