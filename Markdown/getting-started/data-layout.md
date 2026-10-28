# 資料與目錄慣例

## 1. `dataset/` 分層

```
dataset/
├── raw/videos/<camera>/<seq>.mp4      原始影片（依鏡頭歸位）→ datasets.yaml: reid_sources（role: source）
├── eval/cctv_all8/                    評估資料集（人工 GT，唯讀）→ datasets.yaml: cctv_all8（role: eval）
│   ├── annotations_all8/annotation_<seq>/          detection GT + 序列影片
│   └── mot_annotations_all8/mot_annotation_<seq>/  MOT GT
├── reid/                              Re-ID 裁切、排除清單、資料集版本（舊內容已隔離到 temp/）
└── reports/                           稽核報表（例如 reports/reid/）
```

## 2. `config/datasets.yaml`

情境 config 用 `paths.dataset: <id>` 引用，loader 會自動補上 `root`、`run_det_eval_dir`、`run_track_eval_dir`：

| id | role | layout | cameras |
|---|---|---|---|
| `cctv_all8` | `eval`（只用來評估） | `annotated`（`<root>/annotations_all8/annotation_<seq>/<seq>.mp4`） | `C000025` |
| `reid_sources` | `source`（用來產生訓練資料） | `raw`（`<root>/<camera>/<seq>.mp4`） | `C000011`、`C000012`、`C000033`、`C000043`、`C000048`、`C000054`、`C000056`、`C000085`、`C000256`（`C000026` 與評估鏡頭同路口，影片歸位但不列入） |

- eval 與 source 的鏡頭不能重疊（smoke 測試會檢查）；**評估鏡頭 C000025 不能進任何訓練資料**。
- 情境 config 可以另外加 `paths.cameras: [...]`，只跑資料集登記鏡頭中的一部分；列出未登記的鏡頭會直接報錯。
- 想縮小範圍時，也可以在 `paths:` 直接寫 `SOURCE_VIDEO_DIR` 指向單一序列所在目錄。

## 3. 隔離慣例（`temp/`）

- 有疑慮、不可使用的權重或資料，用 `mv` 移到**所在目錄**下的 `temp/`，不刪除。
- **先寫紀錄再搬移**：在 `temp/DO_NOT_USE.md` 記錄原路徑、日期、原因與還原指令。
- `config/` 與 `tests/golden/configs/` 不能引用 `temp/` 內的路徑（smoke 測試會攔截）。
- 目前已隔離：`dataset/reid/` 全部舊內容、`models/yolo-reid/*`、`models/class_reid_weights/{person,scooter}`、`all8` 的限縮版與重複版標註。

## 4. 禁止大量 I/O

`dataset/`、`models/`、`runs/` 在掛載磁碟上，檔案數以百萬計。**不要**做全量 `sha256sum`／`md5sum`、全量 `du`、`find … -exec cat`、`rsync --checksum`。需要核對時，只比對路徑清單與檔案數（`find -type f | wc -l`）。
