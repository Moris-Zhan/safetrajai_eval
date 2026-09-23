# scripts/reid_data/ — Re-ID 資料集工具（資料位於 dataset/reid/）

本文件說明 `dataset/reid/crop_dataset/`、`dataset/reid/reid_category_dataset/`、`dataset/reid/reid_dataset/` 三個目錄的檔案／目錄架構與彼此關係。切分與稽核腳本放在 `scripts/reid_data/`（`build_reid_split.py`、`check_class_consistency.py`、`generate_identity_review.py`、`quarantine_flagged.py`），資料本身仍在 `dataset/reid/`（gitignore）。

> **2026-09-23 更新**：下面描述的 `crop_dataset/`、`reid_category_dataset/`、`reid_dataset/` 與稽核報表，都是舊追蹤器產生的無效資料，已整批隔離到 `dataset/reid/temp/`（見該目錄的 `DO_NOT_USE.md`），不可使用。Re-ID 資料會由後續 change `rebuild-reid-dataset` 從 `dataset/raw/videos/` 重建，屆時本文件與這些腳本的輸入會一併更新。

```
dataset/reid/
├── crop_dataset/               原始裁切影像（來源，唯讀）
├── reid_category_dataset/      --mode category 輸出
└── reid_dataset/                --mode flat 輸出
```

## 1. `dataset/reid/crop_dataset/` — 原始裁切影像（來源，唯讀）

```
dataset/reid/crop_dataset/
├── bus/           {影片檔名}_track{track_id}/*.jpg
├── car/           {影片檔名}_track{track_id}/*.jpg
├── person/        ...
├── scooter/       ...
├── small_truck/   ...
└── truck/         ...
```

- 6 個類別資料夾（`bus`/`car`/`person`/`scooter`/`small_truck`/`truck`）底下每個子資料夾是一個身分，命名為 `{影片檔名}_track{track_id}`，由 `track_eval.py` 的 `save_reid_crops` 功能（`config/botsort_save_reid_crops.yaml`，`reid_crop_dir: ./dataset/reid/crop_dataset`）逐幀裁切寫入，詳見 `markdown/GUIDE.md` 第 6 節。**`track_id` 是 class-aware tracker 逐類別各自配號**——同一支影片同一個編號在不同類別代表不同物體，不能跨類別直接比對資料夾名稱。

> **2026-09-14 更新**：先前這裡記錄過一個 `train/` 子目錄（舊版扁平身分資料集，444 身分，資料夾直接命名為 6 碼數字如 `000100`，跟上面 6 個類別資料夾不是同一套產生流程）與對應的 `numbers.txt`／`res.txt`。確認**這個環境的 `dataset/reid/crop_dataset/` 底下本來就沒有同步這批資料**，只有上面 6 個類別資料夾——不是被移除，是這個工作目錄從未有過。因此 `markdown/ARCHITECTURE.md` 原本規劃的「搬到 `archive/reid_legacy_flat_dataset/`」這個批次操作在此環境無事可做，已從待辦清單移除；若其他環境/主機上確實還有這批舊資料，屬於該處另外處理的範圍，不在本文件描述內。

## 2. `dataset/reid/reid_category_dataset/` — `build_reid_split.py --mode category` 輸出

```bash
python scripts/reid_data/build_reid_split.py --mode category
```

6 個類別**各自獨立**依身分（資料夾）切一次 train/val（同一身分的照片只會出現在其中一邊），輸出：

```
dataset/reid/reid_category_dataset/
├── bus/
│   ├── train/{identity}/*.jpg
│   └── val/{identity}/*.jpg
├── car/{train,val}/{identity}/*.jpg
├── person/{train,val}/{identity}/*.jpg
├── scooter/{train,val}/{identity}/*.jpg
├── small_truck/{train,val}/{identity}/*.jpg
├── truck/{train,val}/{identity}/*.jpg
├── numbers.txt      每個身分一行 "{張數:>6d} {class}/{split}/{identity}"
└── res.txt          tree -h -L 3 輸出（顯示到身分資料夾層級，不展開個別影像）
```

身分資料夾名稱維持原樣（`{影片檔名}_track{track_id}`），同一類別內不會撞名。

## 3. `dataset/reid/reid_dataset/` — `build_reid_split.py --mode flat` 輸出

```bash
python scripts/reid_data/build_reid_split.py --mode flat
```

6 個類別的身分**全部混在一起**依身分切一次 train/val（不分類別），輸出：

```
dataset/reid/reid_dataset/
├── train/{class}_{identity}/*.jpg
├── val/{class}_{identity}/*.jpg
├── numbers.txt      每個身分一行 "{張數:>6d} {split}/{class}_{identity}"
└── res.txt          tree -h -L 2 輸出
```

身分資料夾名稱前綴類別名（如 `car_C000025_..._track52`），避免不同類別的 `track_id` 撞名。

## 目前資料量（`dataset/reid/crop_dataset/`，2026-09-21）

不含舊的 `train/` 子目錄，只計 6 個類別資料夾：

| 類別 | 身分數 | 總 crop 數 | 平均/身分 | 最少 | 最多 |
|---|---:|---:|---:|---:|---:|
| person | 283 | 12742 | 45.0 | 1 | 2635 |
| car | 7021 | 223322 | 31.8 | 1 | 1567 |
| scooter | 5386 | 299852 | 55.7 | 1 | 1352 |
| small_truck | 2091 | 11851 | 5.7 | 1 | 844 |
| bus | 57 | 1256 | 22.0 | 1 | 340 |
| truck | 1172 | 35091 | 29.9 | 1 | 1270 |
| **總計** | **16010** | **584114** | | | |

資料來源：`config/botsort_save_reid_crops.yaml`（中華三路_民生二路 6 支序列）+ `config/botsort_save_reid_crops_video.yaml`（`video/` 底下 5 支不同路口影片 + `bus_video/` 補充的 4 支民族一路/天祥二路路口完整長度影片，`det_corner` 關閉）。`bus` 身分數已從 5 個提升到 57 個（`openspec/changes/reid-bus-video-crops/`，2026-09-21）：既有兩批來源影片路口車流裡幾乎沒拍到公車，`bus_video/` 完整長度影片驗證能實際補到公車身分，其餘類別身分數也因全長影片內容量大幅增加。

## `save_reid_crops` 蒐集目標（`reid_crop_dir`）

`config/*.yaml` 的 `save params.reid_crop_dir` 現在指向 `./dataset/reid/crop_dataset`——跟這裡的 `crop_dataset/` 是同一個目錄，`track_eval.py` 的 `save_reid_crops` 功能會直接把新蒐集的裁切寫進 `{class}/{影片}_track{id}/*.jpg`，與 `reid_category_dataset/`、`reid_dataset/` 這兩個切分輸出目錄不會互相覆蓋（各自獨立）。`scripts/train_reid.py` 的 `--data-root` 預設也指向這裡（訓練時直接讀 class-organized 的原始裁切，不經過切分輸出）。

## 共用行為（`build_reid_split.py` 兩種模式都適用）

- 切分單位是**身分**（整個資料夾），不是影像——避免同一身分的照片同時落在 train 和 val 造成資料外洩。
- 預設 `--val-ratio 0.2 --seed 42`；身分數只有 1 個時整個身分會分進 train，val 為 0（見上方「目前資料量」，`bus` 目前 5 個身分已不會觸發這個邊界情況，但仍是身分數最少的類別）。
- 用 `shutil.copy2` 實體複製影像，不會動到 `dataset/reid/crop_dataset/` 的原始資料。
- 輸出目錄已存在且非空時預設報錯中止，需加 `--overwrite` 才會清空重建。
