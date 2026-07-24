# dataset/reid/ — Re-ID 資料集（原始裁切 + 切分輸出）

本文件說明 `dataset/reid/crop_dataset/`、`dataset/reid/reid_category_dataset/`、`dataset/reid/reid_dataset/` 三個目錄的檔案／目錄架構與彼此關係。切分腳本 `build_reid_split.py` 放在 `dataset/reid/` 底下。

```
dataset/reid/
├── build_reid_split.py       切分腳本（讀 crop_dataset，寫 reid_category_dataset / reid_dataset）
├── README.md                  本文件
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
├── truck/         ...
├── train/          舊的扁平身分資料集，跟上面 6 個類別資料夾無關（見下方說明）
│   ├── train/{6碼數字id}/*.jpg
│   ├── val/{6碼數字id}/*.jpg
│   ├── train.cache
│   └── val.cache
├── numbers.txt      舊資料：train/ 子目錄的身分張數統計（見下方說明）
└── res.txt          舊資料：train/ 子目錄的 tree -h 輸出（見下方說明）
```

- 6 個類別資料夾（`bus`/`car`/`person`/`scooter`/`small_truck`/`truck`）底下每個子資料夾是一個身分，命名為 `{影片檔名}_track{track_id}`，由 `track_eval.py` 的 `save_reid_crops` 功能（`config/botsort_save_reid_crops.yaml`，`reid_crop_dir: ./dataset/reid/crop_dataset`）逐幀裁切寫入，詳見 `markdown/USAGE.md` 第 6 節。**`track_id` 是 class-aware tracker 逐類別各自配號**——同一支影片同一個編號在不同類別代表不同物體，不能跨類別直接比對資料夾名稱。
- `train/` 子目錄是另一批舊資料（身分資料夾直接命名為 6 碼數字如 `000100`），跟上面 6 個類別資料夾**不是同一套產生流程**，`build_reid_split.py` 不會讀取這個子目錄當來源。
- 根目錄下既有的 `numbers.txt`／`res.txt` 描述的是這個 **`train/` 子目錄**（444 個身分，`tree -h -L 3` 輸出），**不是** 6 個類別資料夾的統計，不要跟下面兩個輸出目錄裡同名的新檔案搞混。

## 2. `dataset/reid/reid_category_dataset/` — `build_reid_split.py --mode category` 輸出

```bash
python dataset/reid/build_reid_split.py --mode category
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
python dataset/reid/build_reid_split.py --mode flat
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

## 目前資料量（`dataset/reid/crop_dataset/`，2026-07-24）

不含舊的 `train/` 子目錄，只計 6 個類別資料夾：

| 類別 | 身分數 | 總 crop 數 | 平均/身分 | 最少 | 最多 |
|---|---:|---:|---:|---:|---:|
| person | 35 | 1082 | 30.9 | 1 | 284 |
| car | 531 | 14455 | 27.2 | 1 | 601 |
| scooter | 336 | 14260 | 42.4 | 1 | 590 |
| small_truck | 183 | 2230 | 12.2 | 1 | 448 |
| bus | 5 | 51 | 10.2 | 1 | 24 |
| truck | 95 | 1321 | 13.9 | 1 | 113 |
| **總計** | **1185** | **33399** | | | |

資料來源：`config/botsort_save_reid_crops.yaml`（中華三路_民生二路 6 支序列）+ `config/botsort_save_reid_crops_video.yaml`（`video/` 底下 5 支不同路口影片，各取前 60 秒，`det_corner` 關閉）。`bus` 身分數明顯偏少——這兩批來源影片路口車流裡幾乎沒拍到公車，需要額外找有公車出現的影片才能補足。

## `save_reid_crops` 蒐集目標（`reid_crop_dir`）

`config/*.yaml` 的 `save params.reid_crop_dir` 現在指向 `./dataset/reid/crop_dataset`——跟這裡的 `crop_dataset/` 是同一個目錄，`track_eval.py` 的 `save_reid_crops` 功能會直接把新蒐集的裁切寫進 `{class}/{影片}_track{id}/*.jpg`，與 `reid_category_dataset/`、`reid_dataset/` 這兩個切分輸出目錄不會互相覆蓋（各自獨立）。`module/reid/training/train.py` 的 `--data-root` 預設也指向這裡（訓練時直接讀 class-organized 的原始裁切，不經過切分輸出）。

## 共用行為（`build_reid_split.py` 兩種模式都適用）

- 切分單位是**身分**（整個資料夾），不是影像——避免同一身分的照片同時落在 train 和 val 造成資料外洩。
- 預設 `--val-ratio 0.2 --seed 42`；身分數只有 1 個時整個身分會分進 train，val 為 0（見上方「目前資料量」，`bus` 目前 5 個身分已不會觸發這個邊界情況，但仍是身分數最少的類別）。
- 用 `shutil.copy2` 實體複製影像，不會動到 `dataset/reid/crop_dataset/` 的原始資料。
- 輸出目錄已存在且非空時預設報錯中止，需加 `--overwrite` 才會清空重建。
