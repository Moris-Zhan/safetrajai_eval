# scripts/reid_data/ — Re-ID 資料集工具（資料位於 dataset/reid/）

Re-ID 訓練資料由 change `rebuild-reid-dataset`（2026-09-24）從 `dataset/raw/videos/` 重建。流程是：

**蒐集 → 標記 → 審查 → 排除 → manifest**

裁切只寫一次，之後的標記、審查、排除、切分都只產生紀錄，不搬移、改名或刪除任何裁切。術語（裁切、身分、污染身分、資料集版本、排除、隔離）見 `CONTEXT.md`，排除與隔離的差異見 `docs/adr/0006`。

> 舊的 `crop_dataset/`、`reid_category_dataset/`、`reid_dataset/` 是舊追蹤器產生的無效資料，已隔離到 `dataset/reid/temp/`（見該目錄的 `DO_NOT_USE.md`），不可引用或拿來訓練。

## 目錄

```
dataset/reid/
├── crops/                                   裁切（stage1 的 save_reid_crops 寫入）
│   ├── <class>/<序列>_track<id>/<frame>.jpg
│   └── _index/<序列>.csv                    每張裁切一列的 metadata 索引
│       _index/<序列>.skipped.json           略過統計（ROI 外、最短邊 < 32px）
├── exclusions.csv                           人工審查後的排除清單（apply_review_exclusions.py 寫入）
├── splits/<version>/                        資料集版本
│   ├── manifest.csv                         crop_path,identity,class,camera,date,split
│   ├── summary.json                         各類別／split／鏡頭的身分數與張數
│   └── split_rules.yaml                     本次使用的切分規則複本
└── temp/                                    已隔離的舊資料，不可使用

dataset/reports/reid/                        標記與審查頁（detect_contamination.py、generate_identity_review.py）
```

**訓練時請讀 `splits/<version>/manifest.csv`**，再以 `crop_path`（相對於 `crops/`）讀取影像，不要直接掃描 `crops/`：排除清單、最少張數、每身分最多張數與 train／val 切分，都只在 manifest 中套用。

## 1. 蒐集裁切

```bash
python track_eval.py --config config/reid_crops.yaml
```

- 只從 `config/datasets.yaml` 中 `role: source` 的 `reid_sources` 蒐集；評估鏡頭（C000025）一律拒絕，每支鏡頭都必須有 `config/roi/<camera>.yaml`，否則在追蹤前就報錯，不會寫出任何東西。
- 規則：框中心點在 ROI 內、最短邊 ≥ 32px、每條軌跡依影片時間每 0.5 秒最多 1 張。
- 每支序列在結束時一次寫出索引；索引已存在時報錯，重跑前要先移走舊輸出。
- 工具模型與追蹤參數（dennis@640、`frame_rate: auto`、`gmc_method: sparseOptFlow`、四輪車加重 `center_dist_weight`）的選定依據見 `openspec/changes/archive/2026-09-24-rebuild-reid-dataset/verification.md`。
- 只跑部分鏡頭時，在另建的設定中加上 `paths.cameras: [<camera>, ...]`（只能取 `reid_sources` 登記鏡頭的子集）。

## 2. 污染標記

```bash
python scripts/reid_data/detect_contamination.py
```

- 每張裁切只讀一次，以 DINOv2（timm `vit_small_patch14_dinov2`，第一次使用時從 HF Hub 下載）抽特徵。
- 只標記裁切數 ≥ `config/reid_split.yaml` 的 `min_crops_per_identity` 的身分（不足的身分不會進 manifest）。
- 四種標記：
  - **類別一致性**：只在人／二輪／四輪粗分組之間不一致時標記（`class_consistency.csv`）。
  - **外觀離群**：裁切到身分平均特徵的 cosine 距離超過同類別第 99 百分位（`appearance_outliers.csv`）。
  - **靜止身分**：首末兩張的中心點位移 < 0.5 倍框尺寸且持續 ≥ 5 秒，目標是號誌等誤判（`motion_flags.csv`）。
  - **運動跳動**：相鄰取樣的速度突變，目標是 ID 閃跳（`motion_flags.csv`）。
- 裁切層級的判定類別與離群結果在 `contamination_crops.csv`。
- 全量約 29 萬張時曾發生一次 segmentation fault；若再發生，以 `python -X faulthandler -u ... --batch-size 256` 重跑並保留呼叫堆疊。

## 3. 人工審查頁

```bash
python scripts/reid_data/generate_identity_review.py --audit-per-camera 20 --seed 0
```

- 只包含裁切數 ≥ `min_crops_per_identity` 的身分：任一標記的身分、bus 的全部身分，以及每支鏡頭以固定種子從未標記身分中抽出的隨機抽查身分。
- 依原因分頁（靜止 → 運動跳動 → 外觀離群 → 類別不一致 → bus → 隨機抽查）；靜止分頁可整頁全選。
- 頁面中的勾選代表**排除**：勾身分＝排除整個身分，勾縮圖＝只排除那一張。「產生確認清單」匯出 `class,identity,image,action` 的 CSV。
- 伺服器沒有圖形介面時，在輸出目錄執行 `python -m http.server 8765 --bind 127.0.0.1`，再以 VS Code 或 `ssh -L` 轉發連接埠開啟；CCTV 畫面不要上傳到外部服務。

## 4. 套用排除清單

```bash
python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv          # dry-run
python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv --apply  # 寫入
```

- 合併進 `dataset/reid/exclusions.csv`（可用 `--exclusions` 指定），去除重複並依鍵排序；不動任何裁切。

## 5. 產生資料集版本（manifest）

```bash
python scripts/reid_data/build_reid_split.py --version v1
python scripts/reid_data/build_reid_split.py --version v2 --exclusions dataset/reid/exclusions.csv
```

- 依序套用：排除清單 → 捨棄少於 4 張的身分 → 每個身分依時間均勻抽樣到最多 30 張 → 依鏡頭與日期切分（`config/reid_split.yaml`：C000054 的 20250629 為 val，其他全部為 train）。
- 同一個身分只會出現在一個 split；評估鏡頭永遠不會出現；相同輸入產生逐位元組相同的 manifest；不複製影像。
- 版本目錄已存在時報錯，不覆寫。

## 目前資料量（2026-09-24）

來源：9 支鏡頭、39 支影片（C000054 31 支，C000011／012／033／048／056／256 各 1 支約 10 分鐘，C000043／085 各 1 支 60 秒剪輯）。

| 類別 | 裁切數 | 身分數 | `v1` 身分數（train／val） |
|---|---:|---:|---|
| car | 127,094 | 21,153 | 12,716（11,232／1,484） |
| scooter | 138,228 | 31,257 | 9,820（8,300／1,520） |
| truck | 11,642 | 3,339 | 859（720／139） |
| small_truck | 9,290 | 4,874 | 443（377／66） |
| bus | 1,921 | 409 | 189（143／46） |
| person | 372 | 130 | 29（23／6） |
| **合計** | **288,547** | **61,162** | **24,056（20,795／3,261）** |

- `v1` 是審查前的基準版本（174,344 張裁切），仍含號誌誤判、串錯與殘缺框。全量的審查、排除清單與 `v2` 由後續的自動排除 change 處理。
- person 與 small_truck 的身分明顯不足，是來源影片本身的限制。
- 遮擋造成的碎片（同一實體被切成多個身分）無法只靠追蹤參數接回，離線合併由後續 change 處理。
