# scripts/reid_data/ — Re-ID 資料集工具（資料位於 dataset/reid/）

Re-ID 訓練資料由 change `rebuild-reid-dataset`（2026-09-24）從 `dataset/raw/videos/` 重建，品質把關由 change `auto-reid-exclusions` 改為自動排除加抽查。流程是：

**蒐集 → 標記 → 自動排除 → 抽查 → 排除 → manifest**

裁切只寫一次，之後的標記、排除、抽查、切分都只產生紀錄，不搬移、改名或刪除任何裁切。術語（裁切、身分、污染身分、誤偵測身分、殘缺裁切、標記、排除、自動排除、抽查、不合格率、資料集版本、隔離）見 `CONTEXT.md`；排除與隔離的差異見 `docs/adr/0006`，為什麼以自動排除為主、人工只抽查見 `docs/adr/0007`。

> 舊的 `crop_dataset/`、`reid_category_dataset/`、`reid_dataset/` 是舊追蹤器產生的無效資料，已隔離到 `dataset/reid/temp/`（見該目錄的 `DO_NOT_USE.md`），不可引用或拿來訓練。

## 目錄

```
dataset/reid/
├── crops/                                   裁切（stage1 的 save_reid_crops 寫入）
│   ├── <class>/<序列>_track<id>/<frame>.jpg
│   └── _index/<序列>.csv                    每張裁切一列的 metadata 索引
│       _index/<序列>.skipped.json           略過統計（ROI 外、最短邊 < 32px）
├── auto_exclusions.csv                      自動排除清單（apply_auto_exclusions.py 由標記整份重算）
├── exclusions.csv                           人工排除清單（apply_review_exclusions.py 累加寫入）
├── splits/<version>/                        資料集版本
│   ├── manifest.csv                         crop_path,identity,class,camera,date,split
│   ├── summary.json                         各類別／split／鏡頭的身分數與張數、兩份排除清單的筆數、被排除的鏡頭
│   └── split_rules.yaml                     本次使用的切分規則複本
└── temp/                                    已隔離的舊資料，不可使用

dataset/reports/reid/                        標記（detect_contamination.py）與審查頁、audit_sample.csv（generate_identity_review.py）
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

## 3. 自動排除

```bash
python scripts/reid_data/apply_auto_exclusions.py          # dry-run，列印各規則的身分數與裁切數
python scripts/reid_data/apply_auto_exclusions.py --apply  # 整份覆寫 dataset/reid/auto_exclusions.csv
```

- 規則（只處理 ≥ `min_crops_per_identity` 張的身分；以 pilot 的人工審查校準，涵蓋 95% 的人工排除）：
  - `static`、`motion_jump`：整個身分排除（誤偵測身分、污染身分）。
  - `class_group`：身分中過半裁切被判為四輪、而身分不是四輪（或反之）時整個身分排除；人與二輪之間（例如機車騎士被判為行人）不排除。
  - `appearance_outlier`：只排除被標為離群的單張裁切（多為殘缺裁切）。
- 欄位 `class,identity,image,action,rule`；多條規則以 `+` 串接成一列。每次都從標記完整重算，不寫入人工排除清單；相同輸入產生逐位元組相同的檔案。

## 4. 抽查頁與不合格率

```bash
python scripts/reid_data/generate_identity_review.py --audit-only --seed 0
python scripts/reid_data/summarize_audit.py --confirm-csv <抽查頁匯出的確認清單...>
```

- 母體是自動排除後剩下的 ≥ 4 張身分（被整段排除的身分不進母體，縮圖不含已排除的單張）：
  - 身分數佔全部 ≥ 5% 的主要鏡頭（目前是 C000054）抽 60 個，依類別比例分層、每類至少 2 個；其他鏡頭各抽 15 個，不足則全數。
  - bus 另抽 20 個，person 全數審查；這兩者只寫入排除清單，不計算不合格率。
- 依用途與鏡頭分頁（`identity_camera_audit_<camera>_page*.html`、`identity_bus_audit_page*.html`、`identity_person_review_page*.html`），並輸出 `audit_sample.csv`。
- 頁面中的勾選代表**排除**：勾身分＝整段排除（計入不合格率），勾縮圖＝只排除那一張（不計入）。「產生確認清單」匯出 `class,identity,image,action` 的 CSV，檔名請改成頁名。
- `summarize_audit.py` 計算每支鏡頭（主要鏡頭另外計算 car、scooter）的不合格率：超過 10% 時，主要鏡頭調整規則並以新種子重新抽查（仍超過則該範圍的被標記身分全數人工審查），其他鏡頭以 `--exclude-cameras` 整支排除。
- 依標記原因分頁的舊模式（不帶 `--audit-only`）仍保留，供 pilot 與除錯使用。
- 伺服器沒有圖形介面時，在輸出目錄執行 `python -m http.server 8765 --bind 127.0.0.1`，再以 VS Code 或 `ssh -L` 轉發連接埠開啟；CCTV 畫面不要上傳到外部服務。

## 5. 套用人工排除

```bash
python scripts/reid_data/apply_review_exclusions.py --confirm-csv audit_C000054.csv          # dry-run
python scripts/reid_data/apply_review_exclusions.py --confirm-csv audit_C000054.csv --apply  # 寫入
```

- 合併進 `dataset/reid/exclusions.csv`（可用 `--exclusions` 指定），去除重複並依鍵排序；不動任何裁切。

## 6. 產生資料集版本（manifest）

```bash
python scripts/reid_data/build_reid_split.py --version v2
python scripts/reid_data/build_reid_split.py --version v3 --exclude-cameras C000056   # 整支排除抽查不合格的鏡頭
```

- 依序套用：自動與人工兩份排除清單（`--auto-exclusions`、`--exclusions`，任一份列出即移除）→ 捨棄少於 4 張的身分 → 每個身分依時間均勻抽樣到最多 30 張 → 依鏡頭與日期切分（`config/reid_split.yaml`：C000054 的 20250629 為 val，其他全部為 train）。
- 同一個身分只會出現在一個 split；評估鏡頭永遠不會出現；相同輸入產生逐位元組相同的 manifest；不複製影像。
- 版本目錄已存在時報錯，不覆寫。

## 目前資料量（2026-09-24）

來源：9 支鏡頭、39 支影片（C000054 31 支，C000011／012／033／048／056／256 各 1 支約 10 分鐘，C000043／085 各 1 支 60 秒剪輯）。

| 類別 | 裁切數 | 身分數 | `v1` 身分數（train／val） | `v2` 身分數（train／val） |
|---|---:|---:|---|---|
| car | 127,094 | 21,153 | 12,716（11,232／1,484） | 12,174（10,718／1,456） |
| scooter | 138,228 | 31,257 | 9,820（8,300／1,520） | 8,617（7,188／1,429） |
| truck | 11,642 | 3,339 | 859（720／139） | 797（665／132） |
| small_truck | 9,290 | 4,874 | 443（377／66） | 364（309／55） |
| bus | 1,921 | 409 | 189（143／46） | 184（140／44） |
| person | 372 | 130 | 29（23／6） | 28（23／5） |
| **合計** | **288,547** | **61,162** | **24,056（20,795／3,261）** | **22,164（19,043／3,121）** |

- **訓練請用 `v2`**（156,141 張裁切）：套用自動排除 4,374 列（整段 1,847 個身分、單張 2,527 張）與人工排除 80 列；抽查 223 個身分，各鏡頭的不合格率都不超過 10%（C000054 為 1.7%），沒有鏡頭被排除。細節見 `openspec/changes/archive/*-auto-reid-exclusions/verification.md`。
- `v1` 是審查前的基準版本（174,344 張裁切），仍含號誌誤判、串錯與殘缺框，只供比較。
- person 與 small_truck 的身分明顯不足，是來源影片本身的限制。
- 遮擋造成的碎片（同一實體被切成多個身分）無法只靠追蹤參數接回，離線合併由後續 change 處理。
