# Re-ID 資料工具（`rebuild-reid-dataset` 新流程）

> Re-ID 資料已由 change `rebuild-reid-dataset` 重建完成，現行流程多了自動排除與碎片合併，步驟與資料量見 [`scripts/reid_data/README.md`](../../scripts/reid_data/README.md)，指令索引見 README。舊的 `dataset/reid/` 內容與舊權重都已隔離，不要引用或拿來訓練。下面的前半部是最初的流程說明，保留作為各步驟的細節。

最初的流程：**蒐集 → 標記 → 審查 → 排除 → manifest**（現行流程另有自動排除與碎片合併）

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

## 1. 裁切蒐集（stage1 的 `save_reid_crops`）

- 以 ROI 中心點過濾，最短邊 < 32px 的框略過，每條軌跡每 0.5 秒取一張。
- 索引一次寫出；索引已存在時報錯（不會覆寫）。
- 追蹤前會先檢查：評估鏡頭一律拒絕，每支鏡頭都必須有 ROI 檔，任一不通過就不會產生任何輸出。
- 序列名稱必須能解析出日期，否則報錯。

## 2. 污染偵測

```bash
python scripts/reid_data/detect_contamination.py
python scripts/reid_data/detect_contamination.py --percentile 98 --min-agreement 0.6
```

- 每張裁切只讀一次，以 DINOv2 特徵同時做類別中心點比對與軌跡內外觀離群值偵測。
- 輸出到 `--out-dir`（預設 `dataset/reports/reid/`）：`class_consistency.csv`、`appearance_outliers.csv`、`contamination_crops.csv`。
- 只寫標記檔，不搬移、改名或刪除裁切；DINOv2 權重第一次使用時會從 HF Hub 下載。
- 其他參數：`--crop-root`、`--batch-size`、`--min-outliers`、`--min-fraction`。

## 3. 人工審查頁

```bash
python scripts/reid_data/generate_identity_review.py
python scripts/reid_data/generate_identity_review.py --out-dir dataset/reports/reid/review --page-size 100
```

- 審查範圍是「任一檢查標記的身分＋bus 的全部身分」，依類別分頁；每個身分最多 12 張依時間均勻抽樣的縮圖。
- 頁面上的「產生確認清單」按鈕會匯出 CSV（`class,identity,image,action`；`image` 空白代表整個身分）。

## 4. 套用排除清單

```bash
python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv          # dry-run
python scripts/reid_data/apply_review_exclusions.py --confirm-csv review_exclusions.csv --apply  # 實際寫入
```

- 合併進 `dataset/reid/exclusions.csv`（可用 `--exclusions` 指定），去除重複並依鍵排序。
- 只寫排除清單，不動任何裁切；排除是在產生資料集版本時由 `build_reid_split.py` 套用。

## 5. 產生資料集版本（manifest）

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

## 6. Re-ID 推論端

- `safetrajai/reid/encoders.py`：`ReID`（YOLO 當 encoder）與 `Custom_ReID`（`models/class_reid_weights/`）兩條推論路徑，透過 BoTSORT 的逐類別 `use_reid` 啟用。
- `utils/model.py` 是 pickle 相容 shim，**不可刪除或改名**（舊格式權重靠它反序列化）。
- 不要再引入已經移除的 SOTA_ReID、FastReID、TransReID。

---

以下兩節自 README 搬入（指令表與執行前提）。

## 指令與前提：Re-ID 資料

資料流程依序是 **蒐集 → 標記 → 自動排除 → 抽查 → 人工排除 → 碎片合併 → manifest**。各步驟的細節、輸出目錄與資料量見 [`scripts/reid_data/README.md`](../../scripts/reid_data/README.md)。腳本都在 repo 根目錄執行；寫入類的腳本不帶 `--apply` 時只列印計畫（dry-run）。

| 步驟 | 腳本／設定 | 用途 | 前提 |
|---|---|---|---|
| 1 蒐集 | `config/reid/reid_crops.yaml`（`python track_eval.py --config config/reid/reid_crops.yaml`） | 只跑 stage1，從來源鏡頭蒐集裁切到 `dataset/reid/crops/` | 來源影片（不在 HF，見 README 的「資料與模型」）、YOLO 權重、長時間 |
| 2 標記 | `scripts/reid_data/detect_contamination.py` | 以 DINOv2 標記污染身分，輸出到 `dataset/reports/reid/` | 個別裁切 JPEG；第一次使用從 HF Hub 下載 DINOv2（連網）；HF 還原的資料不足以執行 |
| 3 自動排除 | `scripts/reid_data/apply_auto_exclusions.py [--apply]` | 由標記整份重算 `auto_exclusions.csv` | 只讀報表與 CSV |
| 4 抽查 | `scripts/reid_data/generate_identity_review.py --audit-only` | 產生抽查頁與 `audit_sample.csv`（勾選＝排除） | 個別裁切 JPEG；HF 還原的資料不足以執行 |
| 4 抽查 | `scripts/reid_data/summarize_audit.py --confirm-csv <csv...>` | 每支鏡頭的不合格率（> 10% 要處理） | 只讀 CSV |
| 5 人工排除 | `scripts/reid_data/apply_review_exclusions.py --confirm-csv <csv> [--apply]` | 合併進 `exclusions.csv` | 只讀 CSV |
| 6 碎片合併 | `scripts/reid_data/extract_fragment_features.py` | 只讀碎片候選端點裁切的 DINOv2 特徵 | 個別裁切 JPEG；HF 還原的資料不足以執行 |
| 6 碎片合併 | `scripts/reid_data/build_fragment_merges.py [--sample N --seed S \| --calibrate <csv> \| --apply]` | 產生碎片合併清單 `fragment_merges.csv`；`--sample` 會複製裁切到確認頁，所以需要個別裁切 JPEG | `--calibrate`／`--apply` 只讀特徵與 CSV |
| 7 manifest | `scripts/reid_data/build_reid_split.py --version <v> [--merges dataset/reid/fragment_merges.csv]` | 兩份排除清單都套用，依 `config/reid/reid_split.yaml` 切分 train／val | 產生新版本需要 `crops/`，HF 還原的資料不足以執行（見 [`hf-restore.md`](../getting-started/hf-restore.md) 的 3.4）；直接使用下載的 manifest 即可 |

蒐集用的其他情境設定：

| 設定檔 | 說明 |
|---|---|
| `config/reid/botsort_save_reid_crops.yaml` | BoTSORT＋`save_reid_crops`，只跑 stage1。資料來源是**評估資料集** `cctv_all8`，裁切不得用於訓練 |
| `config/reid/botsort_save_reid_crops_video.yaml` | 同上，來源改為 `reid_sources`、`frame_rate: 30`、輸出放在 `runs/`；只保留作為對照，正式訓練資料請用 `config/reid/reid_crops.yaml` |
| `config/reid/reid_split.yaml` | 資料集版本的切分規則（哪些鏡頭＋日期為 val、每身分最少／最多張數），由 `build_reid_split.py` 讀取 |

## 指令與前提：Re-ID 訓練與驗證（需要 [`hf-restore.md`](../getting-started/hf-restore.md) 第 3 節的資料）

```bash
# val 評估既有模型（mAP／Rank-1，只作參考）
python scripts/reid_train/eval_reid.py \
  --weights models/class_reid_weights/dinov2s_v3/model.pt --version v3

# 重新訓練（30 epoch 約 25～45 分鐘，建議背景執行）
python scripts/reid_train/train_reid.py --version v3 \
  --output models/class_reid_weights/<new_run>

# 調參序列 GT 上的外觀距離分布（需要 `hf-restore.md` 第 2 節的 tune 子集；保留序列會被拒絕）
python scripts/reid_train/fusion_distance_stats.py --out runs/reid_stats/<name>.json
```

新模型要上線，須依 spec `reid-model` 的條件在調參序列上與驗收基準比較：結果不變差，而且 IDSW 減少或 IDP 上升。只有 val mAP 上升不算通過。

其餘兩支腳本：

| 腳本 | 用途 | 前提 |
|---|---|---|
| `scripts/reid_train/pack_manifest.py --version <v> [--reuse-version <v0>]` | 依 manifest 把裁切打包成本機快取 `~/.cache/safetrajai/reid_packs/<v>/`（只從掛載磁碟各讀一次） | **HF 還原後不要執行**：打包檔已還原，而且個別裁切 JPEG 不在 HF，重新打包會失敗（見 [`hf-restore.md`](../getting-started/hf-restore.md) 的 3.4） |
| `scripts/reid_train/acceptance_grid.py {configs,summarize}` | 追蹤驗收：`configs` 以 `config/tracking/botsort_reid_timm.yaml` 為底，產生每個子集的驗收設定；`summarize` 彙總逐類別結果並依 spec 判定（輸出 CSV＋Markdown） | `hf-restore.md` 第 2 節的 tune／holdout 子集、YOLO 權重；執行驗收設定要跑 stage1＋stage2，耗時 |

```bash
# 產生驗收設定 → 以 track_eval.py 執行 → 彙總
python scripts/reid_train/acceptance_grid.py configs --name <run> --set CLASS.KEY=VALUE
python track_eval.py --config runs/reid_domain/accept/configs/<run>_tune.yaml > runs/reid_domain/accept/<run>/tune/run.log 2>&1
python scripts/reid_train/acceptance_grid.py summarize --baseline baseline --runs <run> --out <prefix>
```

holdout 子集只作最終確認，不可拿來調參。目前 `config/tracking/botsort_reid_timm.yaml` 的 6 類 ReID 都關閉。
