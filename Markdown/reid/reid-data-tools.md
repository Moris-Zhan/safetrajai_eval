# Re-ID 資料工具（`rebuild-reid-dataset` 新流程）

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
