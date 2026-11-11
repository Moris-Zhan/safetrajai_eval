# Homography 驗收 spec（`training_eval_audit.md` 補強優先順序 #4）

日期：2026-11-11。

## 背景

`utils/modules/safetrajai/collision/homography/validation.py` 的 `validate_camera()` 目前「驗證」的定義只檢查獨立 check points 數量是否 ≥ `MIN_CHECKPOINTS`（4），完全沒有誤差大小的門檻。這個弱點在 2026-11-11 的調查中被實際資料印證：`C000054` 雖然點數足夠被標成 `validated=True`，但真實誤差（因點位順序記反）高達 51.6m，修正後才是合理的 2.64m（見 `Markdown/homography/checkpoint_validation_2026-11-11.md`）。

這份文件記錄「正式化成驗收 spec」的第一步：**結構先搭好，門檻留待定**。使用者 2026-11-11 明確決定誤差門檻先不定案，等累積更多鏡頭的實測資料後再決定；但「有沒有一個可重複執行、會印出判定結果的 spec」不需要等門檻數字才能動工。

## 做法

新增 `scripts/homography/acceptance_check.py`，做法仿照 `scripts/reid_train/acceptance_grid.py` 的 `summarize`：讀取 `validate_homography.py` 既有輸出的 `models/homography/validation_output/summary.csv`，套用門檻判定，輸出 CSV＋Markdown 報表（`--out` 指定輸出路徑前綴，同時寫 `.csv` 與 `.md`）。

**刻意不改動 `validate_homography.py`／`validate_camera()` 本身**：這份 spec 是獨立的下游報表工具，不碰現有的 `validated` 欄位語意，避免連動影響 golden 比對的既有輸出（golden 基準的 `homography` 類別目前逐位元比對，改動驗證邏輯本身會需要重建基準，超出這次「先搭結構」的範圍）。

### 判定邏輯（`verdict()`）

| 條件 | 判定 |
|---|---|
| `summary.csv` 裡沒有這支鏡頭（沒有矩陣） | `no_matrix` |
| `check_N` 是空值 | `no_checkpoints` |
| `check_N` < `MIN_CHECKPOINTS`（4） | `too_few_checkpoints` |
| 沒有 `check_dist_rmse_m`（`m_per_pixel` 不可用） | `no_meters_per_pixel` |
| 點數與公尺換算都可用，但 `ACCEPTANCE_RMSE_M` 是 `None` | `threshold_tbd` |
| 門檻已設定，`rmse_m ≤ ACCEPTANCE_RMSE_M` | `pass` |
| 門檻已設定，`rmse_m > ACCEPTANCE_RMSE_M` | `fail` |

`ACCEPTANCE_RMSE_M` 目前是 `None`（腳本內常數，有清楚註解說明原因）。之後決定門檻時，只需要把這個常數改成實際數值，不需要改動其他邏輯或報表格式。

### 範圍

只涵蓋 `conflict_analysis` 實際依賴 `meters_per_pixel()` 的 4 支鏡頭：C000025、C000043、C000054、C000085（見 CLAUDE.md「衝突指標分析」一節）。其餘有矩陣但不在這 4 支鏡頭內的路口（例如 C000297）跟衝突指標的可信度無關，不在這份驗收 spec 範圍內。

## 目前的驗收報表（2026-11-11，首次執行）

執行 `python scripts/homography/acceptance_check.py --out Markdown/homography/acceptance_report` 產生，數據與 `checkpoint_validation_2026-11-11.md` 記錄的一致：

| 鏡頭 | check_N | RMSE (m) | 判定 |
|---|---|---|---|
| C000025 | 2 | 4.46 | too_few_checkpoints |
| C000043 | 3 | 9.95 | too_few_checkpoints |
| C000054 | 4 | 2.64 | threshold_tbd |
| C000085 | 6 | 20.22 | threshold_tbd |

原始報表檔案：`Markdown/homography/acceptance_report.csv`、`Markdown/homography/acceptance_report.md`（每次重跑 `acceptance_check.py` 都會覆寫，反映當下的 `summary.csv`）。

## 尚未解決

- **誤差門檻本身未定案**：這是刻意的，等累積更多實測資料（更多鏡頭、更多 check points）後再由使用者決定，不是這次的遺漏。
- **C000025／C000043 的 `too_few_checkpoints`**：需要再補點才能跨過 `MIN_CHECKPOINTS=4`，但 2026-11-11 的決定是「現有資料先夠用，不為了湊資料而回頭收集」，留待之後有需要時再處理。
- **spec 本身只到「讀報表、套門檻、出判定」這一層**，尚未決定門檻定案後是否要把這個判定接回 CI 或 pipeline（例如讓 golden 比對或 PR 檢查自動擋下不合格的鏡頭設定變更），這部分留給門檻定案之後再評估。
