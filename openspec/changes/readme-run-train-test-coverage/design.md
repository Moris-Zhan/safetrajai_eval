# Design

## Context

動機與範圍見 `proposal.md`。盤點時觀察到的現況（2026-10-14）：

- README 第 7 節的指令只涵蓋 `track_eval.py`、`run_evaluation.py` 與 `scripts/reid_train/` 的三支腳本。`scripts/` 共 26 支、`config/*.yaml` 共 14 份，未出現在 README 的分別是 16 支與 6 份。
- 「訓練」在本專案只有 Re-ID 一條線（`scripts/reid_train/train_reid.py`，已列出）。repo 內沒有 YOLO 偵測模型的訓練腳本，偵測權重由外部提供（README 第 3 節已說明）。`grid_search_*` 是追蹤參數調參，不是模型訓練，目前也沒有列出。
- 完整指令表在 `CLAUDE.md`，但那是給 Claude Code 的開發規則；`Markdown/USAGE.md` 的 0.3 節仍寫「`rebuild-reid-dataset` 進行中」，已過時。README 才是 clone 後第一個被讀的文件。
- `tests/test_smoke.py` 已對全部腳本做 `--help` 檢查、對全部 config 做載入檢查，所以「腳本能啟動」已有保護，缺的是「文件有沒有提到它」。

### 盤點：沒有直接單元測試的模組

以 `tests/*.py` 內是否 import 該模組粗略判斷（smoke 測試只做 import 與 `--help`，不算行為測試；結果是估計值，實作時用 `grep` 重新確認）：

| 模組 | 備註 |
|---|---|
| `detection/detector.py` | 逐類別信心門檻過濾，只靠 golden 比對間接覆蓋 |
| `grid_search/core.py` | 「無論成功、跳過或例外都要還原 `CLASS_TRACKING_PARAMS`」的不變式沒有單元測試 |
| `tracking/config.py`、`tracking/matching.py` | `load_track_config()` 必須先於 `create_tracker()` 的順序契約，沒有測試 |
| `evaluation/mot_converter.py`、`excel_export.py`、`dataset_prep.py`、`pipeline.py` | stage2 資料轉換與報表，只靠 golden 比對 |
| `collision/detection_core.py`、`trajectory_adapter.py`、`output.py`、`homography/heatmap.py`、`pick_aid.py`、`compare_video.py` | stage3 TTC／PET 主體，只靠 golden 比對 |
| `visualization/drawing.py`、`cli/run_pipeline.py` | 繪圖與 CLI 參數解析 |
| `reid/dinov2_features.py` | 只在資料工具中使用 |

這些都是候選，不在本 change 補測試。

## Goals / Non-Goals

**Goals:**
- README 成為「指令入口」的單一總表：讀者不必翻 `CLAUDE.md` 就知道每支腳本和每份 config 是做什麼的、怎麼跑、需要什麼前提。
- 用一個輕量 pytest 讓「新增入口卻漏寫文件」在本機測試中立刻失敗。

**Non-Goals:**
- 不複製 `CLAUDE.md` 的全部說明；README 只放一行用途加一個指令範例，細節以連結指向 `scripts/reid_data/README.md`、`Markdown/USAGE.md`、`CLAUDE.md`、`docs/adr/`。
- 不重寫 `Markdown/USAGE.md`，不補測試，不改任何 `src/`、config 內容或 tracker override 方法。
- 不實際執行 grid search、golden 比對或任何 stage（耗時且可能覆寫 `runs/`）；指令範例以 `--help` 與 `--dry-run` 驗證能解析。

## Decisions

### D1：按「使用情境」分節，而不是按目錄列檔案
README 第 7 節擴為：7.1 追蹤＋評估、7.2 調參（grid search）、7.3 衝突事件與 Homography、7.4 Re-ID 資料、7.5 Re-ID 訓練與驗證、7.6 工具（ROI、疊圖）、7.7 測試。每節以表格列「腳本／設定、用途、前提」，再給最短可執行指令。

- 考慮過的替代方案：照 `scripts/` 目錄逐檔列表。拒絕原因：讀者的問題是「我要做 X 該跑什麼」，不是「這個目錄有什麼」；而且 Re-ID 資料流程有順序（蒐集 → 標記 → 排除 → 抽查 → 合併 → manifest），按目錄會失去順序。
- 現有 7.1 的編號與錨點（`#7-執行`）保持不變，避免破壞目錄連結；原 7.2「Re-ID 訓練與驗證」改編號為 7.5，目錄與內部連結同步更新。

### D2：前提用固定欄位標示，不散在段落裡
每個表格都有「前提」欄，用固定詞彙：`連網`、`金鑰:<環境變數名>`、`來源影片`、`權重`、`完整資料（HF 還原不足）`、`長時間`。這樣 spec 要求的「標明前提」可以被一眼掃讀。金鑰只寫環境變數名稱，不寫值。

### D3：覆蓋檢查以「檔名字串出現在 README」判定，豁免清單寫在測試檔內
`tests/test_readme_coverage.py`：列出 `scripts/**/*.py` 與 `config/*.yaml`（不含子目錄），對每個檔名檢查 `README.md` 是否含該檔名（含副檔名的 basename）。`EXEMPT` 是測試檔內的 dict，值為理由字串，理由不得為空。

- 考慮過的替代方案 A：解析 README 的 markdown 結構、要求每個檔案出現在表格。拒絕：過度耦合排版，改一次格式就壞，而檢查的目的只是「沒有被遺忘」。
- 考慮過的替代方案 B：把豁免清單放獨立 YAML。拒絕：只有一個使用者，多一個檔案只會增加跳轉；理由與檔名放一起最容易審查。
- 已知限制：字串比對可能被不相關的敘述誤滿足；用含副檔名的完整檔名（`run_pipeline.py`）降低誤判。

預期豁免清單為空或極短。`scripts/run_pipeline.py` 與 `track_eval.py` 等價，仍應在 README 提一行。

### D4：測試小節以三層呈現，並寫明「何時跑哪一層」
| 層級 | 指令 | 耗時 | 前提 | 何時跑 |
|---|---|---|---|---|
| smoke | `pytest tests/test_smoke.py` | 秒級 | 安裝即可 | 任何結構調整後 |
| 單元 | `pytest`（其餘 `tests/test_*.py`） | 秒～分鐘 | 不需資料／權重 | 提交前 |
| golden | `tests/golden/run_all.sh <run>` ＋ `python tests/golden/compare.py <基準> <run>` | 約 16 分鐘 | 完整資料、YOLO 權重 | 改動演算法或 stage 行為後 |

README 不寫死基準名稱（目前是 `conflict1`，換基準時容易漏改），只說明基準與換基準的原因記在 `tests/golden/TOLERANCE.md`。

目前 `tests/` 中唯一帶 `skipif` 的是 smoke 裡「被隔離的 Re-ID 權重可載入」測試；README 要說明它在權重不存在時會被略過，不是失敗。實作時重新用 `grep` 確認，不憑本文件的記錄。

### D5：README 與 `CLAUDE.md` 的分工
README 面向「clone 後想跑起來的人」，`CLAUDE.md` 面向「改程式的人」。指令範例允許兩邊重複，但規則性內容（凍結的 tracker 方法、禁止大量 I/O、隔離慣例）README 只放連結，不重述，避免兩份文件各自過時。README 第 8 節已有資料使用規則，維持不動。

### D6：已與使用者確認的範圍決定
- `Markdown/USAGE.md` 不在本 change 處理，另開 change。
- 覆蓋檢查放進預設 `pytest`；太嚴格時以含理由的 `EXEMPT` 緩衝，不關閉檢查。豁免清單預設為空，`scripts/run_pipeline.py` 以一行「與 `track_eval.py` 等價」說明，不豁免。
- README 全部留在第 7 節；實作後若超過約 400 行，再評估拆成獨立文件（該決定可事後調整）。
- 原 7.2 改為 7.5，接受外部舊錨點失效；repo 內以 `grep` 確認沒有殘留。
- 範例指令只以 `--help`、`--dry-run` 驗證，不實跑會寫入 `runs/` 的指令。
- 測試缺口不在本 change 補，之後另開 change，優先 `grid_search/core.py`（失敗也要還原參數）與 `tracking/config.py`（先載入設定再建立 tracker）。
- 實作完成後以單一 commit 提交（README 與測試一起），提交前顯示 diff 摘要並等使用者確認。
- README 維持繁體中文，指令與檔名保持英文。

## Risks / Trade-offs

- [README 變長、可讀性下降] → 每支腳本只放一行用途加一個指令；細節用連結；以表格而非段落呈現。
- [覆蓋檢查太寬鬆（字串比對）] → 接受此限制，目的是防遺忘而非審查文件品質；使用含副檔名的完整檔名比對。
- [覆蓋檢查太嚴格，阻擋合理的實驗腳本] → 提供含理由的 `EXEMPT`；理由必填，由 code review 把關。
- [README 指令範例與實際參數不符] → 實作時每個範例都以 `--help`（或 `--dry-run`）實際跑一次確認可解析；不跑會寫入 `runs/`、`dataset/` 的指令。
- [會連網或使用金鑰的指令被誤跑] → 範例旁標明前提；驗證時只跑 `--help`，不呼叫外部服務。
- [`USAGE.md` 過時內容與新 README 矛盾] → README 不引用 `USAGE.md` 的 0.x 節；更新 `USAGE.md` 列為後續事項。

## Migration Plan

純文件與一個新測試，無資料或設定遷移。回復方式：還原 `README.md` 並刪除 `tests/test_readme_coverage.py`。

## Open Questions

- `Markdown/USAGE.md` 是整份更新，還是改為指向 README，留待後續 change 決定（見 D6）。
