# 實作計畫：safetrajai_eval 精簡重構

> 依據：`SPEC.md`（已確認）。新功能另見 `SPEC-features.md`，在本計畫 T9 之後才開始，不在本計畫範圍。
> 任務細項與勾選清單見 `tasks/todo.md`。
> **狀態：已全部完成並合併進 `main`（2026-10-29，merge commit `8b5eb4e`）。** 實際結果與偏離規格之處見 `tasks/delivery-report.md`（例如 golden 原訂不跑，實際在 T5 後與 B1 後各跑一次，皆與基準 identical）。新功能計畫見 `tasks/plan-features.md`。

## 概觀

在不改變功能行為的前提下，完成三件事：

1. **整理目錄架構**：套件 `src/safetrajai` → `utils/modules/safetrajai`、vendored TrackEval → `utils/modules/_vendor/trackeval`、`video/` → `dataset/raw/homography_videos/`、`scripts/` 與 `config/` 依功能分資料夾、移除已搬走的 `openspec/`、`docs/`、`Markdown/` 舊文件與 opsx 工具檔。
2. **精簡冗餘**：刪除註解掉的程式碼與失效引用（A 級）、消除重複邏輯（B 級）。
3. **文件**：同步路徑、拆分 `USAGE.md` 成模組教學、新增 config 使用教學。

基準線：`main` 上 `pytest` = 739 passed, 11 skipped（約 43 秒）。golden 原訂本輪不跑；實際執行結果：pytest 790 passed, 11 skipped，golden 兩次 identical。

## 架構決策

- **結構搬動先於清理，文件最後同步**：目錄搬動（T1–T2d、T2b、T2c、T4）都不改行為，先做完，後續去重複任務才建立在最終目錄上；文件在 T3 一次對齊 `src`、`scripts`、`config` 三種新路徑，避免改兩遍。代價是中間 commit 的文件路徑暫時是舊的，整條分支合併前補齊。
- **套件名不變**：`safetrajai` 與 `trackeval` 的匯入名稱都不改，所以 pickle 路徑、`import utils.model`、所有 `from safetrajai… import …` 不受影響。
- **搬動用 `git mv`**：保留歷史；vendored 檔案相似度必須 100%。
- **不保留舊路徑相容層**：`--config`、腳本、套件路徑都直接一次改乾淨（Q23、Q53）。
- **特徵化測試先於抽取**：B 級每個去重項目先寫測試鎖定現行行為再動手；Excel 以合成資料逐格比對。
- **測試只允許機械式路徑更新**（SPEC 第 6 節）：斷言不改。
- **分支**：`refactor/simplify-codebase`，一任務一 commit，精確 `git add`，絕不使用 `git add -A`。

## 依賴圖

```
T0 分支與規格
 └─ T1 移除已搬走內容 ─┐
                       ├─ T2 搬套件 ─ T2d 搬 trackeval ─ T2b 搬 video/ ─ T2c 搬 scripts ─ T4 分 config
                       │                                                                      │
                       │      ┌───────────────────────────────────────────────────────────────┘
                       │      ▼
                       │   T3 文件路徑同步 ─ T3b 拆 USAGE.md ─ T4b config 教學
                       │
                       └─ T5 A 級清理 ─ T6 B 級去重 ─ T7 顏色修正 ─ T8 雜項 ─ T9 交付報告
```

T5 起的任務建立在最終目錄（T2–T4 完成後）上；T3、T3b、T4b 只動文件，可與 T5 之後的程式碼任務交錯，但為了讓 diff 單純，仍依序執行。

## 階段與檢查點

### 階段 0：準備
- T0：建分支、提交規格與計畫、擷取基準輸出到 scratchpad。

### 階段 1：結構搬動（行為不變）
- T1 移除、T2 搬套件、T2d 搬 trackeval、T2b 搬資料、T2c 搬 scripts、T4 分 config。

**檢查點 A（T4 完成後）**：`pytest` = 739 passed, 11 skipped；三支 grid search 的 `--help`（`--config` 預設字串除外）與 `--dry-run` 與基準相同；`import utils.model`、`import trackeval`、`import safetrajai` 在任意工作目錄可解析；凍結檔案 `git diff --stat` 零更動；與使用者一起審閱再繼續。

### 階段 2：文件
- T3 路徑同步、T3b 拆分 `USAGE.md`、T4b config 教學。

**檢查點 B（T4b 完成後）**：`git grep` 舊路徑零命中（SPEC 成功標準 7、7b、7c、7d、7e）；`README` 覆蓋測試通過；審閱。

### 階段 3：清理與去重
- T5 A 級（A1–A8）、T6 B 級（B2 → B3／B4 → B1 → B5 → B6–B10）、T7 顏色修正、T8 雜項。

**檢查點 C（T6 完成後）**：`pytest` 全過且新增特徵化測試；`--help`／`--dry-run` 比對相同；淨減行數。

### 階段 4：交付
- T9：交付報告（淨減行數、D 級清單、golden 未跑的聲明、殘餘待辦）。

## 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| 重裝 editable 後 `import` 失敗（pyproject 的 package-dir 寫錯） | 高 | T2、T2d 各自在重裝後立刻跑 `pytest` 與三個 import 檢查；失敗即 `git revert` 該 commit 並重裝 |
| `paths.py` 的 `parents[N]` 層數錯誤，`CONFIG_DIR` 指向錯誤位置 | 高 | T2 加入 `PROJECT_ROOT` 解析結果前後相同的檢查 |
| vendored 檔案被意外修改 | 高 | T2d 用 `git diff -M --stat` 確認相似度 100%；不在該目錄跑格式化 |
| 資料搬動（T2b）寫錯路徑 | 中 | 先貼指令與還原方式，同意後才執行；只比對路徑清單、檔案數與 `stat` 大小 |
| 凍結的 tracker override 被改到（A8 註解清理誤傷） | 高 | A8 明確排除 `tracking/class_aware_*.py`、`utils/model.py`、`_vendor/`；跑 smoke 的 AST 檢查 |
| 大量機械式修改（A8、T3、T4）漏掉某些檔案 | 中 | 每個任務結尾用 `git grep` 驗證舊字串零命中 |
| `--help` 因 argparse 重構（B2）改變 | 中 | 動手前先擷取基準；B2 逐位元 diff |
| GateGuard／分類器 hook 阻擋首次 Bash／Write／Edit | 低 | 先在文字中說明事實再重試 |
| 工作樹原本就有使用者的未提交修改（`skills-lock.json` 等） | 中 | 只 `git add` 自己改的檔案；T1 只暫存指定路徑 |
| 上游 LICENSE 抓不到原文 | 低 | 不自己編寫，記入待辦 |

## 待確認事項（進入實作時才需要使用者決定；皆已處理）

1. T2b 資料搬動前，貼出指令與還原方式請使用者同意。
2. T2d 補 LICENSE 需要一次對外網路請求，執行前提醒使用者。
3. T4 掃出的「已無人讀取的 config 鍵」清單，請使用者確認後才刪。
4. A6 `excel_export.py` 末段註解掉程式碼的確切範圍、B8 匯出方式、B10 是否可合併，實作時依實際內容決定並回報。
