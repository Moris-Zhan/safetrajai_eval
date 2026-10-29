# 重構交付報告

分支 `refactor/simplify-codebase`（從 `main` 分出），約 40 個 commit，確切數量以 `git rev-list --count main..refactor/simplify-codebase` 為準（合併前為 41）。所有數字都是 2026-10-29 當天實測，不是推估。

## 1. 結論

- 功能與介面保留：`pytest` 790 passed, 11 skipped（`main` 為 739 passed, 11 skipped，多出的 51 個都是新增的特徵測試）；golden 在 T5 之後與 B1 之後各跑一次，**與基準 `conflict1` 完全相同**（9 個類別、約 6,000 個檔案，差異與最大絕對差都是 0）。
- 程式碼淨減少 **522 行**（第一方 `.py`：17,202 → 16,680，不含 `_vendor` 與 `tests`）。這個幅度不大，因為前提是「保留所有功能與介面」，而且新增了兩個共用模組。
- 追蹤的檔案由 446 個降到 320 個，追蹤的 `.md` 由 163 個降到 48 個，主要來自你移到 `ignore/` 的 `docs/`、`openspec/` 與 opsx 工具檔。
- **只有一項刻意的輸出行為變更**：`visualize_gt_on_video.py` 的類別顏色（T7，見第 4 節）。另有幾處說明文字與錯誤訊息文字的變動，也列在第 4 節。

整體 diff：`git diff -M main --shortstat` → 431 個檔案，+6,070／−15,516。

## 2. 成功標準逐項結果（SPEC 第 9 節）

| # | 標準 | 結果 | 證據 |
|---|---|---|---|
| 1 | `pytest` 全過、passed 數只增不減、既有測試斷言零修改 | 通過 | 790 passed。9 個既有測試檔有改動，逐行檢查全是路徑更新（`config/` 子目錄、`src/` → `utils/modules/`）；`test_no_sys_path_mutation` 排除 `_vendor` 目錄，等同重構前 `third_party/` 本來就不在掃描根目錄內 |
| 2 | 三支 grid search 的 `--help` 與每個類別的 `--dry-run` 與基準相同 | 通過（有說明） | 18 組 `--dry-run`（3 腳本 × 6 類別）逐位元相同。`--help` 的差異只有 A2 刻意修正的文字（`--model` 說明、botsort 的 gmc 預設值、`--max-videos` 預設值，共 7 行文字變更，三支腳本合計）與 `--config` 預設路徑。`_default` 這個網格鍵無法由命令列選到（`--target-class` 有 `choices`），所以沒有對應的 `--dry-run` |
| 3 | 第一方程式碼淨減少、無新增循環 import | 通過 | 淨 −522 行；`tests/test_smoke.py` 會 import 全部模組且通過（沒有另外跑循環依賴分析工具） |
| 4 | `import utils.model`、`from safetrajai…` 在任意工作目錄可解析，`PROJECT_ROOT` 不變 | 通過 | 從 `/tmp` 實際執行：`PROJECT_ROOT` 為專案根目錄，`trackeval` 解析到 `utils/modules/_vendor/trackeval/` |
| 5 | 每個任務與 A／B 子項各有獨立 commit | 通過 | 見第 3 節 |
| 6 | 「絕對不做」清單零違反 | 通過 | `class_aware_botsort.py`／`class_aware_bytetrack.py` 為 100% 純改名（0 行變動）；`utils/model.py` 未動；vendored TrackEval 47 個檔案 `R100`（內容逐位元相同）加 1 個新增的 `LICENSE`；`tests/golden/` 只有 `run_all.sh` 的 4 行路徑更新；未提交 `skills-lock.json` 與 `tools/favorites install.v18.sh`；D 級 9 個符號全部保留 |
| 7 | `docs/superpowers`、`openspec`、`markdown/reference_` 在 `utils/modules/`、`config/`、`scripts/` 零命中 | 通過 | 實測 0 命中；`docs/adr` 也是 0 |
| 7 | `src/safetrajai` 在 T3 清單文件內零命中 | 通過（有一處刻意保留） | `utils/modules/safetrajai/tracking/HANDOFF.md:125` 保留在「路徑搬移歷史」一行中；`SPEC.md`、`tasks/` 本身描述搬移，不算 |
| 7a | `video/` 不存在、3 支影片大小不變 | 通過 | 大小 97,456,550／311,221,801／61,385,808 位元組與搬動前相同；`dataset/raw/videos/` 仍是 12 個鏡頭資料夾；程式碼中 `video/C0` 零命中 |
| 7b | `Markdown/` 結構與設計一致、連結有效、拆分前後正文行集合相同 | 通過 | 44 個 `.md` 檢查，0 個壞連結；拆分比對在 T3c 完成 |
| 7c | `Markdown/config/` 5 份文件存在且經驗證 | 通過 | T4b 時以程式比對 363 個識別字、11 個情境檔載入、CLI 旗標存在 |
| 7d | `scripts/` 根目錄無 `.py`、`run_all.sh` 語法正確、`config/homography/*.yaml` 內容不變 | 通過 | `scripts/*.py` 數量 0；`bash -n` 通過；T4 時 53 個 yaml 的 `yaml.safe_load` 逐檔相同 |
| 7e | `third_party/` 不存在、`import trackeval` 指向新位置、內容 100% 相同 | 通過 | `third_party/` 不存在；`git grep third_party` 在指定目錄零命中 |
| 8 | （原規格）golden 本輪未跑，須標示「未跑」 | **與規格不同，已執行** | 見第 5 節 |

## 3. Commit 對照（一個任務一個 commit，可單獨還原）

| 階段 | commit |
|---|---|
| T0 規格、計畫 | `0dd1b88` |
| T1 移除 openspec／舊文件／opsx 工具 | `68523d8`、`f9debfe` |
| T2 套件搬動、vendored TrackEval 搬動與授權 | `abd948b`、`82d4c99`、`e53ce62` |
| T2b、T2c、T4 影片、腳本、設定檔重新分層 | `919a757`、`60910a0`、`8f74c33`、`ef511a7` |
| T3、T3b、T3c、T4b 文件 | `13228d5`、`635cfcf`、`6d4e4de`、`c73f0df`、`cbb1300` |
| T5 A1–A8、`FALSE` import | `fbaa641`、`40c8c19`、`9f4ae2b`、`2846fbf`、`9b79c16`、`c40f898`、`a1c01b7`、`6577fdb`、`7902dfd`、`ec00930`、`65d6803`（另有 `ad9e5cd` 記錄進度） |
| T6 B4、B3、B8、B2、B7、B9、B5、B6、B1 | `37987d3`、`ff86641`、`1cf9c70`、`5610c91`、`4b1aabf`、`871679d`、`00bb141`、`d15927d`、`d6f4a47` |
| T7 顏色修正（行為變更） | `f9e04e4` |
| T8 改名、進度記錄、本報告前的文件修正 | `3c16397`、`5a26f87` |

B10（`file_md5`／`md5_file`）評估後不做：兩者回傳型別不同（字串對兩個值），不改呼叫端無法合併。

## 4. 可觀察到的行為變更（全部列出）

1. **T7，輸出影片顏色**（`f9e04e4`）：`visualize_gt_on_video.py` 的框與圖例原本每個類別拿到下一個編號的顏色（person 畫成天藍、car 畫成綠……），現在 person 黃、car 天藍、scooter 綠、small_truck 紫、bus 藍紫、truck 橘。`visualize_gt_on_class_video.py` 的輸出 0 幀改變；`--color_by id` 時只有圖例色塊改變。
2. **`--help` 文字**：三支 grid search 腳本的 `--model`（改為「目前沒有作用」）、`--max-videos`（預設 6）、botsort 的 `--gmc_method`（預設 sift）；`generate_identity_review.py --audit-only` 的說明少了 ADR 引用。
3. **三個錯誤訊息**少了結尾的「見 ADR …」，規則文字不變：`config_loader` 的 `det_corner` 遷移提示、`fragment_merge` 的 `time_margin_s` 不再支援、`tracking/config.py` 的融合參數改名。
4. **ROI 選點器產生的檔頭**：`make_roi_picker.py` 之後產生的 ROI 檔頭不再含 `docs/adr/0005`，與已改過的 10 個 `config/roi/*.yaml` 一致。
5. **副作用**：`trackeval.utils.get_code_path()` 現在指向 `utils/modules/_vendor`，所以 `Evaluator` 預設的 `error_log.txt` 只在評估拋出例外時才會寫到那裡；runner 已明確指定 GT／TRACKERS／OUTPUT 資料夾，正常執行不受影響。

## 5. 驗證方式與沒有涵蓋的部分

**golden 與規格的差異**：SPEC 預設「golden 本輪不跑，須標示未跑」，並把「跑 golden」列為先問再做。實際上在你同意之下跑了兩次：
- T5 結束後（`runs/refactor_golden/t5_check_20261029`）：涵蓋 T0–T5 的所有搬移與清理。
- B1 之後（`runs/refactor_golden/b1_check_20261029`）：涵蓋 B2–B9 與 B1。
- 兩次都用 `python tests/golden/compare.py runs/refactor_golden_archive/conflict1 <run>` 與基準比對，結果 identical。

**golden 之外的實測**：三支 grid search 腳本各用 1 支影片在 B2 前後與 B1 後各真實執行一次，分數、參數、patch 檔、結束碼逐項相同；視覺化腳本用合成影片逐幀比對；Excel 報表用合成資料逐儲存格比對。

**沒有涵蓋**：
- `run_trackeval_mot` 的預設類別清單只能間接驗證（需要完整 TrackEval 輸入），由 golden 的 stage2 涵蓋。
- 沒有跑 E 級長函式拆分（本輪排除）。
- 沒有針對循環 import 的專用工具檢查，只靠 smoke 測試的全模組 import。

## 6. D 級疑似死碼（9 個，全部保留，供你日後決定）

這 9 個符號在全部 `.py` 中只出現在自己的定義處；動態呼叫、notebook、外部腳本無法由靜態掃描排除，所以沒有刪。

- `save_homography`、`transform_points_batch`、`warp_and_overlay`（`collision/homography/__init__.py`）
- `list_available_tags`、`load_prebuilt_homography`、`save_verification_image`（`collision/homography/points.py`）
- `merge_map`（`reid/fragment_merge.py`）
- `keep_mot_gt_valid_ids`（`evaluation/mot_converter.py`）
- `clip_xywh`（`evaluation/utils.py`）

## 7. 沿途發現、未處理的問題（都不是這次重構造成的）

**行為上可能不是原意（我都只鎖住現狀，沒有修改）**
1. `visualize_gt_on_video.py` 與 `visualize_gt_on_class_video.py` 的 `main()` 都**忽略** `--trackeval_root` 與 `--video_path`，一律掃描寫死的 `dataset/eval/cctv_all8/annotations_all8/...`，與 `CLAUDE.md` 描述的用法不符。
2. `visualize_gt_on_class_video.py` 的類別編號依目錄讀取順序臨時指派，所以顏色取決於資料夾列舉順序，與真正的類別無關。
3. Excel 補列（MISSING）的排序依 tracker 名稱字母序，不是類別契約順序；`sort_rows_grouped` 的類別順位只在同一個 tracker 內才有作用。
4. DET 曲線降採樣：長度超過 `curve_max_points` 時步長為 `n // maxn`，實際點數可能超過上限（25 點、上限 10 → 13 點）。
5. `score_params()` 的 `gmc_method` 預設值是 `"ecc"`，但三支腳本實際傳入的值各不同（`none`／`sparseOptFlow`／CLI 的 `sift`）。
6. `CLASS_COLORS` 還留著 6、7 兩個舊類別（bicycle、motorcycle），不在 6 類契約內，現在對應不到任何 GT 類別。

**尚未統一的重複**
7. 影片尺寸讀取還有 7 處不在 SPEC 範圍：`sequence_tracker.py:140`、`core.py:328`、`compare_video.py:70`、兩支 visualize 腳本各 1 處、`tracking_pipeline.resolve_frame_rate` 另外讀 fps。
8. 兩支 visualize 腳本的 `main()`、`_draw_legend`、`_draw_frame_counter` 等仍有重複，只有 B5 規定的四個符號已合併。

## 8. 殘餘待辦

1. `tests/golden/TOLERANCE.md:95` 仍寫 `compare.py conflict1 <run>`（該資料夾受保護，沒有修改）；`CLAUDE.md` 與 `Markdown/testing/testing.md` 已改為路徑寫法。
2. `Sub_Module.md` 已過時。
3. 文件盤點：`docs/`、`Handoff/` 等已移到 `ignore/`，哪些要重寫尚未決定；`Markdown/` 內部分既有敘述（「已知限制」中的 `track_buffer` 單位、`config/homography_points`、「進行中／尚未執行」）是原樣搬動、尚未重新釐清。
4. 上游 TrackEval 授權：已在 `e53ce62` 補上 MIT `LICENSE`，此項完成。
5. 移到 scratchpad 的 `src/safetrajai.egg-info` 與 `scripts/roi/__pycache__` 沒有刪除；`runs/refactor_golden_archive/` 內保留 8 組舊 golden 結果，`runs/refactor_golden/` 內有兩次新 golden 輸出。
6. `SPEC-features.md` 的啟動：F1 網頁選點器 → F2 計數線 → F3 鳥瞰影片 → F4 盲區，預設全部關閉；需等你核准合併後才開始（檢查點 D）。

## 9. 需要你決定

1. 是否核准合併 `refactor/simplify-codebase`（檢查點 D）。
2. T7 的 small_truck／bus／truck 顏色要跟著「編號位置」（目前）還是「類別名稱」（原註解的對應）。
3. 第 7 節的行為問題要不要另開 change 處理，優先順序如何。
4. `visualize_gt_on_class_video.py` 的顏色是否一併改成依契約編號。
