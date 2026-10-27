# Tasks

## 1. tracking/config.py 的單元測試

- [x] 1.1 新增 `tests/test_track_config.py`，含還原全域狀態的 fixture（四個 dict 的 `deepcopy` 快照、`_DEFAULT_CONF_THRESH[0]`、`_TRACK_CONFIG_RESOLVED`，測試後原地還原）與 `write_config(tmp_path, ...) -> Path`（2 個類別的最小合法 YAML）。還原邏輯做成 context manager，由 autouse fixture 使用。驗證：兩個自我測試（載入另一份設定後可還原、dict 物件身分不變，不依賴執行順序），`pytest tests/test_track_config.py -q` 通過。
- [x] 1.2 結構驗證與格式（spec「設定內容缺漏或型別錯誤時載入失敗」「接受章節式與扁平式兩種格式」）：缺少或空的 `class_names`／`class_tracking_params`／`class_conf_thresh`、`default_tracking_params` 非 mapping、根節點非 mapping 各一個案例（`parametrize`），加章節式與扁平式等價。驗證：每個 Scenario 對應一個測試，全部通過。
- [x] 1.3 失敗不污染（spec「已知的驗證失敗不改變已載入的全域參數」）：`class_names` 含非整數鍵、有效 `track_buffer` 缺少／布林／浮點／小於 1。斷言四個 dict 與 `get_track_config_path()` 失敗前後相等。不重複 `test_reid_fusion.py` 已涵蓋的融合參數案例。驗證：通過。
- [x] 1.4 成功載入（spec「成功載入後以原地更新取代全域參數」「設定檔路徑解析」）：既有參考看到新內容、信心門檻轉 `float`、`default_conf_thresh` 未設定為 0.15；絕對路徑、`monkeypatch.chdir` 後的相對路徑、`None` 與空字串載入 `config/bytetrack.yaml`、不存在路徑拋 `FileNotFoundError` 且全域不變。驗證：通過。
- [x] 1.5 查表與命令列（spec「查表函式在缺值時退回預設」「命令列只取出 --track-config」）：`get_params_for_class` 以 ID／名稱／未知 ID／`numpy` 整數查詢、`get_class_name` 的 `class_<id>`、`get_conf_thresh_for_class` 與 `get_min_conf_thresh`、`DEFAULT_CONF_THRESH` 與未知屬性拋 `AttributeError`、`reload_track_config_from_cli` 混有其他參數與無參數。驗證：通過。
- [x] 1.6 順序契約（spec「BoTSORT 在建立時固定類別集合」）：沿用 `test_tracker_retention.py` 建立 BoTSORT 的方式（所有類別 `use_reid: false`），建立後載入類別集合不同的設定，斷言 `tracker.CLASS_KEYS` 不變；再用同一份新設定重建 tracker，斷言 `CLASS_KEYS` 更新。驗證：通過，且測試沒有呼叫任何凍結的 override 方法。
- [x] 1.7 以 `xfail(strict=True)` 記錄 G1：不在 `class_names` 的類別有非 mapping 的設定、`class_conf_thresh` 含非數字，都應在失敗後保持全域不變。驗證：`pytest tests/test_track_config.py -q` 顯示 2 xfailed 且其餘通過；加 `--runxfail` 時這兩個測試的失敗原因確實是全域被改動（不是測試寫錯）。
- [x] 1.8 更新 `README.md` 7.7 的單元測試分組表，新增「追蹤設定」一列列出 `tests/test_track_config.py`。驗證：`pytest tests/test_readme_coverage.py -q` 仍通過，`grep -n test_track_config README.md` 有結果。

## 2. grid_search/core.py 的單元測試

- [x] 2.1 新增 `tests/test_grid_search_core.py`：`ParameterGrid`（展開順序、長度為乘積、`combo_id` 與鍵順序無關且值不同時不同）、`extract_assa_mean`（tuple 解包、`COMBINED_SEQ` 優先、`ndarray` 取平均、純量、`None`／非 dict／缺 `HOTA`／缺 `AssA` 回傳 `None`）。以 `xfail(strict=True)` 記錄 G4（找不到目標類別時應回傳 `None`）。驗證：`pytest tests/test_grid_search_core.py -q` 通過，G4 顯示 xfailed。
- [x] 2.2 GT 與影片輔助函式：`_setup_trial_gt`（複製、覆寫舊檔、`seqinfo.ini` 在根目錄或 `gt/` 下、缺 `gt/gt.txt` 回傳 `False`）、`_resolve_gt_source_root`（精確目錄優先、找不到回傳 `None`）、`discover_grid_search_videos`（預設掃描只列有 mp4 的序列、glob 覆寫、`max_videos` 截斷）。全部使用 `tmp_path` 與 `SimpleNamespace` cfg。以 `xfail(strict=True)` 記錄 G3（前 20 字元相同的兄弟序列，不得回傳別段序列的目錄）。驗證：通過，G3 顯示 xfailed。
- [x] 2.3 `score_params`：依 design D3 替換重度依賴；斷言 trial 期間目標類別參數是合併後的內容並保留未指定欄位、其他類別不變；正常結束、例外、GT 缺漏跳過三條路徑都還原；例外時回傳 `-inf`；多序列分數排除零分後取第 25 百分位、無有效分數回傳 `-inf`。預先以 `monkeypatch.setitem` 設定已知參數。以 `xfail(strict=True)` 記錄 G2（目標類別原本不存在時，還原後該鍵應不存在）。驗證：通過，G2 顯示 xfailed。
- [x] 2.4 `GridSearchRunner`：替換 `score_params`，在 `tmp_path` 寫真實 CSV。涵蓋 `resume=True` 只對未記錄的組合評分且結果與最佳取自全部組合、`resume=False` 全部重跑、每個 trial 追加一列且標頭只寫一次、平手不更新最佳、`results_` 與回傳值結構。以 `xfail(strict=True)` 記錄 G5（`-inf` 的 trial 在 resume 時應重跑）。驗證：通過，G5 顯示 xfailed。
- [x] 2.5 更新 `README.md` 7.7 的單元測試分組表，在「追蹤設定」一列加入 `tests/test_grid_search_core.py`（列名改為「追蹤設定與 grid search」）。驗證：`pytest tests/test_readme_coverage.py -q` 通過，`grep -n test_grid_search_core README.md` 有結果。

## 3. 整合檢查

- [x] 3.1 在 conda env `track_eval` 執行完整 `pytest`。驗證：全部通過，xfailed 恰為 6（G1 兩個，G2～G5 各一個），skipped 數量與實作前相同（11）。
- [x] 3.2 加 `--runxfail` 單獨跑兩份新測試檔。驗證：失敗的測試恰為上述 6 個 xfail 測試，且每個失敗訊息符合 design 表格中對應缺口的描述。
- [x] 3.3 檢查順序獨立與全域無外洩：以相反的檔案順序（`tests/test_grid_search_core.py tests/test_track_config.py tests/test_tracker_retention.py tests/test_reid_fusion.py`）跑一次，再跑完整套件一次。驗證：兩次都全部通過。
- [x] 3.4 對照兩份 spec 逐條核對：每個 Scenario 都能指出對應的測試名稱。驗證：產出對照清單，沒有未覆蓋的 Scenario。
- [x] 3.5 檢查 diff：`git status --short` 只應有兩個新測試檔、`README.md` 與本 change 目錄；`src/`、`config/`、`third_party/` 無變動。向使用者顯示 diff 摘要並等待確認後，以單一 commit 提交（`test: add unit tests for track config and grid search core`）。驗證：使用者確認，且提交後 `pytest` 為綠燈。

## Workflow follow-up

- 歸檔前確認 `openspec validate add-config-and-grid-search-unit-tests --strict` 通過。
- 後續另開 change：修正 G1～G5（修正時移除對應的 `xfail`，並把期望行為補進 spec）；更正 `CLAUDE.md` 對載入順序的描述（實際是用到預設或舊參數，不是空的）。
