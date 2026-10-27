# Proposal

## Why

`src/safetrajai/tracking/config.py`（282 行）與 `src/safetrajai/grid_search/core.py`（792 行）是整條調參與追蹤流程的共用基礎，但幾乎沒有直接的單元測試：

- `config.py` 只被間接測到——`test_reid_fusion.py`（融合參數驗證不污染全域）、`test_tracker_retention.py`（`track_buffer` 驗證、原地更新）、`test_smoke.py`（所有 config 可載入）。結構驗證、路徑解析、查表函式與退回預設值的行為都沒有測試。
- `grid_search/core.py` 完全沒有測試。`CLAUDE.md` 把「每個 trial 無論成功、跳過或例外都必須還原參數」與「`--resume` 只能跳過完全相同、已完成的組合」列為不可違反的規則，目前只靠 golden 比對（約 16 分鐘、需要完整資料）間接覆蓋。

這兩處的失誤都是**不會報錯**的那一類：參數沒還原，後續 trial 或後續流程悄悄用到被改過的參數；載入設定失敗卻留下半更新的全域狀態。需要能在幾秒內跑完、不依賴資料與權重的測試來防護。

讀程式碼時另外發現幾個疑似缺陷（見 `design.md` 的「已觀察到的缺口」）。本 change 只補測試，不修改 `src/`；缺口以 `xfail(strict=True)` 記錄期望行為，修正另開 change。

## What Changes

- 新增 `tests/test_track_config.py`，涵蓋 `tracking/config.py`：
  - 結構驗證（根節點、`class_names`、`class_tracking_params`、`default_tracking_params`、`class_conf_thresh` 缺漏或型別錯誤）、`class_tracking:` 章節與扁平格式兩種輸入。
  - 路徑解析順序與找不到檔案的錯誤、`path=None`／空字串使用預設設定。
  - 查表函式：`get_params_for_class`（整數、字串、未知類別退回預設）、`get_class_name`、`get_conf_thresh_for_class`、`get_min_conf_thresh`、`DEFAULT_CONF_THRESH`（含預設值 0.15）。
  - `reload_track_config_from_cli` 只解析 `--track-config` 並忽略其他參數。
  - 順序契約：BoTSORT 在建立時固定當下的類別集合（`CLASS_KEYS`），之後再載入設定不會更新已建立的 tracker。
- 新增 `tests/test_grid_search_core.py`，涵蓋 `grid_search/core.py`：
  - `ParameterGrid`（排序、長度、`combo_id` 與鍵順序無關）、`extract_assa_mean`（tuple 解包、`COMBINED_SEQ` 優先、陣列取平均、缺值回傳 `None`）。
  - `_setup_trial_gt`、`_resolve_gt_source_root`、`discover_grid_search_videos`（以 `tmp_path` 與假 cfg，不碰真實資料）。
  - `score_params` 的參數修補與還原：成功、跳過、例外三種路徑都還原；重度依賴（YOLO、cv2、TrackEval）以 `monkeypatch` 取代。
  - `GridSearchRunner` 的 resume 語意、CSV 寫入與最佳結果追蹤。
  - 測試用的 fixture 在每個測試前後還原 `config` 的全域 dict，避免污染其他測試。
- 疑似缺陷以 `xfail(strict=True)` 記錄（不改 `src/`）：載入失敗後全域被清空、`score_params` 對原本不存在的類別還原成 `{}`、GT 目錄前綴 fallback 取到別段序列、`extract_assa_mean` 退回任意類別、失敗的 trial 在 resume 時被當成已完成。

## Capabilities

### New Capabilities
- `track-config-loading`: 追蹤設定（`class_tracking`）的載入、驗證、查表與全域狀態更新契約。
- `grid-search-trials`: grid search 單一 trial 的參數修補與還原、評分，以及 resume 的跳過規則。

### Modified Capabilities
<!-- 無：既有 spec（tracker-retention、reid-fusion）的需求沒有改變 -->

## Impact

- 新增 `tests/test_track_config.py`、`tests/test_grid_search_core.py`；不改動 `src/`、`config/`、`third_party/`、tracker override 方法（ADR 0004）。
- 測試不讀 `dataset/`、`models/`、`runs/`，不需要網路、GPU 或權重，目標是幾秒內跑完。
- 後續另開 change：修正 design 列出的缺口（修正時同步移除對應的 `xfail`）；`CLAUDE.md` 對「載入順序」的描述（「悄悄用到空的或舊的參數」）與實際行為（import 時已載入預設 `bytetrack.yaml`，所以是用到預設或舊參數）不一致，一併更正。
