# Design

## Context

動機與範圍見 `proposal.md`；要驗證的行為見 `specs/track-config-loading/spec.md` 與 `specs/grid-search-trials/spec.md`。

現況（2026-10-14）：

- `tracking/config.py` 以**模組層級的全域 dict** 保存設定（`CLASS_NAMES`、`CLASS_TRACKING_PARAMS`、`DEFAULT_TRACKING_PARAMS`、`CLASS_CONF_THRESH`，另有 `_DEFAULT_CONF_THRESH[0]` 與 `_TRACK_CONFIG_RESOLVED`），載入時原地 `clear()`＋`update()`，且 **import 時就會載入預設 `config/bytetrack.yaml`**（檔尾）。所以任何測試呼叫 `load_track_config` 都會改變整個 pytest session 共用的狀態。
- 全域的消費端：`ClassAwareBOTSORT.__init__` 在建立時取 `CLASS_KEYS` 快照並為 `use_reid` 類別建立編碼器；`ClassAwareBYTETracker` 與 `Detector` 則在 `update`／`detect` 時**即時**呼叫 `get_params_for_class`、`get_conf_thresh_for_class`，建立時不固定任何參數。
- `grid_search/core.py` 的 `score_params` 直接修補 `CLASS_TRACKING_PARAMS[target_class]`，並在 `finally` 還原；其餘重度依賴（`Detector`／YOLO、`cv2.VideoCapture`、`save_mot_format`、`run_trackeval_mot`）都從模組層級 import，可以用 `monkeypatch` 在模組上替換。
- 既有測試已涵蓋部分 `config.py` 行為：`test_reid_fusion.py`（融合參數無效時不污染全域）、`test_tracker_retention.py`（`track_buffer` 驗證、原地更新維持參考、各類別取值）、`test_smoke.py`（所有 config 可載入）。新測試**不重複**這些項目。

### 已觀察到的缺口（讀程式碼得到，尚未用測試驗證）

| # | 位置 | 觀察 | 期望行為 |
|---|---|---|---|
| G1 | `config.py:192-215` | `class_tracking_params` 中不在 `class_names` 的類別若值不是 mapping，或 `class_conf_thresh` 的值不是數字，錯誤發生在全域已被 `clear()` 之後，失敗後全域是半更新狀態 | 驗證全部在變更全域前完成 |
| G2 | `core.py:430-437, 537` | 目標類別原本不在 `CLASS_TRACKING_PARAMS` 時，`finally` 把它還原成 `{}`；鍵留在原地，`get_params_for_class` 之後回傳 `{}` 而不是預設參數 | 還原後該鍵不存在 |
| G3 | `core.py:197-200` | 找不到精確的序列目錄時，以序列名前 20 字元 glob 並取排序第一個；C000025 的 6 段序列前 20 字元相同，可能取到別段序列的 GT | 有歧義時回傳 `None` |
| G4 | `core.py:145-150` | 找不到目標類別時退回「第一個 dict」，可能是 `cls_comb_cls_av` 等彙總項，回傳別的分數 | 回傳 `None` |
| G5 | `core.py:629, 716-718` | 失敗的 trial（分數 `-inf`）一樣寫入 CSV，`--resume` 會把它當成已完成，不再重試 | 失敗的 trial 在 resume 時重跑（依 `CLAUDE.md`「已經完成的組合」的解讀，需確認） |

另有三項不測試、僅記錄：`core.py:444` 硬編碼 `"mot_annotations_all8"`（`dataset_layout` 的說明要求呼叫端不再硬編碼；此處只影響暫存目錄名稱）；`run()` 的 `remain = total - done` 在 CSV 含有不在目前網格內的組合時會算錯（只影響顯示）；`_append_csv` 在續寫既有 CSV 且參數鍵不同時，欄位與標頭會錯位。

實作時另發現一項**環境風險**（不是 `src/` 缺陷，因此不用 `xfail`）：`import trackeval` 會把 `np.bool` 重新綁定成內建的 `bool`（numpy 2.2.6 原為 `np.bool_`），而 `numpy.testing` 內部呼叫 `np.bool(array)`，同一個 pytest session 之後所有 `assert_allclose` 都會拋 `ValueError`。`test_grid_search_core.py` 在收集階段 import `core`（連帶 import TrackEval）就會觸發，造成排在前面的 `test_conflict_metrics`、`test_homography_georef`、`test_roi_and_layout` 共 13 個測試失敗。TrackEval 是唯讀的（ADR 0002），所以改在該測試檔 import 之後還原 `np.bool`，並以 `test_importing_core_does_not_leave_numpy_bool_rebound` 防止日後被移除；`np.int`、`np.float`、`np.object`、`np.str` 在 numpy 2 本來就不存在，不處理。往後任何在收集階段就 import `trackeval_runner` 的測試檔都會遇到同樣的問題。

`CLAUDE.md` 說載入順序錯了會「悄悄用到空的或舊的參數」，但 `config.py` 在 import 時已載入預設設定，實際是用到**預設或舊**參數，不會是空的。

## Goals / Non-Goals

**Goals:**
- 兩份新測試檔在秒級跑完，不讀 `dataset/`、`models/`、`runs/`，不需網路、GPU 或權重。
- 測試對全域狀態零污染：任何測試前後，`config` 的全域與其他測試看到的相同。
- 疑似缺陷有可追蹤的記錄，而且修正後會被提醒移除標記。

**Non-Goals:**
- 不修改 `src/` 的任何程式碼，不修缺陷。
- 不重構既有測試（`test_tracker_retention.py` 自己的快照輔助函式維持原樣）。
- 不測 `_run_tracking_for_video`（迴圈內是 cv2、偵測與追蹤的整合，golden 比對覆蓋），不測 `print_summary`／`generate_config_patch` 的輸出格式。
- 不測 import 時的預設載入（`importlib.reload` 會讓其他模組持有的舊 dict 失效；子行程要付出 torch import 的時間）；`load_track_config(None)` 已涵蓋預設路徑。

## Decisions

### D1：全域狀態以「快照＋原地還原」隔離
`tests/test_track_config.py` 內定義 fixture：測試前對四個 dict 做 `deepcopy` 並記下 `_DEFAULT_CONF_THRESH[0]` 與 `_TRACK_CONFIG_RESOLVED`；測試後以 `clear()`＋`update()` 原地還原並寫回兩個純量。`test_grid_search_core.py` 的 `score_params` 測試則用 `monkeypatch.setitem(config.CLASS_TRACKING_PARAMS, ...)` 預先設定已知值，讓 teardown 由 pytest 還原——即使被測程式沒有正確還原也不會外洩。

- 替代方案：`monkeypatch.setattr(config, "CLASS_TRACKING_PARAMS", {})` 換掉整個物件。拒絕：`class_aware_botsort` 等模組以 `from .config import CLASS_TRACKING_PARAMS` 持有舊參考，換物件會讓它們看不到變動，反而掩蓋「原地更新」這項契約。
- 替代方案：放進共用 `tests/conftest.py` 的 autouse fixture。拒絕：會改變所有既有測試的執行環境；兩個檔案只有一處需要，直接放在使用處。

### D2：設定用 `tmp_path` 現場產生，小而完整
輔助函式 `write_config(tmp_path, ...) -> Path` 產生只有 2 個類別的最小合法 YAML（含 `track_buffer`），各測試以覆寫欄位製造缺漏或錯誤。不讀 `config/*.yaml`，所以測試不會因情境設定被調整而失敗。

### D3：`score_params` 在模組縫上替換重度依賴
`monkeypatch.setattr` 替換 `core.Detector`、`core._run_tracking_for_video`、`core.cv2.VideoCapture`、`core.save_mot_format`、`core.run_trackeval_mot`、`core._resolve_gt_source_root`、`core._setup_trial_gt`、`core.extract_assa_mean`。假的 `_run_tracking_for_video` 會在被呼叫時複製一份 `CLASS_TRACKING_PARAMS[target]`，供測試斷言「trial 期間看到修補後的參數」。`cfg` 用 `types.SimpleNamespace`，只放被讀取的欄位。

- 替代方案：用 `cv2.VideoWriter` 產生真實的小影片。拒絕：慢、依賴編碼器，而且被測的是參數還原與分數彙總，不是影片讀取。

### D4：`GridSearchRunner` 替換 `score_params`，用 `tmp_path` 寫真實 CSV
假的 `score_params` 依序回傳預先設定的分數，並記錄被呼叫時的 `params`；`output_root` 與 `result_csv` 都在 `tmp_path`。CSV 的讀寫走真實路徑，才能驗證 resume。

### D5：缺陷用 `xfail(strict=True)` 記錄，測試名稱帶缺口編號
每個缺口一個測試，寫的是**期望行為**，並標 `@pytest.mark.xfail(strict=True, reason="G<n>: ...")`。目前為預期失敗、套件維持綠燈；日後修正程式碼後，strict 會讓該測試變成失敗並提醒移除標記。`xfail` 測試**不**對應 spec 需求（spec 只含已驗證為真的行為）；修正缺陷的 change 要把期望行為補進 spec。G5 的期望行為屬於對 `CLAUDE.md` 的解讀，見 Open Questions。

- 替代方案：特徵化測試（把現況寫成斷言）。拒絕：會把可能錯誤的行為寫成「正確」，修正時還得改測試。

### D6：順序契約只測 BoTSORT 的建立時快照
以既有 `test_tracker_retention.py` 的方式建立 BoTSORT（所有類別 `use_reid: false`，不載入模型），記下 `CLASS_KEYS`，再載入類別集合不同的設定，斷言 tracker 的 `CLASS_KEYS` 不變。ByteTrack 與 `Detector` 是即時讀取，不構成順序契約，只在本文件記錄。只觀察、不呼叫凍結的 override 方法（ADR 0004）。

### D7：風格
依使用者偏好：函式式、所有函式加型別標註、變數名稱完整（`restoredParams` 而非 `rp`）、註解只用英文且只在不明顯處；以 `pytest.mark.parametrize` 收斂相似案例，但不為少於三個的重複抽出輔助函式。

## Risks / Trade-offs

- [測試污染全域後影響其他檔案] → D1 的還原 fixture；驗證時整個套件各以正常順序與調換檔案順序跑一次。
- [對 `core` 的模組層級名稱做 `monkeypatch`，內部改名就會壞] → 縫的數量控制在 D3 列出的幾個；改名時測試會明確報錯，不會悄悄通過。
- [`xfail(strict=True)` 在無關的重構讓缺陷「意外消失」時失敗] → 這正是設計目的；失敗訊息帶缺口編號。
- [期望行為由我推定，可能與維護者的意圖不同] → 只有 G1～G4 是明確的資料完整性問題；G5 的語意列入 Open Questions，確認前維持 `xfail`，不寫進 spec。
- [測試對 `print` 輸出沒有斷言] → 刻意不斷言，避免與格式耦合。
- [收集階段 import TrackEval 會污染 `numpy.testing`] → 見 Context 的環境風險；在 `test_grid_search_core.py` 還原 `np.bool` 並加防護測試，驗證時必須跑完整套件（單獨跑新檔案看不出這個問題）。

## Migration Plan

只新增兩個測試檔，無資料或設定遷移。回復方式：刪除這兩個檔案。

## Open Questions

- G5：失敗的 trial（`-inf`）在 `--resume` 時應該重跑，還是視為已完成？目前依 `CLAUDE.md`「只能跳過……已經完成的組合」解讀為重跑；若維護者認為「有結果就算完成」，改成特徵化測試即可，不影響其他任務。
- `CLAUDE.md` 的載入順序描述要不要在修正缺陷的 change 一併更正（見 Context 末段）。
