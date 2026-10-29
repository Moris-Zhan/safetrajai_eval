# 追蹤情境與 `class_tracking`

> 涵蓋 `config/tracking/` 與 `config/grid_search/`。區塊的共通說明見 [overview.md](overview.md)。

## 1. 我該用哪一個檔

| 我想要… | 用這個檔 | tracker | Re-ID |
|---|---|---|---|
| 基準：ByteTrack 完整流程（stage1＋stage2） | `config/tracking/bytetrack.yaml` | ByteTrack | 無 |
| BoTSORT，Re-ID 關閉 | `config/tracking/botsort_noreid.yaml` | BoTSORT | 6 類都關 |
| BoTSORT，Re-ID 全開 | `config/tracking/botsort_reid.yaml` | BoTSORT | 6 類都開 |
| BoTSORT＋本地訓練的 Re-ID 模型 | `config/tracking/botsort_reid_timm.yaml` | BoTSORT | 目前 6 類都關，等同驗收基準 |
| 跑 ByteTrack 的 grid search | `config/grid_search/grid_search_bytetrack.yaml` | ByteTrack | 無 |
| 跑 BoTSORT 的 grid search | `config/grid_search/grid_search_botsort.yaml` | BoTSORT | 6 類都關 |

執行方式都是 `python track_eval.py --config <檔>`；grid search 檔由 `scripts/grid_search/` 的腳本以 `--config` 讀取（預設值：`grid_search_bytetrack.py` 與 `grid_search_botsort.py` 各自讀對應的 `grid_search_*.yaml`，`grid_search_botsort_reid.py` 讀 `config/tracking/botsort_reid.yaml`）。

## 2. 與 `bytetrack.yaml` 的差異

下列情境檔和 `bytetrack.yaml` 大部分內容相同，所以只列出差異；表中沒有提到的鍵，值都與 `bytetrack.yaml` 相同（以 `yaml.safe_load` 的結果比對）。BoTSORT 的檔案另外多出 BoTSORT 專屬的鍵：每個類別的 `model`、`appearance_thresh`、`proximity_thresh`、`use_reid`，以及 `default_tracking_params` 裡的 `model`；`botsort_reid.yaml` 的 `default_tracking_params.use_reid` 也是 `true`。

| 檔案 | `tracking_params` | 其他區塊 | `class_tracking` |
|---|---|---|---|
| `botsort_noreid.yaml` | `tracker: botsort` | — | 新增 BoTSORT 專屬鍵（見第 3 節）；`use_reid` 全為 `false`；`track_buffer` 60／90／60／90／90／90 |
| `botsort_reid.yaml` | `tracker: botsort` | — | 同上，但 `use_reid` 全為 `true`；`track_buffer` 全為 120。它對應 `grid_search_botsort_reid.py` 的 `PARAM_GRIDS` 固定值 |
| `botsort_reid_timm.yaml` | `tracker: botsort`、`frame_rate: auto`、`gmc_method: sparseOptFlow` | `save params.save: false` | `use_reid` 全為 `false`；`track_buffer` 10／15／10／15／15／15；四輪車的 `center_dist_weight` 加重（car 0.6，small_truck、bus、truck 0.4） |
| `grid_search_bytetrack.yaml` | 同基準 | `save params.save: false` | 與基準相同 |
| `grid_search_botsort.yaml` | `tracker: botsort` | `save params.save: false` | 與 `botsort_noreid.yaml` 相同 |

`track_buffer` 的順序是 `person`、`car`、`scooter`、`small_truck`、`bus`、`truck`；基準的 6 類都是 120。

## 3. `class_tracking:` 的結構

```yaml
class_tracking:
  class_names:            # 類別 id → 名稱（契約：person=0 … truck=5）
  class_tracking_params:  # 逐類別參數表
    car: { track_buffer: 120, match_thresh: 0.7, ... }
  default_tracking_params:  # 沒有該類別設定時的預設參數
  class_conf_thresh:      # 逐類別偵測信心門檻
  default_conf_thresh: 0.15   # 沒寫時為 0.15
```

載入時會檢查：`class_names`、`class_tracking_params`、`class_conf_thresh` 不可為空，`default_tracking_params` 必須是 mapping，每個類別的 `class_tracking_params` 必須是 mapping。偵測後會**依類別分組**才做三階段匈牙利匹配，所以不同類別不會共用同一個匹配矩陣。

### 逐類別參數

| 鍵 | 適用 | 說明 |
|---|---|---|
| `track_buffer` | 兩種 tracker | 保留期限，**正整數幀數**，不依幀率換算。每個類別都必須有有效值（沒有類別設定時取預設參數），浮點數與布林值會報錯 |
| `match_thresh`、`second_match_thresh`、`unconfirmed_match_thresh`、`new_track_thresh` | 兩種 tracker | 三階段匹配與新軌跡的門檻，每個類別各自一組 |
| `center_dist_weight`、`norm_factor` | 兩種 tracker | 中心點距離融合：`center_dist_weight > 0` 才啟用，正規化後的中心點歐式距離會與 IoU 距離線性加權 |
| `fuse_score` | 兩種 tracker | `true` 時把偵測信心分數融合進距離矩陣 |
| `use_reid` | BoTSORT | 該類別是否使用 Re-ID，預設關閉 |
| `model`、`appearance_thresh`、`proximity_thresh` | BoTSORT | Re-ID 模型路徑與外觀、IoU 接近度門檻。`appearance_thresh` 是**相似度方向**：容許的外觀距離上限為 `1 − 值`，值越大越嚴 |

`tracking_params.gmc_method` 只在 BoTSORT 有作用；ByteTrack 的檔案雖然也寫了這個鍵，但沒有效果。

### 選用的 Re-ID 融合參數

未設定時距離矩陣逐位元不變。設定了會檢查型別與範圍，不合法就報錯：

| 鍵 | 範圍 | 說明 |
|---|---|---|
| `reid_center_gate_px` | `> 0` | 只對遺失中的軌跡，以中心距離閘門擴大外觀的作用範圍 |
| `reid_center_gate_px_per_frame` | `>= 0` | 閘門放寬量：每幀像素 × 遺失幀數 |
| `appearance_veto_sim` | 0 到 1 | 外觀否決，同樣是相似度方向 |

實測上，閘門讓 car、scooter 的 IDSW 明顯增加，否決沒有效果，所以目前沒有任何情境啟用它們；這些鍵只在做 Re-ID 實驗時才用。

### 什麼時候可以啟用 Re-ID

啟用條件：在調參序列上與驗收基準比較，**不變差，而且 IDSW 減少或 IDP 上升**；只有 AssA 或 TP 上升、沒有 GT 的類別不啟用；沒有類別符合時全部關閉。目前 `botsort_reid_timm.yaml` 的 6 類都關閉。啟用 Re-ID 會讓每幀追蹤時間增加數倍。

## 4. Grid search 的起始設定

`grid_search_*.yaml` 的 `class_tracking` 只是「起始值」：每次試驗都會用腳本裡的 `PARAM_GRIDS` 動態覆寫對應欄位，內容與 `bytetrack.yaml`、`botsort_noreid.yaml` 相同，差別只在把 `save` 關掉。腳本的參數與使用流程見 [Grid Search](../grid_search/grid-search.md)。

## 5. 做自己的實驗

1. 複製最接近的基準檔到同資料夾並改名，例如 `config/tracking/my_experiment.yaml`。
2. 只改要試的鍵；改 `class_tracking_params` 時，每個類別都要有合法的 `track_buffer`。
3. 在 `README.md` 提到新檔名（`tests/test_readme_coverage.py` 會檢查）。
4. 驗證載入：

   ```bash
   python -c "from safetrajai.tracking.config import load_track_config; print(load_track_config('config/tracking/my_experiment.yaml'))"
   ```

5. 先用單一序列試跑，確認 `HOTA`、`AssA`、`IDSW` 與基準的差異，再決定要不要擴大範圍。
