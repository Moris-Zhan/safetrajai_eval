# Re-ID 設定（`config/reid/`）

> 這個資料夾有兩種檔案：**情境檔**（交給 `track_eval.py` 蒐集裁切）與**工具設定**（由 `scripts/reid_data/` 的腳本讀取）。整個資料流程（蒐集 → 標記 → 排除 → 抽查 → 碎片合併 → manifest）見 [Re-ID 資料工具](../reid/reid-data-tools.md)。

## 1. 檔案與用途

| 檔案 | 類型 | 用途 | 誰讀它 |
|---|---|---|---|
| `reid_crops.yaml` | 情境檔 | **正式**的 Re-ID 訓練資料蒐集：在來源資料集 `reid_sources` 上追蹤，只跑 stage1 | `track_eval.py --config` |
| `botsort_save_reid_crops_video.yaml` | 情境檔 | 同樣用 `reid_sources`，但 `frame_rate: 30`，裁切輸出放 `runs/reid_crops_video/crops`，不寫進 `dataset/reid/`；保留作對照 | `track_eval.py --config` |
| `botsort_save_reid_crops.yaml` | 情境檔 | 情境 3 的原始版本，資料集是評估用的 `cctv_all8` | `track_eval.py --config` |
| `reid_split.yaml` | 工具設定 | Re-ID 資料集版本的切分規則 | `build_reid_split.py`、`apply_auto_exclusions.py`、`detect_contamination.py`、`generate_identity_review.py`、`summarize_audit.py`（參數 `--rules`） |
| `fragment_merge.yaml` | 工具設定 | 遮擋碎片合併的候選條件與外觀門檻 | `extract_fragment_features.py`、`build_fragment_merges.py`（參數 `--config`） |

> ⚠️ `botsort_save_reid_crops.yaml` 的資料集是 `cctv_all8`，其鏡頭 `C000025` 是評估鏡頭。開啟 `save_reid_crops` 時，蒐集前的檢查會直接拒絕（`評估鏡頭不得用來蒐集 Re-ID 裁切`）。要蒐集裁切請用 `reid_crops.yaml`，或把 `paths.dataset` 改成 `reid_sources`。

## 2. 蒐集裁切：`reid_crops.yaml`

```bash
python track_eval.py --config config/reid/reid_crops.yaml
```

與 `bytetrack.yaml` 相比的主要差異：

| 鍵 | 值 | 原因 |
|---|---|---|
| `paths.dataset` | `reid_sources` | 只從 `role: source` 的鏡頭蒐集 |
| `stages.run_stage2` | `false` | 只跑 stage1 |
| `save params.save_reid_crops` | `true` | 開啟裁切蒐集；`save`、`save_txt` 關閉以減少磁碟用量 |
| `tracking_params` | `tracker: botsort`、`frame_rate: auto`、`gmc_method: sparseOptFlow` | `auto` 依影片實際幀率；`sparseOptFlow` 比 `ecc` 快（`ecc` 是 CPU 瓶頸） |
| 四輪車的 `center_dist_weight` | car 0.6，small_truck、truck、bus 0.4 | 四輪車加重中心距離 |

蒐集規則固定在程式裡，不能從 config 調整：框中心點要在 ROI 內、最短邊至少 32 px、每條軌跡每 0.5 秒最多取一張。輸出在 `dataset/reid/crops/{class}/{seq}_track{id}/{frame:06d}.jpg`，另有每支序列一份索引 `dataset/reid/crops/_index/<seq>.csv`。**索引已存在時會報錯**，重跑前要先把舊輸出移走。

蒐集前會檢查：

- 評估鏡頭一律拒絕，不能進任何訓練資料。
- 每支要處理的鏡頭都必須有 `config/roi/<camera>.yaml`，缺少時在追蹤前就報錯，不會退回整張影像（見 [per-camera-configs.md](per-camera-configs.md)）。

想縮小範圍時，在 `paths` 加 `cameras: [<camera>, ...]`，只能取 `reid_sources` 登記鏡頭的子集。

## 3. 切分規則：`reid_split.yaml`

```yaml
val:
  - camera: C000054
    dates: ["20250629"]
min_crops_per_identity: 4
max_crops_per_identity: 30
```

- 列在 `val` 的「鏡頭＋日期」組合會進 val，其餘來源鏡頭全部進 train；評估鏡頭永遠不會出現在任何資料集版本。
- `min_crops_per_identity`：裁切數少於此值的身分不會進 manifest，也不會被標記。
- `max_crops_per_identity`：每個身分依時間均勻抽樣到最多這個張數。
- 訓練一律讀 `dataset/reid/splits/<version>/manifest.csv`，不要直接掃 `crops/`。產生版本：

  ```bash
  python scripts/reid_data/build_reid_split.py --version <v>
  ```

## 4. 碎片合併：`fragment_merge.yaml`

| 鍵 | 說明 |
|---|---|
| `track_config` | 取得逐類別 `track_buffer` 的 config 路徑，**相對於專案根目錄**，目前是 `config/reid/reid_crops.yaml`。搬動或改名這個檔時要同步修改 |
| `margin_frames` | 候選窗的餘裕幀數。候選窗 = `track_buffer + margin_frames`（幀，不依幀率換算） |
| `position_factor` | B 的起點距 A 末段外插位置的上限，單位是框尺寸（面積平方根，取兩者較大）的倍數 |
| `endpoint_crops` | 比外觀時，A 最後與 B 最前各取幾張裁切的特徵平均 |
| `threshold` | DINOv2 特徵平均的 cosine 相似度門檻，已用人工確認的樣本校準 |
| `sample` | 校準門檻時的人工確認抽樣設定（分層分位數、各層數量、主要鏡頭、目標準確率） |

同一序列、同一類別內，時間、位置與外觀三個條件同時符合才會合併；只靠時間與位置不可靠（車流密集的鏡頭會誤配）。流程：

```bash
python scripts/reid_data/extract_fragment_features.py
python scripts/reid_data/build_fragment_merges.py --sample 80 --seed 0   # 抽樣給人工確認
python scripts/reid_data/build_fragment_merges.py --calibrate <csv>      # 以確認結果校準門檻
python scripts/reid_data/build_fragment_merges.py --apply                # 寫出合併清單
```

合併清單只寫 `dataset/reid/fragment_merges.csv`，不動裁切；產生資料集版本時用 `build_reid_split.py --merges dataset/reid/fragment_merges.csv` 套用。

## 5. 常見任務

| 我想要… | 做法 |
|---|---|
| 蒐集 Re-ID 裁切 | `python track_eval.py --config config/reid/reid_crops.yaml`；先確認每支鏡頭都有 ROI，且 `dataset/reid/crops/_index/` 沒有舊索引 |
| 只蒐集某幾支鏡頭 | `paths.cameras: [<camera>, ...]` |
| 改 val 的鏡頭或日期 | 改 `reid_split.yaml` 的 `val`，重新跑 `build_reid_split.py --version <新版本>`（不要覆寫舊版本） |
| 改合併門檻 | 改 `fragment_merge.yaml` 的 `threshold`，並重新用人工確認的樣本校準 |
| 確認設定檔可讀 | `python scripts/reid_data/build_reid_split.py --help`、`python scripts/reid_data/build_fragment_merges.py --help` |
