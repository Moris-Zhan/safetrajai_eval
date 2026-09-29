# Proposal

## Why

change `train-reid-timm` 訓練出的 Re-ID 模型（`dinov2s_v3`，val mAP 0.783）在 C000025 的追蹤上沒有實質改善：`ClassAwareBOTSORT.get_dists_class_aware()` 只讓外觀距離作用在 IoU 距離 ≤ `1 − proximity_thresh` 的配對上，再取 `min(運動距離, 外觀距離)`。Re-ID 因此只能降低「本來就重疊」的配對成本，無法重新關聯遮擋後位置已偏移的遺失中的軌跡，而這正是 Re-ID 最該發揮作用的情況。第一階段匹配的候選池本來就包含遺失中的軌跡s，成本只經由這個可修改的函式計算，所以不需要動凍結的 `update()` 就能改善。

## What Changes

- **中心距離閘門（只作用在遺失中的軌跡）**：新增逐類別參數。開啟後，遺失中的軌跡與偵測的配對只要「IoU 在 `proximity_thresh` 內」或「中心距離在閘門內」其中之一成立，就可取得外觀成本（聯集，新行為是舊行為的超集）。中心距離取「預測框中心」與「最後一次 Tracked 時的中心」兩者中較近者，以同時涵蓋遮擋期間繼續移動與停下的物體。閘門半徑以像素計，可隨遺失秒數放寬。融合仍取 `min(運動距離, 外觀距離)`；Tracked 與未確認的 track 維持原本的 IoU proximity 規則。
- **外觀否決**：新增逐類別參數（預設關閉），開啟後外觀距離大於否決門檻的配對一律不可配，適用所有配對。
- **門檻校準工具**：新增 `scripts/reid_train/fusion_distance_stats.py`，以調參序列的 GT 量測各類別正樣本（同一條 GT 軌跡、間隔 ≥ 1 秒）與硬負樣本（同幀、同類別、閘門範圍內的不同 ID）的外觀距離分布，並列出 val 分布作對照；只接受調參子集，不讀保留序列，不寫任何訓練資料。
- **相容性**：兩個功能都只在類別設定新參數且 `use_reid: true` 時生效；未設定時距離矩陣與現在逐位元相同，既有情境設定與 golden 輸出不變。追蹤器 override 的方法（`__init__`、`update`、`remove_duplicate_stracks`、`init_track`）不改（ADR 0004）。
- **啟用條件收緊**：啟用 Re-ID 除了不變差，還必須有身分層級的改善（IDSW 減少或 IDP 上升）；只有 AssA 或 TP 上升不算。現行 `config/botsort_reid_timm.yaml` 的 car（只有 AssA +0.006）因此不再合格。
- **驗收**：沿用 `train-reid-timm` 的 D5（R1 驗收基準、C000025 調參／保留序列、`dinov2s_v3`），逐類別試少量由校準得出的參數組合；bus（調參序列無 GT）維持關閉。沒有類別通過時，`config/botsort_reid_timm.yaml` 全部類別 `use_reid: false`，並以負面結果收尾。
- **不做**：
  - 不修改 `update()` 等凍結方法，也不新增獨立的 lost-track 重新配對階段。
  - 不改第二階段（低分偵測，只用 IoU）與 ByteTrack。
  - 不重新訓練 Re-ID 模型，不做加權融合（`λ·運動 + (1−λ)·外觀`）。
  - 沒有類別通過時不再做第二輪更寬的調參。

## Capabilities

### New Capabilities

- `reid-fusion`：追蹤器在 class-aware 匹配中如何把外觀距離與運動距離融合，包括遺失中的軌跡的中心距離閘門、外觀否決、參數驗證、門檻校準的資料範圍，以及未設定新參數時的相容行為。

### Modified Capabilities

- `reid-model`：「只在追蹤不變差的類別啟用 Re-ID」改為：不變差是必要條件，另外必須 IDSW 減少或 IDP 上升才啟用；指標都不變或只有 AssA／TP 上升的類別，以及調參序列上沒有 GT 的類別，不啟用。

## Impact

- **程式碼**：`src/safetrajai/tracking/class_aware_botsort.py` 的 `get_dists_class_aware()`（不在凍結範圍）；`src/safetrajai/tracking/matching.py` 新增融合純函式；`src/safetrajai/tracking/config.py` 的參數驗證；新增 `scripts/reid_train/fusion_distance_stats.py` 與單元測試。
- **設定**：`class_tracking_params` 新增選用的逐類別參數；`config/botsort_reid_timm.yaml` 依驗收結果更新（可能變成全部關閉）。其他情境設定不變。
- **資料**：校準工具會讀 C000025 調參序列的影片與 GT 來量測距離，不產生任何訓練資料。
- **文件**：CLAUDE.md 的 Tracker 章節、`src/safetrajai/tracking/class_aware_botsort_flow.md`／`HANDOFF.md`。
- **驗證輸出**：`runs/reid_accept/`（沿用 `train-reid-timm` 的子集與工具）。
