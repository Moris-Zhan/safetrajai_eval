# Verification

## 總結

- 依賴保留期限的兩個衍生量都改以保留期限幀數計算（ADR 0008），並共用 `retention_frames()`，它的換算方式與兩個追蹤器的凍結 `update()` 相同。
  - Re-ID 閘門：`reid_center_gate_px_per_frame` × 遺失幀數。
  - 碎片合併：候選窗為保留期限幀數＋`margin_frames`，外插用每幀速度。
- 舊參數名稱 `reid_center_gate_px_per_s`、`time_margin_s` 出現時會明確報錯。凍結方法與情境 config 都沒有改，golden 不受影響。
- 驗收條件是「候選清單不變」，實際結果**不成立**：有 4 支非 10fps 來源序列，而且時間戳不規則，導致合併紀錄 734 → 732。使用者決定**不重建** `fragment_merges.csv` 與 `v3`，詳見第 2 節。
- 文件已更新：CLAUDE.md、`class_aware_botsort_flow.md`、`scripts/reid_data/README.md`。
- 範圍外、未處理：`safetrajai/reid/fusion_stats.py` 的 `positive_pairs(min_gap_s=1.0)`（校準用的正樣本間隔）仍以秒計算。它屬於 ADR 0008 所列的衍生量類型，但不在本 change 的範圍內，留待之後處理。

## 1. 程式與測試（任務 1.1～1.4）

- `tests/test_retention_frames.py`：`retention_frames()` 對照兩個追蹤器 `update()` 的算式（BoTSORT 3 × 5 → 15；ByteTrack 10 / 30 × 90 → 30）、未知追蹤器與非整數幀率報錯。
- `tests/test_reid_fusion.py`：閘門放寬改為每幀（80 ＋ 8 × 7 幀 → 136 ≥ 120 px，在 5fps 與 10fps 下結果相同；每幀 5 → 115 < 120 px 不取得外觀成本）；`reid_center_gate_px_per_s` 載入時報錯並指出新名稱，全域參數不變。修改前 4 個測試失敗，修改後通過。
- `tests/test_reid_fragment_merge.py`：改以幀號建構合成資料；新增候選窗 40 幀（BoTSORT car 3、ByteTrack car 90，10fps、`margin_frames` 10；第 40 幀在窗內、第 41 幀在窗外）、5fps 序列候選窗 25 幀、序列幀率取中位數（單一不規則時間戳不影響）、每幀速度外插、`load_retention()`、`time_margin_s` 報錯。修改前無法匯入（`load_retention` 不存在），修改後 22 個測試通過；兩支腳本 `--help` 正常。
- 全部 `pytest`：318 passed、8 skipped（既有的「script does not iterate classes」）；tracker override 方法的 AST 檢查通過。

## 2. 碎片合併候選清單比對（任務 2.1，design D4）

輸入與產生 `v3` 時相同：`dataset/reid/crops/_index/*.csv`（最後修改 2026-09-24）、`auto_exclusions.csv`（09-24）、`exclusions.csv`（09-24）、`fragment_features.npz`（09-25）；`v3` 產生於 09-28。只讀索引、排除清單與既有特徵檔，不讀影像。以 `git show HEAD` 取出修改前的 `fragment_merge.py` 與 `fragment_merge.yaml` 作為舊規則。

先確認舊規則＋既有特徵可逐筆重現 `fragment_merges.csv`（734 筆，相同）。

### 結果：候選清單不同

| 項目 | 修改前（秒） | 修改後（幀） |
|---|---|---|
| 軌跡 | 59,310 | 59,310（鍵與裁切順序完全相同） |
| 候選 | 46,144 | 46,090（共同 46,083；只在舊 61、只在新 7） |
| 端點裁切 | 66,421 | 66,370 |
| 合併紀錄 | 734 | 732（可計分的候選；少 2 筆、多 0 筆） |

### 原因

1. **來源序列不都是 10fps**（由索引推得）：39 支序列中 35 支 10fps，另有 C000011（14fps）、C000012（15fps）、C000043（9fps）、C000056（14fps）。
   - 在 14／15fps 序列，`margin_frames: 10` 只相當於 0.67～0.71 秒，候選窗變成 3.67～3.71 秒（原本 4 秒）。
   - 只在舊規則的 61 組中有 50 組來自這些序列（14fps 33 組、15fps 17 組），都是間隔 3.75～4.0 秒的配對。
2. **10fps 序列的時間戳不完全等於幀號 / 10**（例如第 41 幀間隔只有 3.999 秒）。
   - 只在舊規則的 11 組：間隔 41 幀但不到 4 秒，或外插距離比恰在 2.0 附近（例如 1.9997），差異來自每秒 vs. 每幀速度的數值差。
   - 只在新規則的 7 組：原因相同但方向相反——間隔 40 幀但超過 4 秒（例如 4.199 秒），或舊比值略大於 2.0（2.0031、2.1032）；另有 1 組 scooter 兩端時間戳相同（舊間隔 0 秒被排除，新間隔 1 幀）。
3. `distance_ratio` 在共同候選上的最大差值為 0.19（時間戳不規則造成的每秒與每幀速度差）。

### 對合併清單的影響

- 新規則少了 2 筆合併紀錄，都在非 10fps 序列：`car` C000012（track6591 → track6559）、`truck` C000056（track7450 → track7348）。沒有新增的合併。
- 只在新規則的 7 組候選中，6 組相似度都低於門檻 0.911305；剩下 1 組（`truck` C000085 track106 → track149）的端點特徵不在既有特徵檔中，要重跑 `extract_fragment_features.py`（只補 3 張端點裁切）才能確定是否會合併。

### 結論

依 D4，候選清單不同，不自動重建，回報使用者決定。

**使用者決定（2026-09-30）：不重建。** `fragment_merges.csv` 與 `v3` 維持以修改前的秒數規則產生（差異為 734 筆中的 2 筆，都在非 10fps 序列）；下次重建合併清單或資料集版本時才套用幀數規則，屆時合併清單預期會少上述 2 筆，C000085 truck track106 → track149 需先補抽端點特徵再判定。
