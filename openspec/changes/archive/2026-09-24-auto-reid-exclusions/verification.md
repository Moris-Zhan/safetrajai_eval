# 驗證紀錄：auto-reid-exclusions

## 總結（任務 3.2，2026-09-24）

| 項目 | 結果 | 詳見 |
|---|---|---|
| 自動排除規則 | 靜止、運動跳動整段；四輪與非四輪的類別不一致整段；外觀離群單張；人與二輪之間不排除 | design D1、D2 |
| pilot 校準 | 8／8 個整段排除的身分、84／88 張人工排除（95.5%）被涵蓋 | 一 |
| scooter 被判為 person | 30 個樣本中需要整段排除 1 個（< 3），維持不自動排除 | 三 |
| 全量自動排除 | 4,374 列：整段 1,847 個身分、單張 2,527 張，共 22,748 張（10.1%） | 五 |
| 人工排除 | 80 列：pilot 53＋scooter 確認 3＋全量抽查 24 | 四、六 |
| 抽查 | 223 個身分（鏡頭 175、bus 20、person 28）；不合格率最高 C000085 10.0%、C000054 1.7%，都不超過門檻 | 六、七 |
| `v2` | 22,164 個身分、156,141 張（train 19,043／val 3,121 個身分）；驗收全部通過、可重現、`v1` 未變動 | 八 |
| 人工工作量 | 約 250 個身分（scooter 確認 30＋抽查 223），原本全量審查約 3,000 個 | — |

**後續事項**

1. **遮擋碎片合併**（另開 change）：C000012、C000048、C000056 等鏡頭的碎片仍各自是獨立身分。
2. **資料量**：person 只有 28 個身分、small_truck 364 個；要不要合併類別或補充來源，由 `train-reid-timm` 決定。
3. **規則維護**：之後增加來源影片或調整標記時，重跑 `apply_auto_exclusions.py` 與抽查即可；抽查不合格時依 design D5 處理，不改回全量人工審查（`docs/adr/0007`）。

## 一、pilot 校準（任務 1.6，design D1）

以 `safetrajai.reid.auto_exclusion.build_auto_exclusions()` 對 pilot 資料（`runs/rebuild-reid-dataset/pilot_final/` 的索引與標記，最少張數 4）產生自動排除，與使用者的 6 份人工確認清單（`review_results/`）比對：

| 項目 | 結果 | 驗收條件 |
|---|---|---|
| 自動排除列數 | 100（整段 33 個身分、單張 67 張） | — |
| 人工整段排除的身分 | 8／8 被涵蓋 | 全部涵蓋 ✅ |
| 人工排除的裁切 | 84／88 被涵蓋（95.5%） | ≥ 90% ✅ |
| 自動排除移除的裁切 | 418／5,035（8.3%） | — |
| 排除後仍 ≥ 4 張的身分 | 485／519 | — |

- 各規則的列數：外觀離群（單張）67、運動跳動 21、靜止 9、靜止＋類別不一致 3。
- 數字與 design D1 的分析一致；漏掉的 4 張分布在 3 個未被標記的身分中，由抽查處理。

## 二、單元測試（任務 1.7）

`pytest`：194 passed、6 skipped。新增 `tests/test_reid_auto_exclusion.py`（8 個）、`tests/test_reid_audit_summary.py`（3 個），`tests/test_reid_review.py`、`tests/test_reid_manifest.py` 各新增抽查與兩份排除清單的測試。smoke 測試會把內容含 `CLASSES` 字串的 `scripts/reid_data/*.py` 當成逐類別迭代的腳本，`summarize_audit.py` 的常數因此命名為 `PER_CLASS_SCOPES`。

## 三、scooter 被判為 person 的抽樣確認（任務 2.1，design D2）

全量 scooter 身分中，裁切判定類別過半為 person、且被類別一致性標記的 ≥ 4 張身分共 886 個；以固定種子 0 抽 30 個（C000054 27 個，C000012、C000033、C000256 各 1 個），審查頁與樣本清單在 `runs/auto-reid-exclusions/scooter_person_check/`（`sample.csv` 含各身分被判為 person 的比例）。

- 使用者確認（`results/review_exclusions.csv`）：需要整段排除 **1 個**（C000033 track13414），另有 2 個身分各排除 1 張單張裁切。
- 需要整段排除的身分 1／30 < 3，**維持 D2**：人與二輪之間的類別不一致不自動排除。以 3/n 規則估計，這 886 個身分中需要整段排除的比例 95% 上限約 10%。
- 這 3 筆人工結果與 pilot 的確認清單一起併入人工排除清單（任務 2.2）。

## 四、併入人工排除清單（任務 2.2）

以 `apply_review_exclusions.py`（先 dry-run、再 `--apply`）依序併入 pilot 的 6 份確認清單與任務 2.1 的確認清單，建立 `dataset/reid/exclusions.csv`（原本不存在，無需備份）。

- 7 份清單去重後共 56 列（整段排除 9 個身分：pilot 8＋2.1 的 1；單張 47 張），寫入後清單列數為 56，一致。
- `dataset/reid/crops/` 的檔案數寫入前後都是 288,625（288,547 張裁切＋78 份索引與略過統計），未變動。

## 五、全量自動排除清單（任務 2.3）

`python scripts/reid_data/apply_auto_exclusions.py --apply` 產生 `dataset/reid/auto_exclusions.csv`（4,374 列、421 KB）。

| 規則 | 整段排除的身分 | 單張 | 涉及裁切 |
|---|---:|---:|---:|
| static | 575 | 0 | 9,491 |
| motion_jump | 1,129 | 0 | 9,956 |
| class_group | 330 | 0 | 2,479 |
| appearance_outlier | 0 | 2,527 | 2,527 |
| **合計**（規則重疊只算一次） | **1,847** | **2,527** | **22,748**（≥ 4 張身分共 225,011 張的 10.1%） |

- 與 design D1 的估計一致：整段 1,847 個身分、共移除 22,748 張。單張為 2,527 張，比估計的 2,875 張少 348 張，因為落在整段排除身分中的離群裁切已由整段排除涵蓋，不另列一列。
- 連續兩次 `--apply` 的輸出逐位元組相同；`exclusions.csv` 寫入前後完全相同；`dataset/reid/crops/` 的檔案數前後都是 288,625。

## 六、全量抽查（任務 2.4）

`generate_identity_review.py --audit-only --seed 0` 在 `dataset/reports/reid/review/` 產生 223 個身分的審查頁與 `audit_sample.csv`：C000054 60 個（car 30、scooter 23、truck 4、small_truck 3，依類別比例分層）、C000011／012／033／043／048／056／256 各 15 個、C000085 10 個（不足 15 個，全數）、bus 20 個、person 28 個（全數）。

- 使用者匯出 11 份確認清單到 `runs/auto-reid-exclusions/audit_results/`（依頁面命名），以 `apply_review_exclusions.py` 併入 `exclusions.csv`：新增 24 列（整段 4 個身分、單張 20 張），清單由 56 列增為 80 列；併入前的清單備份在 `runs/auto-reid-exclusions/exclusions_before_audit.csv`；裁切檔案數前後都是 288,625。
- bus 抽查與 person 審查都沒有需要排除的項目。
- 整段排除的 4 個身分：C000048 car track1324、C000085 car track477、C000256 car track2000、C000054 truck track429。

`summarize_audit.py` 的不合格率（只計整段排除）：

| 範圍 | 不合格／抽查 | 不合格率 |
|---|---:|---:|
| C000054（主要鏡頭）整體 | 1／60 | 1.7% |
| C000054 car | 0／30 | 0.0% |
| C000054 scooter | 0／23 | 0.0% |
| C000011、C000012、C000033、C000043、C000056 | 0／15 | 0.0% |
| C000048、C000256 | 1／15 | 6.7% |
| C000085 | 1／10 | 10.0% |

## 七、不合格處理（任務 2.5）

沒有任何範圍的不合格率超過 10%（C000085 等於 10%，門檻為「超過」），**無需處理**：不調整規則、不重新抽查、不排除任何鏡頭。以 3/n 規則估計，C000054 整段問題的殘留比例 95% 上限約 5%。

## 八、`v2` manifest（任務 2.6）

`python scripts/reid_data/build_reid_split.py --version v2`（兩份排除清單都套用，沒有被排除的鏡頭）。

| 項目 | `v1` | `v2` |
|---|---:|---:|
| 身分 | 24,056 | 22,164 |
| 裁切 | 174,344 | 156,141 |
| train 身分／裁切 | 20,795／149,903 | 19,043／133,081 |
| val 身分／裁切 | 3,261／24,441 | 3,121／23,060 |

- `v2` 各類別身分數：car 12,174、scooter 8,617、truck 797、small_truck 364、bus 184、person 28。
- `summary.json`：`exclusions` 為 auto 4,374、manual 80；`excluded_cameras` 為空。
- 驗收：評估鏡頭的列數為 0；每個身分 4～30 張；C000054 的 0629 全在 val，val 中沒有其他日期或鏡頭；沒有身分同時出現在 train 與 val；manifest 中沒有任何被整段排除的身分（0），也沒有任何被排除的單張（0）；以相同輸入在暫存目錄重建的 manifest 逐位元組相同；`splits/` 內沒有任何影像檔。
- `v1` 的 `manifest.csv`、`summary.json`、`split_rules.yaml` 大小與修改時間都未改變（只有上層 `splits/` 目錄因新增 `v2` 而更新時間）；`dataset/reid/crops/` 的檔案數仍為 288,625。
