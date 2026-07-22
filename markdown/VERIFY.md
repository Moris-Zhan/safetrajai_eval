# VERIFY.md

紀錄已經做過的實驗驗證與結論，避免重複測試或誤改已驗證過的設定。每筆結論都附上資料來源，之後有新結論請照同樣格式補充。

## ReID 開關（person vs. scooter）
- **結論**：開啟 ReID 對 person 的追蹤有幫助，但會損害 scooter 追蹤並增加 IDSW（ID switch）。
- **目前設定**：`trackers/__init__.py` 的 `DEFAULT_TRACKER_CONFIG['with_reid']` 預設 `False`。
- **來源**：`trackers/__init__.py` 程式碼註解；細節推論見 `trackers/HANDOFF.md` 需求 4。
- **待補充**：實際數據（各類別 HOTA/IDSW 變化量）目前只存在於程式碼註解描述，沒有對應的 Excel/CSV 可查— 需要的話應重新用 `tracker_grid_search_botsort_reid.py` 針對 person / scooter 各跑一次 ReID on/off 對照並存檔。

## GMC（Generalized Motion Compensation）方法比較 — BoTSORT + NoReID
- **結論**：以全部 6 類別 × 6 序列的 `mean_AssA` 平均比較，`sift` 排名最佳，`ecc`、`none`、`sparseOptFlow`、`orb` 依序在後，但彼此差距極小（第三位小數才有差異），**實務上五種 GMC 方法表現接近，非決定性差異**。
- **來源**：`botsort_noreid/botsort_noreid_gmc_comparison_by_class_and_sequence.md`（逐類別、逐序列明細）、對應的 `.xlsx`、`gmc_*.txt`（各方法原始結果）。
- **目前設定**：`config.yaml` 的 `tracking_params.gmc_method: ecc`。

## FPS / 推論解析度對 ByteTrack 準確度的影響
- **來源**：`fps_compare/fps_bytetrack_comparison_by_class_and_sequence.md`、`fps_bytetrack_comparison_by_sequence.md`（imgsz=640，比較 10/20/30 fps）、對應 `.xlsx`。
- **結論**：待補充 — 檔案內已有逐序列逐類別數據，尚未整理出整體結論摘要。

## 追蹤參數查表機制（class-aware matching）
- **結論**：確認「逐類別分組匹配」相較原版 Ultralytics 全域統一匹配，可解決跨類別誤配問題（如 person 配到 scooter track）。此為架構性修改，已落地於 `trackers/class_aware_botsort.py` / `class_aware_bytetrack.py`。
- **來源**：`trackers/HANDOFF.md`（完整原理與程式碼對照表）。
- **待補充**：沒有找到「改前 vs 改後」的量化比較數據（例如跨類別誤配次數），若需要作為佐證應另外統計。

## 待補充
> 以下項目目前沒有找到對應的驗證紀錄，若已經做過請補上來源；若還沒做，建議列進 `NEED.md`：
> - `config.yaml` 的 `det_corner` 偵測區域多邊形設定對指標的影響
> - `fuse_score` 開關（目前全域關閉）對各類別的影響
> - `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據
