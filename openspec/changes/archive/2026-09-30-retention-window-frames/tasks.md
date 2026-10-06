# Tasks

## 1. 程式修改

- [x] 1.1 新增 `retention_frames()`（D1）；以單元測試對照兩個追蹤器的算式：BoTSORT `track_buffer` 3、幀率 5 → 15；ByteTrack `track_buffer` 90、幀率 10 → 30；未知追蹤器報錯
- [x] 1.2 先改測試：`tests/test_reid_fusion.py` 的閘門放寬測試改為 `reid_center_gate_px_per_frame`（spec「閘門隨遺失時間放寬」：80 ＋ 8 × 7 幀 → 136），並新增舊名稱 `reid_center_gate_px_per_s` 被拒絕的測試；確認修改前失敗，再修改 `_gated_appearance()` 與 `config.py`（D2），確認通過
- [x] 1.3 先改測試：`tests/test_reid_fragment_merge.py` 改以幀號建構合成資料，涵蓋 spec「遮擋後的碎片被合併」（10fps、BoTSORT car 3、`margin_frames` 10 → 候選窗 40 幀）與「候選窗依追蹤器換算」（ByteTrack car 90 → 40 幀），以及序列幀率推算、每幀外插；確認修改前失敗，再修改 `fragment_merge.py`、`config/fragment_merge.yaml` 與兩支 `scripts/reid_data/` 腳本（D3），確認通過且 `--help` 正常
- [x] 1.4 跑 `pytest`，確認全部通過且 tracker override 方法的 AST 檢查仍通過

## 2. 驗收

- [x] 2.1 依 D4 以產生 `v3` 時的索引與排除清單比對修改前後的碎片合併候選清單；相同則記錄「`fragment_merges.csv` 與 `v3` 不需重建」，不同則列出差異並回報使用者；結果寫入 `verification.md`

## 3. 文件與收尾

- [x] 3.1 更新 CLAUDE.md（`track_buffer` 的說明改為「設定值原單位、衍生量以保留期限幀數計算（ADR 0008）」、碎片合併的描述）、`src/safetrajai/tracking/class_aware_botsort_flow.md` 的閘門參數名稱、`scripts/reid_data/README.md` 中碎片合併的時間條件
- [x] 3.2 撰寫 `verification.md` 的總結；跑 `pytest` 與 `openspec validate retention-window-frames --strict` 確認通過
