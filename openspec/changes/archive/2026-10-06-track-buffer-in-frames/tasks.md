# Tasks

## 1. 凍結方法的例外與 AST 檢查

- [x] 1.1 在 `tests/test_smoke.py` 加入明列的例外表與 BASE 版 AST 替換（design D2），並新增「例外表每項恰好替換 1 次」與「例外以外的改動仍被拒絕」的測試（以暫存的竄改原始碼驗證）；確認修改追蹤器前 `test_tracker_override_bodies_unchanged` 失敗（目前檔案仍是舊敘述）
- [x] 1.2 只改兩個 `update()` 的保留期限敘述（D1）：BoTSORT 為 `int(cls_buffer_sec)`、ByteTrack 為 `int(cls_buffer)`，不動任何註解與其他敘述；以 `git diff` 確認兩個檔案各只有 1 行變更，且 1.1 的測試通過
- [x] 1.3 新增單元測試：同一類別 `track_buffer: 15` 時，BoTSORT 與 ByteTrack 在 5fps 與 10fps 下都在遺失第 16 幀移除、第 15 幀仍保留（spec「兩個追蹤器的保留期限相同」「保留期限不隨影片幀率變化」），測試通過

## 2. 設定檢查與 config 換算

- [x] 2.1 先寫測試：缺少 `track_buffer`、`0`、負數、`2.5`、`90.0`、`true` 都在載入時報錯，訊息含類別與 `track_buffer`，且全域參數不變；確認失敗後在 `load_track_config()` 實作 D3 檢查，測試通過
- [x] 2.2 依 design D4 換算 `config/*.yaml` 的所有追蹤情境（含補上 `bytetrack`、`botsort_reid`、`grid_search_bytetrack` 每類別 120），在 `track_buffer` 旁註明單位（幀）與換算來源、移除「秒」的註解；以一支 scratchpad 腳本對每個 config 比對「修改前依舊算式與該 config 的 frame_rate 算出的幀數」與「修改後的值」，除 `reid_crops.yaml` 的非 10fps 序列外全部相同
- [x] 2.3 同樣換算 `tests/golden/configs/*.yaml`，以 2.2 的腳本確認全部等價
- [x] 2.4 跑 `pytest`，確認全部 config 載入成功（spec「所有情境設定都通過檢查」）且全部通過

## 3. 衍生量

- [x] 3.1 先改 `tests/test_reid_fragment_merge.py`：候選窗為 `track_buffer + margin_frames`（spec「遮擋後的碎片被合併」：30 + 10 → 40）、兩種追蹤器設定的候選窗相同（「候選窗依追蹤器換算」）、10fps 與 14fps 序列的候選窗相同（「候選窗不隨序列幀率變化」）、`load_retention()` 只回傳逐類別幀數；移除依追蹤器換算與序列幀率推算的測試；確認失敗後修改 `fragment_merge.py`（移除 `retention_frames` 的使用、`tracker` 參數、`sequence_fps()` 與 `Track.fps`）與兩支 `scripts/reid_data/` 腳本（D5），測試通過且兩支腳本 `--help` 正常
- [x] 3.2 移除 `retention_frames()` 與 `tests/test_retention_frames.py`；`grep` 確認 `src/`、`scripts/`、`tests/` 不再引用 `retention_frames`
- [x] 3.3 跑 `pytest`，確認全部通過（含 AST 檢查）

## 4. 文件

- [x] 4.1 新增 `docs/adr/0009`（D6），ADR 0008 開頭標示已由 0009 取代，ADR 0004 加入兩個例外敘述並連到 0009；確認三份 ADR 互相引用正確
- [x] 4.2 更新 CLAUDE.md、CONTEXT.md（只改「保留期限」定義與換算說明兩處）、`class_aware_botsort_flow.md`、`class_aware_bytetrack_flow.md`、`HANDOFF.md`、`scripts/reid_data/README.md`、`config/fragment_merge.yaml` 的註解；`grep` 確認這些文件不再把 `track_buffer` 描述為秒或 30fps 基準

## 5. 驗收

- [x] 5.1 向使用者確認後跑 `tests/golden/run_all.sh <run>`（約 16 分鐘），以 `python tests/golden/compare.py reid1 <run>` 比對，所有項目差異為 0（spec「golden 比對不變」）；結果寫入 `verification.md`
- [x] 5.2 撰寫 `verification.md` 總結（含 2.2 的換算比對表與 `reid_crops.yaml` 的已知差異）；跑 `pytest` 與 `openspec validate track-buffer-in-frames --strict` 確認通過
