# Tasks

## 1. 程式修改

- [x] 1.1 先寫測試：以隨機的 tracks／detections（含遺失中的軌跡）固定目前 `get_dists_class_aware()` 的輸出，作為「未設定新參數時逐位元不變」的回歸測試；在修改前跑一次確認通過
- [x] 1.2 `matching.py` 新增融合純函式（D1）；以合成矩陣的單元測試涵蓋 spec 的情境：遺失中軌跡的列 IoU 為 0 但在閘門內可取得外觀成本、閘門外只用運動距離、IoU 重疊但在閘門外仍可取得（聯集）、閘門放寬、非遺失中軌跡的列不受閘門影響、外觀差異大不給外觀成本、否決設為 1、未設否決不影響
- [x] 1.3 `get_dists_class_aware()` 接上新函式（D1、D2）：`use_reid` 為 false 或新參數都沒設定時原樣走現有程式碼；以 1.1 的回歸測試與「`use_reid: false` 但設定閘門」的測試驗證不變，並以實際的 `BOTrack` 驗證：Tracked 時記錄 `_reid_last_center`、遺失時取預測中心與最後位置的較近者、沒有記錄值時只用預測中心、遺失秒數與閘門半徑的計算
- [x] 1.4 `load_track_config()` 驗證新參數（D3）；以單元測試涵蓋 `appearance_veto_sim: 1.5`、負的 `reid_center_gate_px_per_s`、`reid_center_gate_px: 0`，錯誤訊息含類別與參數名稱
- [x] 1.5 新增 `scripts/reid_train/fusion_distance_stats.py`（D4）；以合成的 MOT GT 單元測試驗證正樣本（同 ID、間隔 ≥ 1 秒）與硬負樣本（同幀、同類別、2×`norm_factor` 內）的配對規則、少於 30 對時的退回與標示、保留序列路徑被拒絕；`--help` 正常
- [x] 1.6 跑 `pytest`，確認全部通過且 tracker override 方法的 AST 檢查仍通過

## 2. 候選門檻

- [x] 2.1 以 `fusion_distance_stats.py` 對調參序列（`dinov2s_v3`）與 `v3` val 量測各類別的外觀距離百分位數，依 D4 定出寬／嚴 `appearance_thresh` 與兩個否決門檻（標示使用退回規則的類別）；結果寫入 `verification.md`

## 3. 追蹤驗收（調參序列，`dinov2s_v3`）

- [x] 3.1 擴充 `train-reid-timm` 的驗收 config 產生器以支援新參數；重跑驗收基準（與候選同批），確認基準數字與 `train-reid-timm` 相同（調參 IDSW 6、保留 IDSW 3）；結果寫入 `verification.md`
- [x] 3.2 D5 的 A 閘門（20 組）與 C 否決（10 組）在調參序列上平行執行，逐類別依 `reid-model` spec 判定（不變差，且 IDSW 減少或 IDP 上升）並記錄每幀追蹤時間；結果寫入 `verification.md`
- [x] 3.3 D5 的 B 放寬（A 有通過的類別各 1 組）與 D 合併（同類別閘門＋否決、所有啟用類別一起開），確認合併後調參序列整體不變差；沒有類別通過時跳過合併並記錄；結果寫入 `verification.md`
- [x] 3.4 在保留序列上跑基準與最終設定，依 spec 只否決整體；更新 `config/botsort_reid_timm.yaml`（啟用類別的新參數與 `use_reid`；沒有類別通過時全部 `use_reid: false`，檔頭註明原因），以 smoke 測試的 config 載入檢查通過、`config/botsort_reid.yaml` 與 golden configs 未改變作為驗證；結果寫入 `verification.md`

## 4. 相容性與文件

- [x] 4.1 告知使用者後跑 `tests/golden/run_all.sh <run>`（約 16 分鐘），以 `python tests/golden/compare.py reid1 <run>` 確認差異為 0；結果寫入 `verification.md`
- [x] 4.2 更新 CLAUDE.md 的 Tracker 章節（新參數、融合方式、啟用條件、驗收結果）與常用指令（`fusion_distance_stats.py`）、`src/safetrajai/tracking/class_aware_botsort_flow.md` 與 `HANDOFF.md` 中 `get_dists_class_aware()` 的說明
- [x] 4.3 撰寫 `verification.md` 的總結；跑 `pytest` 與 `openspec validate reid-center-gate-fusion --strict` 確認通過
