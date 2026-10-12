# Tasks

## 1. 量測與關卡規則

- [x] 1.1 先寫測試：`tests/test_reid_fusion_stats.py` 加入 spec「沒有身分的框不參與量測」（id -1 的框不出現在任何配對、跳過數正確）與「同一組物體的多幀配對只算一組」（41 對 → 2 組；使用退回規則時為 0）；確認修改前失敗，再修改 `fusion_stats.py`（D1、D2 的物體配對數）與 `fusion_distance_stats.py`（輸出 `skipped_unlabeled_boxes`），確認通過
- [x] 1.2 先改測試：關卡情境改以物體配對計算——「物體配對不足的類別不列入下降檢查」（small_truck 2 組、下降 0.03 → 通過，回報列在 `unchecked_classes`）、「物體配對足夠的類別下降過多時不通過」（person 20 組、下降 0.03）、「car 或 scooter 證據不足時不通過」（scooter 6 組）；既有關卡測試改用物體配對欄位後仍通過；確認修改前失敗，再修改 `separability_gate`（D2），確認通過
- [x] 1.3 跑 `pytest`，確認全部通過且 tracker override 方法的 AST 檢查仍通過；更新 CLAUDE.md Re-ID 章節的關卡規則描述（物體配對、id < 0）

## 2. 重新量測與判定

- [x] 2.1 以新程式重跑 6 份量測到 `runs/reid_domain/v2/`（D3）；確認參考的物體配對數：car 754、scooter 356（探索時含 id -1 的 367）、small_truck 2、person 20，跳過的框為 scooter 2、small_truck 2；結果寫入 `verification.md`
- [x] 2.2 以 `seed1` 與 `dinov2s_v3`（緊框）重算雜訊底線，對 4 個候選判定關卡並以 `select_best` 選出最佳者（或記錄沒有通過）；並列舊判定（`align-reid-training-domain`）與新判定、各類別物體配對數與 `unchecked_classes`；結果寫入 `verification.md`

## 3. 追蹤驗收工具（只在 2.2 有候選通過時）

- [x] 3.1 先寫測試：逐類別判定的純函式涵蓋 `reid-model` spec 的情境（IDSW 增加不通過、只有 AssA 上升不通過、IDSW 減少且其他不下降通過、無 GT 不通過）與彙總方式（IDP 由 IDTP／IDFP 加總計算、AssA 依 GT 框數加權）；再新增 `scripts/reid_train/acceptance_grid.py` 的 `configs`／`summarize`（D4），以單元測試驗證 `--set` 套用、無效參數報錯；`pytest`（含 `scripts/**/*.py --help` smoke）通過；沒有候選通過時記錄跳過
- [x] 3.2 以 `acceptance_grid.py` 產生基準與 1 組候選設定，在調參序列冒煙執行，確認基準數字可重現並量測每幀追蹤時間；估算 40 組全量時間後回報使用者，再進行 4.1；沒有候選通過時記錄跳過

## 4. 追蹤驗收（只在 2.2 有候選通過時）

- [x] 4.1 以最佳候選重跑 `fusion_distance_stats.py` 取得候選門檻，執行 D5 的 40 組網格與同批基準，逐類別依 `reid-model` spec 判定並記錄每幀追蹤時間；結果寫入 `verification.md`；沒有候選通過時記錄跳過
- [x] 4.2 有類別通過時做合併驗證與保留序列，更新 `config/botsort_reid_timm.yaml`（啟用類別、模型路徑、參數與檔頭說明）；沒有類別通過時維持 6 類關閉、只更新檔頭；以 smoke 的 config 檢查通過、`config/botsort_reid.yaml` 與 golden configs 未改變作為驗證；結果寫入 `verification.md`

## 5. 收尾

- [x] 5.1 更新 CLAUDE.md 的 Re-ID 章節與 `src/safetrajai/reid/HANDOFF.md`（驗收工具、判定結果、`botsort_reid_timm.yaml` 現況）；撰寫 `verification.md` 的總結；跑 `pytest` 與 `openspec validate separability-gate-object-pairs --strict` 確認通過
