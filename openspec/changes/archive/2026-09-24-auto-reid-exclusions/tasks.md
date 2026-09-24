# Tasks

## 1. 程式修改

- [x] 1.1 新增 `src/safetrajai/reid/auto_exclusion.py`：依 D1、D2 的規則（靜止整段、運動跳動整段、外觀離群單張、四輪與非四輪之間的類別不一致整段；人與二輪之間不排除；只處理 ≥ 最少張數的身分；多規則合併為一列、`rule` 以 `+` 串接；依鍵排序）由標記產生排除紀錄；以單元測試涵蓋 spec「自動排除以既有標記產生排除紀錄」的 6 個情境作為驗證
- [x] 1.2 新增 `scripts/reid_data/apply_auto_exclusions.py`：讀 `dataset/reports/reid/` 的標記，預設 dry-run 印出各規則的身分數與裁切數，`--apply` 才完整覆寫 `dataset/reid/auto_exclusions.csv`；以測試用資料驗證連續兩次產生的檔案逐位元組相同、`exclusions.csv` 與裁切檔案清單不變，且 `--help` 正常
- [x] 1.3 `build_reid_split.py` 新增 `--auto-exclusions`（預設 `dataset/reid/auto_exclusions.csv`，可不存在）與 `--exclude-cameras`，兩份清單串接後交給 `build_manifest()`，`summary.json` 增加兩份清單的排除筆數與被排除的鏡頭；以單元測試涵蓋「兩份排除清單都套用」、「小鏡頭抽查不合格被排除在版本外」與「可重現」情境作為驗證
- [x] 1.4 `generate_identity_review.py` 新增 `--audit-only`（D5）：排除已被整段自動排除的身分、縮圖不含已排除的單張；身分數 ≥ 5% 的鏡頭抽 60 個並依類別比例分層（每類至少 2 個）、其他鏡頭各抽 15 個（不足全數）、bus 另抽 20 個、person 全數；固定種子、同一身分只出現一次；已在人工排除清單中審過的身分標示「已人工審查」；另輸出 `audit_sample.csv`（身分、鏡頭、類別、用途）；以單元測試涵蓋 spec「污染標記與抽查審查只產生紀錄」的抽查相關情境（已自動排除不進審查、主要鏡頭分層、小鏡頭全數、bus 抽查、person 全數、可重現）作為驗證
- [x] 1.5 新增 `scripts/reid_data/summarize_audit.py`：讀 `audit_sample.csv` 與使用者的確認清單，計算每支鏡頭、主要鏡頭的 car 與 scooter 的不合格率（只計整段排除；bus 抽查與 person 審查不計），列出超過 10% 的範圍；以單元測試涵蓋「單張問題不計入不合格」與超過門檻的判定作為驗證
- [x] 1.6 以 pilot 資料（`runs/rebuild-reid-dataset/pilot_final/`）重現 D1：以新模組產生 pilot 的自動排除，與 6 份人工確認清單比對，須涵蓋全部 8 個整段排除的身分與 ≥ 90% 的人工排除裁切；數字寫入本 change 的 `verification.md`
- [x] 1.7 跑 `pytest` 確認全部通過

## 2. 全量資料

- [x] 2.1 D2 的一次性確認：從全量 scooter 被判為 person（類別不一致標記）的身分中，以固定種子抽 30 個產生審查頁給使用者目視；需要整段排除 < 3 個時維持 D2；≥ 3 個時依 D2 以樣本訂出門檻（分不出界線則全部排除），先更新 spec、design 再修改規則並重跑 1.1 的測試；結果寫入 `verification.md`
- [x] 2.2 以 `apply_review_exclusions.py` 將 pilot 的 6 份確認清單併入 `dataset/reid/exclusions.csv`（先 dry-run，再 `--apply`）；以清單列數等於 6 份清單去重後的列數、裁切檔案清單不變作為驗證
- [x] 2.3 以 `apply_auto_exclusions.py --apply` 產生全量 `dataset/reid/auto_exclusions.csv`；以各規則的身分數、裁切數與 design D1 的估計一致（整段約 1,847 個身分、單張約 2,875 張，若 2.1 修改了規則則另行說明差異）、裁切檔案清單不變作為驗證，數字寫入 `verification.md`
- [x] 2.4 以 `--audit-only` 產生全量審查頁（C000054 60 個、其他 8 支鏡頭各 15 個或全數、bus 20 個、person 全數，約 230 個），以 localhost 網頁服務交給使用者審查；確認清單併入 `exclusions.csv`；以 `summarize_audit.py` 計算不合格率，結果寫入 `verification.md`
- [x] 2.5 依 D5 處理不合格：主要鏡頭（或其 car、scooter）> 10% 時，回報使用者並調整一輪規則、以新種子重新抽查同一範圍，仍 > 10% 則將該範圍的被標記身分全數送人工審查；其他鏡頭 > 10% 時列入 `--exclude-cameras`；全部合格時本任務記錄「無需處理」；處理過程與結果寫入 `verification.md`
- [x] 2.6 產生 `v2` manifest（含 2.5 決定的 `--exclude-cameras`）；以 `summary.json` 顯示兩份清單的排除筆數與被排除的鏡頭、評估鏡頭為 0、每身分 4～30 張、C000054 的 0629 全在 val，manifest 不含任何被整段排除的身分與被排除鏡頭的列、`v1` 目錄內容未變、裁切檔案未移動作為驗證

## 3. 文件與收尾

- [x] 3.1 更新 `scripts/reid_data/README.md`（自動排除、抽查與不合格處理的流程，`v2` 資料量）與 CLAUDE.md 的 Re-ID 章節、常用指令；確認用語與 `CONTEXT.md` 的新術語一致（標記、自動排除、抽查、不合格率、誤偵測身分、殘缺裁切），`docs/adr/0007` 已存在
- [x] 3.2 撰寫 `verification.md` 的總結（規則、pilot 校準、scooter／person 確認、抽查與不合格處理、`v2` 摘要）；跑 `pytest` 與 `openspec validate --strict` 確認通過
