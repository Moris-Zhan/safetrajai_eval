# Tasks

## 1. 程式修改

- [x] 1.1 新增 `src/safetrajai/reid/fragment_merge.py`：D1 的候選產生（時間、外插位置、排除整段的身分不參與）、D3 的一對一貪婪配對與 union-find 串接、合併清單輸出（依鍵排序）；以單元測試涵蓋 spec 新增需求的 6 個情境作為驗證
- [x] 1.2 新增 `config/fragment_merge.yaml`（位置倍數 2、時間餘裕 1 秒、端點張數 3、外觀門檻待校準）與 `scripts/reid_data/extract_fragment_features.py`（只讀候選端點裁切、DINOv2 特徵存成 `fragment_features.npz`）；以測試用小型裁切集驗證只讀取端點裁切、裁切目錄檔案清單不變、`--help` 正常
- [x] 1.3 新增 `scripts/reid_data/build_fragment_merges.py`：讀索引、排除清單、特徵與設定，預設 dry-run 印出各鏡頭的候選數與合併數，`--apply` 才寫 `dataset/reid/fragment_merges.csv`；另提供 `--sample 80 --seed 0` 依 D4 分層抽樣產生並排確認頁；以測試驗證輸出可重現、抽樣可重現且分層正確
- [x] 1.4 `manifest.build_manifest()` 支援合併對照、`build_reid_split.py` 新增 `--merges` 並在 `summary.json` 記錄合併數；以單元測試涵蓋「合併後的碎片共同計算張數」、「排除先於合併」與「可重現」情境作為驗證
- [x] 1.5 跑 `pytest` 確認全部通過

## 2. 全量資料

- [x] 2.1 以 `extract_fragment_features.py` 抽取全量候選端點特徵；記錄候選對數、讀取的裁切數與耗時，確認裁切檔案數不變，寫入 `verification.md`
- [x] 2.2 產生 80 組候選確認頁（依相似度分層、中高兩層約 60 組、非 C000054 至少 10 組），以 localhost 網頁服務交給使用者確認；依 D4 計算門檻與門檻以上的準確率；達不到 95% 時回報使用者並暫停；結果寫入 `verification.md` 與 `config/fragment_merge.yaml`
- [x] 2.3 以校準後的門檻產生全量 `dataset/reid/fragment_merges.csv`；以各鏡頭的合併鏈數、被合併的原始身分數、連續兩次輸出逐位元組相同、索引與排除清單與裁切檔案都未改變作為驗證，寫入 `verification.md`
- [x] 2.4 產生 `v3` manifest（`--merges dataset/reid/fragment_merges.csv`）；以 `summary.json` 的合併數、評估鏡頭為 0、每身分 4～30 張、C000054 的 0629 全在 val、沒有身分同時在 train 與 val、可重現、`v2` 未改變作為驗證；比較 `v2` 與 `v3` 各類別的身分數與張數，寫入 `verification.md`

## 3. 文件與收尾

- [x] 3.1 更新 `scripts/reid_data/README.md`（合併流程與 `v3`）與 CLAUDE.md 的 Re-ID 章節、常用指令；確認用語與 `CONTEXT.md` 已更新的「身分」「碎片」「碎片合併」一致
- [x] 3.2 撰寫 `verification.md` 的總結；跑 `pytest` 與 `openspec validate --strict` 確認通過
