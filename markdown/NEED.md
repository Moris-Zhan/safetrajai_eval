# NEED.md

待辦與未滿足需求清單。以下項目是整理程式碼時發現的落差，非正式規劃，請專案負責人確認優先順序與是否要處理。

## 待確認 / 待清理
- [x] 2026-07-23：專案已依 Detection / Tracker / Re-ID 三大模組重組，`trackers/` → `tracker/`，`utils/track_eval/` + `utils/eval_utils.py` → `evaluation/`。詳見 `CLAUDE.md`「架構」章節與 `Project.md`。
- [x] 2026-07-23：`utils/train_utils.py` 已搬到 `reid/training/train_utils.py`（歸檔性質，仍無腳本引用，孤兒程式碼原樣搬移）。`utils/model.py` **維持原路徑不動**——搬移過程中發現 `models/class_reid_weights/{person,scooter}/best_full_model.pt` 是用 `torch.save()` 整物件序列化，pickle 內容寫死 `utils.model.xxx` 模組路徑，且會被 `tracker/class_aware_botsort.py` 的 `Custom_ReID` 分支在啟用對應設定時載入，並非真正孤兒；不可搬移，否則載入時會 `ModuleNotFoundError`。
- [x] 2026-07-23：`trackers/trackeval/`（重複副本）已歸檔到 `archive/trackers_trackeval_duplicate/`，根目錄 `trackeval/` 是實際使用的版本。
- [x] 2026-07-23：`past/`、`trackers/past_conf/`、`trackers/past_py_conf/` 已歸檔到 `archive/`（`past_track_eval_orign.py`、`trackers_past_conf/`、`trackers_past_py_conf/`），保留供對照。
- [ ] `reid_models/`（vendored fast-reid 風格套件）同樣因 pickle 相容性**固定留在根目錄**，未搬進 `reid/`；若未來要重組，需先用舊路徑載入權重再用新路徑重新存檔，而非直接搬檔案。
- [ ] `archive/` 底下的舊碼是否還需要保留，或可進一步刪除，留給專案負責人決定。

## 流程缺口
- [x] 2026-07-23：已導入 git 版本控制（見 commit `7061f97` 起）。實驗版本仍靠 `track_config_*.yaml` 檔名區分，尚未改用 git branch/tag 管理，可再評估。
- [ ] 沒有自動化測試，也沒有「基準指標」可供迴歸比對（見 `TEST.md` 待補充）；評估是否要建立至少一組基準 HOTA/AssA 數值。
- [ ] 沒有 `requirements.txt` / `pyproject.toml`，環境重建靠人工記憶依賴套件；建議補上。
- [ ] **Detection 模組缺訓練腳本**：`detection/` 目前只有推論用的 `Detector`（`detection/detector.py`），沒有 YOLO 訓練腳本——這本來就不在專案原始範圍內（`models/yolo/*.pt` 是外部訓練產物）。待日後有專門的 YOLO 訓練腳本，再一併整理進 `detection/`（此項由使用者於 2026-07-23 明確要求先記錄、暫不生成佔位腳本）。
- [ ] Tracker / Re-ID 模組同樣沒有「測試腳本」，目前是用 `tracker/grid_search/*.py` 與 `reid/grid_search_botsort_reid.py` 兼作超參數搜尋與驗證，沒有獨立的單元測試或迴歸測試腳本；是否要另外補上，待評估。

## 待驗證（見 VERIFY.md 對應段落）
- [ ] ReID on/off 的量化數據補齊（目前只有程式碼註解結論，無 Excel/CSV 佐證）。
- [ ] `fps_compare/` 現有數據尚未整理出整體結論摘要。
- [ ] `det_corner` 偵測區域多邊形設定對指標的影響。
- [ ] `fuse_score` 開關（目前全域關閉）對各類別的影響。
- [ ] `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據。

## 待補充
> 此檔案是根據現有程式碼與文件整理出的「已知落差」，不包含尚未提出的新功能需求。若有新的功能/優化需求，請直接加在下方，並標註提出人與日期。
