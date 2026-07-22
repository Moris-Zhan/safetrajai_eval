# NEED.md

待辦與未滿足需求清單。以下項目是整理程式碼時發現的落差，非正式規劃，請專案負責人確認優先順序與是否要處理。

## 待確認 / 待清理
- [ ] `utils/model.py`、`utils/train_utils.py` 目前沒有任何腳本引用，確認是否還需要（舊 ReID 訓練流程殘留），若不需要可考慮移除或另外歸檔。
- [ ] `trackers/trackeval/` 與根目錄 `trackeval/` 內容重複，確認是否可移除其中一份，避免未來改動時改錯份。
- [ ] `past/`、`trackers/past_conf/`、`trackers/past_py_conf/` 內的舊版程式碼是否還需要保留對照，或可歸檔到別處。

## 流程缺口
- [ ] 沒有版本控制（非 git repo），實驗版本靠 `track_config_*.yaml` 檔名區分，容易搞混或遺失歷史；評估是否要導入 git。
- [ ] 沒有自動化測試，也沒有「基準指標」可供迴歸比對（見 `TEST.md` 待補充）；評估是否要建立至少一組基準 HOTA/AssA 數值。
- [ ] 沒有 `requirements.txt` / `pyproject.toml`，環境重建靠人工記憶依賴套件；建議補上。

## 待驗證（見 VERIFY.md 對應段落）
- [ ] ReID on/off 的量化數據補齊（目前只有程式碼註解結論，無 Excel/CSV 佐證）。
- [ ] `fps_compare/` 現有數據尚未整理出整體結論摘要。
- [ ] `det_corner` 偵測區域多邊形設定對指標的影響。
- [ ] `fuse_score` 開關（目前全域關閉）對各類別的影響。
- [ ] `track_buffer` 類別覆蓋值（如 scooter=90 vs 全域 120）的實際效果數據。

## 待補充
> 此檔案是根據現有程式碼與文件整理出的「已知落差」，不包含尚未提出的新功能需求。若有新的功能/優化需求，請直接加在下方，並標註提出人與日期。
