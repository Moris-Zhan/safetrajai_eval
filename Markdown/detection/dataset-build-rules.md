# 偵測訓練資料集製作規則

狀態：v3 為止的規則已定案（使用者 2026-11-23、2026-11-24 逐項回覆「都先照建議」或指定修改）。v4 為計畫，尚未執行。

用途：說明從原始影片到 YOLO 微調資料集（`dataset/detection/exports/<version>/`）的每個步驟與裁決規則，以及評估要怎麼做才公平。類別邊界（什麼算 car、scooter、small_truck…）另見 [`labeling-guide.md`](labeling-guide.md)，CVAT 操作見 [`cvat-operations.md`](cvat-operations.md)。

## 1. 總流程

1. 從原始影片取樣影格（第 2 節）。
2. 用目前權重 `models/yolo/yolo11x_dennis.pt` 產生偽標籤，匯入 CVAT（task 3），由人工修正（第 3 節）。
3. 套用遮罩與忽略區（第 4 節）。
4. 補充 person 資料（第 6 節）與其他類別（第 7 節）。
5. 依鏡頭切分訓練與驗證（第 8 節），匯出版本。
6. 以固定訓練 epoch、最後一個 epoch 的權重，在公平的評估鏡頭上與 dennis 比較（第 8 節）。

## 2. 資料來源與取樣

- 原始影片在 `dataset/raw/videos/<鏡頭>/`；`dataset/raw/videos` 下的所有影片都可以被標註（使用者確認）。
- pilot_v1：10 支影片、9 個鏡頭，每 1.5 秒取一幀，共 3,308 幀（偽標籤 67,545 個框）。
- 評估鏡頭 C000025（附人工 GT `cctv_all8`）**永遠不進訓練資料**；smoke 測試會檢查。
- C000026 納入訓練來源（v4），需要先補 ROI 與偵測遮罩並用網頁確認，再取樣。
- 公平評估鏡頭是 C000012 與 C000056：dennis 沒看過（使用者確認）。dennis 很可能看過 C000025，所以 C000025 上的 baseline 偏高，只當參考。

## 3. 人工審查與修正規則

- 標籤以 CVAT 匯出的快照為準，不用原始偽標籤 txt；匯出時會把快照複製到版本資料夾，確保能重現。
- 被刪除的整組框（群組刪除）記錄在 `runs/cvat_review/box_issues.csv`，原座標留在最早期備份快照（`runs/cvat_review/backups/task3_pre_corrections_*.json`）。
- 類別邊界依 [`labeling-guide.md`](labeling-guide.md)；人工 GT 內容不可修改。

## 4. 遮罩與忽略區

- 每個鏡頭的遮罩多邊形在 `config/detection_masks/<鏡頭>.json`，匯出時塗成灰色（114）。
- 一個框有 50% 以上面積落在遮罩內就丟掉；保留的框不會被塗掉。
- 「看不清」的框也塗灰，當作忽略區。
- **群組刪除幀（v3 起拿回）**：被刪的群框用備份快照的原座標塗灰當忽略區，不再整張排除；已標好的框不會被塗掉。`export_training_set.py --restore-group-deleted <快照>`。

## 5. 版本規則

- 已訓練過的版本不覆蓋；試匯出才用 `--overwrite`。
- 每個版本的 `export_record.json` 記錄各類別框數、被遮罩丟掉的框、挖掘幀統計。
- 實驗預算以資料版本重新計算，上一版的額度用完不延用。

## 6. person 補充（挖掘與人工審查）

目的：pilot_v1 的 person 很少（訓練只有 155 個框、約 90 個不同的人），微調後 person 的準確率比 dennis 低。

1. `mine_person_frames.py`：用 dennis 掃描原始影片，保留有 person 偵測（信心 ≥ 0.25）的影格；已在 manifest 的影格跳過；評估鏡頭（C000012、C000056、C000025）不讀。
2. `serve_person_review.py`：審查頁，每個框兩組設定。
   - **類別按鈕**：人／機車／car／其他（其他旁的下拉選 small_truck、bus、truck）。預設「人」。
   - **狀態下拉**：正常／誤檢／看不清／框不完整。預設「正常」。
3. 狀態的意義與匯出處理：

| 狀態 | 意義 | 匯出處理 |
| --- | --- | --- |
| 正常 | 框裡就是所選類別 | 標成該類別 |
| 誤檢 | 框裡沒有該標註的東西（誤偵測，不是標錯類別） | 丟掉這個框，類別不算 |
| 看不清 | 小或模糊，無法判定 | 該框範圍塗灰，類別不算 |
| 框不完整 | 知道類別，但框只框到一部分 | 類別照樣有效；有修框就用修正座標當「正常」，沒修框則整張排除並列入 `partial_pending.json` |

4. **修框**：按「改框」在圖上拖出正確範圍，原框紅色虛線、修正框綠色；修好的框自動變「正常」。座標以原始影像像素存在答案的 `corrections`。
5. 勾「有人沒被框到」的整張排除（那個人會變成假的背景）。
6. 儲存值：類別字串（`person`、`scooter`…）、`none`（誤檢）、`unsure`、`partial:<類別>`；舊版 `not_person` 讀成 `none`，舊版 `partial` 讀成 `partial:person`。
7. **不採用外觀比對自動套用答案**：DINOv2 外觀比對（`propagate_person_answers.py`）在留一法（LOO）校準的一致率是 95.5%，但抽查實際只對 23%（跨鏡頭失效），所以全部改人工審查。若要再用，必須改成留一鏡頭（leave-one-camera-out）校準。
8. 審查時發現的人為錯誤要回頭檢查：本輪有 92 個交通號誌桿被誤標成 small_truck（快速鍵對應改版造成的可能性，未證實），已改為誤檢；備份在 `person_review_responses.before_pole_fix.jsonl`。

## 7. 挖掘幀的其他類別

挖掘幀只審查了 person 候選框。其他類別用 dennis 預測補上：信心 ≥ 0.5，且與已審查保留的框 IoU ≥ 0.5 的丟掉（人工審查優先）。這是偽標籤，信心低於 0.5 的物體在這些幀會被當成背景。挖掘幀只進訓練集，不進驗證集；驗證序列的影格一律跳過。

### 防止 person 遺忘的匯出選項（v3 資料上的實驗，預設都關閉）

- `--person-repeat N`：在版本資料夾寫 `train_repeat.txt`，含 person 框的訓練幀列 N 次，`data.yaml` 的 `train` 改指向它（絕對路徑）。影像檔本身不複製。
- `--tiny-person-pseudo`：在 pilot 訓練幀補上 dennis 的 person 偽標籤（信心 ≥ `--tiny-person-conf`，預設 0.25；有效高度 ≤ `--tiny-person-max-height`，預設 32 像素）。與已保留的框或塗灰區域 IoU ≥ 0.5 的丟掉。這些框沒有人工審查，CVAT 審查時被刪掉的誤檢可能被加回來，所以只當實驗，結果要與 person 誤報數一起看。挖掘幀不套用（那些幀的 person 已人工審查）。
- 兩者的設定都記在 `export_record.json`（`person_repeat`、`tiny_person_pseudo`）。

## 8. 切分與評估公平性

- 以鏡頭／序列切分，不以影格隨機切：驗證 = C000012 + C000056（`export_training_set.py` 的預設）。
- 評估協議：訓練固定 epoch，取**最後一個 epoch**，不用公平鏡頭挑 best.pt；回報 C000012 + C000056 合併的逐類別結果。
- 注意 truck 樣本稀少（C000056 沒有 truck、C000012 只有 26 個），truck 的差異不可過度解讀。
- C000025 只當參考，並註明 dennis 可能見過它。
- 以資料版本為單位重新跑 dennis 與前一版的基準，因為驗證集的組成會變。

## 9. 版本

| 版本 | 組成 | 訓練幀 | 驗證幀 |
| --- | --- | --- | --- |
| ft_v2 | pilot_v1 排除群組刪除幀 | 2,075 | 505 |
| ft_v3 | pilot_v1 全 3,308 幀（群組刪除幀拿回並塗灰）＋挖掘的 person 幀 377 | 2,869 | 816 |
| ft_v4（計畫） | v3 ＋ 新取樣（C000054 另 29 支各約 100 幀、C000026 約 400 幀、以 0.75 秒偏移重新取樣既有影片各約 400 幀） | 約 10,200 幀（約 v2 的 3 倍） | — |

v3 只比 v2 多約 1.4 倍的訓練幀，達不到預期的 3～4 倍；規模要靠 v4 的新取樣。

## 10. 重現

```bash
# 匯出 ft_v3（驗證 = C000012 + C000056 是預設）
python scripts/detection_data/export_training_set.py --version ft_v3 \
  --restore-group-deleted runs/cvat_review/backups/task3_pre_corrections_20261113T033126Z.json \
  --mined-person-frames

# 訓練（與 E0 相同設定，固定 50 epoch、不早停）
python scripts/detection_data/train_finetune.py --export ft_v3 --name ft_v3_e0 \
  --optimizer AdamW --lr0 0.0002 --workers 0 --epochs 50 --patience 50
```

訓練當機（segfault）時，用同一個指令加 `--resume` 接續。`optimizer` 保持 auto 時 Ultralytics 會忽略 `lr0`，所以要明確指定 AdamW。
