# 偵測訓練資料集製作規則

狀態：v3 為止的規則已定案（使用者 2026-11-23、2026-11-24 逐項回覆「都先照建議」或指定修改）。v4 為計畫，尚未執行；2026-11-25 起，之後的方向以 [`cctv-all8-finetune-plan.md`](cctv-all8-finetune-plan.md) 為準（目標：在 cctv_all8 上贏過 dennis）。

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

### v3 雜訊底線與 person 遺忘實驗結果（2026-11-24，`runs/finetune/eval_v3c/results.json`）

所有權重都用 last.pt 評估。驗證集是 ft_v3 的 816 幀，三份匯出的驗證影像完全相同。數字是 mAP50-95，C000012 與 C000056 合併計算：

| 權重 | 整體 | person | car | scooter | small_truck | bus | truck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dennis（未微調） | 0.428 | 0.476 | 0.889 | 0.658 | 0.194 | 0.255 | 0.094 |
| v2_e0 | 0.523 | 0.334 | 0.876 | 0.659 | 0.550 | 0.516 | 0.202 |
| v3_e0（seed 0） | 0.497 | 0.300 | 0.875 | 0.661 | 0.508 | 0.495 | 0.140 |
| v3_s1（seed 1） | 0.524 | 0.369 | 0.882 | 0.672 | 0.512 | 0.543 | 0.168 |
| v3_rep3（seed 0，person 幀 ×3） | 0.491 | 0.323 | 0.876 | 0.659 | 0.483 | 0.503 | 0.098 |
| v3_tiny（seed 0，小 person 偽標籤） | 0.502 | 0.351 | 0.868 | 0.669 | 0.514 | 0.463 | 0.147 |
| v3_tiny_s1（seed 1，小 person 偽標籤） | 0.501 | 0.289 | 0.880 | 0.666 | 0.489 | 0.510 | 0.172 |

- **雜訊底線**：只換 seed，整體分數就差 0.027，person 差 0.069。v2 與 v3 的差距（0.026）落在這個雜訊內，所以「v3 比 v2 差」不成立。之後比較變體時，整體差距小於約 0.03、person 差距小於約 0.07，單一 seed 都不能算有差，要用至少兩個 seed 比較。
- **person 遺忘確實存在**：所有微調結果的 person 都在 0.30～0.37，明顯低於 dennis 的 0.476。
- **rep3**：person 比 v3_e0 高 0.023，在雜訊內，整體分數沒有改善。沒有證據顯示有效。
- **tiny**：兩個 seed 都沒有超出雜訊的改善。
  - 合併計算的 person：seed 0 是 0.351，seed 1 是 0.289；v3 兩個 seed 是 0.300 和 0.369。
  - C000012 的 person：seed 0 曾到 0.337，但 seed 1 只有 0.165，低於 v3 的兩個 seed（0.184、0.229）。所以 seed 0 的改善是 seed 運氣。
  - 偽標籤會增加 person 誤報，見下表。
- **person 的漏檢與誤報**（`runs/finetune/errors_v3/errors.json`）：
  - 信心門檻 0.15，與追蹤用的相同；IoU ≥ 0.5 才算配對。
  - 「小」指換算到 640 輸入後高度 ≤ 32 px。
  - 驗證集只有 145 個 person（C000012 有 49 個、C000056 有 96 個），所以 person 的 mAP 波動很大。

  | 權重 | C000012 漏檢（其中小的） | C000012 誤報 | C000056 漏檢 | C000056 誤報 |
  | --- | --- | --- | --- | --- |
  | dennis | 8（8） | 15 | 27 | 212 |
  | v3_e0 | 41（40） | 3 | 45 | 28 |
  | v3_s1 | 33（32） | 50 | 42 | 25 |
  | v3_rep3 | 33（32） | 16 | 41 | 32 |
  | v3_tiny | 22（21） | 23 | 48 | 71 |
  | v3_tiny_s1 | 33（32） | 24 | 43 | 43 |

  - 遺忘的幾乎都是小 person：C000012 的 person 全是小目標。
  - 誤報數本身也受 seed 影響很大：v3 只換 seed，C000012 的誤報就從 3 變成 50。
- **dennis 在 C000056 的 212 個 person 誤報抽查**（2026-11-25，`runs/cvat_review/fp_review_{sample.json,responses.jsonl}`）：
  - 依信心分三層（< 0.25 有 79 個、0.25～0.5 有 95 個、≥ 0.5 有 38 個），每層隨機抽 17 個（seed 0）。使用者答了 48 題，結果依層大小加權。
  - **騎士一律算 scooter**（使用者裁決，2026-11-25；與 [`labeling-guide.md`](labeling-guide.md)「騎士不另外標 person」一致）。使用者原本把 11 個機車騎士判成「GT 漏標」，這些騎士的 GT 都有 scooter 框（IoU 0.81～0.99），所以改計為「其他類別」。照原作答，漏標比例會是 33%。

  | 判定 | 加權比例 |
  | --- | --- |
  | GT 真的漏標 person | 10.0% |
  | 真的誤報（不是道路使用者） | 62.3% |
  | 騎士（GT 是 scooter 框） | 23.2% |
  | 看不清 | 3.4% |
  | GT 已有框，只是框不合 | 1.2% |

  - 漏標比例 10% 低於事先定好的 30% 門檻，所以不需要補標驗證集的 person，person 遺忘的結論不變。樣本只有 48 題，這只能當粗估。
  - 約四分之一的 dennis 誤報是「把騎士框成 person」，這是 dennis 和本專案 GT 慣例不同。微調後模型的 person 誤報變少，有一部分是學會了這個慣例，不能當成遺忘。遺忘的證據是小 person 的漏檢增加和 person AP 下降。
- **結論**：person 幀 ×3 和小 person 偽標籤，在兩個 seed 的雜訊底線下都看不出效果，兩者都不採用。用目前 145 個 person 的驗證集，也分辨不出小幅改善。要處理小 person 遺忘，需要更多有人工標註的小 person，包括訓練和驗證；這比調整訓練取樣更直接。
- rep3 與 tiny 訓練途中 segfault 過好幾次，都是從 last.pt 續跑完成，見第 10 節。

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

```bash
# 評估（last.pt；逐鏡頭與合併的逐類別 mAP，結果合併寫進 results.json）
python scripts/detection_data/eval_finetune.py --export ft_v3 --out runs/finetune/eval_v3c \
  dennis=models/yolo/yolo11x_dennis.pt v3_e0=runs/finetune/ft_v3_e0/weights/last.pt

# 逐鏡頭、逐類別的 TP／FP／FN（信心門檻 0.15／0.25／0.5，FN 依有效高度 ≤ 32 px 拆分）
python scripts/detection_data/count_detection_errors.py --export ft_v3 --out runs/finetune/errors_v3 \
  dennis=models/yolo/yolo11x_dennis.pt v3_e0=runs/finetune/ft_v3_e0/weights/last.pt

# 誤報抽查：先輸出逐框誤報清單（最低信心門檻），再開分層抽樣的審查頁（連接埠 8771，權杖沿用 runs/cvat_review/server_token）
python scripts/detection_data/count_detection_errors.py --export ft_v3 --out runs/finetune/errors_v3 --dump-fp person \
  dennis=models/yolo/yolo11x_dennis.pt
python scripts/cvat_review/serve_fp_review.py --fp runs/finetune/errors_v3/fp_dennis_person.json --camera C000056
python scripts/cvat_review/serve_fp_review.py --fp runs/finetune/errors_v3/fp_dennis_person.json --camera C000056 --summary
```

訓練當機（segfault）時，用同一個指令加 `--resume` 接續；`optimizer` 保持 auto 時，Ultralytics 會忽略 `lr0`，所以要明確指定 AdamW。

segfault 從第一輪（2026-11-19）就存在。2026-11-24 重建 env 後變得很頻繁，每 2～15 分鐘一次。查到的情況：

- 當掉時所在的執行緒每次不同：有時是主執行緒等待 GPU 的地方，有時是 autograd 反向傳播的執行緒。所以問題在原生程式碼，與 OpenCV 無關。
- `CUDA_LAUNCH_BLOCKING=1` 無法避免。
- nvidia-smi 看不到降頻或錯誤；這張卡沒有 ECC。
- 根本原因還沒找到，可能是驅動 580、PyTorch 2.7.1 在 Blackwell 上的問題，或硬體。

在原因查明以前，訓練要用有重試的續跑迴圈。評估前先確認 run 資料夾裡有 `train_record.json`，代表 50 個 epoch 都跑完。2026-11-24 的佇列曾在 rep3 和 tiny 沒跑完時就評估，那次的結果已經作廢重跑。
