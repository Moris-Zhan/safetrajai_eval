# config/homography_checkpoints/

`safetrajai/collision/homography/validation.py` 用來驗證既有 Homography 矩陣準確度的
**check points**（獨立於算矩陣用的訓練點之外的驗證點）。一台**鏡頭**一份，檔名是鏡頭編號
（序列名稱第一個空白前的字串，與 `config/roi/`、`config/homography/` 相同）：

```
config/homography_checkpoints/<camera>.yaml
```

⚠️ **這不是拿來算 Homography 矩陣的訓練點**（訓練點是 `models/homography/matrices/
pts_cctv_<tag>.npy` / `pts_sate_<tag>.npy`，已經算好、不需要也不應該重新點選）。
check points 必須是額外挑選、跟訓練點不同的點位，才是有意義的 out-of-sample 驗證。
規則：

- 與訓練點 CCTV 座標相同的 check point 會被**排除**，排除數列在報表的 `check_excluded`。
- 排除後少於 **4 點**時，報表的 `validated` 為 False、`reason` 為 `too_few_checkpoints`（統計仍會輸出）。
  建議 8 點以上，統計才穩定。
- 報表另外列出訓練點的擬合殘差（`train_residual_*`）。只有 4 個訓練點時 DLT 恰好可解，
  殘差接近 0 是預期結果，**不代表矩陣準確**。
- 鏡頭的 `m_per_pixel` 可用時（`config/homography/<camera>.yaml` 有緯度且 `scale_check: passed`），
  報表會同時輸出公尺欄位（`_m`）。

## 目前狀態

由 `scripts/migrate_homography_config.py` 從舊的 tag 鍵資料轉入：

- `C000025.yaml`：2 點，從舊的 tag 鍵 YAML 原樣複製（檔頭有信心度說明，是目視估計），點數不足。
- `C000297.yaml`：10 點，從 `models/homography/checkpoints/check_pts_{cctv,sat}_<tag>.npy` 轉入，數值逐位元保留。

只有**本專案實際有 CCTV 來源影片**的 4 台鏡頭才有辦法補點：
`C000025`、`C000043`、`C000054`、`C000085`。其餘既有矩陣的鏡頭沒有影片可看，
報表會是 `no_checkpoints`，不代表矩陣有問題，只是缺乏驗證資料。

## 怎麼填

1. 打開 `models/homography/pick_aid/<tag>_pick_aid.png`（左邊 CCTV 第一幀、右邊衛星底圖，
   兩邊都疊了每 100px 一條的座標格線方便讀數）。
2. 在兩張圖上找至少 4 組（建議 8 組以上）能明確辨認的同一個實際地點
   （例如：斑馬線角、分向島尖端、明顯的地面標線交點），**不要跟訓練點重複**。
3. 讀出兩邊對應的像素座標，建立或補充 `config/homography_checkpoints/<camera>.yaml`：

```yaml
cctv_points:
  - [x1, y1]
  - [x2, y2]
  ...
sat_points:
  - [x1, y1]
  - [x2, y2]
  ...
```

`cctv_points` / `sat_points` 必須一一對應、順序相同、數量相等。

4. 執行 `python scripts/validate_homography.py`，會輸出到 `models/homography/validation_output/`：
   - `summary.csv`：每台鏡頭一列（`check_*` 驗證誤差、`train_residual_*` 擬合殘差、公尺欄位）
   - overlay 圖（CCTV warp 疊到衛星圖，綠點=GT、紅點=預測）
   - 誤差熱力圖（IDW/Linear/RBF 空間內插，check points ≥ 4 個才會產生）
