# config/homography_checkpoints/

`module/collision/homography_validation.py` 用來驗證既有 Homography 矩陣準確度的
**check points**（獨立於算矩陣用的訓練點之外的驗證點），一個路口一份，檔名對應
`models/homography/matrices/H_<tag>.npy` 的 `<tag>`：

```
config/homography_checkpoints/<tag>.yaml
```

⚠️ **這不是拿來算 Homography 矩陣的訓練點**（訓練點是 `models/homography/matrices/
pts_cctv_<tag>.npy` / `pts_sate_<tag>.npy`，已經算好、不需要也不應該重新點選）。
check points 必須是額外挑選、跟訓練點不同的點位，才是有意義的 out-of-sample 驗證；
拿訓練點驗證會循環論證（誤差趨近於 0，見 `markdown/Project.md`「Homography 資料」章節）。

## 目前狀態

只有**本專案實際有 CCTV 來源影片**的 6 個路口才有辦法點 check points（其餘 20 個既有
矩陣路口沒有影片可看，物理上選不了，永遠是「未驗證」狀態，見
`module/collision/homography_validation.py` 模組說明）：

- `C000025 中華三路 民生二路(東北角)-2-20-1280-1280`
- `C000043 民族一路 九如一路-2-20-1280-1280`
- `C000054 民族一路 天祥二路-2-20-1280-1280`
- `C000085 中正路 三多路(建軍)-2-20-1280-1280`
- `C000294 一心凱旋西南角-2-20-1280-1280`
- `C000297 一心 凱旋(西北角)-2-20-1280-1280`

這個目錄目前是空的（還沒有任何 `.yaml`），代表以上 6 個路口都還沒做正式驗證。

## 怎麼填

1. 打開 `models/homography/pick_aid/<tag>_pick_aid.png`（已為上述 6 個路口產生好，
   左邊 CCTV 第一幀、右邊衛星底圖，兩邊都疊了每 100px 一條的座標格線方便讀數）。
2. 在兩張圖上找 4~8 組（建議不少於 6 組，統計才穩定）你能明確辨認的同一個實際地點
   （例如：斑馬線角、分向島尖端、明顯的地面標線交點），**不要跟
   `models/homography/matrices/pts_cctv_<tag>.npy` 裡已經用過的訓練點重複**。
3. 讀出兩邊對應的像素座標，建立 `config/homography_checkpoints/<tag>.yaml`：

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

4. 填完後跑 `module/collision/homography_validation.py::validate_tag()`（或後續會加的
   批次驗證腳本）會自動：
   - 算出 reprojection error 統計（像素）：mean/std/median/RMSE/p95/max，
     以及誤差最大的幾個點
   - 存一張 overlay 圖（CCTV warp 疊到衛星圖，綠點=GT、紅點=預測，越接近代表矩陣越準）
   - 存三張誤差熱力圖（IDW/Linear/RBF 空間內插，check points ≥ 4 個才會產生）

## 其餘 20 個既有矩陣路口

`037-`、`BEV0001`、`C000011/012/022/026/033/035/047/048/056/082/094/096/102/132/
203/221/222/256` 這 20 個路口只有 `models/homography/matrices/H_<tag>.npy` 跟
`models/homography/satellite_images/<tag>.png`，沒有 CCTV 影片，不會也不應該在這個
目錄下建立 `.yaml`——`homography_validation.validate_tag()` 對這些路口會回傳
`{"validated": False, "reason": "no_checkpoints"}`，不代表矩陣有問題，只是缺乏驗證資料。
