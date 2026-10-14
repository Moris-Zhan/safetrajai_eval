# Tasks

## 1. 前置確認與設定

- [x] 1.1 確認 change `camera-homography-georef` 已實作：可以依鏡頭解析 Homography，`meters_per_pixel()` 對 C000025 回傳有限值（C000025 的經緯度已填，`scale_check` 為 `passed`）。不符合時停下並告知使用者，不繼續 5.3 之後需要真實資料的任務（1.2～5.2 的合成資料任務仍可進行）
- [x] 1.2 先寫 `tests/test_conflict_metrics.py` 的設定測試：
  - 沒有 `conflict_analysis:` 區塊時 `enabled=False`；
  - `smooth_window_s` ≤ 0、`min_fps_for_acceleration` ≤ 0、門檻為負時報錯；
  - 角度區間不連續或沒有涵蓋 0～180 時報錯；
  - `class_pairs` 含未知類別時報錯，預設值是排除 person–person 的 20 種組合（6 類兩兩組合加同類共 21 種，扣掉 1 種）；
  - `object_length_m`、`object_mass_kg` 缺任一類別時報錯；
  - 預設 `ttc_distance_m=2.0`、`ttc_distance_mode=fixed`、`pet_max_gap_s=0`（沿用參考實作，ADR 0011）；`ttc_distance_mode` 不是 `fixed`／`class_length` 時報錯。

  確認失敗後新增 `collision/conflict/config.py`，並在 `config_loader.py` 讀取這個區塊，測試通過
- [x] 1.3 新增 `config/conflict_analysis.yaml`（以 `collision_detect.yaml` 為底，列出全部參數與預設值，註解標明單位，風格依既有 config）；`pytest` 確認 smoke 的全部 config 載入通過

## 2. 軌跡與運動學

- [x] 2.1 先寫測試：底邊中點為 `(left + w/2, top + h)`；ROI 以框中心判斷；car 與 scooter 相同的 track_id 被視為不同軌跡；以單位矩陣 H 與 m/px=0.1 驗證公尺座標。確認失敗後實作 `tracks.py`，測試通過
- [x] 2.2 先寫測試（spec「等速直線」「軌跡中斷」「平滑視窗換算」「低幀率序列」）：
  - 等速 10 m/s 時，中間各幀速率誤差小於 1e-6、加速度為 0、端點未定義；
  - 等加速 2 m/s²（30 fps）時，內部加速度誤差小於 1e-6；
  - `smooth_window_s=0.6` 在 4.9 fps 與 30 fps 時分別換算為 3 幀與 19 幀；
  - fps 低於 `min_fps_for_acceleration` 時加速度全部為 NaN，速度不受影響；
  - 幀差大於 `max_gap_frames` 時切段；
  - 速率低於門檻時航向未定義。

  確認失敗後實作 `kinematics.py`，測試通過

## 3. 逐幀配對指標

- [x] 3.1 先寫測試（spec「超出配對半徑」「預設排除行人對行人」「兩條軌跡都靜止」「區間邊界」「接近相向」「一方靜止」）：
  - 半徑外不配對；不同類別也配對；預設不產生 person–person，但 person–car 照常產生；
  - 兩條軌跡都靜止時不配對；
  - 夾角 30.5 度為 Lane-change、170 度為 Crossing（預設三類，沒有 Head-on）；
  - cos 浮點誤差略大於 1 時不產生 NaN；
  - 一方靜止時為 Undefined。

  確認失敗後實作 `pairs.py`，測試通過
- [x] 3.2 先寫測試（spec「正面接近」「依類別調整碰撞距離（選用模式）」「一開始就重疊」「正在遠離」「只有加速度才會接近」）：
  - 相距 22 m、各 10 m/s 相向，預設碰撞距離 2.0 m 時，TTC 為 1.0±0.02 秒；
  - `ttc_distance_mode=class_length` 時，car–car 與 car–scooter 的碰撞距離分別為 5.0 m 與 3.4 m；
  - t=0 中心距離 1.5 m 時標成 `overlap_at_start`，TTC 未定義；
  - 正在遠離時 TTC 為無限大；
  - 只靠加速度才會接近時 TTC 與 MTTC 都是無限大（MTTC 以 TTC 命中為前提，30 fps 合成資料）；TTC 有限時 MTTC 有值；加速度未定義時 MTTC 未定義；
  - 碰撞點是兩外推位置的中點；
  - 追撞時前車為前導。

  確認失敗後實作 `ttc.py`，測試通過
- [x] 3.3 先寫測試（spec「三車同向」「DRAC」「超限」「低幀率序列」「CI 數值」「加速度只取大小」）：
  - 三車同向時最近前後車正確；
  - 中心距離 10 m、後車 15 m/s、前車 10 m/s 時 DRAC 為 1.25 m/s²；接近速度 ≤ 0 時 DRAC 未定義；
  - DRAC 1.25、MaxD 0.8 時 `drac_exceeds_maxd` 為真、`drac_excess` 為 0.45；同一配對的超限比例正確；
  - 加速度未定義時 DR、MaxD、DRAC 超限、CI、CI_m 未定義，DRAC 照常計算；
  - 直接呼叫 CI 函式：V_F=15、V_L=10、加速度皆 0、MTTC=2.0 時 CI 為 31.25；前車加速度向量與航向同向或反向（大小 2）時 CI 相同；
  - CI_m 使用類別質量。

  確認失敗後實作 `following.py`，測試通過

## 4. 軌跡層級：PET 與追撞鏈

- [x] 4.1 先寫測試（spec「同時出現過的配對」「沒有同時出現（預設）」「沒有同時出現（選用模式）」）：
  - 同時出現過的兩條軌跡，PET 為 1.2 秒、先到者正確；
  - 預設（`pet_max_gap_s=0`）下，沒有同時出現的配對不計算 PET；`pet_max_gap_s=5` 時 PET 為 1.2 秒；
  - 停留時間重疊時 PET 為 0；
  - 最近距離大於 `pet_zone_m` 時沒有 PET。

  確認失敗後實作 `pet.py`，測試通過
- [x] 4.2 先寫測試（spec「一列車隊」）：三台相隔 8 m 的同向車屬於同一追撞鏈；相隔 30 m 時分屬不同追撞鏈；四個方向扇區的邊界值（45、135、225、315 度）分類正確；同一追撞鏈、同一行進方向內依位置排序。確認失敗後實作 `chains.py`，測試通過

## 5. 事件、輸出與串接

- [x] 5.1 先寫測試（spec「連續命中」）：連續 12 幀命中時合併成 1 個事件；中間空 1 幀（≤ `event_gap_frames`）時仍是 1 個；空 5 幀時為 2 個；門檻統計以事件數計算；只有 `overlap_at_start` 列的配對不產生事件，但 `overlap_at_start_pairs` 加 1（spec「一開始就重疊不算事件」）；事件 ID 與配對內的順序固定。確認失敗後實作 `events.py`，測試通過
- [x] 5.2 實作 `output.py` 與 `run_conflict_analysis_for_sequence()`，並在 `run_stage3()` 中依 `enabled` 呼叫。先寫測試（暫存目錄、合成的 MOT 檔與單位矩陣）：
  - 輸出檔齊全，事件 CSV 有 `_m`、`_map_px`、`_cctv_px` 欄位；
  - `m_per_pixel` 不可用時只輸出 `summary.json`，`skipped_reason` 正確；
  - 關閉時不產生 `conflict/` 目錄；
  - 連跑兩次的輸出逐位元相同。

  測試通過
- [x] 5.3 向使用者確認後，以 `config/conflict_analysis.yaml` 對 C000025 的一段序列跑 stage1＋stage3（在 config 的 `paths:` 加 `SOURCE_VIDEO_DIR` 限縮到單一序列）。記錄到 `verification.md`：
  - 事件數、TTC／PET 門檻統計（4.9 fps，所以 MTTC 與 CI 應全部未定義，並確認 summary 註明原因）；
  - `overlap_at_start_pairs` 的數量；
  - 追撞鏈大小分布；
  - 執行時間。

  抽查 5 個事件：以一支 scratchpad 腳本從序列影片取出每個事件最小 TTC 的那一幀，畫上兩條軌跡的框與 CCTV 像素碰撞點，存成圖片，目視確認位置與類型是否合理（`visualize_gt_on_video.py` 只畫 GT 與軌跡，不畫事件）；依追撞鏈分布與使用者確認 `chain_dist_m`，修改時同步更新 design 第 6 節與 config 註解

## 6. 驗收與文件

- [x] 6.1 驗證 spec「開啟分析時既有輸出不變」：對 5.3 的同一份 stage1 輸出，分別在分析開啟與關閉下跑 stage3，以 `tests/golden/compare.py` 的比對函式或 `cmp` 確認 `ttc_events.*`、`pet_events.*`、log CSV、熱區圖逐位元相同；結果寫入 `verification.md`
- [x] 6.2 驗證 spec「重現性」：以 5.3 的設定再跑一次 stage3，`conflict/` 目錄以 `cmp` 逐檔比對（只有該序列的少數檔案）完全相同
- [x] 6.3 向使用者確認後跑 `tests/golden/run_all.sh <run>`，以 `python tests/golden/compare.py georef1 <run>` 比對（`georef1` 是前置 change 建立的基準），確認既有 5 個情境沒有多出任何檔案、所有類別差異為 0；結果寫入 `verification.md`
- [x] 6.4 **使用者目視確認**：把 5.3 抽查的 5 張事件圖交給使用者確認。通過後才把 `conflict_analysis` 加入 golden（design 第 11 節）：
  - `tests/golden/run_all.sh` 的 `SCENARIOS` 加入 `conflict_analysis`，`tests/golden/configs/` 加入對應設定；
  - `compare.py` 新增 `conflict` 類別；
  - 跑兩次建立基準 `conflict1`，自比差異為 0；
  - 與 `georef1` 比對時，只有新情境的檔案多出來，其餘類別差異為 0；
  - 在 TOLERANCE.md 記錄 `georef1` → `conflict1` 的差異來源，並把 CLAUDE.md 的基準名稱改為 `conflict1`。

  使用者不同意時，記錄在 `verification.md`，這一項不做
- [x] 6.5 更新文件：
  - CONTEXT.md：衝突事件、衝突角度、MTTC、DRAC 超限、前導與跟隨、前車與後車、追撞鏈、行進方向等術語已在規劃階段加入，確認與實作後的 spec 一致；新增 ADR 0011 的連結已在 ADR 本身，不另外重複；
  - CLAUDE.md：「衝突事件／Homography」一節加上子步驟、輸出目錄、只限有 m/px 的鏡頭、指標只當技術訊號；
  - `config/conflict_analysis.yaml` 檔頭寫明用法。

  跑 `pytest` 與 `openspec validate conflict-metrics --strict`，確認通過
