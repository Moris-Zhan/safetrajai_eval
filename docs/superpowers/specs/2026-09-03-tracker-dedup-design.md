# module/tracker/ 去重複 + track_buffer 語意統一 — 設計文件

- 日期:2026-09-03
- **狀態:已作廢(2026-09-03)**——使用者明確決定:`class_aware_botsort.py`/`class_aware_bytetrack.py` 內繼承自 `ultralytics.trackers`(`BOTSORT`/`BYTETracker`)的 override 方法(`__init__`、`update()`、`remove_duplicate_stracks()`、`init_track()`)**程式碼本身一字不可更動,也不可搬離**。這條約束直接推翻本文件的兩個提案:(1) 第 3 節把 `update()` 內的 stage1-4 匹配邏輯抽到 `class_aware_common.py`——需要修改/搬移 `update()`,禁止;(2) 第 4 節 track_buffer 語意統一(改動 BoTSORT `update()` 內的換算公式)——同樣需要修改 `update()`,禁止。兩者均不執行。約束已記錄進 `.claude/rules/tracker.md`,`markdown/NEED.md` 的 track_buffer 項目已回頭撤銷「已決定」狀態。本文件以下內容保留供歷史對照,不再是待實作的設計。
- 範圍:五模組(detection/collision/grid_search/reid/tracker)+ `track_eval.py` 主流程整體重構的**第五個子專案**。這是**核心 matching 邏輯**的改動,風險評等高於前四輪,驗證要求也相應提高。
- 本文件涵蓋兩件事:(1) `update()` 結構去重複(行為保留,逐位元組不變);(2) `track_buffer`/`max_time_lost` 語意統一為幀數基準(**行為變更**,使用者已明確決定,見下節)。兩者一起做是因為語意統一後,track_buffer 換算本身也變成兩邊完全相同的邏輯,可以一起抽進共用模組。

## 0. 範圍邊界

- **`track_buffer`/`max_time_lost` 統一以「30fps 基準幀數」為基準**(使用者於本輪決策過程中先決定用秒數、後反轉為幀數基準,以下為最終決定):`class_aware_bytetrack.py` 現行語意(`cls_max_time_lost = int(self.frame_rate / 30.0 * cls_buffer)`)維持不變;`class_aware_botsort.py` 現行語意(`int(cls_buffer_sec * self.frame_rate)`,秒數)**改成跟 ByteTrack 相同的幀數公式**。這是 `markdown/NEED.md` 記載的已知落差,使用者已於本輪明確決定採統一方向(而非兩邊分開設計)。**這是行為變更,不是純結構重構**——見第 4 節「行為變更影響」。⚠️ **這次改動的一方是 BoTSORT**,而 8 個情境設定檔裡有 6 個用 BoTSORT(`botsort_noreid`/`botsort_reid`/`botsort_save_reid_crops`/`botsort_save_reid_crops_video`/`grid_search_botsort`/`collision_detect`),只有 2 個用 ByteTrack——影響面比「改 ByteTrack」大很多,詳見第 4 節。
- **不改變 class 繼承結構**:`ClassAwareBOTSORT(BOTSORT)`、`ClassAwareBYTETracker(BYTETracker)` 的繼承關係不變,不引入 mixin 或多重繼承(已評估 Mixin 方案,使用者選擇風險較低的共用函式方案)。
- **不改變 `get_dists_class_aware()`、`init_track()` 的實作**——兩者本來就是各自獨立、依 tracker 特性不同的方法(BoTSORT 含 ReID/GMC,ByteTrack 不含),維持現狀,`update()` 呼叫方式不變。
- **不改變任何匹配閾值(`match_thresh`/`second_match_thresh`/`unconfirmed_match_thresh`/`new_track_thresh`)或 `config/*.yaml` 裡的既有 `track_buffer` 數值本身**——只改變 ByteTrack 對這個數值的**解讀公式**,不改變使用者已設定的數字。

## 1. 背景與問題

逐段比對 `ClassAwareBOTSORT.update()`(~290 行本體)與 `ClassAwareBYTETracker.update()`(~250 行本體),確認以下區塊**完全相同或近乎相同**:

- Stage 1(高信心)逐類別分組匹配迴圈(`=== CLASS-AWARE MATCHING ===`)
- Stage 2(低信心)逐類別分組匹配迴圈(固定用 `matching.iou_distance()`,兩邊本來就相同,不含 ReID)
- Stage 3(unconfirmed track)逐類別分組匹配迴圈
- Stage 4 新 track 初始化迴圈(逐偵測套用 `get_params_for_class(cls)['new_track_thresh']`)
- `remove_duplicate_stracks()`(static method,逐字元相同)
- `_center_distance()`(逐字元相同)

不同的只有:GMC 補償區塊(僅 BoTSORT)、`track_buffer`/`max_time_lost` 換算公式(見第 0、4 節,本輪決定統一)、`get_dists_class_aware()`/`init_track()` 內部實作(本來就各自獨立,維持不動)。

## 2. 目標 / 非目標

**目標**
- 把「完全相同」的區塊抽成 `module/tracker/class_aware_common.py` 的共用函式,兩個 tracker class 的 `update()` 改為呼叫這些函式。
- 把 `track_buffer`/`max_time_lost` 換算公式統一為「30fps 基準幀數」語意,BoTSORT 版本改用與 ByteTrack 相同的公式——統一後這段邏輯也一併抽進共用模組。
- `update()` 保留各自骨架(GMC 區塊、呼叫共用函式的順序)——不合併成單一 `update()`,降低多重繼承相關風險。
- 除了 track_buffer 語意變更本身預期造成的差異外,其餘匹配行為(matches、track 狀態轉移)**逐位元組不變**。

**非目標**
- 不合併/整併 GMC、`get_dists_class_aware()`、`init_track()`。
- 不改變 `create_tracker()` 工廠函式介面或 `DEFAULT_TRACKER_CONFIG`。
- 不重新調整 `config/*.yaml` 裡既有的 `track_buffer` 數值本身(語意變更後這些數值的實際效果會不同,是否要重新調參是後續評估項目,不在本次範圍)。
- 不處理 `matching.py`/`config.py` 的其他調整。

## 3. 新設計

### 3.1 `module/tracker/class_aware_common.py`(新檔)

```python
def remove_duplicate_stracks(stracksa, stracksb):
    """逐字搬移現行兩邊完全相同的 static method 實作，行為不變。"""

def center_distance(tracks, detections, norm_factor: float = 200.0):
    """逐字搬移現行兩邊完全相同的 _center_distance() 實作（本來就不依賴 self），
    呼叫端從 self._center_distance(...) 改成 class_aware_common.center_distance(...)。"""

def match_stage1_by_class(strack_pool, detections, get_dists_fn, frame_id):
    """Stage 1（高信心）逐類別分組匹配。
    get_dists_fn(tracks, dets, cls_id) 由呼叫端傳入 self.get_dists_class_aware
    （兩個 tracker 各自的方法本身不變，只是呼叫方式從內嵌改成傳函式參照）。
    match_thresh 固定用 get_params_for_class(cls_id)['match_thresh']（兩邊本來就相同）。
    回傳 (activated_stracks, refind_stracks, matched_track_indices, matched_det_indices)
    —— matched_* 索引供呼叫端計算 stage2 的輸入。"""

def match_stage2_by_class(r_tracked_stracks, detections_second, frame_id):
    """Stage 2（低信心）逐類別分組匹配。固定用 matching.iou_distance() 與
    get_params_for_class(cls_id)['second_match_thresh']。內部直接處理
    「剩餘未匹配 → lost」，回傳 (activated_stracks, refind_stracks, lost_stracks)。"""

def match_stage3_unconfirmed_by_class(unconfirmed, detections_remain, get_dists_fn, frame_id):
    """Stage 3（unconfirmed）逐類別分組匹配，get_dists_fn 同 stage1。內部直接處理
    「剩餘未匹配 → removed」，回傳
    (activated_stracks, removed_stracks, matched_det_remain_indices)。"""

def activate_new_tracks(detections_remain, u_detection_indices, kalman_filter, frame_id):
    """Stage 4：逐偵測套用 get_params_for_class(cls)['new_track_thresh']，達門檻才呼叫
    track.activate(kalman_filter, frame_id)。回傳 activated_stracks。"""

def track_buffer_frames(cls_buffer, frame_rate: float, global_max_time_lost: int) -> int:
    """【2026-09-03 使用者決定：語意統一為「30fps 基準幀數」】
    cls_buffer is None 時回傳 global_max_time_lost（沿用現行 fallback 行為）；
    否則回傳 int(frame_rate / 30.0 * cls_buffer)。
    這是 ByteTrack 現行公式；BoTSORT 呼叫端改用這個函式取代原本的
    int(cls_buffer_sec * frame_rate)（秒數）公式——是本次唯一預期改變數值輸出的函式，
    且影響面較大（見第 4 節：8 個情境檔中 6 個用 BoTSORT）。"""
```

### 3.2 `update()` 內 track_buffer 清理迴圈改動

`class_aware_botsort.py`(**改動,數值輸出改變**——從秒數公式改成跟 ByteTrack 相同的 30fps 基準幀數公式):
```python
# 改前
for track in self.lost_stracks:
    params = get_params_for_class(int(track.cls))
    cls_buffer_sec = params.get('track_buffer', None)
    if cls_buffer_sec is not None:
        cls_max_time_lost = int(cls_buffer_sec * self.frame_rate)   # 秒數語意
    else:
        cls_max_time_lost = self.max_time_lost
    if self.frame_id - track.end_frame > cls_max_time_lost:
        track.mark_removed()
        removed_stracks.append(track)

# 改後
for track in self.lost_stracks:
    params = get_params_for_class(int(track.cls))
    cls_max_time_lost = class_aware_common.track_buffer_frames(
        params.get('track_buffer', None), self.frame_rate, self.max_time_lost
    )
    if self.frame_id - track.end_frame > cls_max_time_lost:
        track.mark_removed()
        removed_stracks.append(track)
```

`class_aware_botsort.py` 目前另外留有一段用 `#` 註解掉的更舊版 `track_buffer` 邏輯(第 365-373 行,語意又跟現行秒數版本不同,已被取代但沒刪除)——本次一併移除這段死碼,避免和新的共用函式並存造成混淆。

`class_aware_bytetrack.py`(**數值輸出不變**,只改呼叫共用函式以消除重複):
```python
for track in self.lost_stracks:
    params = get_params_for_class(int(track.cls))
    cls_max_time_lost = class_aware_common.track_buffer_frames(
        params.get('track_buffer', None), self.frame_rate, self.max_time_lost
    )
    if self.frame_id - track.end_frame > cls_max_time_lost:
        track.mark_removed()
        removed_stracks.append(track)
```

### 3.3 `update()` 其餘部分改動結構

Stage 1-4 改為呼叫 `class_aware_common.*`(見 `3.1`),`update()` 骨架(score 分組、`init_track`、GMC 區塊(僅 BoTSORT)、七步驟 bookkeeping、`return`)維持現行程式碼不動,只是內部呼叫換成共用函式(含 `remove_duplicate_stracks`)。

### 3.4 影響檔案清單

| 檔案 | 改動 |
|---|---|
| `module/tracker/class_aware_common.py`(新) | 承接 stage1-4 匹配邏輯、`remove_duplicate_stracks`/`center_distance`、統一後的 `track_buffer_frames` |
| `module/tracker/class_aware_botsort.py` | `update()` 改呼叫共用函式;track_buffer 清理迴圈改呼叫 `track_buffer_frames()`(**數值輸出改變**,見第 4 節);移除死碼註解區塊 |
| `module/tracker/class_aware_bytetrack.py` | `update()` 改呼叫共用函式;track_buffer 清理迴圈改呼叫 `track_buffer_frames()`(數值輸出不變) |

不動 `module/tracker/__init__.py`、`config.py`、`matching.py`、`module/reid/*`、`config/*.yaml`(數值本身不動)。

## 4. 行為變更影響:BoTSORT 的 `track_buffer` 語意(⚠️ 影響 6/8 個情境檔)

以 `track_buffer: 2.0`、30fps 序列為例:

| | 舊語意(秒數) | 新語意(30fps 基準幀數,本次改動後) |
|---|---|---|
| BoTSORT `cls_max_time_lost` | `int(2.0 × 30) = 60` 幀(2 秒) | `int(30/30 × 2.0) = 2` 幀(≈0.07 秒) |
| ByteTrack `cls_max_time_lost` | `int(30/30 × 2.0) = 2` 幀(不受本次改動影響) | 同左,不變 |

**這次改動的一方是 BoTSORT**,而 `config/botsort_noreid.yaml`、`config/botsort_reid.yaml`、`config/botsort_save_reid_crops.yaml`、`config/botsort_save_reid_crops_video.yaml`、`config/grid_search_botsort.yaml`、`config/collision_detect.yaml`(沿用 `botsort_noreid.yaml` 追蹤參數)共 6 個情境檔都用 BoTSORT,只有 `config/bytetrack.yaml`、`config/grid_search_bytetrack.yaml` 2 個不受影響。

`config/*.yaml` 現有各類別 `track_buffer` 數值(依 NEED.md 記載約 2.0~3.0,原本是設計給秒數語意用的)本次改動後,BoTSORT 的 lost-track 存活時間會**大幅縮短(約 30 倍)**,直接影響:
- 短暫遮擋/離開偵測區域後,track 可能來不及等到物件重新出現就被判定 removed,IDSW 可能上升。
- 也可能減少「已離場物件殘留 lost 狀態太久、被新物件誤接回」的情況(需要用實際資料驗證,不能只憑理論推測方向)。
- **stage3 碰撞偵測(`config/collision_detect.yaml`)也會受影響**,因為它沿用 `botsort_noreid.yaml` 的追蹤參數,track 存活時間變化會反映到軌跡完整度,間接影響 TTC/PET 計算的輸入品質。

**這個方向性影響需要用第 6 節的驗證計畫實際量測,不能假設一定變好或變壞。因為影響面涵蓋 6 個情境檔,建議實作前再次跟使用者確認是否要調整既有 `track_buffer` 數值,或先接受新語意跑一輪基準再決定。**

## 5. 錯誤處理與邊界情況

- `match_stage1_by_class`/`match_stage3_unconfirmed_by_class` 的 `get_dists_fn` 參數必須傳入**已綁定的 bound method**(`self.get_dists_class_aware`),確保呼叫時仍能存取該 tracker 實例的 `self.cls_encoder`(BoTSORT)等狀態。
- 抽出的函式一律不吞例外、不新增預設值——搬移過程中不可新增現行程式碼沒有的 `try/except` 或 fallback。
- `activate_new_tracks` 的 `u_detection_indices` 語意(「detections_remain 裡的索引」而非全域索引)需要在函式文件字串裡明確標注,避免呼叫端傳錯索引空間。
- `track_buffer_frames()` 的 `cls_buffer_sec is None` fallback 行為(回傳 `global_max_time_lost`)兩邊都要維持,不可在統一過程中誤刪。

## 6. 驗證計畫(比照 `.claude/rules/tracker.md` 的要求,較前四輪更嚴格,且需區分兩類改動)

**A. 結構去重複部分(必須逐位元組不變)**
1. 對 `config/bytetrack.yaml`(ByteTrack,不受 track_buffer 語意改動影響)挑 1-2 支序列,改動前後跑完整 pipeline,逐類別 HOTA/AssA/IDSW 數值必須完全一致。

**B. track_buffer 語意統一部分(BoTSORT,預期數值改變,需記錄實際影響——涵蓋 6 個情境檔,優先驗證)**
2. 對 `config/botsort_noreid.yaml`、`config/botsort_reid.yaml` 挑 2-3 支序列(涵蓋不同類別,尤其 person/scooter 這類遮擋頻繁的類別),分別跑「改動前(舊秒數語意)」與「改動後(新 30fps 基準幀數語意)」,記錄每個類別的 HOTA/AssA/IDSW 變化方向與幅度,寫進 `markdown/VERIFY.md`(依專案慣例,量化結論要有資料佐證,不能只憑程式碼推論)。
3. 用疊圖(`save: true`)肉眼比對至少一支序列,確認新語意下的 track 存活時間符合預期(縮短後是否造成不該發生的 ID 中斷)。
4. 若第 2 步發現 HOTA/AssA 明顯劣化,回頭跟使用者確認是否要一併調整 `config/*.yaml` 的 `track_buffer` 數值(不在本次程式碼變更範圍,但可能是必要的後續動作)。

**C. 迴歸覆蓋**
5. 三支 `module/grid_search/*.py`(待前面幾輪 spec 落地後)各跑一次 `--max-videos 1` 的 trial:ByteTrack 版本 `HOTA.AssA_mean` 應與改動前一致;BoTSORT 版本預期會因 track_buffer 語意改變而變化,記錄新的基準值供後續 grid search 参考。
6. `config/collision_detect.yaml`(沿用 BoTSORT 參數)跑一次 stage3,確認 track_buffer 縮短後軌跡完整度是否仍足以支撐 TTC/PET 計算(不要求數值不變,但要確認沒有因軌跡斷裂導致明顯失真)。

**任何 A 類(ByteTrack/純結構部分)的數值對不上,視為重構引入非預期行為變更,必須先定位原因並修正,不可視為誤差略過。B 類(BoTSORT track_buffer)的數值變化是預期中的,重點是記錄變化方向與幅度,不是要求「不變」。**

## 7. 風險總結

- **高風險(相對前四輪)**:直接改動核心 matching 程式碼路徑,且本次額外包含一項刻意的行為變更(ByteTrack track_buffer 語意),需要嚴格區分「應該不變」與「預期改變」兩類驗證。
- **結構去重複的風險點**:抽出函式時把索引空間(全域 vs 局部)、bound method 綁定、或某個隱含的 `self` 狀態搬錯,這類錯誤通常不會拋例外,只會靜默改變追蹤結果。
- **track_buffer 語意變更的風險點**:`config/*.yaml`(6 個 BoTSORT 情境檔)現有數值是在舊秒數語意下調校/沿用的,語意改變後這些數值的「原意」不再成立,lost-track 存活時間大幅縮短,可能需要後續重新調參(不在本次範圍,但應在 `markdown/NEED.md` 追蹤)。這是本輪影響面最大的一項改動,建議優先驗證(見第 6 節 B 類)。
- **依賴**:與 `2026-09-03-grid-search-dedup-design.md`(共用同一條 `create_tracker()` → `update()` 路徑)有間接關聯,建議先完成本輪驗證再驗證 grid_search 的 trial 結果一致性。
