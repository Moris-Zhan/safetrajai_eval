# Trackers 程式碼交接說明

本文件說明 `trackers/` 資料夾中的程式碼如何從 Ultralytics 繼承並修改，供接手工程師參考。

---

## 一、Ultralytics 原先的作法

### 1.1 繼承架構

Ultralytics 提供兩個追蹤器，它們之間是繼承關係：

```
BaseTrack → STrack → BOTrack          (單一物件的 track 表示)
BYTETracker → BOTSORT                  (追蹤器主體)
```

- **BYTETracker** 是基礎追蹤器，使用 IoU 做匹配
- **BOTSORT** 繼承 BYTETracker，額外加上 ReID（外觀特徵）和 GMC（攝影機運動補償）

關鍵事實：**BOTSORT 沒有 override `update()`**。它完全沿用 `BYTETracker.update()` 的流程，只透過 override `get_dists()` 和 `init_track()` 來注入 ReID 能力。

### 1.2 原版 `update()` 每幀流程

```
YOLO 偵測結果進入
        │
        ▼
┌─ 分組：高信心 (≥ track_high_thresh) / 低信心 (介於 low~high 之間) ─┐
│                                                                    │
│  【階段一】First Association                                        │
│   高信心偵測 vs 全部 tracks (tracked + lost)                         │
│   距離計算：get_dists() → 匈牙利演算法                               │
│   閾值：match_thresh（全域統一，如 0.8）                              │
│                                                                    │
│  【階段二】Second Association                                       │
│   低信心偵測 vs 階段一未匹配的 tracked tracks                         │
│   距離計算：純 IoU（不用 ReID）                                      │
│   閾值：寫死 0.5                                                    │
│                                                                    │
│  【階段三】Unconfirmed Tracks                                       │
│   階段一剩餘的高信心偵測 vs 只出現過一幀的 tracks                      │
│   閾值：寫死 0.7                                                    │
│                                                                    │
│  【階段四】初始化新 Track                                            │
│   仍未匹配的高信心偵測 → 建立新 track                                 │
│   閾值：new_track_thresh（全域統一）                                  │
│                                                                    │
│  【階段五】清理                                                     │
│   lost track 超過 max_time_lost 幀 → 永久移除                       │
│   max_time_lost = frame_rate / 30 × track_buffer（全域統一）         │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 原版 `get_dists()` 的差異

這是 BYTETracker 與 BOTSORT 唯一的行為差異點：

```python
# BYTETracker.get_dists() — 只用 IoU
def get_dists(self, tracks, detections):
    dists = matching.iou_distance(tracks, detections)        # cost = 1 - IoU
    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)       # 融合偵測信心分數
    return dists

# BOTSORT.get_dists() — IoU + ReID 外觀特徵
def get_dists(self, tracks, detections):
    dists = matching.iou_distance(tracks, detections)
    dists_mask = dists > (1 - self.proximity_thresh)         # 空間太遠的 pair 標記
    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)
    if self.args.with_reid and self.encoder is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0  # 外觀太不像 → 排除
        emb_dists[dists_mask] = 1.0                                 # 空間太遠 → 排除
        dists = np.minimum(dists, emb_dists)                        # 取 IoU 和 ReID 的較小值
    return dists
```

### 1.4 原版的核心限制

原版所有閾值都是**全域統一**的，不區分物件類別：

| 參數 | 原版行為 |
|------|----------|
| `match_thresh` | 所有類別用同一個值（如 0.8） |
| `appearance_thresh` | 所有類別用同一個值（如 0.4） |
| `new_track_thresh` | 所有類別用同一個值 |
| `second_match_thresh` | 寫死 0.5，不可配置 |
| `unconfirmed_match_thresh` | 寫死 0.7，不可配置 |
| `track_buffer` | 所有類別用同一個值 |
| `with_reid` | 全域開或關，不能只對某些類別開 |

而且，**所有類別的偵測和 tracks 混在一起做匹配**，理論上可能發生跨類別錯配（例如 person 匹配到 scooter 的 track）。

---

## 二、我們想修改的需求

我們的應用場景是 CCTV 監控，同時追蹤 person、car、scooter、small_truck、bus、truck 六種物件。不同類別的物理特性差異很大：

| 特性 | person | scooter | car/truck |
|------|--------|---------|-----------|
| 尺寸 | 小 | 小 | 大 |
| 移動速度 | 慢 | 快 | 中 |
| 遮擋頻率 | 高 | 高 | 低 |
| 外觀多樣性 | 高 | 中 | 低 |

**核心需求：不同類別需要不同的追蹤策略。** 具體來說：

1. **不同類別要有不同的匹配閾值** — scooter 體積小、移動快，IoU 天然較低，需要更寬鬆的 `match_thresh`；car 體積大、IoU 穩定，可以用較嚴格的閾值
2. **避免跨類別錯配** — 原版混合匹配可能把 person 配到重疊位置的 scooter track
3. **不同類別要有不同的 track 存活時間** — scooter 移動快，丟失過久更可能已離場，需要比全域更短的 buffer 以避免幽靈 track
4. **ReID 要能按類別開關** — 實驗發現 ReID 對 person 有幫助但會損害 scooter 追蹤
5. **階段二、階段三的閾值也要可按類別配置** — 原版寫死 0.5 和 0.7，無法調整

---

## 三、如何實作來滿足需求

### 3.1 檔案結構

```
utils/modules/safetrajai/tracking/    # 現況路徑（本文件寫成時仍叫 trackers/，後續歷經 trackers/ → tracker/ → 舊版 tracker 模組 → src/safetrajai/tracking/ → utils/modules/safetrajai/tracking/ 多次搬移）
├── __init__.py                  # 工廠函式 + 全域基礎參數
├── config.py                    # 類別專屬參數定義（核心配置檔）
├── class_aware_botsort.py       # 繼承 BOTSORT，override update()
└── class_aware_bytetrack.py     # 繼承 BYTETracker，override update()
```

### 3.2 `config.py` — 類別專屬參數表

為每個類別定義獨立的參數組：

```python
CLASS_TRACKING_PARAMS = {
    'person':      { 'match_thresh': 0.85, 'second_match_thresh': 0.5,  'track_buffer': 120, ... },  # 保留期限 120 幀
    'scooter':     { 'match_thresh': 0.90, 'second_match_thresh': 0.75, 'track_buffer': 90,   ... },  # 90 < 全域 120，適度縮短
    'car':         { 'match_thresh': 0.85, 'second_match_thresh': 0.5,  'track_buffer': 120, ... },  # 保留期限 120 幀
    ...
}
```

透過 `get_params_for_class(cls_id)` 函式查詢，傳入 class ID 或 class name 都可以，未知類別回傳 `DEFAULT_TRACKING_PARAMS`。

`config.py` 另外定義了**類別專屬偵測信心門檻** `CLASS_CONF_THRESH`，用於推論階段（inference script）的前處理過濾：

```python
CLASS_CONF_THRESH = {
    'person': 0.15,
    'car': 0.15,
    'scooter': 0.10,       # 降低門檻，救回更多 scooter 偵測
    'small_truck': 0.15,
    'bus': 0.15,
    'truck': 0.15,
}
```

推論腳本先用 `get_min_conf_thresh()`（所有類別中的最小值）作為 YOLO 的全域 conf，確保低門檻類別的偵測不被丟棄。然後在 YOLO 輸出後，逐一檢查每個偵測的類別，用 `get_conf_thresh_for_class(cls_id)` 過濾不達門檻的偵測，才送進 tracker。這讓 scooter 可以用更低的信心門檻（0.10）而不影響其他類別。

### 3.3 `__init__.py` — 工廠函式 + 全域參數

全域參數定義在 `DEFAULT_TRACKER_CONFIG`，模擬 Ultralytics 原本從 YAML 讀取的 `args` 格式：

```python
DEFAULT_TRACKER_CONFIG = {
    'track_high_thresh': 0.15,
    'track_low_thresh': 0.05,
    'track_buffer': 120,
    'match_thresh': 0.85,       # 全域預設（被 config.py 的類別參數覆蓋）
    'with_reid': False,
    'fuse_score': False,        # 關閉信心融合，避免低信心偵測（如 scooter）被額外懲罰
    ...
}
```

`create_tracker()` 工廠函式將 config dict 包成 `SimpleNamespace` 傳給父類建構子：

```python
def create_tracker(tracker_type='botsort', frame_rate=30, **kwargs):
    config = DEFAULT_TRACKER_CONFIG.copy()
    config.update(kwargs)
    args = SimpleNamespace(**config)
    if tracker_type == 'botsort':
        return ClassAwareBOTSORT(args=args, frame_rate=frame_rate)
    elif tracker_type == 'bytetrack':
        return ClassAwareBYTETracker(args=args, frame_rate=frame_rate)
```

### 3.4 `class_aware_botsort.py` / `class_aware_bytetrack.py` — 核心修改

兩個檔案的 class-aware 匹配邏輯相同，差異在於：(1) BoTSORT 版新增 `get_dists_class_aware()` 支援類別級 ReID 開關；(2) ByteTrack 版刻意省略 GMC 分支（ByteTrack 本身不使用 GMC）。以下用 BoTSORT 說明，每個修改點都對應一個需求：

#### 修改 1：按類別分組匹配（解決需求 1 + 2）

原版在階段一做一次全域匹配：

```python
# 原版
dists = self.get_dists(strack_pool, detections)
matches = matching.linear_assignment(dists, thresh=self.args.match_thresh)  # 單一閾值
```

改為對每個類別分別匹配：

```python
# 我們的版本 (class_aware_botsort.py:94-141)
for cls_id in all_classes:
    params = get_params_for_class(cls_id)
    cls_match_thresh = params['match_thresh']    # 每個類別自己的閾值

    cls_detections = [同類別的偵測]
    cls_tracks = [同類別的 tracks]

    dists = self.get_dists_class_aware(cls_tracks, cls_detections, cls_id)
    matches = matching.linear_assignment(dists, thresh=cls_match_thresh)
```

**這個改動同時解決了跨類別錯配問題**：因為每次迴圈只處理同一類別的偵測和 tracks，不同類別之間不可能被匹配到。

階段二和階段三也同樣改為按類別分組，且閾值改為可配置：

```python
# 階段二 (class_aware_botsort.py:147-180)
cls_second_thresh = params.get('second_match_thresh', 0.5)   # 原版寫死 0.5

# 階段三 (class_aware_botsort.py:190-219)
cls_unconf_thresh = params.get('unconfirmed_match_thresh', 0.7)  # 原版寫死 0.7
```

#### 修改 2：類別級 ReID 開關（解決需求 4，僅 BoTSORT）

新增 `get_dists_class_aware()` 方法，取代原版的 `get_dists()`：

```python
# class_aware_botsort.py:291-321
def get_dists_class_aware(self, tracks, detections, cls_id):
    params = get_params_for_class(cls_id)
    use_reid_cls = params.get('use_reid', False)          # 類別級開關
    appearance_thresh = params.get('appearance_thresh', ...)  # 類別級閾值

    dists = matching.iou_distance(tracks, detections)
    ...
    # 必須全域 with_reid=True 且該類別 use_reid=True 才啟用
    if self.args.with_reid and use_reid_cls and self.encoder is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - appearance_thresh)] = 1.0  # 用類別專屬閾值
        ...
```

對比原版 `BOTSORT.get_dists()` 只有全域的 `self.appearance_thresh`，不接受 `cls_id` 參數。

change `reid-center-gate-fusion` 在這裡加了兩個選用的逐類別參數：`reid_center_gate_px`（只對遺失中的軌跡，以中心距離閘門擴大外觀的作用範圍）與 `appearance_veto_sim`（外觀否決）。設定任一個時改走 `_gated_appearance()` → `matching.fuse_appearance()`；都沒設定時上面的程式碼逐位元不變（`tests/test_reid_fusion.py` 以修改前演算法的複製比對）。流程細節見 `class_aware_botsort_flow.md` 第四節。

ByteTrack 版本不需要這個方法（ByteTrack 本身就不用 ReID），所以直接呼叫繼承的 `self.get_dists()`。

#### 修改 3：類別級 track buffer（解決需求 3）

原版在清理階段用全域統一的 `self.max_time_lost`：

```python
# 原版
for track in self.lost_stracks:
    if self.frame_id - track.end_frame > self.max_time_lost:
        track.mark_removed()
```

改為查詢每個 track 所屬類別的 buffer：

```python
# 我們的版本 (class_aware_botsort.py:244-254)
for track in self.lost_stracks:
    params = get_params_for_class(int(track.cls))
    cls_buffer = params.get('track_buffer', None)
    cls_max_time_lost = self.max_time_lost                         # 全域預設 120
    if cls_buffer is not None:
        cls_max_time_lost = int(cls_buffer)  # 類別保留期限幀數（ByteTrack；BoTSORT 變數為 cls_buffer_sec）
    if self.frame_id - track.end_frame > cls_max_time_lost:
        track.mark_removed()
```

注意：以上原版 Ultralytics 段落保留 upstream 行為說明。本專案逐類別 `track_buffer` 已統一為正整數幀數（ADR 0009），不依 `self.frame_rate` 換算。載入設定時每個類別必須有有效值，使全域 fallback 不可達；BoTSORT 的 `cls_buffer_sec` 為歷史名稱，方法內舊註解維持原文。

#### 修改 4：類別級新 track 閾值（解決需求 5）

```python
# 我們的版本 (class_aware_botsort.py:231-242)
for inew in u_detection:
    track = detections_remain[inew]
    params = get_params_for_class(int(track.cls))
    new_track_thresh = params['new_track_thresh']     # 原版用全域 self.args.new_track_thresh
    if track.score < new_track_thresh:
        continue
    track.activate(self.kalman_filter, self.frame_id)
```

---

## 修改對照總表

| `update()` 中的位置 | 原版 Ultralytics | 我們的修改 | 對應程式碼行 |
|---------------------|-----------------|-----------|-------------|
| 階段一匹配 | 全類別混合，`match_thresh` 統一 | 按類別分組，各自 `match_thresh` | `class_aware_botsort.py:94-141` |
| 階段一距離計算 | `get_dists()` 全域 `appearance_thresh` | `get_dists_class_aware()` 類別級閾值 + ReID 開關（僅 BoTSORT） | `class_aware_botsort.py:291-321` |
| 階段二匹配 | 全類別混合，閾值寫死 0.5 | 按類別分組，可配置 `second_match_thresh` | `class_aware_botsort.py:147-180` |
| 階段三匹配 | 全類別混合，閾值寫死 0.7 | 按類別分組，可配置 `unconfirmed_match_thresh` | `class_aware_botsort.py:190-219` |
| 新 track 初始化 | 全域 `new_track_thresh` | 類別級 `new_track_thresh` | `class_aware_botsort.py:231-242` |
| Track 存活時間 | 全域 `max_time_lost` | 類別級 `track_buffer` 覆蓋 | `class_aware_botsort.py:244-254` |
| Track 去重複 | 只看 IoU，不看 class（跨類別會互刪） | `remove_duplicate_stracks()` 加入 class 檢查，只在同類別間去重 | 兩個 tracker 的 `remove_duplicate_stracks()` |
| `__init__` | 父類不保存 `frame_rate` | override `__init__` 儲存 `self.frame_rate` | 兩個 tracker 的 `__init__` |
| 其餘邏輯 | — | **主要流程對齊 upstream**（Kalman predict、狀態管理等沿用父類）。刻意差異：ByteTrack 版省略 GMC 分支（ByteTrack 本身不使用 GMC） | — |
