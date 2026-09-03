---
paths:
  - 'module/tracker/**/*.py'
  - 'track_eval.py'
---

# Tracker 開發規範

## 核心行為

- 對外只透過 `create_tracker(tracker_type, frame_rate, **overrides)` 建立追蹤器；支援 `bytetrack` 與 `botsort`。
- 追蹤必須依類別分組再進行匹配，禁止不同類別共用 matching matrix。這是避免 person、scooter、車輛跨類誤配的核心設計。
- 每個類別可覆寫 `match_thresh`、`second_match_thresh`、`unconfirmed_match_thresh`、`new_track_thresh`、`track_buffer`、`use_reid` 與外觀匹配參數；未設定時才可回退 default。

## 設定載入與 ID

- `load_track_config()` 必須在 `create_tracker()` 前呼叫。順序錯誤可能不報錯，但會靜默使用舊或預設參數，必須視為失敗。
- 每支影片會新建 tracker；`track_id` 只在該影片內唯一，跨影片可重複。不要把 ID 當作全資料集身份。
- 新增參數時，確保 YAML 載入、class lookup、factory override 與兩種 tracker 的行為一致。

## BoTSORT 與 ByteTrack 差異

- BoTSORT 的 `track_buffer` 目前以秒數處理，會乘上 frame rate；ByteTrack 以 30fps 基準幀數處理。同一數值不代表相同 lost-track 存活時間。這個落差**已知但不會被修正**（見下方「不可更動的核心方法」）。
- GMC 僅對 BoTSORT 有意義；Re-ID 僅由 BoTSORT 的逐類別 `use_reid` 啟用。

## 不可更動的核心方法（2026-09-03，使用者明確指示）

- `class_aware_botsort.py::ClassAwareBOTSORT`（繼承 `ultralytics.trackers.bot_sort.BOTSORT`）與 `class_aware_bytetrack.py::ClassAwareBYTETracker`（繼承 `ultralytics.trackers.byte_tracker.BYTETracker`）是本模組僅有的兩個真正繼承自 `ultralytics.trackers` 的類別。
- 這兩個類別 override 自 ultralytics 基底類別的方法——`__init__`、`update()`、`remove_duplicate_stracks()`（static）、`init_track()`（僅 `ClassAwareBOTSORT`）——**程式碼本身一字不可更動，也不可搬離**（不可抽成共用函式、不可移到其他檔案、不可改寫成呼叫其他模組）。這是使用者的明確指示，非本專案自行推導的保守原則。
- 這條限制**優先於**任何重構/去重複目標：即使兩邊 `update()` 高度重複、即使 `track_buffer` 語意不一致這類已知落差，只要修正方案需要碰到上述方法的程式碼，就一律不執行。`2026-09-03-tracker-dedup-design.md`（去重複＋track_buffer 統一）因此已作廢。
- `get_dists_class_aware()`、`_center_distance()` 不受此限制——這两個是本專案新增的方法，不是 ultralytics 基底類別的 override（父類別沒有同名方法），修改時仍適用本檔案其餘規則（依類別分組、驗證 HOTA/AssA/IDSW 等）。
- `module/tracker/matching.py`、`config.py`、`__init__.py` 不受此限制——它們不繼承 `ultralytics.trackers`（`matching.py` 只 import `ultralytics.utils.metrics` 的工具函式，非 class 繼承）。

## 驗證

- 修改 matching、track state 或閾值時，先跑單序列／目標類別，再比較同一基準的 HOTA、AssA、IDSW。
- 以追蹤疊圖檢查 ID switch、ID 消失／重生與跨類別誤配；只通過語法檢查不足以驗證追蹤行為。
