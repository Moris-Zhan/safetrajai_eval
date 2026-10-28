# class_aware_bytetrack.py 流程說明

`ClassAwareBYTETracker` 繼承 Ultralytics `BYTETracker`，override `__init__`、`update`、`get_dists`（改名 `get_dists_class_aware`）、`remove_duplicate_stracks`。**不 override `init_track`**——直接沿用 `BYTETracker` 原生版本，因為 ByteTrack 不需要 ReID 特徵，`init_track` 不用額外處理 `img`。與 BoTSORT 版的完整對照見 `class_aware_botsort_flow.md` 第五節；核心原理對照見 `HANDOFF.md`。

## 一、`__init__(args, frame_rate=30)`

只做一件事：呼叫 `super().__init__()` 後多存一個 `self.frame_rate = frame_rate`，保留原初始化行為；`update()` 的保留期限已直接使用幀數，不再使用此值換算。**沒有** BoTSORT 版的 Kalman dt_scale 重新校正、沒有 GMC、沒有逐類別 ReID encoder 建立。

## 二、`update(results, img=None, feats=None)` 逐幀主流程

`img`、`feats` 兩個參數都保留在簽章上（維持與 BoTSORT 版呼叫端相容），但**完全沒被使用**——ByteTrack 不用影像做外觀比對，也沒有 GMC 需要用到 `img`。

模組化流程圖（每個 `subgraph` 對應 `update()` 裡的一個邏輯區塊；跟 `class_aware_botsort_flow.md` 對照可以直接看出少了哪些區塊）：

```mermaid
flowchart TD
    Start(["update results, img, feats"]) --> Split

    subgraph Split["① 偵測分組"]
        S1["依 track_high_thresh / track_low_thresh<br/>切成 高信心 dets / 低信心 dets_second"]
    end

    Split --> Prep

    subgraph Prep["② 前處理"]
        P1["init_track（BYTETracker 原生，不含 ReID）<br/>建立本幀 detections"]
        P2["tracked_stracks 拆成<br/>unconfirmed / tracked"]
        P3["strack_pool = tracked + self.lost_stracks"]
        P4["multi_predict：Kalman 預測位置"]
        P1 --> P2 --> P3 --> P4
    end

    Prep --> Stage1

    subgraph Stage1["③ 第一階段：高信心偵測 vs strack_pool（class-aware）"]
        A1["逐類別（偵測∪track 出現的類別）"]
        A2["get_dists_class_aware<br/>只有 IoU + 中心距離 + fuse_score"]
        A3["linear_assignment<br/>thresh = match_thresh（該類別）"]
        A4["配對成功：Tracked → update()<br/>非 Tracked → re_activate()"]
        A1 --> A2 --> A3 --> A4
    end

    Stage1 --> Stage2

    subgraph Stage2["④ 第二階段：低信心偵測 vs 未配對且 Tracked 的 track"]
        B1["逐類別 iou_distance（單純 IoU）"]
        B2["linear_assignment<br/>thresh = second_match_thresh"]
        B3["未配對 track → mark_lost()"]
        B1 --> B2 --> B3
    end

    Stage2 --> Stage3

    subgraph Stage3["⑤ 第三階段：unconfirmed track vs 剩餘偵測"]
        C1["逐類別 get_dists_class_aware<br/>IoU + 中心距離 + fuse_score"]
        C2["linear_assignment<br/>thresh = unconfirmed_match_thresh"]
        C3["未配對 unconfirmed → mark_removed()<br/>（直接淘汰，不會變成 lost）"]
        C1 --> C2 --> C3
    end

    Stage3 --> NewTrack

    subgraph NewTrack["⑥ 新增 track"]
        D1["剩餘偵測逐一比對<br/>new_track_thresh（該類別）"]
        D2["score 足夠 → track.activate()"]
        D1 --> D2
    end

    NewTrack --> LostBuffer

    subgraph LostBuffer["⑦ Lost track 存活判斷"]
        E1["cls_buffer = track_buffer（該類別）"]
        E2["max_time_lost = int(cls_buffer)<br/>（單位：幀）"]
        E3["超過時限 → mark_removed()"]
        E1 --> E2 --> E3
    end

    LostBuffer --> Merge

    subgraph Merge["⑧ 狀態合併（7 個子步驟，與 BoTSORT 版相同邏輯）"]
        F1["tracked_stracks 只留 Tracked 狀態"]
        F2["併入 activated / refind"]
        F3["lost_stracks 移除已回到 tracked 的"]
        F4["加入新 lost、移除已 removed 的"]
        F5["remove_duplicate_stracks<br/>（只在同類別內去重）"]
        F1 --> F2 --> F3 --> F4 --> F5
    end

    Merge --> End(["回傳 activated / unactivated / removed 三個 array"])
```

跟 `class_aware_botsort_flow.md` 的圖對照，少了「GMC 運動補償」這個前處理步驟，第一、三階段的 `get_dists_class_aware` 也少了 ReID 那一段。

## 三、`get_dists_class_aware(tracks, detections, cls_id)` — 距離矩陣怎麼算

比 BoTSORT 版單純很多，依 `cls_id` 查表後只做：

1. **基礎 IoU 距離**：`matching.iou_distance(tracks, detections)`。
2. **中心點距離融合**（`center_dist_weight > 0` 才啛用）：`_center_distance()` 算正規化中心點歐式距離，跟 IoU 距離線性加權。
3. **`fuse_score`**（`True` 才啟用）：把偵測信心分數揉進距離矩陣。

**沒有** `proximity_thresh` gating、**沒有** ReID embedding 融合這兩步——`proximity_thresh` 這個欄位即使在 track_config 裡有設定，對 ByteTrack 也完全不起作用（只有 BoTSORT 的 `get_dists_class_aware` 會讀它）。

## 四、⚠️ 注意事項

1. **與 BoTSORT 共用同一份 `track_config*.yaml` 時，`use_reid`/`model`/`appearance_thresh`/`proximity_thresh` 這幾個欄位對 ByteTrack 完全是死欄位**（讀不到，也不會報錯，靜默忽略）。這是 ByteTrack 本身設計上不支援 ReID 的必然結果，不是 bug，但容易誤以為調整這些參數對 ByteTrack 有效。
2. **`track_buffer` 與 BoTSORT 統一為正整數幀數**（ADR 0009）：保留期限直接為 `int(cls_buffer)`，不依影片幀率換算。設定載入時檢查每個已宣告類別，沒有類別設定才使用預設參數；缺少或不是正整數時報錯。凍結方法只有這個敘述與 BoTSORT 對應敘述開放修改，其餘包含 `__init__` 與舊註解都維持原文。
