# class_aware_botsort.py 流程說明

`ClassAwareBOTSORT` 繼承 Ultralytics `BOTSORT`，override `__init__`、`init_track`、`update`、`get_dists`（改名 `get_dists_class_aware`）、`remove_duplicate_stracks`。核心邏輯與原版的對照見 `HANDOFF.md`；本檔案專注在「一次 `update()` 呼叫內部實際怎麼跑」的逐步流程。

## 一、`__init__(args, frame_rate=30)`

1. 呼叫 `super().__init__()`，沿用 BOTSORT 原生初始化（Kalman filter、`proximity_thresh`/`appearance_thresh` 等全域參數）。
2. **Kalman filter 依 fps 重新縮放**：原版 Kalman 轉移矩陣的速度項係數預設對應 30fps（`dt=1`）。若實際 `frame_rate` 不是 30，計算 `dt_scale = 30.0 / frame_rate`，把 `kf._motion_mat` 對角線上代表速度的 4 個係數（`(0,4) (1,5) (2,6) (3,7)`）乘上 `dt_scale`。frame_rate 越低，每幀間隔時間越長，同樣像素位移對應的「每幀速度」應該放大，否則 Kalman 預測位置會落後實際物體。
3. 建 `self.gmc = GMC(method=args.gmc_method)`（BoTSORT 專屬，ByteTrack 沒有這行）。
4. **逐類別建立 ReID encoder**（`self.cls_encoder: dict[cls_id, encoder|None]`）：對每個 `CLASS_TRACKING_PARAMS` 裡的類別查 `use_reid`，若為 `True` 才建立對應 encoder，否則該類別的 `cls_encoder[cls_id] = None`。依 `model` 路徑字串判斷用哪個 encoder 類別（見 `safetrajai/reid/encoders.py`）：
   - 路徑含 `reid_model_weights` → `SOTA_ReID`
   - 路徑含 `class_reid_weights` → `Custom_ReID`
   - 其餘 → `ReID`（YOLO 本身當 encoder）

## 二、`init_track(dets, scores, cls, img=None)`

把 YOLO 偵測框轉成 `BOTrack` 物件列表。**逐一偵測**檢查該偵測所屬類別是否 `use_reid=True` 且有設定 `model`：
- 是 → 用該類別的 `cls_encoder[cls_id]` 對這個偵測框 crop 出的影像算特徵向量 `f`，建立 `BOTrack(xyxy, s, c, f)`（帶特徵）。
- 否 → `BOTrack(xyxy, s, c)`（不帶特徵，之後 `get_dists_class_aware` 就不會對這個 track 做外觀比對）。

⚠️ 這裡是**逐偵測**查 `use_reid`（用該偵測的 class），不是一次整批查，所以同一幀裡不同類別的偵測可以有的帶特徵、有的不帶。

## 三、`update(results, img=None)` 逐幀主流程

模組化流程圖（每個 `subgraph` 對應 `update()` 裡的一個邏輯區塊）：

```mermaid
flowchart TD
    Start(["update results, img"]) --> Split

    subgraph Split["① 偵測分組"]
        S1["依 track_high_thresh / track_low_thresh<br/>切成 高信心 dets / 低信心 dets_second"]
    end

    Split --> Prep

    subgraph Prep["② 前處理"]
        P1["init_track 依類別套用 ReID 特徵<br/>建立本幀 detections"]
        P2["tracked_stracks 拆成<br/>unconfirmed / tracked"]
        P3["strack_pool = tracked + self.lost_stracks"]
        P4["multi_predict：Kalman 預測位置"]
        P5["GMC：warp 補償攝影機運動<br/>套用到 strack_pool / unconfirmed"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    Prep --> Stage1

    subgraph Stage1["③ 第一階段：高信心偵測 vs strack_pool（class-aware）"]
        A1["逐類別（偵測∪track 出現的類別）"]
        A2["get_dists_class_aware<br/>IoU + 中心距離 + ReID（依類別）"]
        A3["linear_assignment<br/>thresh = match_thresh（該類別）"]
        A4["配對成功：Tracked → update()<br/>非 Tracked → re_activate()"]
        A1 --> A2 --> A3 --> A4
    end

    Stage1 --> Stage2

    subgraph Stage2["④ 第二階段：低信心偵測 vs 未配對且 Tracked 的 track"]
        B1["逐類別 iou_distance（不含 ReID）"]
        B2["linear_assignment<br/>thresh = second_match_thresh"]
        B3["未配對 track → mark_lost()"]
        B1 --> B2 --> B3
    end

    Stage2 --> Stage3

    subgraph Stage3["⑤ 第三階段：unconfirmed track vs 剩餘偵測"]
        C1["逐類別 get_dists_class_aware（含 ReID）"]
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
        E1["cls_buffer_sec = track_buffer（該類別）"]
        E2["max_time_lost = int(cls_buffer_sec)<br/>（單位：幀；變數為歷史名稱）"]
        E3["超過時限 → mark_removed()"]
        E1 --> E2 --> E3
    end

    LostBuffer --> Merge

    subgraph Merge["⑧ 狀態合併（7 個子步驟）"]
        F1["tracked_stracks 只留 Tracked 狀態"]
        F2["併入 activated / refind"]
        F3["lost_stracks 移除已回到 tracked 的"]
        F4["加入新 lost、移除已 removed 的"]
        F5["remove_duplicate_stracks<br/>（只在同類別內去重）"]
        F1 --> F2 --> F3 --> F4 --> F5
    end

    Merge --> End(["回傳 activated / unactivated / removed 三個 array"])
```

## 四、`get_dists_class_aware(tracks, detections, cls_id)` — 距離矩陣怎麼算

依 `cls_id` 查表取得該類別參數，依序疊加：

1. **基礎 IoU 距離**：`matching.iou_distance(tracks, detections)`（cost = 1 − IoU）。
2. **proximity gating**：`dists_mask = dists > (1 - proximity_thresh)`，先記下哪些 pair 距離太遠（之後若有 ReID 會直接排除，不管外觀多像）。
3. **中心點距離融合**（`center_dist_weight > 0` 才啟用）：算正規化中心點歐式距離 `_center_distance()`，跟 IoU 距離做線性加權 `(1-w)*iou + w*center`。
4. **`fuse_score`**（設定為 `True` 才啟用）：把偵測的信心分數揉進距離矩陣，讓低信心偵測的配對成本變高。
5. **ReID 外觀比對**（`use_reid=True` 且該類別有 encoder 才啟用）：
   - `emb_dists = embedding_distance(tracks, detections) / 2.0`
   - 超過 `appearance_thresh` 的 pair 設為 1.0（無法匹配）
   - 步驟 2 記下「距離太遠」的 pair 也強制設為 1.0
   - `dists = np.minimum(dists, emb_dists)`：兩個距離取更小值（IoU 夠近 **或** 外觀夠像，其一即可配對）
6. **選用的閘門與否決**（change `reid-center-gate-fusion`，類別設定了新參數且 `use_reid=True` 才啟用；都沒設定時完全走步驟 5）：改由 `_gated_appearance()` 呼叫 `matching.fuse_appearance()`。
   - `reid_center_gate_px`（＋`reid_center_gate_px_per_frame` × 遺失幀數；遺失幀數＝`self.frame_id − track.frame_id`）：**只對遺失中的軌跡（Lost）**，中心距離在閘門內的 pair 也可取得外觀成本（與 IoU proximity 取聯集）。中心距離取「預測中心」與「最後一次 Tracked 時的中心」（記在 `track._reid_last_center`，因為 `STrack` 不保存觀測框）的較近者。Tracked 與未確認的 track 仍只看 IoU proximity。
   - `appearance_veto_sim`：外觀距離 > `1 − appearance_veto_sim` 的 pair 成本設為 1（所有 track）。
   - 門檻方向：`appearance_thresh` 與 `appearance_veto_sim` 都是相似度方向，值越大越嚴。
   - 在 C000025 上驗收的結果：閘門讓 car、scooter 串錯明顯增加，否決沒有效果，所以目前沒有任何情境啟用。

## 五、與 `class_aware_bytetrack.py` 的差異

| 項目 | BoTSORT | ByteTrack |
|---|---|---|
| GMC（攝影機運動補償） | 有（`self.gmc.apply()`） | 無，BYTETracker 本身不支援 |
| ReID 外觀比對 | 有（`init_track` 依類別建特徵、`get_dists_class_aware` 依類別融合 embedding） | 無，`get_dists_class_aware` 只有 IoU + 中心距離 + fuse_score |
| Kalman dt_scale 重新校正 | 有（`__init__` 手動改 `kf._motion_mat`） | 無 |
| `init_track` 簽章 | `init_track(dets, scores, cls, img=None)`，需要 `img` 來裁切算 ReID 特徵 | 沿用 BYTETracker 原生 `init_track`，不需要 `img` |
| **`track_buffer` 的單位** | `cls_max_time_lost = int(cls_buffer_sec)`（幀） | `cls_max_time_lost = int(cls_buffer)`（幀） |

## 六、保留期限統一為幀數

兩個追蹤器的逐類別 `track_buffer` 都是正整數幀數；例如 15 在 5fps 與 10fps 下都於遺失第 16 幀移除，第 15 幀仍可重新關聯。載入時檢查每個類別的有效值，使全域 fallback 不會被已宣告的類別用到。

BoTSORT 的 `cls_buffer_sec` 是歷史名稱，現在代表幀數。`update()` 中仍提到秒與乘上幀率的註解已過時，但依使用者授權範圍只修改算式那一行，註解原文保留。`__init__` 與 Kalman 的 `dt_scale` 不變。
