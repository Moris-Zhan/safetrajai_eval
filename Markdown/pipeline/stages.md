# 各階段說明

## 1. Stage1：偵測 + class-aware 追蹤

- `safetrajai/detection/`：`Detector` 做單幀 YOLO11 推論，再依逐類別信心門檻（`class_conf_thresh`）過濾。
- `safetrajai/tracking/`：唯一入口是 `create_tracker()`。偵測結果**先依類別分組，再做三階段匈牙利匹配**，每個類別的參數由 `get_params_for_class()` 查表取得。
- `load_track_config()` 必須在 `create_tracker()`／`Detector` 建立之前呼叫。順序錯了不會報錯，只會悄悄用到空的或舊的參數。
- 改動 tracker 前，先讀 `utils/modules/safetrajai/tracking/HANDOFF.md` 與 `class_aware_*_flow.md`。`ClassAwareBOTSORT`／`ClassAwareBYTETracker` 的 override 方法（`__init__`、`update()`、`remove_duplicate_stracks()`、`init_track()`）**不可更動**（使用者明確指示，詳見 `CLAUDE.md` 的 Tracker 一節）。

## 2. Stage2：TrackEval 評估

只對既有的 stage1 輸出重跑評估：

```bash
python scripts/pipeline/run_evaluation.py --config <config>
```

處理順序：

1. 準備 GT shadow（輸出目錄中的副本）。
2. **忽略區域前處理**（`evaluation/ignore_region.py`，UA-DETRAC 式）：
   1. GT 中心點在 ROI 外的標為 ignore。
   2. 預測與全部 GT 以 IoU ≥ 0.5 做一對一匹配。
   3. 匹配到 ROI 內 GT 的預測保留。
   4. 其餘預測中，匹配到 ignore GT 或中心點在 ROI 外的移除。
   5. 移除 ignore GT。
3. 呼叫 vendored TrackEval（HOTA／CLEAR／Identity），輸出 Excel 報表。
4. `eval_flags.run_det_eval: true` 時，以人工 GT 組 `det_eval/<seq>/` 資料集，交給 `det_model.val()`（imgsz 取 `tracking_params.imgsz`）。報表會標示「ROI：未套用（全畫面）」。

所有中間產物只寫入輸出目錄，不會改動來源 GT 或 stage1 軌跡。

## 3. Stage3：衝突事件（TTC／PET）

```bash
python track_eval.py --config config/collision/collision_detect.yaml
```

- 程式碼中的 `collision` 指的是**衝突事件**（TTC／PET 低於門檻的接近事件），不是實際碰撞。
- 輸入只讀 stage1 的 7 欄軌跡，不讀 GT；讀入後逐列以 ROI 中心點過濾。
- 參數在 config 的 `collision:` 區塊：`homography_points_dir`、`max_ttc_frames`、`iou_threshold`、`lookback_sec`、`overlap_threshold`。
- 事件結果請讀 `ttc_events.{json,csv}`、`pet_events.{json,csv}`，不要依賴舊 log 的 `colision_occur_sec`（PET 的這個欄位可能記的是 frame）。
- Homography 查找順序：`models/homography/` 既有鏡頭矩陣 → `config/homography_points/<seq>.yaml` 人工點選 → 都沒有時直接用 CCTV 像素座標。
- 誤差單位是像素，不是公尺；未經實測校正的事件數只能當作技術訊號，不能當成正式的安全結論。
