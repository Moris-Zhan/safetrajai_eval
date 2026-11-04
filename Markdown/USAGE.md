# safetrajai 使用說明（USAGE）

> 整理日期：2026-09-24
> 依據：repo 現況（各腳本 `--help`、`config/`、`utils/modules/safetrajai/`）。
> 術語見 `CONTEXT.md`。

---

## 1. 文件目錄

- [安裝](getting-started/install.md)
- [資料與目錄慣例](getting-started/data-layout.md)
- [從 Hugging Face 還原資料與模型](getting-started/hf-restore.md)
- [評測資料集與追蹤基準分數](validation/baseline-scores.md)
- [主流程：`track_eval.py`](pipeline/main-flow.md)
- [各階段說明](pipeline/stages.md)
- [ROI（關注區域）](roi/roi-guide.md)
- [追蹤參數 Grid Search](grid_search/grid-search.md)
- [Re-ID 資料工具（`rebuild-reid-dataset` 新流程）](reid/reid-data-tools.md)
- [驗證與視覺化工具](validation/validation-and-viz.md)
- [測試與回歸驗證](testing/testing.md)
- Config 使用教學
  - [Config 總覽](config/overview.md)
  - [追蹤情境與 `class_tracking`](config/tracking-scenarios.md)
  - [Re-ID 設定](config/reid-configs.md)
  - [Stage3 設定](config/collision-configs.md)
  - [以鏡頭為鍵的設定](config/per-camera-configs.md)

---

## 2. 常見情境速查

| 我想要… | 做法 |
|---|---|
| 跑基準評估 | `python track_eval.py --config config/tracking/bytetrack.yaml` |
| 只重算 stage2 | `python scripts/pipeline/run_evaluation.py --config <config>` |
| 只跑某支鏡頭 | 另建 config，加上 `paths.cameras: [<camera>]`（必須是資料集登記鏡頭的子集） |
| 開 detection 評估 | `eval_flags.run_det_eval: true`（使用人工 GT，全畫面、不套 ROI） |
| 跑衝突事件 | `python track_eval.py --config config/collision/collision_detect.yaml` |
| 新鏡頭畫 ROI | `python scripts/picker/make_roi_picker.py --camera <id>` → 本機瀏覽器點選 → 存成 `config/roi/<id>.yaml`（完整步驟見 [ROI 教學](roi/roi-guide.md) 第 2 節） |
| 調某類別的追蹤參數 | `grid_search_*.py --target-class <cls> --dry-run` → `--max-videos 1` → 正式跑 |
| 隔離一份有問題的權重 | 先寫 `temp/DO_NOT_USE.md`，再 `mv` 到同目錄的 `temp/` |
| 確認沒有改壞 | `pytest`；行為可能有變時再跑 golden 並與 `conflict1` 比對 |

---

## 3. 已知限制

- `botsort_reid.yaml` 的 small_truck／scooter Re-ID 權重是暫代方案，要等 `rebuild-reid-dataset` 產出新模型後替換（此條尚未重新查證）。
- detection 評估目前不套用 ROI（全畫面）。
- `track_buffer` 在 BoTSORT 與 ByteTrack 都是正整數保留期限幀數，不依幀率換算；舊的秒數／30fps 換算已取代。
- Homography 以鏡頭為鍵（`config/homography/<camera>.yaml`）；舊的 `config/homography_points/<seq>.yaml` 不再讀取，存在時報錯。
