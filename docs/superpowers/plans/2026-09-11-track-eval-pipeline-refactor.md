# track_eval.py 主流程重構 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **2026-09-11 修訂**：`dataset-path-refactor` 分支已合併進 master（`53a4501`）。本計畫原本假設尚未合併、路徑組裝暫時沿用舊字串邏輯的版本已作廢；本版本對照合併後的實際程式碼重寫，原 Task 1（移除補幀死碼）已因合併而完成，本版本移除該 Task，其餘 Task 全面改用已存在的 `evaluation/dataset_layout.py`。

**Goal:** 把 `VideoSequenceTracker.run()`（~350 行單一 method，混雜影片讀寫、偵測追蹤主迴圈、label/image 存檔、Re-ID 裁切蒐集、三段幾乎重複的標註繪圖邏輯、`cv2.imshow`、影片寫檔與 ffmpeg 轉檔）拆成協調（orchestration）+ 一組私有方法 + 兩個協作物件；消除三段重複的標註繪圖邏輯；`TrackingEvalPipeline.run_stage1()`/`run_stage3()` 職責拆分。純結構重構，對外 CLI（`python track_eval.py --config <path>`）與每個 stage 的輸出檔案格式/內容不變。

**Architecture:** 新增 `utils/visualization.py::draw_track_box()` 統一「多邊形過濾→畫框→標註」邏輯；`track_eval.py` 新增 `ReidCropCollector`（Re-ID 裁切蒐集）與 `VideoWriterWithFfmpeg`（影片寫檔+ffmpeg 轉檔）兩個協作物件；`VideoSequenceTracker.run()` 拆成 `_open_video()`/`_prepare_output_dirs()`/`_run_frame_loop()`/`_process_frame()`/`_annotate_frame()`/`_print_summary()`；`TrackingEvalPipeline.run_stage1()` 拆出 `_process_single_video()`；`run_stage3()` 的 stage1 存在性檢查拆成獨立函式 `_ensure_stage1_output_exists()`。所有路徑組裝（序列目錄命名、GT 來源解析、stage1 序列列舉）一律呼叫已合併的 `evaluation/dataset_layout.py`，不自行拼字串。

**Tech Stack:** Python 3、既有 `opencv-python`/`ultralytics.utils.plotting`（不新增套件）。本專案沒有 pytest（見 `CLAUDE.md`：「無測試」），驗證步驟一律用**可直接執行的 assert 腳本**（放 scratchpad 不進版控）+ 手動跑 1-2 支序列比對輸出（labels/images/trackers 逐行比對、Demo.mp4 肉眼比對）。

**Spec:** `docs/superpowers/specs/2026-09-03-track-eval-pipeline-refactor-design.md`（第 0 節「前提依賴」所描述的前提現已成立：dataset-path-refactor 已合併）。

## Global Constraints

- 固定 6 類別與 ID 不可變動：`person=0`、`car=1`、`scooter=2`、`small_truck=3`、`bus=4`、`truck=5`（`.claude/rules/project-and-pipeline.md`）。
- **對外 CLI 與輸出格式契約不變**：`python track_eval.py --config <path>` 介面不變；`labels/`、`images/`、`trackers/mot_challenge/...`、`data.yaml`、`Demo.mp4`（含標註樣式：matched 框內嵌 track_id、unmatched 紅框+文字、removed 黑框+文字）的內容與格式必須與重構前逐位元組/逐幀一致。這是純結構重構，不是行為變更。
- **一律呼叫 `evaluation/dataset_layout.py`，不自行拼路徑字串**：`dataset_layout.seq_name_from_video_path()`、`stage1_seq_dir()`、`stage2_seq_dir()`、`source_gt_dir(source_video_dir, run_track_eval_dir, seq_name)`（注意簽章含 `run_track_eval_dir` 參數，不同 dataset 的 SOURCE 端 GT 子目錄名稱不同，不可寫死 `"mot_annotations_all8"`）、`list_stage1_sequences()`。OUTPUT 端序列目錄**不含**前綴（`<seq>/`），SOURCE 端既有資料維持原本 `annotation_<seq>/`／`mot_annotation_<seq>/` 命名不變。
- **GT 來源不一定存在**：`video_seqs` 這類 dataset 本來就沒有人工標註（見 `config/datasets.yaml`），`_process_single_video()` 複製 GT 前必須先用 `gt_src.exists()` 判斷，不存在時印警告並略過，不可讓 `shutil.copytree()` 對不存在的來源拋 `FileNotFoundError` 中斷整條 pipeline（已於合併時修好，本計畫拆分方法時不可退化回無條件複製）。
- `draw_track_box()` 要同時涵蓋 matched（嵌入式文字 `embed_label`）與 unmatched/removed（疊加式文字 `overlay_label`+`overlay_anchor`）兩種現行畫法；`embed_label`/`overlay_label` 兩者互斥，呼叫方依三種情境各自決定要用哪一種，不可混用。
- `VideoWriterWithFfmpeg` 的 `cv2.VideoWriter(...)` 建構子沿用現行**寫死的 `5`**（不是傳入的 `fps` 參數）——這是現行程式碼本來就有的行為（原始碼裡有一行被註解掉的 `fps` 版本），重構時**不修正這個既有問題**，只搬移位置，維持輸出影片逐幀節奏與重構前一致。
- `_annotate_frame()` 的簽章比 spec 草稿少一個 `total_frames` 參數：現行「左上角顯示幀號」的文字只用當前幀號（`show_frame_id`），從未用到總幀數，因此拆分時不引入這個實際上不會被使用的參數（避免死參數），此為對 spec 草稿的刻意小幅偏離，記錄於本文件 Self-Review。
- `module/tracker/**/*.py` 的 `ClassAwareBOTSORT`/`ClassAwareBYTETracker` override 方法一字不可更動、不可搬離（`.claude/rules/tracker.md`）——本計畫不觸碰這些檔案，只呼叫既有的 `create_tracker()`/`tracker.update()` 對外介面。
- 不執行全量影片、不覆寫 `runs/` 既有輸出——每個驗證步驟都先在限量序列（1-2 支）上跑，且用獨立的 `OUTPUT_ROOT`（如暫時修改 config 或用 `--config` 指向複本）避免覆寫既有基準結果。

---

## File Structure

```
utils/visualization.py   [修改] 新增 draw_track_box()
track_eval.py            [修改] 新增 ReidCropCollector、VideoWriterWithFfmpeg；
                          VideoSequenceTracker.run() 拆分；TrackingEvalPipeline.run_stage1()/
                          run_stage3() 拆分
```

不動 `module/*`、`evaluation/*`（純呼叫既有的 `evaluation/dataset_layout.py`，不修改它）。

---

### Task 1: `utils/visualization.py::draw_track_box()` + `VideoSequenceTracker._annotate_frame()`

**Files:**
- Modify: `utils/visualization.py`（新增函式）
- Modify: `track_eval.py`（`VideoSequenceTracker` 新增 `_annotate_frame()` 方法，`run()` 內對應區塊改呼叫它）

**Interfaces:**
- Consumes: 無新依賴（`ultralytics.utils.plotting.Annotator`/`colors` 已在 `track_eval.py` 匯入）。
- Produces（供 Task 4 的 `run()` 拆分使用）: `draw_track_box(annotator, box, det_corner, color, *, embed_label=None, overlay_label=None, overlay_anchor=None) -> None`、`VideoSequenceTracker._annotate_frame(self, frame, tracks, unmatched, removed, frame_id) -> np.ndarray`。

- [ ] **Step 1: 新增 `draw_track_box()`**

在 `utils/visualization.py` 新增：

```python
def draw_track_box(
    annotator,
    box,
    det_corner,
    color,
    *,
    embed_label: str | None = None,
    overlay_label: str | None = None,
    overlay_anchor: tuple[int, int] | None = None,
) -> None:
    """
    統一的「多邊形過濾 → 畫框 → 標註」邏輯，取代 VideoSequenceTracker.run() 內
    matched/unmatched/removed 三段重複程式碼。

    - det_corner 為 None 時不做多邊形過濾（全部畫）；不為 None 時，只在框中心點落在
      多邊形內才畫（沿用現行 cv2.pointPolygonTest 判斷）。
    - embed_label：文字直接嵌入 box_label()（matched track 目前的畫法：框內嵌 track_id）。
    - overlay_label + overlay_anchor：額外用 annotator.text() 疊加文字（unmatched/removed
      目前的畫法：紅框/黑框 + 框外文字標籤）。兩種模式互斥，由呼叫方依三種情境各自決定。

    Parameters
    ----------
    box : tuple[float, float, float, float]
        (x1, y1, x2, y2)，像素座標。
    """
    x1, y1, x2, y2 = box

    if det_corner is not None and len(det_corner) > 0:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if cv2.pointPolygonTest(det_corner, (float(cx), float(cy)), False) < 0:
            return

    annotator.box_label([x1, y1, x2, y2], embed_label or "", color=color)

    if overlay_label is not None and overlay_anchor is not None:
        annotator.text(
            overlay_anchor, overlay_label,
            txt_color=(255, 255, 255), anchor="top",
            box_color=color,
        )
```

- [ ] **Step 2: 驗證 `draw_track_box()` 的過濾/畫框邏輯（用假 Annotator 記錄呼叫，不需真的開視窗）**

```python
# scratchpad/test_draw_track_box.py — 手動驗證腳本，不進版控
import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/moris/Dcoker_Project/Project/safetrajai_eval")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from utils.visualization import draw_track_box


class FakeAnnotator:
    def __init__(self):
        self.box_calls = []
        self.text_calls = []

    def box_label(self, box, label, color):
        self.box_calls.append((tuple(box), label, color))

    def text(self, anchor, label, txt_color, anchor_pos=None, box_color=None, **kwargs):
        self.text_calls.append((anchor, label, box_color))


# 1. det_corner=None：一律畫框（matched 模式，embed_label）
ann = FakeAnnotator()
draw_track_box(ann, (10, 10, 20, 20), None, (0, 255, 0), embed_label="7")
assert ann.box_calls == [((10, 10, 20, 20), "7", (0, 255, 0))]
assert ann.text_calls == []

# 2. det_corner 有值、框中心點在多邊形內：畫框（overlay 模式）
polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
ann2 = FakeAnnotator()
draw_track_box(
    ann2, (10, 10, 20, 20), polygon, (0, 0, 255),
    overlay_label="unmat: car", overlay_anchor=(20, 20),
)
assert ann2.box_calls == [((10, 10, 20, 20), "", (0, 0, 255))]
assert ann2.text_calls == [((20, 20), "unmat: car", (0, 0, 255))]

# 3. det_corner 有值、框中心點在多邊形外：完全不畫
ann3 = FakeAnnotator()
draw_track_box(
    ann3, (200, 200, 220, 220), polygon, (0, 0, 0),
    overlay_label="Rm: 3", overlay_anchor=(220, 200),
)
assert ann3.box_calls == []
assert ann3.text_calls == []

print("ALL OK")
```

Run: `python3 scratchpad/test_draw_track_box.py`
Expected: `ALL OK`

- [ ] **Step 3: 在 `track_eval.py` 匯入 `draw_track_box`**

Read 現有 `track_eval.py:68`：
```python
from utils.visualization import draw_polygon_and_save, save_fst_frame
```

改為：
```python
from utils.visualization import draw_polygon_and_save, save_fst_frame, draw_track_box
```

- [ ] **Step 4: 新增 `VideoSequenceTracker._annotate_frame()`，改寫三段畫框邏輯**

Read 現有 `track_eval.py`（用 `grep -n "annotator = Annotator"` 定位，`track_eval.py:275` 起，縮排在 `if self.save or self.show:` 底下的整段，約 40 行）：
```python
            # 繪製標注
            if self.save or self.show:
                annotator = Annotator(frame, line_width=2)

                for track in tracks:
                    if len(track) >= 7:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])

                        # 如果有指定偵測區域多邊形 (det_corner is not None 且長度 > 0)
                        if self.det_corner is not None and len(self.det_corner) > 0:
                            # 計算目標 (track) 邊框的中心點 (cx, cy)
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            # 使用 cv2.pointPolygonTest 判斷中心點是否在多邊形內
                            result = cv2.pointPolygonTest(self.det_corner, (float(cx), float(cy)), False)
                            if result >= 0:
                                # 如果中心點在多邊形內，則畫框和 id 標記
                                annotator.box_label(
                                    [x1, y1, x2, y2], f"{track_id}", color=colors(track_id, True)
                                )
                        # 如果沒有偵測區域 (det_corner is None)，則對所有目標都畫框和 id 標記
                        elif self.det_corner is None:
                            annotator.box_label(
                                    [x1, y1, x2, y2], f"{track_id}", color=colors(track_id, True)
                                )



                if self.show_unmatched:
                    for track in unmatched:
                        if len(track) >= 7:
                            x1, y1, x2, y2 = track[:4]
                            score    = track[5]
                            cls_id   = int(track[6])
                            cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                            # label_id = f"{cls_name}-{score:.2f}"
                            label_id = f"unmat: {cls_name}"
                            # 判斷是否有指定偵測區域 (det_corner 不為 None 且長度大於 0)
                            if self.det_corner is not None and len(self.det_corner) > 0:
                                # 計算未配對目標邊框的中心點 (cx, cy)
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                # 使用 cv2.pointPolygonTest 判斷中心點是否落於多邊形內
                                result = cv2.pointPolygonTest(self.det_corner, (float(cx), float(cy)), False)
                                if result >= 0:
                                    # 若在多邊形內，畫紅色方框及顯示標籤 (unmatched)
                                    annotator.box_label([x1, y1, x2, y2], "", color=(0, 0, 255))
                                    annotator.text(
                                        (int(x2), int(y2)), label_id,
                                        txt_color=(255, 255, 255), anchor="top",
                                        box_color=(0, 0, 255),
                                    )
                            # 若未指定偵測區域，則對所有 unmatched 畫紅框與標籤
                            elif self.det_corner is None:
                                annotator.box_label([x1, y1, x2, y2], "", color=(0, 0, 255))
                                annotator.text(
                                    (int(x2), int(y2)), label_id,
                                    txt_color=(255, 255, 255), anchor="top",
                                    box_color=(0, 0, 255),
                                )


                if self.show_removed:
                    for track in removed:
                        if len(track) >= 7:
                            x1, y1, x2, y2 = track[:4]
                            track_id = int(track[4])
                            score    = track[5]
                            # label = f"{track_id}-{score:.2f}"
                            label = f"Rm: {track_id}"
                            # 判斷是否有指定偵測區域 (det_corner 不為 None 且長度大於 0)
                            if self.det_corner is not None and len(self.det_corner) > 0:
                                # 計算被移除目標的邊框中心點座標
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                # 判斷中心點是否落在多邊形內 (result >= 0 代表在內)
                                result = cv2.pointPolygonTest(self.det_corner, (float(cx), float(cy)), False)
                                if result >= 0:
                                    # 在移除目標的邊框畫上黑色方框與標籤
                                    annotator.box_label([x1, y1, x2, y2], "", color=(0, 0, 0))
                                    annotator.text(
                                        (int(x1), int(y1)), label,
                                        txt_color=(255, 255, 255), anchor="top",
                                        box_color=(0, 0, 0),
                                    )
                            # 若未指定偵測區域，則對所有 removed 畫黑框與標籤
                            elif self.det_corner is None:
                                annotator.box_label([x1, y1, x2, y2], "", color=(0, 0, 0))
                                annotator.text(
                                    (int(x1), int(y1)), label,
                                    txt_color=(255, 255, 255), anchor="top",
                                    box_color=(0, 0, 0),
                                )


                annotated_frame = annotator.result()

                # 在左上角顯示幀號
                # text = f"frame: {frame_id}"
                text = f"frame: {show_frame_id}"
                org  = (8, 22)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, ft = 0.6, 1
                (tw, th), bl = cv2.getTextSize(text, font, fs, ft)
                cv2.rectangle(
                    annotated_frame,
                    (org[0] - 3, org[1] - th - 3),
                    (org[0] + tw + 3, org[1] + bl + 3),
                    (0, 0, 0), -1,
                )
                cv2.putText(annotated_frame, text, org, font, fs, (0, 255, 0), ft, cv2.LINE_AA)
```

改為（先新增方法，Step 5 才處理呼叫端）——在 `VideoSequenceTracker` class 內新增方法：

```python
    def _annotate_frame(self, frame, tracks, unmatched, removed, frame_id) -> np.ndarray:
        """呼叫 utils.visualization.draw_track_box() 分別處理 matched/unmatched/removed，
        取代原本內嵌的三段重複迴圈。左上角幀號文字邏輯不變。"""
        annotator = Annotator(frame, line_width=2)

        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                draw_track_box(
                    annotator, (x1, y1, x2, y2), self.det_corner,
                    color=colors(track_id, True),
                    embed_label=f"{track_id}",
                )

        if self.show_unmatched:
            for track in unmatched:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[:4]
                    cls_id   = int(track[6])
                    cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                    draw_track_box(
                        annotator, (x1, y1, x2, y2), self.det_corner,
                        color=(0, 0, 255),
                        overlay_label=f"unmat: {cls_name}",
                        overlay_anchor=(int(x2), int(y2)),
                    )

        if self.show_removed:
            for track in removed:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[:4]
                    track_id = int(track[4])
                    draw_track_box(
                        annotator, (x1, y1, x2, y2), self.det_corner,
                        color=(0, 0, 0),
                        overlay_label=f"Rm: {track_id}",
                        overlay_anchor=(int(x1), int(y1)),
                    )

        annotated_frame = annotator.result()

        # 在左上角顯示幀號
        text = f"frame: {frame_id}"
        org  = (8, 22)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, ft = 0.6, 1
        (tw, th), bl = cv2.getTextSize(text, font, fs, ft)
        cv2.rectangle(
            annotated_frame,
            (org[0] - 3, org[1] - th - 3),
            (org[0] + tw + 3, org[1] + bl + 3),
            (0, 0, 0), -1,
        )
        cv2.putText(annotated_frame, text, org, font, fs, (0, 255, 0), ft, cv2.LINE_AA)

        return annotated_frame
```

- [ ] **Step 5: 呼叫端改用 `_annotate_frame()`**

把 Step 4 引用的原始區塊整段替換為：
```python
            # 繪製標注
            if self.save or self.show:
                annotated_frame = self._annotate_frame(frame, tracks, unmatched, removed, show_frame_id)
```

- [ ] **Step 6: 語法檢查**

Run: `python3 -m py_compile utils/visualization.py track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 7: 手動驗證——`save: true` 跑 1 支序列，肉眼比對 `Demo.mp4` 標註畫面**

在對應 config 開啟 `save: true`（或用 `--config` 指向一份暫時開啟 `save` 的複本），跑：
```bash
python track_eval.py --config config/bytetrack.yaml
```
Expected：`Demo.mp4` 的標註畫面與重構前一致——matched track 框內嵌 track_id、（若 `show_unmatched`/`show_removed` 開啟）unmatched 紅框+`unmat: <class>` 文字、removed 黑框+`Rm: <id>` 文字，位置/顏色不變。

- [ ] **Step 8: Commit**

```bash
git add utils/visualization.py track_eval.py
git commit -m "Add draw_track_box(), extract VideoSequenceTracker._annotate_frame()

Unifies matched/unmatched/removed drawing logic that was duplicated three
times inline. Visual output unchanged (verified against config/bytetrack.yaml
Demo.mp4 output).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `ReidCropCollector` 協作物件

**Files:**
- Modify: `track_eval.py`（新增 `ReidCropCollector` class；`VideoSequenceTracker.__init__` 建立實例；`run()` 內對應區塊改呼叫它）

**Interfaces:**
- Consumes: `module.tracker.CLASS_NAMES`（已匯入）。
- Produces（供 Task 4 使用）: `ReidCropCollector(crop_dir, det_corner)`，方法 `collect(tracks, clean_frame, seq_stem, frame_id, width, height) -> None`。

- [ ] **Step 1: 新增 `ReidCropCollector` class**

在 `track_eval.py` 的 `VideoSequenceTracker` class 之前新增：

```python
class ReidCropCollector:
    """蒐集 Re-ID 訓練用裁切圖（save_reid_crops=True 時啟用）。

    存到 crop_dir/{class}/{seq_stem}_track{track_id}/{frame_id:06d}.jpg，
    用「seq_stem_track_id」當身分資料夾，避免不同影片的 track_id 重來
    （tracker 每支影片重建一次）互相撞名。
    """

    def __init__(self, crop_dir: Path, det_corner: np.ndarray | None):
        self.crop_dir = crop_dir
        self.det_corner = det_corner

    def collect(
        self,
        tracks,
        clean_frame: np.ndarray,
        seq_stem: str,
        frame_id: int,
        width: int,
        height: int,
    ) -> None:
        """對每個 track：多邊形過濾（與 draw/annotate 邏輯一致）→ 裁切 → 存檔。"""
        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                cls_id   = int(track[6])

                if self.det_corner is not None and len(self.det_corner) > 0:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if cv2.pointPolygonTest(self.det_corner, (float(cx), float(cy)), False) < 0:
                        continue

                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(width, int(x2)), min(height, int(y2))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue

                crop = clean_frame[yi1:yi2, xi1:xi2]
                cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                identity_dir = self.crop_dir / cls_name / f"{seq_stem}_track{track_id}"
                identity_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(identity_dir / f"{frame_id:06d}.jpg"), crop)
```

- [ ] **Step 2: `VideoSequenceTracker.__init__` 建立 `self._reid_collector`**

Read 現有 `track_eval.py:128-130`（`__init__` 尾端）：
```python
        self.det_corner = det_corner
        self.save_reid_crops = save_reid_crops
        self.reid_crop_dir = Path(reid_crop_dir) if reid_crop_dir else None
```

改為：
```python
        self.det_corner = det_corner
        self.save_reid_crops = save_reid_crops
        self.reid_crop_dir = Path(reid_crop_dir) if reid_crop_dir else None
        self._reid_collector = (
            ReidCropCollector(self.reid_crop_dir, det_corner)
            if save_reid_crops and self.reid_crop_dir is not None
            else None
        )
```

- [ ] **Step 3: `run()` 內改呼叫 `self._reid_collector.collect(...)`**

Read 現有 `track_eval.py:245-269`（用 `grep -n "Re-ID 訓練資料"` 定位）：
```python
            # Re-ID 訓練資料：把每個追蹤到的物件裁切存檔
            # 存到 reid_crop_dir/{class}/{影片名}_track{track_id}/{frame:06d}.jpg
            # 用「影片名_track_id」當身分資料夾，避免不同影片的 track_id 重來（tracker 每支影片重建一次）互相撞名
            if self.save_reid_crops and self.reid_crop_dir is not None:
                for track in tracks:
                    if len(track) >= 7:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        cls_id   = int(track[6])

                        # 是否落在偵測區域多邊形內（與 draw/annotate 邏輯一致）
                        if self.det_corner is not None and len(self.det_corner) > 0:
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            if cv2.pointPolygonTest(self.det_corner, (float(cx), float(cy)), False) < 0:
                                continue

                        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                        xi2, yi2 = min(width, int(x2)), min(height, int(y2))
                        if xi2 <= xi1 or yi2 <= yi1:
                            continue

                        crop = clean_frame[yi1:yi2, xi1:xi2]
                        cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                        identity_dir = self.reid_crop_dir / cls_name / f"{stem_name}_track{track_id}"
                        identity_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(identity_dir / f"{show_frame_id:06d}.jpg"), crop)
```

改為：
```python
            if self._reid_collector is not None:
                self._reid_collector.collect(tracks, clean_frame, stem_name, show_frame_id, width, height)
```

- [ ] **Step 4: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 5: 手動驗證——`save_reid_crops: true` 跑 1 支序列，比對裁切圖數量/檔名**

用 `config/botsort_save_reid_crops.yaml`（限量 1 支序列），重構前後各跑一次，比對 `dataset/reid/crop_dataset/<class>/<seq>_track<id>/` 底下的檔案數量與檔名。

Run: `python track_eval.py --config config/botsort_save_reid_crops.yaml`
Expected: 裁切圖數量、檔名（`{frame_id:06d}.jpg`）與重構前一致。

- [ ] **Step 6: Commit**

```bash
git add track_eval.py
git commit -m "Extract ReidCropCollector from VideoSequenceTracker.run()

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `VideoWriterWithFfmpeg` 協作物件

**Files:**
- Modify: `track_eval.py`（新增 `VideoWriterWithFfmpeg` class；`run()` 內影片寫檔/ffmpeg 轉檔區塊改呼叫它）

**Interfaces:**
- Consumes: 無新依賴（`cv2`、`os` 已匯入）。
- Produces（供 Task 4 使用）: `VideoWriterWithFfmpeg(output_path, fps, width, height)`，方法 `write(frame) -> None`、`close() -> None`。

- [ ] **Step 1: 新增 `VideoWriterWithFfmpeg` class**

在 `track_eval.py` 的 `ReidCropCollector` 之後新增：

```python
class VideoWriterWithFfmpeg:
    """包裝 cv2.VideoWriter，close() 時額外做 ffmpeg H.264 轉檔
    （失敗則保留原始 mp4v 檔並印警告，行為與現行一致）。

    ⚠️ 建構子的 cv2.VideoWriter fps 沿用現行寫死值 5（不是傳入的 fps 參數）——
    這是既有行為，重構時不修正，只搬移位置。
    """

    def __init__(self, output_path: Path, fps: int, width: int, height: int):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(output_path), fourcc, 5, (width, height))
        print(f"Output video : {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
        if not self.output_path.exists():
            return

        ffmpeg_out = self.output_path.with_stem(self.output_path.stem + "_h264")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{self.output_path}" '
            f'-vcodec libx264 -pix_fmt yuv420p -acodec aac "{ffmpeg_out}"'
        )
        print(f"  ffmpeg    : {ffmpeg_cmd}")
        ret = os.system(ffmpeg_cmd)
        if ret == 0 and ffmpeg_out.exists():
            self.output_path.unlink()
            ffmpeg_out.rename(self.output_path)
            print(f"  video     : {self.output_path}")
        else:
            print("  ffmpeg conversion failed; keeping original mp4v file.")
```

- [ ] **Step 2: `run()` 內影片寫檔設定改用 `VideoWriterWithFfmpeg`**

Read 現有 `track_eval.py:175-185`（用 `grep -n "影片輸出設定"` 定位）：
```python
        # ── 影片輸出設定 ─────────────────────────────────────────
        video_writer     = None
        output_video_path = None
        if self.save:
            output_video_path = output_path / "Demo.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                # str(output_video_path), fourcc, fps, (width, height)
                str(output_video_path), fourcc, 5, (width, height)
            )
            print(f"Output video : {output_video_path}")
```

改為：
```python
        # ── 影片輸出設定 ─────────────────────────────────────────
        video_writer = None
        if self.save:
            video_writer = VideoWriterWithFfmpeg(output_path / "Demo.mp4", fps, width, height)
```

- [ ] **Step 3: 主迴圈寫入呼叫不變**

Read 現有：
```python
            # 如果已開啟 video_writer，將多邊形疊加到 annotated_frame 並寫入影片檔
            if video_writer:
                # 疊加偵測多邊形（不保存，只進行繪製）
                draw_polygon_and_save(annotated_frame, pts=self.det_corner, output_path=None, save_img=False)
                # 將已標注與多邊形的影格寫到影片
                video_writer.write(annotated_frame)
```

不變（`VideoWriterWithFfmpeg.write()` 與 `cv2.VideoWriter.write()` 呼叫方式一致，這段程式碼不用改）。

- [ ] **Step 4: 迴圈結束後的釋放/ffmpeg 轉檔改用 `.close()`**

Read 現有 `track_eval.py:413-417`（用 `grep -n "cap.release()"` 定位，`run()` 尾端第一個 `cap.release()`）：
```python
        cap.release()
        if video_writer:
            video_writer.release()
        if self.show:
            cv2.destroyAllWindows()
```

改為：
```python
        cap.release()
        if video_writer:
            video_writer.close()
        if self.show:
            cv2.destroyAllWindows()
```

刪除原本尾端的 ffmpeg 轉檔區塊（`track_eval.py:434-448`，用 `grep -n "ffmpeg 轉檔"` 定位）：
```python
        # ffmpeg 轉檔（H.264 相容）
        if self.save and output_video_path and output_video_path.exists():
            ffmpeg_out = output_video_path.with_stem(output_video_path.stem + "_h264")
            ffmpeg_cmd = (
                f'ffmpeg -y -i "{output_video_path}" '
                f'-vcodec libx264 -pix_fmt yuv420p -acodec aac "{ffmpeg_out}"'
            )
            print(f"  ffmpeg    : {ffmpeg_cmd}")
            ret = os.system(ffmpeg_cmd)
            if ret == 0 and ffmpeg_out.exists():
                output_video_path.unlink()
                ffmpeg_out.rename(output_video_path)
                print(f"  video     : {output_video_path}")
            else:
                print("  ffmpeg conversion failed; keeping original mp4v file.")
```
整段刪除（邏輯已搬進 `VideoWriterWithFfmpeg.close()`，由 Step 4 前段的 `video_writer.close()` 觸發）。

- [ ] **Step 5: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 6: 手動驗證——`save: true` 跑 1 支序列，確認 `Demo.mp4` 產生且 ffmpeg 轉檔訊息與重構前一致**

Run: `python track_eval.py --config config/bytetrack.yaml`（`save: true`）
Expected: console 印出 `Output video : .../Demo.mp4`、`ffmpeg    : ffmpeg -y -i ...`、`video     : .../Demo.mp4`（或轉檔失敗時的警告訊息），`Demo.mp4` 檔案存在且可播放。

- [ ] **Step 7: Commit**

```bash
git add track_eval.py
git commit -m "Extract VideoWriterWithFfmpeg from VideoSequenceTracker.run()

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `VideoSequenceTracker.run()` 拆分為協調方法

**Files:**
- Modify: `track_eval.py`（`VideoSequenceTracker` 新增 `_open_video()`/`_prepare_output_dirs()`/`_process_frame()`/`_run_frame_loop()`/`_print_summary()`；`run()` 改寫為純協調）

**Interfaces:**
- Consumes: `_annotate_frame()`（Task 1）、`self._reid_collector`（Task 2）、`VideoWriterWithFfmpeg`（Task 3）、`evaluation.dataset_layout`（已於 `track_eval.py` 頂部匯入，見 `track_eval.py:72`：`from evaluation import dataset_layout`）。
- Produces: `run()` 對外簽章、回傳值（`output_path, images_path, labels_path`）與行為完全不變。

此為本計畫風險最高的一步——把 Task 1-3 完成後仍是單一大迴圈的 `run()` 拆成多個私有方法。拆分只搬移程式碼位置，不改變任何判斷式或計算邏輯。

- [ ] **Step 1: 新增 `_open_video()`**

在 `VideoSequenceTracker` class 內（`__init__` 之後）新增：

```python
    def _open_video(self, video_path: str):
        """開啟影片來源，回傳 (cap, fps, width, height, total_frames)。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video        : {video_path}")
        print(f"Video        : {width}x{height} @ {fps}fps, {total_frames} frames")

        return cap, fps, width, height, total_frames
```

- [ ] **Step 2: 新增 `_prepare_output_dirs()`（呼叫 `dataset_layout`，OUTPUT 端序列目錄無前綴）**

```python
    def _prepare_output_dirs(self, output_dir: Path, video_path: str):
        """建立 <seq>/、labels/、images/ 目錄，複製原始影片，
        並設定 self._stem_name（縮短過的檔名前綴，供 label/image/reid 檔名使用）。

        回傳 (output_path, images_path, labels_path)。OUTPUT 端序列目錄不含前綴
        （見 evaluation/dataset_layout.py），SOURCE 端既有命名不受影響。
        """
        seq_stem    = dataset_layout.seq_name_from_video_path(video_path)
        output_path = dataset_layout.stage1_seq_dir(output_dir, seq_stem)
        output_path.mkdir(parents=True, exist_ok=True)

        labels_path = output_path / "labels" / seq_stem
        labels_path.mkdir(parents=True, exist_ok=True)

        images_path = output_path / "images" / seq_stem
        images_path.mkdir(parents=True, exist_ok=True)

        # 將原始影片複製一份到輸出目錄（供 seqinfo.ini 參考）
        video_dst = output_path / Path(video_path).name
        if not video_dst.exists():
            shutil.copy(video_path, video_dst)

        # 縮短 stem 用於檔案命名（去掉中文地名）
        self._stem_name = seq_stem.replace(" 中華三路 民生二路(東北角)_", "_")

        return output_path, images_path, labels_path
```

- [ ] **Step 3: 新增 `_process_frame()`**

```python
    def _process_frame(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        frame_id: int,
        images_path: Path,
        labels_path: Path,
        width: int,
        height: int,
    ) -> np.ndarray | None:
        """偵測 → 追蹤 → 存 label/image → reid 裁切蒐集 → 回傳（若需要）已標註的 frame。

        frame_id 是逐幀遞增的計數器（對應現行的 show_frame_id），用於輸出檔名與進度輸出。
        """
        st = time.time()
        boxes = self.detector.detect(frame, iou=self.iou, imgsz=self.imgsz)
        en = time.time()
        self._avg_pred_time.append(en - st)

        if len(boxes) > 0:
            tracks, unmatched, removed = self.tracker.update(boxes.cpu().numpy(), frame)
        else:
            tracks = unmatched = removed = np.array([])

        tr_st = time.time()
        self._avg_track_time.append(tr_st - en)

        if self.save_txt:
            label_file = labels_path / f"{self._stem_name}_f{frame_id:06d}.txt"
            img_file   = images_path / f"{self._stem_name}_f{frame_id:06d}.jpg"

            cv2.imwrite(str(img_file), clean_frame)
            with open(label_file, "w") as f:
                for track in tracks:
                    if len(track) >= 7:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        score    = track[5]
                        cls_id   = int(track[6])
                        cx = (x1 + x2) / 2 / width
                        cy = (y1 + y2) / 2 / height
                        w  = (x2 - x1) / width
                        h  = (y2 - y1) / height
                        f.write(
                            f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                            f" {track_id} {score:.4f}\n"
                        )

        if self._reid_collector is not None:
            self._reid_collector.collect(tracks, clean_frame, self._stem_name, frame_id, width, height)

        annotated_frame = None
        if self.save or self.show:
            annotated_frame = self._annotate_frame(frame, tracks, unmatched, removed, frame_id)

        if frame_id % 10 == 0:
            print(
                f"  [{frame_id}/{self._total_frames}] "
                f"pred={self._avg_pred_time[-1]:.3f}s  "
                f"track={self._avg_track_time[-1]:.3f}s"
            )

        return annotated_frame
```

- [ ] **Step 4: 新增 `_run_frame_loop()`**

```python
    def _run_frame_loop(
        self,
        cap,
        output_path: Path,
        images_path: Path,
        labels_path: Path,
        video_writer,
        width: int,
        height: int,
        total_frames: int,
    ) -> None:
        """逐幀讀取、處理、（若需要）顯示/寫入影片，直到影片結束或使用者按 'q'。"""
        self._total_frames   = total_frames
        self._avg_pred_time  = []
        self._avg_track_time = []

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            clean_frame = deepcopy(frame)

            if frame_id == 0:
                draw_polygon_and_save(clean_frame, output_path, pts=self.det_corner, save_img=True)

            frame_id += 1

            annotated_frame = self._process_frame(
                frame, clean_frame, frame_id, images_path, labels_path, width, height,
            )

            if self.show and annotated_frame is not None:
                cv2.imshow("Class-Aware Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if video_writer and annotated_frame is not None:
                draw_polygon_and_save(annotated_frame, pts=self.det_corner, output_path=None, save_img=False)
                video_writer.write(annotated_frame)
```

- [ ] **Step 5: 新增 `_print_summary()`**

```python
    def _print_summary(self) -> None:
        COLOR_RESET   = "\033[0m"
        COLOR_YELLOW  = "\033[33m"
        COLOR_CYAN    = "\033[36m"
        COLOR_MAGENTA = "\033[35m"
        COLOR_GREEN   = "\033[32m"

        tracker_name_str = f"{COLOR_MAGENTA}[Algorithm]{COLOR_RESET} {COLOR_CYAN}{self.tracker_type.upper()}{COLOR_RESET}"
        print(f"\n{COLOR_YELLOW}Tracking complete — {len(self._avg_track_time)} frames{COLOR_RESET}")
        print(f"  {COLOR_MAGENTA}pred  avg{COLOR_RESET} : {COLOR_GREEN}{sum(self._avg_pred_time)  / len(self._avg_pred_time):.4f}s{COLOR_RESET}")
        print(f"  {COLOR_MAGENTA}track avg{COLOR_RESET} : {COLOR_GREEN}{sum(self._avg_track_time) / len(self._avg_track_time):.4f}s{COLOR_RESET}")
        print(tracker_name_str)
```

- [ ] **Step 6: 改寫 `run()` 為純協調**

Read 現有 `run()`（Task 1-3 完成後的完整內容，即目前 `track_eval.py` 裡從 `def run(self, video_path...)` 到 `return output_path, images_path, labels_path` 的整段，`track_eval.py:132` 起），整段替換為：

```python
    def run(self, video_path: str, output_dir: Path) -> tuple[Path, Path, Path]:
        """
        將所有結果輸出到 output_dir。

        Returns
        -------
        output_path : Path
            <seq> 目錄（OUTPUT 端無前綴，見 evaluation/dataset_layout.py）
        images_path : Path
            output_path/images/<seq>
        labels_path : Path
            output_path/labels/<seq>
        """
        cap, fps, width, height, total_frames = self._open_video(video_path)
        output_path, images_path, labels_path = self._prepare_output_dirs(output_dir, video_path)

        video_writer = None
        if self.save:
            video_writer = VideoWriterWithFfmpeg(output_path / "Demo.mp4", fps, width, height)

        self._run_frame_loop(
            cap, output_path, images_path, labels_path, video_writer, width, height, total_frames,
        )

        cap.release()
        if video_writer:
            video_writer.close()
        if self.show:
            cv2.destroyAllWindows()

        self._print_summary()

        return output_path, images_path, labels_path
```

- [ ] **Step 7: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 8: 手動驗證——1-2 支序列完整比對（labels/images/trackers 逐行、Demo.mp4 肉眼）**

Run（重構前後各跑一次，用不同 `OUTPUT_ROOT` 避免互相覆蓋）：
```bash
python track_eval.py --config config/bytetrack.yaml
```
Expected：
- 輸出目錄仍是 `<seq>/`（無前綴），`labels/<seq>/*.txt`、`images/<seq>/*.jpg` 檔案數量與檔名與重構前完全一致。
- `labels/<seq>/*.txt` 內容逐行比對一致（可用 `diff -r` 比對重構前後兩次輸出目錄）。
- `trackers/mot_challenge/.../data/<seq>.txt` 內容一致。
- `Demo.mp4`（若 `save: true`）標註畫面與 Task 1 Step 7 驗證的樣式一致。

- [ ] **Step 9: Commit**

```bash
git add track_eval.py
git commit -m "VideoSequenceTracker.run(): split into orchestration + private methods

_open_video/_prepare_output_dirs/_process_frame/_run_frame_loop/_print_summary.
run()'s external signature, return value, and output format are unchanged
(verified: labels/images/trackers diff-clean against pre-refactor output).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: `TrackingEvalPipeline.run_stage1()` 拆出 `_process_single_video()`

**Files:**
- Modify: `track_eval.py`（`TrackingEvalPipeline` 新增 `_process_single_video()`；`run_stage1()` 迴圈本體改呼叫它）

**Interfaces:**
- Consumes: `VideoSequenceTracker`（Task 1-4 已拆分完成）、`evaluation.dataset_layout`。
- Produces: `run_stage1()` 對外簽章、回傳值（`det_output_dir`）與行為不變。

- [ ] **Step 1: 新增 `_process_single_video()`**

Read 現有 `run_stage1()` 迴圈本體（`track_eval.py:493` 起，用 `grep -n "for video_path in target_mp4s"` 定位）：
```python
        for video_path in target_mp4s:
            print(f"\n{'='*70}")
            print(f"  VIDEO: {video_path}")
            print(f"{'='*70}")

            seq_tracker = VideoSequenceTracker(
                model_path   = cfg.det_model_path,
                tracker_type = cfg.tracker,
                conf         = cfg.conf,
                iou          = cfg.iou,
                imgsz        = cfg.imgsz,
                frame_rate   = cfg.frame_rate,
                gmc_method   = cfg.gmc_method,
                save         = cfg.save,
                save_txt     = cfg.save_txt,
                show         = cfg.show,
                show_unmatched = cfg.show_unmatched,
                show_removed   = cfg.show_removed,
                det_corner    = cfg.det_corner,
                save_reid_crops = cfg.save_reid_crops,
                reid_crop_dir   = cfg.reid_crop_dir,
            )
            output_path, images_path, labels_path = seq_tracker.run(video_path, det_output_dir)

            # 取影片尺寸（從已複製的來源影片）
            cap    = cv2.VideoCapture(video_path)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # 儲存 MOT 格式（同時在 output_path/trackers/ 下建立 per-class txt）
            save_mot_format(
                imgs_path   = images_path,
                labels_path = labels_path,
                output_file = output_path / f"{output_path.stem}.txt",
                img_width   = width,
                img_height  = height,
                det_corner  = cfg.det_corner,
            )

            # 生成 DET data.yaml
            make_det_eval_dataset(output_path, CLASS_NAMES)

            # 將 trackers/ 由 annotations_all8 複製到 runs/track_eval/mot_annotations_all8/<seq>/trackers/
            mot_seq_dir = dataset_layout.stage2_seq_dir(track_output_dir, output_path.stem)
            shutil.copytree(
                output_path / "trackers",  # annotations_all8/<seq>
                mot_seq_dir / "trackers",  # mot_annotations_all8/<seq>
                dirs_exist_ok=True,
            )
            # 準備 GT 驗證資料集（SOURCE 端既有命名不變，source_gt_dir() 回傳值已含尾端 /gt）。
            # 並非所有 dataset 都有 GT（例如 video_seqs 本來就沒有標註，見 config/datasets.yaml），
            # 來源不存在時略過並警告，不當成錯誤中斷整條 pipeline。
            gt_src = dataset_layout.source_gt_dir(cfg.SOURCE_VIDEO_DIR, cfg.run_track_eval_dir, output_path.stem)
            if gt_src.exists():
                shutil.copytree(gt_src, mot_seq_dir / "gt", dirs_exist_ok=True)
            else:
                print(f"  ⚠️  {output_path.stem}: 找不到 GT 來源 {gt_src}，略過（此 dataset 可能本來就沒有 GT 標註）")
            print(f"  trackers → {mot_seq_dir / 'trackers'}")
```

改為（先新增方法，Step 2 才改 `run_stage1()` 呼叫它）——在 `TrackingEvalPipeline` class 內新增：

```python
    def _process_single_video(self, video_path: str, det_output_dir: Path, track_output_dir: Path) -> None:
        """對單一影片：建立 VideoSequenceTracker → run() → save_mot_format() →
        make_det_eval_dataset() → 複製 trackers/ 到 stage2 → 複製 SOURCE GT 到 stage2
        （GT 來源不存在時略過並警告，不中斷整條 pipeline——見本文件 Global Constraints）。
        """
        cfg = self.cfg

        print(f"\n{'='*70}")
        print(f"  VIDEO: {video_path}")
        print(f"{'='*70}")

        seq_tracker = VideoSequenceTracker(
            model_path   = cfg.det_model_path,
            tracker_type = cfg.tracker,
            conf         = cfg.conf,
            iou          = cfg.iou,
            imgsz        = cfg.imgsz,
            frame_rate   = cfg.frame_rate,
            gmc_method   = cfg.gmc_method,
            save         = cfg.save,
            save_txt     = cfg.save_txt,
            show         = cfg.show,
            show_unmatched = cfg.show_unmatched,
            show_removed   = cfg.show_removed,
            det_corner    = cfg.det_corner,
            save_reid_crops = cfg.save_reid_crops,
            reid_crop_dir   = cfg.reid_crop_dir,
        )
        output_path, images_path, labels_path = seq_tracker.run(video_path, det_output_dir)

        # 取影片尺寸（從已複製的來源影片）
        cap    = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 儲存 MOT 格式（同時在 output_path/trackers/ 下建立 per-class txt）
        save_mot_format(
            imgs_path   = images_path,
            labels_path = labels_path,
            output_file = output_path / f"{output_path.stem}.txt",
            img_width   = width,
            img_height  = height,
            det_corner  = cfg.det_corner,
        )

        # 生成 DET data.yaml
        make_det_eval_dataset(output_path, CLASS_NAMES)

        # 將 trackers/ 由 annotations_all8 複製到 runs/track_eval/mot_annotations_all8/<seq>/trackers/
        mot_seq_dir = dataset_layout.stage2_seq_dir(track_output_dir, output_path.stem)
        shutil.copytree(
            output_path / "trackers",  # annotations_all8/<seq>
            mot_seq_dir / "trackers",  # mot_annotations_all8/<seq>
            dirs_exist_ok=True,
        )
        # 準備 GT 驗證資料集（SOURCE 端既有命名不變，source_gt_dir() 回傳值已含尾端 /gt）。
        # 並非所有 dataset 都有 GT（例如 video_seqs 本來就沒有標註，見 config/datasets.yaml），
        # 來源不存在時略過並警告，不當成錯誤中斷整條 pipeline。
        gt_src = dataset_layout.source_gt_dir(cfg.SOURCE_VIDEO_DIR, cfg.run_track_eval_dir, output_path.stem)
        if gt_src.exists():
            shutil.copytree(gt_src, mot_seq_dir / "gt", dirs_exist_ok=True)
        else:
            print(f"  ⚠️  {output_path.stem}: 找不到 GT 來源 {gt_src}，略過（此 dataset 可能本來就沒有 GT 標註）")
        print(f"  trackers → {mot_seq_dir / 'trackers'}")
```

- [ ] **Step 2: `run_stage1()` 迴圈改呼叫 `_process_single_video()`**

把 Step 1 讀到的原始迴圈本體整段替換為：
```python
        for video_path in target_mp4s:
            self._process_single_video(video_path, det_output_dir, track_output_dir)
```

- [ ] **Step 3: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 4: 手動驗證——完整跑一次 stage1（1-2 支序列），確認輸出目錄結構不變**

Run: `python track_eval.py --config config/bytetrack.yaml`
Expected: `runs/track_eval/annotations_all8/<seq>/` 與 `runs/track_eval/mot_annotations_all8/<seq>/` 均正確產生（無前綴），內容與 Task 4 Step 8 驗證的結果一致。

再用 `config/botsort_save_reid_crops_video.yaml`（`video_seqs`，無 GT）跑 1 支序列：
Expected: 印出「找不到 GT 來源...略過」警告，程式正常繼續完成，不拋 `FileNotFoundError`。

- [ ] **Step 5: Commit**

```bash
git add track_eval.py
git commit -m "TrackingEvalPipeline.run_stage1(): extract _process_single_video()

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: `TrackingEvalPipeline.run_stage3()` 新增 `_ensure_stage1_output_exists()`

**Files:**
- Modify: `track_eval.py`（`TrackingEvalPipeline` 新增 `_ensure_stage1_output_exists()`；`run_stage3()` 開頭改呼叫它）

**Interfaces:**
- Consumes: `evaluation.dataset_layout.list_stage1_sequences()`。
- Produces: `run_stage3()` 對外行為（soft-skip 語意：找不到 stage1 輸出時印警告並直接 `return`，不 raise）不變。

- [ ] **Step 1: 新增 `_ensure_stage1_output_exists()`**

Read 現有 `run_stage3()` 開頭（`track_eval.py:572` 起，用 `grep -n "def run_stage3"` 定位）：
```python
    def run_stage3(self):
        """
        ...
        """
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        seq_dirs = dataset_layout.list_stage1_sequences(det_output_dir)
        if not seq_dirs:
            print(f"⚠️  No stage1 output found under: {det_output_dir}（stage3 需要先跑過 stage1）")
            return
```

改為（新增方法 + 呼叫端）——在 `run_stage3()` 之前新增：
```python
    def _ensure_stage1_output_exists(self, det_output_dir: Path) -> bool:
        """檢查 det_output_dir 下是否有任何序列輸出。沒有則印警告並回傳 False——
        這是合法的 soft-skip 情境（見 config/collision_detect.yaml 的「只重跑 stage3、
        沿用之前 stage1 舊輸出」用法：run_stage1: false + run_stage3: true）。
        行為與現行完全一致，只是抽成獨立、可單獨測試的函式。
        """
        seq_dirs = dataset_layout.list_stage1_sequences(det_output_dir)
        if not seq_dirs:
            print(f"⚠️  No stage1 output found under: {det_output_dir}（stage3 需要先跑過 stage1）")
            return False
        return True
```

`run_stage3()` 開頭改為：
```python
    def run_stage3(self):
        """
        ...
        """
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        if not self._ensure_stage1_output_exists(det_output_dir):
            return

        seq_dirs = dataset_layout.list_stage1_sequences(det_output_dir)
```

（`run_stage3()` 內接下來仍需要 `seq_dirs` 這個變數本身，所以檢查通過後重新呼叫一次 `list_stage1_sequences()`——與 `_ensure_stage1_output_exists()` 內部重複一次掃描，但兩者是各自獨立、語意清楚的操作，比起把 `seq_dirs` 當回傳值傳出來、讓函式名稱與回傳型別不一致更符合函式單一職責；序列數量在 `run_stage3()` 執行期間不會變動，重複掃描沒有正確性風險，只有可忽略的效能成本。）

- [ ] **Step 2: 語法檢查**

Run: `python3 -m py_compile track_eval.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 手動驗證——「有」stage1 輸出時 stage3 正常執行**

用 `config/collision_detect.yaml`（`run_stage1: false`、`run_stage3: true`，沿用 Task 5 Step 4 的 stage1 輸出）：
```bash
python track_eval.py --config config/collision_detect.yaml
```
Expected: 不印 `⚠️  No stage1 output found`，正常跑完 stage3。

- [ ] **Step 4: 手動驗證——「沒有」stage1 輸出時 stage3 正確 soft-skip**

暫時指向一個空的 `OUTPUT_ROOT`（或先清空一份複本的 `annotations_all8/` 目錄）跑同一個 config：
Expected: 印出 `⚠️  No stage1 output found under: ...（stage3 需要先跑過 stage1）`，程式正常結束（`return`），不拋例外。

- [ ] **Step 5: Commit**

```bash
git add track_eval.py
git commit -m "TrackingEvalPipeline.run_stage3(): extract _ensure_stage1_output_exists()

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: 完整管線回歸驗證

**Files:**
- 無程式碼修改，純驗證。

**Interfaces:**
- 無。

- [ ] **Step 1: 完整跑一次三階段（1-2 支序列），比對重構前後輸出**

```bash
python track_eval.py --config config/bytetrack.yaml   # stage1 + stage2
```
Expected：
- `labels/`、`images/`、`trackers/` 輸出檔案數量與內容與重構前逐行一致。
- Excel（`trackeval_output/`）的 HOTA/AssA/CLEAR 數值與重構前一致（純結構重構不應影響追蹤/評估數值）。

- [ ] **Step 2: `save: true` + `save_reid_crops: true` 同時開啟，確認兩個協作物件互不干擾**

用一份暫時同時開啟 `save: true` 與 `save_reid_crops: true` 的 config 複本跑 1 支序列：
Expected: `Demo.mp4` 與裁切圖同時正確產生，內容與 Task 1/2 各自的驗證結果一致，彼此不互相覆蓋或遺漏。

- [ ] **Step 3: stage3-only rerun 兩種情況都驗證**

```bash
python track_eval.py --config config/collision_detect.yaml   # 有 stage1 舊輸出
```
Expected: 同 Task 6 Step 3。

清空/指向空目錄後再跑一次：
Expected: 同 Task 6 Step 4。

- [ ] **Step 4: 用 `video_seqs`（無 GT）資料集完整跑一次，確認不 crash**

```bash
python track_eval.py --config config/botsort_save_reid_crops_video.yaml
```
Expected: 印出「找不到 GT 來源...略過」警告後正常完成（`stages.run_stage2: false`，不會跑到 Excel 輸出），exit code 0。

- [ ] **Step 5: 確認 `module/tracker` 核心 matching 邏輯完全未受影響**

Run: `git diff --stat module/tracker/`
Expected: 無輸出（本計畫全程未修改 `module/tracker/` 底下任何檔案）。

- [ ] **Step 6: 清除驗證產生的暫存輸出**

先確認範圍，只刪除本次驗證新增的內容，不動既有 `runs/` 基準結果：
```bash
find runs/track_eval -newer docs/superpowers/plans/2026-09-11-track-eval-pipeline-refactor.md
```
確認清單只包含本次驗證產生的內容後才刪除，若不確定則保留、交由使用者自行清理。

---

## Self-Review

1. **Spec coverage**：`2026-09-03-track-eval-pipeline-refactor-design.md` 第 3.1 節（`draw_track_box()`）→ Task 1；第 3.2 節（`ReidCropCollector`/`VideoWriterWithFfmpeg`）→ Task 2-3；第 3.3 節（`VideoSequenceTracker` 拆分）→ Task 4；第 3.4 節（`run_stage1()` 拆分）→ Task 5；第 3.5 節（stage3 檢查）→ Task 6；第 5 節驗證計畫四點→ Task 7 逐一涵蓋（labels/images/trackers 比對、Demo.mp4 肉眼比對、reid crops 數量、stage3-only 兩種情境、無 GT dataset、Excel 數值一致）。第 0 節「前提依賴」（dataset 路徑重構）：已合併，本版本全面採用 `evaluation/dataset_layout.py`，不再有暫緩段落。
2. **2026-09-11 修訂記錄**：原 Task 1（移除補幀死碼）因 dataset-path-refactor 合併而已完成，本版本移除該 Task 並將原 Task 2-8 重新編號為 Task 1-7；`_prepare_output_dirs()`、`_process_single_video()`、`_ensure_stage1_output_exists()` 三處的程式碼內容與行號參照，已對照合併後的實際 `track_eval.py`（`53a4501`）重新核對。
3. **Placeholder scan**：所有 Task 的程式碼區塊均為完整、可直接貼上的實作，無「TODO」/「依需求調整」等佔位文字。
4. **Type consistency**：`_process_frame(self, frame, clean_frame, frame_id, images_path, labels_path, width, height) -> np.ndarray | None`、`_annotate_frame(self, frame, tracks, unmatched, removed, frame_id) -> np.ndarray`、`_run_frame_loop(...)` 呼叫 `_process_frame`/`_annotate_frame` 時的參數名稱與型別在 Task 1、4 之間一致；`ReidCropCollector.collect(tracks, clean_frame, seq_stem, frame_id, width, height)` 在 Task 2、4 之間一致。
5. **對 spec 草稿的刻意偏離**（已記錄於 Global Constraints，非疏漏）：`_annotate_frame()` 省略 spec 草稿列出的 `total_frames` 參數——現行「左上角幀號」文字只用當前幀號，從未用到總幀數，保留這個參數會是永遠不被使用的死參數。
