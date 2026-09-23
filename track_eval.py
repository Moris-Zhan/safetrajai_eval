"""
track_eval.py — YOLO class-aware tracking + TrackEval evaluation pipeline

目錄結構設計
============
來源（唯讀）:
  SOURCE_VIDEO_DIR/
    annotations_all8/
      annotation_<seq>/
        <seq>.mp4

第一階段輸出（tracking 結果）:
  OUTPUT_ROOT/
    annotations_all8/              ← run_det_eval_dir
      annotation_<seq>/
        images/<seq>/*.jpg
        labels/<seq>/*.txt
        trackers/mot_challenge/CCTV-val/YOLO11x__<class>/data/<seq>.txt
        data.yaml
        <seq>.mp4  (原始影片複本)
        Demo.mp4   (標注影片，save=True 時)
    mot_annotations_all8/          ← run_track_eval_dir
      mot_annotation_<seq>/
        trackers/...  (從 annotations_all8 symlink / copy 過來)

第二階段輸出（eval 結果）:
  OUTPUT_ROOT/
    trackeval_output/              ← trackeval_output_dir
      mot_annotation_<seq>/
        det_metrics_<seq>_new.xlsx
        trackeval_metrics_<seq>_MULTI_new.xlsx

第三階段輸出（碰撞偵測結果，需 stages.run_stage3: true，預設關閉，見 config/collision_detect.yaml）:
  OUTPUT_ROOT/
    collision_output/              ← collision_output_dir
      <seq>/
        ttc_events.json / ttc_events.csv / collision_ttc_log.csv / ttc_collision_heatmap.png
        pet_events.json / pet_events.csv / collision_pet_log.csv / pet_collision_heatmap.png

架構
====
Detector（module/detection/detector.py）  — YOLO 推論 + 逐類別信心過濾
VideoSequenceTracker（本檔）        — 單一影片的 class-aware tracking 迴圈（偵測→追蹤→標註→存檔）
TrackingEvalPipeline（本檔）        — 三階段主流程（stage1 tracking → stage2 TrackEval 評估 → stage3 碰撞偵測）
"""

import os
import time
import shutil
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from safetrajai.detection.detector import Detector
from safetrajai.tracking import create_tracker, CLASS_NAMES
from safetrajai.tracking.config import (
    get_track_config_path,
    load_track_config,
)
from safetrajai.collision import run_collision_detection_for_sequence
from safetrajai.visualization.drawing import draw_polygon_and_save, save_fst_frame, draw_track_box
from safetrajai.evaluation.mot_format import save_mot_format, make_det_eval_dataset
from safetrajai.evaluation.pipeline import run_eval_pipeline
from safetrajai.evaluation.config_loader import load_config
from safetrajai.evaluation import dataset_layout


# =============================================================================
# 第一階段：VideoSequenceTracker — 單一影片的 class-aware tracking
# =============================================================================

def make_min_size_quality_filter(min_width: int, min_height: int) -> Callable[[np.ndarray], bool]:
    """Tier 1 品質過濾器範例：裁切寬或高小於門檻值就捨棄。

    quality_filters 掛載點吃的是 Callable[[np.ndarray], bool]（True=保留），
    用法：ReidCropCollector(crop_dir, det_corner, quality_filters=[make_min_size_quality_filter(20, 20)])
    預設（不傳 quality_filters）行為與現行完全相同，見 openspec/changes/reid-training-data-pipeline/。
    """
    def _filter(crop: np.ndarray) -> bool:
        h, w = crop.shape[:2]
        return w >= min_width and h >= min_height
    return _filter


class ReidCropCollector:
    """蒐集 Re-ID 訓練用裁切圖（save_reid_crops=True 時啟用）。

    存到 crop_dir/{class}/{seq_stem}_track{track_id}/{frame_id:06d}.jpg，
    用「seq_stem_track_id」當身分資料夾，避免不同影片的 track_id 重來
    （tracker 每支影片重建一次）互相撞名。
    """

    def __init__(
        self,
        crop_dir: Path,
        det_corner: np.ndarray | None,
        quality_filters: list[Callable[[np.ndarray], bool]] | None = None,
    ):
        self.crop_dir = crop_dir
        self.det_corner = det_corner
        self.quality_filters = quality_filters

    def collect(
        self,
        tracks,
        clean_frame: np.ndarray,
        seq_stem: str,
        frame_id: int,
        width: int,
        height: int,
    ) -> None:
        """對每個 track：多邊形過濾（與 draw/annotate 邏輯一致）→ 裁切 → 品質過濾（可選）→ 存檔。"""
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

                if self.quality_filters and not all(f(crop) for f in self.quality_filters):
                    continue

                cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                identity_dir = self.crop_dir / cls_name / f"{seq_stem}_track{track_id}"
                identity_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(identity_dir / f"{frame_id:06d}.jpg"), crop)


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


class VideoSequenceTracker:
    """
    對單一影片執行 class-aware tracking。

    對應原本的 run_class_aware_tracking() 函式，改為類別：
    __init__ 建立 Detector + tracker（每支影片各自獨立的偵測器/追蹤器實例，
    與原本函式每次呼叫都重新建立的行為一致），run() 執行逐 frame 迴圈。
    """

    def __init__(
        self,
        model_path: str,
        tracker_type: str = "bytetrack",
        conf: float = 0.15,
        iou: float = 0.5,
        imgsz: int = 1280,
        frame_rate: int = 30,
        gmc_method: str = "sparseOptFlow",
        save: bool = False,
        save_txt: bool = True,
        show: bool = False,
        show_unmatched: bool = False,
        show_removed: bool = False,
        det_corner: np.ndarray = None,
        save_reid_crops: bool = False,
        reid_crop_dir: str = "./dataset/reid/crop_dataset",
    ):
        self.detector = Detector(model_path)
        print(f"Loaded model : {model_path}")

        self.tracker = create_tracker(
            tracker_type=tracker_type,
            frame_rate=frame_rate,
            gmc_method=gmc_method,
            track_high_thresh=conf,
            track_low_thresh=conf * 0.3,
        )
        print(f"Tracker                : {tracker_type} (class-aware)")
        print(f"frame_rate             : {frame_rate} (class-aware)")
        print(f"track_high_thresh      : {conf} (class-aware)")

        self.tracker_type = tracker_type
        self.iou = iou
        self.imgsz = imgsz
        self.save = save
        self.save_txt = save_txt
        self.show = show
        self.show_unmatched = show_unmatched
        self.show_removed = show_removed
        self.det_corner = det_corner
        self.save_reid_crops = save_reid_crops
        self.reid_crop_dir = Path(reid_crop_dir) if reid_crop_dir else None
        self._reid_collector = (
            ReidCropCollector(self.reid_crop_dir, det_corner)
            if save_reid_crops and self.reid_crop_dir is not None
            else None
        )

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


# =============================================================================
# TrackingEvalPipeline — 兩階段主流程
# =============================================================================

class TrackingEvalPipeline:
    """
    stage1：掃描 SOURCE_VIDEO_DIR 下所有影片，逐一執行 class-aware tracking，
            並將 trackers/ 輸出複製到 mot_annotations_all8/ 下供第二階段讀取。
    stage2：把第一階段輸出轉成 TrackEval 期望的目錄結構，計算 HOTA/CLEAR/Identity，
            合併結果輸出 Excel。
    """

    def __init__(self, cfg):
        self.cfg = cfg

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

    def run_stage1(self) -> Path:
        """
        Returns
        -------
        det_output_dir : Path
            annotations_all8 目錄（第二階段的 base_dir/run_det_eval_dir）
        """
        cfg = self.cfg
        det_output_dir   = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir
        track_output_dir = cfg.OUTPUT_ROOT / cfg.run_track_eval_dir
        det_output_dir.mkdir(parents=True, exist_ok=True)
        track_output_dir.mkdir(parents=True, exist_ok=True)

        # 搜尋來源影片（跳過 Demo.mp4）
        mp4s = [str(p) for p in dataset_layout.discover_source_sequences(cfg.SOURCE_VIDEO_DIR, cfg.run_det_eval_dir)]

        if not mp4s:
            print(f"⚠️  No source videos found under: {cfg.SOURCE_VIDEO_DIR / cfg.run_det_eval_dir}")
            return det_output_dir

        print(f"\n✅ Found {len(mp4s)} source video(s)")

        # 取要處理的切片（測試時可調整）
        target_mp4s = mp4s[:]   # ← 改這裡控制要跑哪幾支

        for video_path in target_mp4s:
            self._process_single_video(video_path, det_output_dir, track_output_dir)

        return det_output_dir

    def run_stage2(self):
        """
        第二階段：讀取 OUTPUT_ROOT 內第一階段的產出，跑 TrackEval 評估，
        結果寫到 OUTPUT_ROOT/trackeval_output/ 下。
        """
        cfg = self.cfg

        print(f"\n{'='*70}")
        print(f"  STAGE 2 — TrackEval")
        print(f"  base_dir           : {cfg.base_dir}")
        print(f"  run_det_eval_dir   : {cfg.run_det_eval_dir}")
        print(f"  run_track_eval_dir : {cfg.run_track_eval_dir}")
        print(f"  trackeval_output   : {cfg.trackeval_output_dir}")
        print(f"{'='*70}\n")

        run_eval_pipeline(cfg)

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

    def run_stage3(self):
        """
        第三階段：軌跡碰撞偵測（TTC/PET，module/collision/）。

        讀取 stage1 產生的逐類別軌跡檔（annotation_<seq>/trackers/mot_challenge/
        <benchmark>-<split>/<tracker_name>__<class>/data/<seq>.txt，格式見
        CLAUDE.md「軌跡格式契約」），對每支序列跑一次
        module.collision.run_collision_detection_for_sequence()，輸出寫到
        cfg.collision_output_dir/<seq_name>/ 底下。

        需要 stage1 先跑過（run_det_eval_dir 下要有 annotation_<seq>/trackers/... 與
        first_frame_with_box.jpg），若同一次執行 stages.run_stage1 也是 true，
        會直接讀到本次剛產生的結果；若只想單獨重跑 stage3，把 run_stage1 關掉即可
        沿用上一次的 stage1 輸出。
        """
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        if not self._ensure_stage1_output_exists(det_output_dir):
            return

        seq_dirs = dataset_layout.list_stage1_sequences(det_output_dir)

        print(f"\n{'='*70}")
        print(f"  STAGE 3 — Collision Detection (TTC/PET)")
        print(f"  collision_output   : {cfg.collision_output_dir}")
        print(f"  homography_points  : {cfg.homography_points_dir}")
        print(f"{'='*70}\n")

        for seq_dir in seq_dirs:
            seq_name = seq_dir.stem
            video_path = seq_dir / f"{seq_name}.mp4"
            if not video_path.exists():
                print(f"  ⚠️  {seq_name}: 找不到 {video_path}，略過（stage1 是否有跑完？）")
                continue

            cap = cv2.VideoCapture(str(video_path))
            fps    = cap.get(cv2.CAP_PROP_FPS) or cfg.frame_rate or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            run_collision_detection_for_sequence(
                seq_output_dir=seq_dir,
                cfg=cfg,
                fps=fps,
                bounds=(height, width),
            )

    def run(self):
        """依 config 的 stages.run_stage1 / run_stage2 / run_stage3 決定要跑哪幾階段
        （stage1/stage2 預設都跑，stage3 預設關閉，需在 config 明確開啟）。"""
        if getattr(self.cfg, "run_stage1", True):
            self.run_stage1()
        else:
            print("⏭️  stage1 skipped（config 的 stages.run_stage1: false）")

        if getattr(self.cfg, "run_stage2", True):
            self.run_stage2()
        else:
            print("⏭️  stage2 skipped（config 的 stages.run_stage2: false）")

        if getattr(self.cfg, "run_stage3", False):
            self.run_stage3()
        else:
            print("⏭️  stage3 skipped（config 的 stages.run_stage3: false，預設關閉）")


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO class-aware tracking + TrackEval evaluation"
    )

    # 影片儲存控制
    parser.add_argument("--config", default="config/bytetrack.yaml", help="流程用 config yaml（路徑、eval、class_tracking 等）")
    parser.add_argument(
        "--track-config",
        default=None,
        help=(
            "class-aware tracker 設定（CLASS_TRACKING_PARAMS、置信度門檻等）YAML 路徑。"
            "不指定時使用 --config 內建的 class_tracking 章節。"
        ),
    )

    args = parser.parse_args()

    load_track_config(args.track_config or args.config)
    print(f"Loaded track config: {get_track_config_path()}")

    # 允許 CLI 覆蓋路徑常數
    cfg = load_config(args.config)

    pipeline = TrackingEvalPipeline(cfg)
    pipeline.run()
