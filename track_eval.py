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
import sys
import time
import shutil
import argparse
import glob
from copy import deepcopy
from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.detection.detector import Detector
from module.tracker import create_tracker, CLASS_NAMES
from module.tracker.config import (
    get_track_config_path,
    load_track_config,
)
from module.collision import run_collision_detection_for_sequence
from utils.visualization import draw_polygon_and_save, save_fst_frame
from evaluation.mot_format import save_mot_format, make_det_eval_dataset
from evaluation.pipeline import run_eval_pipeline
from evaluation.config_loader import load_config


# =============================================================================
# 第一階段：VideoSequenceTracker — 單一影片的 class-aware tracking
# =============================================================================

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

    def run(self, video_path: str, output_dir: Path) -> tuple[Path, Path, Path]:
        """
        將所有結果輸出到 output_dir。

        Returns
        -------
        output_path : Path
            annotation_<seq> 目錄
        images_path : Path
            output_path/images/<seq>
        labels_path : Path
            output_path/labels/<seq>
        """
        orig_frames = None

        # 讀取 frame_comparison.json 取得 interpolated_frame_ids
        comparison_json_path = Path(video_path).with_name(f"{Path(video_path).stem}_frame_comparison.json")
        interpolated_frame_ids = set()
        if comparison_json_path.exists():
            with open(comparison_json_path, "r", encoding="utf-8") as f:
                comparison_data = json.load(f)
                # 根據常見格式，假設 {"interpolated_frame_ids": [id1, id2, ...]}
                interpolated_frame_ids = set(comparison_data["interpolated_frame_ids"])
                orig_frames = comparison_data["summary"]["orig_frames"]
        else:
            print(f"Warning: {comparison_json_path} not found. No frames will be excluded.")

        def should_skip_frame(frame_id: int) -> bool:
            return frame_id in interpolated_frame_ids

        # 用法參考：
        # if should_skip_frame(cur_frame_id): continue # 在處理 frame 時跳過

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = int(cap.get(cv2.CAP_PROP_FPS))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video        : {video_path}")
        if orig_frames != None:
            total_frames = orig_frames
            print(f"Video        : {width}x{height} @ {fps}fps, {orig_frames} Original frames")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video        : {width}x{height} @ {fps}fps, {total_frames} frames")

        # Video        : 640x480 @ 4fps, 72 frames
        # Video        : 640x480 @ 30fps, 449 frames
        # print(f"Video        : {width}x{height} @ {fps}fps, {total_frames} frames")

        seq_stem    = Path(video_path).stem
        output_path = output_dir / f"annotation_{seq_stem}"
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
        stem_name = seq_stem.replace(" 中華三路 民生二路(東北角)_", "_")

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

        frame_id      = 0
        avg_pred_time = []
        avg_track_time = []

        # ── 主迴圈 ───────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            clean_frame = deepcopy(frame)  # 預先複製一份原始 frame，供後續標註圖與 image 儲存保持無污染
            st = time.time()  # 記錄推論前的時間，之後可用於性能分析

            if frame_id == 0:
                draw_polygon_and_save(clean_frame, output_path, pts=self.det_corner, save_img=True)

            frame_id += 1

            # 推論
            boxes = self.detector.detect(frame, iou=self.iou, imgsz=self.imgsz)
            en = time.time()
            avg_pred_time.append(en - st)

            # 追蹤
            if len(boxes) > 0:
                tracks, unmatched, removed = self.tracker.update(boxes.cpu().numpy(), frame)
            else:
                tracks = unmatched = removed = np.array([])

            tr_st = time.time()
            # avg_track_time.append(tr_st - en)

            # 檢查是否跳過當前幀
            if should_skip_frame(frame_id):
                continue
            else:
                avg_track_time.append(tr_st - en)
                show_frame_id = len(avg_track_time)
                # 儲存 label / image
                if self.save_txt:
                    # label_file = labels_path / f"{stem_name}_f{frame_id:06d}.txt"
                    # img_file   = images_path / f"{stem_name}_f{frame_id:06d}.jpg"
                    label_file = labels_path / f"{stem_name}_f{show_frame_id:06d}.txt"
                    img_file   = images_path / f"{stem_name}_f{show_frame_id:06d}.jpg"

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

                if self.show:
                    cv2.imshow("Class-Aware Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # 如果已開啟 video_writer，將多邊形疊加到 annotated_frame 並寫入影片檔
                if video_writer:
                    # 疊加偵測多邊形（不保存，只進行繪製）
                    draw_polygon_and_save(annotated_frame, pts=self.det_corner, output_path=None, save_img=False)
                    # 將已標注與多邊形的影格寫到影片
                    video_writer.write(annotated_frame)


                # if frame_id % 10 == 0:
                #     print(
                #         f"  [{frame_id}/{total_frames}] "
                #         f"pred={avg_pred_time[-1]:.3f}s  "
                #         f"track={avg_track_time[-1]:.3f}s"
                #     )

                if show_frame_id % 10 == 0:
                    print(
                        f"  [{show_frame_id}/{total_frames}] "
                        f"pred={avg_pred_time[-1]:.3f}s  "
                        f"track={avg_track_time[-1]:.3f}s"
                    )

        cap.release()
        if video_writer:
            video_writer.release()
        if self.show:
            cv2.destroyAllWindows()

        # 加上顏色並標示所使用的演算法 (bytetrack或botsort)
        COLOR_RESET = "\033[0m"
        COLOR_YELLOW = "\033[33m"
        COLOR_CYAN = "\033[36m"
        COLOR_MAGENTA = "\033[35m"
        COLOR_GREEN = "\033[32m"

        tracker_name_str = f"{COLOR_MAGENTA}[Algorithm]{COLOR_RESET} {COLOR_CYAN}{self.tracker_type.upper()}{COLOR_RESET}"
        # print(f"\n{COLOR_YELLOW}Tracking complete — {frame_id} frames{COLOR_RESET}")
        print(f"\n{COLOR_YELLOW}Tracking complete — {show_frame_id} frames{COLOR_RESET}")
        print(f"  {COLOR_MAGENTA}pred  avg{COLOR_RESET} : {COLOR_GREEN}{sum(avg_pred_time)  / len(avg_pred_time):.4f}s{COLOR_RESET}")
        print(f"  {COLOR_MAGENTA}track avg{COLOR_RESET} : {COLOR_GREEN}{sum(avg_track_time) / len(avg_track_time):.4f}s{COLOR_RESET}")
        # print(f"  {COLOR_MAGENTA}output    {COLOR_RESET}: {COLOR_CYAN}{output_path}{COLOR_RESET}")
        print(tracker_name_str)

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
        mp4_pattern = str(cfg.SOURCE_VIDEO_DIR / cfg.run_det_eval_dir / "annotation_*" / "**" / "*.mp4")

        mp4s = sorted(
            p for p in glob.glob(mp4_pattern, recursive=True)
            if not Path(p).stem.startswith("Demo")
            if "_org.mp4" not in p
        )

        if not mp4s:
            print(f"⚠️  No source videos found under: {cfg.SOURCE_VIDEO_DIR / cfg.run_det_eval_dir}")
            return det_output_dir

        print(f"\n✅ Found {len(mp4s)} source video(s)")

        # 取要處理的切片（測試時可調整）
        target_mp4s = mp4s[:]   # ← 改這裡控制要跑哪幾支

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

            # 將 trackers/ 由annotations_all8複製到 runs/track_eval/mot_annotations_all8/mot_annotation_<seq>/trackers/
            mot_seq_dir = track_output_dir / f"mot_annotation_{output_path.stem.replace('annotation_', '')}"
            shutil.copytree(
                output_path / "trackers",  # annotations_all8
                mot_seq_dir / "trackers",  # mot_annotations_all8
                dirs_exist_ok=True,
            )
            # 準備GT 驗證資料集
            src_gt_dir = cfg.SOURCE_VIDEO_DIR / "mot_annotations_all8" / f"mot_annotation_{output_path.stem.replace('annotation_', '')}"
            shutil.copytree(
                src_gt_dir / "gt",  # dataset/中華三路_民生二路(東北角)yolo驗證用_all8/mot_annotations_all8/mot_annotation_<seq>
                mot_seq_dir / "gt", # runs/track_eval/mot_annotations_all8/mot_annotation_<seq>
                dirs_exist_ok=True,
            )
            print(f"  trackers → {mot_seq_dir / 'trackers'}")

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

    def run_stage3(self):
        """
        第三階段：軌跡碰撞偵測（TTC/PET，module/collision/）。

        讀取 stage1 產生的逐類別軌跡檔（annotation_<seq>/trackers/mot_challenge/
        <benchmark>-<split>/<tracker_name>__<class>/data/<seq>.txt，格式見
        markdown/TRAJECTORY_FORMAT.md 第 1 節），對每支序列跑一次
        module.collision.run_collision_detection_for_sequence()，輸出寫到
        cfg.collision_output_dir/<seq_name>/ 底下。

        需要 stage1 先跑過（run_det_eval_dir 下要有 annotation_<seq>/trackers/... 與
        first_frame_with_box.jpg），若同一次執行 stages.run_stage1 也是 true，
        會直接讀到本次剛產生的結果；若只想單獨重跑 stage3，把 run_stage1 關掉即可
        沿用上一次的 stage1 輸出。
        """
        cfg = self.cfg
        det_output_dir = cfg.OUTPUT_ROOT / cfg.run_det_eval_dir

        seq_dirs = sorted(det_output_dir.glob("annotation_*"))
        if not seq_dirs:
            print(f"⚠️  No stage1 output found under: {det_output_dir}（stage3 需要先跑過 stage1）")
            return

        print(f"\n{'='*70}")
        print(f"  STAGE 3 — Collision Detection (TTC/PET)")
        print(f"  collision_output   : {cfg.collision_output_dir}")
        print(f"  homography_points  : {cfg.homography_points_dir}")
        print(f"{'='*70}\n")

        for seq_dir in seq_dirs:
            seq_name = seq_dir.stem.replace("annotation_", "")
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
