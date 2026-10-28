"""Stage1 single-sequence class-aware tracking loop."""

import time
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from safetrajai.detection.detector import Detector
from safetrajai.tracking import create_tracker, CLASS_NAMES
from safetrajai.visualization.drawing import draw_polygon_and_save, draw_track_box
from safetrajai.evaluation import dataset_layout
from safetrajai.pipeline.reid_crops import ReidCropCollector, frame_time_s
from safetrajai.pipeline.video_io import VideoWriterWithFfmpeg
from safetrajai.roi import resolve_roi


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
        save_reid_crops: bool = False,
        reid_crop_dir: str = "./dataset/reid/crops",
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
        # ROI 屬於鏡頭，在 run() 依影片解析（見 safetrajai/roi.py）
        self.roi: np.ndarray | None = None
        self.save_reid_crops = save_reid_crops
        self.reid_crop_dir = Path(reid_crop_dir) if reid_crop_dir else None
        self._reid_collector: ReidCropCollector | None = None

    def _annotate_frame(self, frame, tracks, unmatched, removed, frame_id) -> np.ndarray:
        """呼叫 safetrajai.visualization.drawing.draw_track_box() 分別處理 matched/unmatched/removed，
        取代原本內嵌的三段重複迴圈。左上角幀號文字邏輯不變。"""
        annotator = Annotator(frame, line_width=2)

        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                draw_track_box(
                    annotator, (x1, y1, x2, y2), self.roi,
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
                        annotator, (x1, y1, x2, y2), self.roi,
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
                        annotator, (x1, y1, x2, y2), self.roi,
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
        time_s: float,
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
            self._reid_collector.collect(tracks, clean_frame, frame_id, time_s, width, height)

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
        time_s = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            time_s = frame_time_s(cap.get(cv2.CAP_PROP_POS_MSEC), frame_id + 1, self._fps, time_s)

            clean_frame = deepcopy(frame)

            if frame_id == 0:
                draw_polygon_and_save(clean_frame, output_path, pts=self.roi, save_img=True)

            frame_id += 1

            annotated_frame = self._process_frame(
                frame, clean_frame, frame_id, images_path, labels_path, width, height, time_s,
            )

            if self.show and annotated_frame is not None:
                cv2.imshow("Class-Aware Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if video_writer and annotated_frame is not None:
                draw_polygon_and_save(annotated_frame, pts=self.roi, output_path=None, save_img=False)
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
        self._fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30
        output_path, images_path, labels_path = self._prepare_output_dirs(output_dir, video_path)

        seq_name = dataset_layout.seq_name_from_video_path(video_path)
        self.roi = resolve_roi(seq_name, (width, height))
        self._reid_collector = None
        if self.save_reid_crops and self.reid_crop_dir is not None:
            self._reid_collector = ReidCropCollector(self.reid_crop_dir, self.roi, seq_name)

        video_writer = None
        if self.save:
            video_writer = VideoWriterWithFfmpeg(output_path / "Demo.mp4", fps, width, height)

        self._run_frame_loop(
            cap, output_path, images_path, labels_path, video_writer, width, height, total_frames,
        )

        cap.release()
        if self._reid_collector is not None:
            print(f"  reid crops: {self._reid_collector.finalize()}")
        if video_writer:
            video_writer.close()
        if self.show:
            cv2.destroyAllWindows()

        self._print_summary()

        return output_path, images_path, labels_path
