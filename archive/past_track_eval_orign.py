import os
import sys
import time
import argparse
from copy import deepcopy
from pathlib import Path
import glob


import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers import create_tracker, CLASS_NAMES
from trackers.config import (
    get_conf_thresh_for_class,
    get_min_conf_thresh,
    get_track_config_path,
    load_track_config,
)
from utils.track_utils import draw_polygon_and_save, save_fst_frame, save_mot_format, make_det_eval_dataset, translate_path

from utils.eval_utils import run_eval_pipeline
from utils.track_eval.config_loader import load_config

import shutil
eval_destination = "./dataset/中華三路_民生二路(東北角)yolo驗證用_all8"

track_eval_destination = "./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/mot_annotations_all8"

DEFAULT_MODEL = "./runs/detect/train/weights/best.pt"
# DEFAULT_MODEL = "./models/yolo-reid/yolo11x-reid-ultra.pt"


# DEFAULT_MODEL = "./runs/triplet-yolo/train-007/best.pt"
# DEFAULT_VIDEO = "dataset/C000025 中華三路 民生二路(東北角)_20251209145630_1.mp4"
# DEFAULT_VIDEO = "dataset/C000025_clip_6_15.mp4"
DEFAULT_VIDEO = "./dataset/中華三路_民生二路(東北角)yolo驗證用_all8/annotations_all8/annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1/C000025 中華三路 民生二路(東北角)_20251209145630_15s_1.mp4"

DEFAULT_VIDEO_DIR = "dataset/中華三路_民生二路(東北角)yolo驗證用_all8"


def run_class_aware_tracking(
    output_dir,
    model_path=DEFAULT_MODEL,
    video_path=DEFAULT_VIDEO,
    tracker_type='botsort',
    conf=0.15,
    iou=0.5,
    imgsz=1280,
    save=True,
    save_txt=True,
    show=False,
    show_unmatched=False,
    show_removed=False,
):
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")

    tracker = create_tracker(
        tracker_type=tracker_type,
        frame_rate=30,
        track_high_thresh=conf,
        track_low_thresh=conf * 0.3,
    )
    print(f"Created {tracker_type} tracker with class-aware parameters")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # normalized_video_path = translate_path(video_path)
    # output_path = Path(output_dir) / f"{Path(normalized_video_path).stem}_{tracker_type}"
    output_path = Path(output_dir) / f"annotation_{Path(video_path).stem}"

    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_path = (output_path / "labels" / Path(video_path).stem)
    labels_path.mkdir(parents=True, exist_ok=True)

    images_path = (output_path / "images" / Path(video_path).stem)
    images_path.mkdir(parents=True, exist_ok=True)

    stem_name = Path(video_path).stem.replace(' 中華三路 民生二路(東北角)_', '_')


    # INSERT_YOUR_CODE
    # 將video_path檔案複製到output_path底下
    import shutil
    video_dst_path = output_path / Path(video_path).name
    if not video_dst_path.exists():
        shutil.copy(video_path, video_dst_path)

    video_writer = None
    if save:
        # video_name = Path(normalized_video_path).stem + "_tracked.mp4"
        # video_name = f"Demo_{Path(video_path).stem}.mp4"
        video_name = f"Demo.mp4"
        output_video_path = output_path / video_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"Output video: {output_video_path}")

    frame_id = 0
    avg_pred_time = []
    avg_track_time = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = deepcopy(frame)
        st = time.time()

        if frame_id == 0:
            draw_polygon_and_save(deepcopy(frame), output_path)
            save_fst_frame(deepcopy(frame), output_path)
        else:
            draw_polygon_and_save(frame, output_path, show_coord=False)

        frame_id += 1

        min_conf = get_min_conf_thresh()
        results = model.predict(frame, conf=min_conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        en = time.time()
        avg_pred_time.append(en - st)

        boxes = results.boxes
        if len(boxes) > 0:
            keep = [
                i for i in range(len(boxes))
                if float(boxes.conf[i]) >= get_conf_thresh_for_class(int(boxes.cls[i]))
            ]
            boxes = boxes[keep] if keep else boxes[:0]

        if len(boxes) > 0:
            tracks, unmatched, removed = tracker.update(boxes.cpu().numpy(), frame)
        else:
            tracks, unmatched, removed = np.array([]), np.array([]), np.array([])

        tr_st = time.time()
        avg_track_time.append(tr_st - en)

        if save_txt:
            # label_file = labels_path / f"frame_{frame_id:06d}.txt"
            # img_file = images_path / f"frame_{frame_id:06d}.jpg"

            label_file = labels_path / f"{stem_name}_f{frame_id:06d}.txt"
            img_file = images_path / f"{stem_name}_f{frame_id:06d}.jpg"

            cv2.imwrite(str(img_file), clean_frame)
            with open(label_file, 'w') as f:
                for track in tracks:
                    if len(track) >= 7:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        score = track[5]
                        cls_id = int(track[6])
                        cx = (x1 + x2) / 2 / width
                        cy = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {track_id} {score:.4f}\n")

        if save or show:
            annotator = Annotator(frame, line_width=2)
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[:4]
                    track_id = int(track[4])
                    score = track[5]
                    cls_id = int(track[6])
                    cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                    # annotator.box_label([x1, y1, x2, y2], f"{track_id} - {cls_name}", color=colors(track_id, True))
                    annotator.box_label([x1, y1, x2, y2], f"{track_id}", color=colors(track_id, True))

            if show_unmatched:
                for track in unmatched:
                    if len(track) >= 7:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        score = track[5]
                        cls_id = int(track[6])
                        cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

                        # label = f"{cls_name} {score:.2f}"
                        label_id = f"{cls_name}-{score:.2f}"
                        # label_score = f"{score:.2f}"
                        # 第一列顯示ID和類別
                        annotator.box_label([x1, y1, x2, y2], "", color=(0, 0, 255))
                        annotator.text((int(x2), int(y2)), label_id, txt_color=(255, 255, 255), anchor="top", box_color=(0, 0, 255))
                        # 第二列（下方）顯示信心分數
                        # box_height = y2 - y1
                        # offset = int(0.85 * box_height)
                        # annotator.box_label([x1, y1 + offset, x2, y2 + offset], label_score, color=(0, 0, 255))

            if show_removed:
                for track in removed:
                    if len(track) >= 7:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        score = track[5]
                        cls_id = int(track[6])
                        cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

                        # label = f"drop: {score:.2f}"          
                        label = f"{track_id}-{score:.2f}"    
                        annotator.box_label([x1, y1, x2, y2], "", color=(0, 0, 0))
                        annotator.text((int(x1), int(y1)), label, txt_color=(255, 255, 255), anchor="top", box_color=(0,0,0))
                        # label = f"{cls_name} {score:.2f}"      
                        # annotator.box_label([x1, y2, x2, y2+20], label, color=(0, 0, 0))    
                        # annotator.box_label([x1, y2, x2, y2 + (y2-y1)], label, color=(0, 0, 0))

            annotated_frame = annotator.result()

            text = f"frame: {frame_id}"
            org = (8, 22)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(annotated_frame, (org[0] - 3, org[1] - th - 3), (org[0] + tw + 3, org[1] + baseline + 3), (0, 0, 0), -1)
            cv2.putText(annotated_frame, text, org, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        if show:
            cv2.imshow("Class-Aware Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if video_writer:
            video_writer.write(annotated_frame)

        if frame_id % 10 == 0:
            print(f"Processing frame {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%)")
            print(f"[Debug] model.predict time: {avg_pred_time[-1]:.4f} seconds")
            print(f"[Debug] tracker.update time: {avg_track_time[-1]:.4f} seconds")

    cap.release()
    if video_writer:
        video_writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nTracking complete!")
    print(f"Average model.predict time: {sum(avg_pred_time) / len(avg_pred_time):.4f} seconds")
    print(f"Average tracker.update time: {sum(avg_track_time) / len(avg_track_time):.4f} seconds")
    print(f"  - Output directory: {output_path}")
    print(f"  - Labels saved to: {labels_path}")
    if save:
        print(f"  - Video saved to: {output_video_path}")

    if save and os.path.exists(output_video_path):
        ffmpeg_output = os.path.splitext(output_video_path)[0] + "_ffmpeg.mp4"
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{output_video_path}" -vcodec libx264 -pix_fmt yuv420p -acodec aac "{ffmpeg_output}"'
        )
        print(f"Converting video with ffmpeg: {ffmpeg_cmd}")
        result = os.system(ffmpeg_cmd)
        if result == 0:
            print(f"ffmpeg conversion successful. Saved to: {ffmpeg_output}")
        else:
            print("ffmpeg conversion failed.")

        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        if os.path.exists(ffmpeg_output):
            os.rename(ffmpeg_output, output_video_path)

    return output_path, images_path, labels_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO inference with class-aware tracking")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Video path")
    # parser.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"], help="Tracker type")
    parser.add_argument("--tracker", default="bytetrack", choices=["botsort", "bytetrack"], help="Tracker type (botsort or bytetrack)")
    parser.add_argument("--conf", type=float, default=0.15, help="Tracker confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    # parser.add_argument("--output", default="./runs/detect/class_aware_track", help="Output directory")
    parser.add_argument("--output", default="./runs/detect/annotations_all8", help="Output directory")
    parser.add_argument("--save", action="store_true", default=False, help="Save video")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Don't save video")
    parser.add_argument("--show", action="store_true", help="Show video")
    parser.add_argument("--show_unmatched", default=True, action="store_true", help="Show unmatched obj")
    parser.add_argument("--show_removed", default=True, action="store_true", help="Show removed obj")

    parser.add_argument("--mot-output", default=True, help="Save MOT format output file")
    parser.add_argument("--save_txt", action="store_true", default=True, help="Save MOT text label")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml file")
    parser.add_argument(
        "--track-config",
        default="trackers/track_config.yaml",
        help="class-aware tracker YAML（CLASS_TRACKING_PARAMS、置信度門檻）",
    )
    args = parser.parse_args()

    load_track_config(args.track_config)
    print(f"Loaded track config: {get_track_config_path()}")

    mp4s = sorted(glob.glob(os.path.join(DEFAULT_VIDEO_DIR, "annotations_all8", "annotation_*", "**", "*.mp4"), recursive=True))
    mp4s.append("dataset/C000025_clip_6_15.mp4")

    # for video_path in mp4s[-2:]:
    for video_path in mp4s[1:2]:
        if video_path == "dataset/C000025_clip_6_15.mp4": continue
        output_path, images_path, labels_path = run_class_aware_tracking(
            model_path=args.model,
            video_path=video_path,
            tracker_type=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=False,
            save_txt=args.save_txt,
            show=False,
            show_unmatched=args.show_unmatched,
            show_removed=args.show_removed,
            output_dir=args.output,
        )
        print(output_path)
        # continue

        if args.mot_output:
            cap = cv2.VideoCapture(args.video)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            save_mot_format(
                # output_path / "labels",
                images_path,
                labels_path,
                output_path / f"{str(output_path.stem)}.txt",
                width,
                height,
            )

        make_det_eval_dataset(output_path, CLASS_NAMES)

        # copy to run_track_eval_dir
        shutil.copytree(
            output_path / "trackers",
            os.path.join(track_eval_destination, f"mot_{output_path.name}", "trackers"),
            dirs_exist_ok=True
        )


    # # 將剛剛完成的整個 annotation_xxx 資料夾 (output_path 的 parent 目錄) 複製到評估資料集目的地 (eval_destination)
    # # 這樣可以方便後續進行評估或整理
    # # copy to run_det_eval_dir
    shutil.copytree(
        str(Path(output_path).parent),
        os.path.join(eval_destination, Path(output_path).parent.name),
        dirs_exist_ok=True
    )

    cfg = load_config(args.config)
    run_eval_pipeline(cfg)