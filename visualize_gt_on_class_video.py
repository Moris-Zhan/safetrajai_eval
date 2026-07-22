"""
visualize_gt_on_class_video.py
==============================
將 trackers/mot_challenge/<benchmark>-<split>/YOLO11x__<class>/data/<seq>.txt
的所有類標註，合併畫在對應的來源影片上，輸出帶有 GT bbox / track-id / class 標籤的影片。

目錄假設
--------
trackeval_root/
  seqinfo.ini
  <seq_name>.mp4              ← 原始影片（可由 video_path 手動指定覆蓋）
  trackers/
    mot_challenge/
      <benchmark>-<split>/
        YOLO11x__<class>/             ← 每一類一個資料夾
          data/
            <seq>.txt                ← 只有6欄：frame, id, x, y, w, h, conf

GT txt 格式（每類一檔，無 class_id 欄）
-----------
frame, id, x, y, w, h, conf      # 無 class_id，僅7欄，class 從資料夾 YOLO11x__<class> 推出

使用方式
--------
python visualize_gt_on_class_video.py --trackeval_root ... --output_video ... \
    --video_path ... --show_class_filter person car
"""

import os
import argparse
import configparser
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import glob
import re

# ── 每類別固定顏色（BGR）────────────────────────────────────────────────────
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0:  (255, 200,  50),   # person     → 黃
    1:  ( 50, 200, 255),   # car        → 天藍
    2:  ( 50, 255, 130),   # scooter    → 綠
    3:  (255,  80, 200),   # bus        → 紫
    4:  ( 80, 130, 255),   # truck      → 藍紫
    5:  (255, 150,  30),   # small_truck→ 橘
    6:  (255,  50,  50),   # bicycle    → 紅
    7:  (150, 255,  50),   # motorcycle → 黃綠
    -1: (200, 200, 200),   # unknown
}

def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """依 track_id 決定 bbox 顏色（id 內唯一顏色）。"""
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]
    return palette[track_id % len(palette)]

def parse_seqinfo(seqinfo_path: Path) -> Dict:
    cfg = configparser.ConfigParser()
    cfg.read(str(seqinfo_path), encoding="utf-8")
    sec = cfg["Sequence"] if "Sequence" in cfg else {}
    return {
        "name":        sec.get("name", seqinfo_path.parent.name),
        "imWidth":     int(sec.get("imWidth", 1920)),
        "imHeight":    int(sec.get("imHeight", 1080)),
        "frameRate":   float(sec.get("frameRate", 30)),
        "seqLength":   int(sec.get("seqLength", 0)),
        "imDir":       sec.get("imDir", "img1"),
        "imExt":       sec.get("imExt", ".jpg"),
    }

def load_all_class_gt_txts(
    trackeval_root: Path,
    class_filter: Optional[List[str]] = None,
) -> Tuple[Dict[int, List[dict]], Dict[int, str], set]:
    """
    掃描 trackers 資料夾，讀取所有類的標註，合併進同一個 dict。
    只吃 6(7)欄：frame, id, x, y, w, h, conf。
    回傳 (gt_by_frame, class_names, visible_classes)
    """
    gt_by_frame = defaultdict(list)
    class_names = dict()  # class_id(int): class_name
    class_name2id = dict()
    visible_classes = set()
    cid_count = 0

    # 支援 mot_challenge 內多個 YOLO11x__<class>
    trackers_root = trackeval_root.glob('trackers/mot_challenge/*')
    trackers_root = [d for d in trackers_root if d.is_dir()]
    data_txts = []
    for bench_dir in trackers_root:
        for class_dir in bench_dir.glob('YOLO11x__*/data/*.txt'):
            data_txts.append(class_dir)

    # class_dirs: Path for each .../YOLO11x__<class>/data/<seq>.txt
    for class_txt in data_txts:
        # 取得類別名
        # 取到 class_txt 的前兩層目錄
        class_dir = class_txt.parent.parent   # .../YOLO11x__<class>
        class_name = class_dir.name.replace('YOLO11x__', '')       

        if class_filter and class_name not in class_filter:
            continue
        if class_name not in class_name2id:
            class_id = cid_count
            class_name2id[class_name] = class_id
            class_names[class_id] = class_name
            cid_count += 1
        else:
            class_id = class_name2id[class_name]

        # parse each txt    
        print(f"read {class_name} ....")    
        with open(class_txt, "r", encoding="utf-8") as f:
            # num_lines = sum(1 for line in f if line.strip() and not line.strip().startswith("#"))            
            # if num_lines > 0:
            #     print(f"{class_name}類別的追蹤物件數量：", num_lines)

            # 依照上一個num_lines計算規則先將文件指標歸零，然後重新逐行讀取
            # f.seek(0)

            for line in f:     
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r'[,\s]+', line)
                if len(parts) < 6:
                    continue
                frame_id  = int(parts[0])
                track_id  = int(parts[1])
                x         = float(parts[2])
                y         = float(parts[3])
                w         = float(parts[4])
                h         = float(parts[5])
                conf      = float(parts[6]) if len(parts) > 6 else 1.0
                # conf=0 → 忽略
                if conf == 0:
                    continue
                # visibility, class_id 欄都無
                gt_by_frame[frame_id].append({
                    "id":         track_id,
                    "x":          x,
                    "y":          y,
                    "w":          w,
                    "h":          h,
                    "conf":       conf,
                    "class_id":   class_id,
                    "class_name": class_name,
                    "visibility": 1.0,
                })
                visible_classes.add(class_name)            
        
    total = sum(len(v) for v in gt_by_frame.values())
    print(f"✅ GT loaded: {len(gt_by_frame)} frames, {total} annotations across {len(class_names)} classes")
    return gt_by_frame, class_names, visible_classes

def draw_gt_on_frame(
    frame: np.ndarray,
    gt_annots: List[dict],
    show_id: bool = True,
    show_class: bool = True,
    show_conf: bool = False,
    color_by: str = "class",
    alpha: float = 0.0,
) -> np.ndarray:
    overlay = frame.copy() if alpha > 0 else None
    for ann in gt_annots:
        x1 = int(ann["x"])
        y1 = int(ann["y"])
        x2 = int(ann["x"] + ann["w"])
        y2 = int(ann["y"] + ann["h"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        if color_by == "id":
            color = _color_for_id(ann["id"])
        else:
            color = CLASS_COLORS.get(ann["class_id"], CLASS_COLORS[-1])
        if alpha > 0 and overlay is not None:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        parts = []
        if show_class:
            parts.append(ann["class_name"])
        if show_id:
            parts.append(f"#{ann['id']}")
        if show_conf:
            parts.append(f"{ann['conf']:.2f}")
        label = " ".join(parts)
        if label:
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness  = 1
            (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
            lx1 = x1
            ly1 = max(0, y1 - th - 4)
            lx2 = x1 + tw + 4
            ly2 = y1
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                frame, label,
                (lx1 + 2, ly2 - 2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )
    if alpha > 0 and overlay is not None:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def _draw_frame_counter(frame: np.ndarray, frame_id: int, total: int) -> None:
    text = f"GT  frame {frame_id}/{total}"
    org  = (8, 24)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.55, 1
    (tw, th), bl = cv2.getTextSize(text, font, fs, ft)
    cv2.rectangle(frame, (4, 4), (4 + tw + 6, 4 + th + bl + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, org, font, fs, (0, 255, 80), ft, cv2.LINE_AA)

def _draw_legend(
    frame: np.ndarray,
    class_names: Dict[int, str],
    visible_classes: set,
) -> None:
    items = [
        (cid, cname)
        for cid, cname in class_names.items()
        if cname in visible_classes
    ]
    if not items:
        return
    x0, y0 = frame.shape[1] - 140, 8
    row_h   = 18
    pad     = 4
    box_h   = row_h * len(items) + pad * 2
    cv2.rectangle(frame, (x0 - pad, y0), (frame.shape[1] - pad, y0 + box_h), (0, 0, 0), -1)
    for i, (cid, cname) in enumerate(items):
        color = CLASS_COLORS.get(cid, CLASS_COLORS[-1])
        y = y0 + pad + i * row_h + row_h // 2
        cv2.rectangle(frame, (x0, y - 5), (x0 + 12, y + 5), color, -1)
        cv2.putText(frame, cname, (x0 + 16, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

def visualize_gt_on_video(
    trackeval_root: str,
    output_video: str,
    video_path: Optional[str] = None,
    class_names: Optional[Dict[int, str]] = None,
    show_class_filter: Optional[List[str]] = None,
    show_id: bool = True,
    show_class: bool = True,
    show_conf: bool = False,
    color_by: str = "class",
    bbox_alpha: float = 0.0,
    show_legend: bool = True,
    max_frames: Optional[int] = None,
) -> str:
    root = Path(trackeval_root)

    # ── 解析 seqinfo ─────────────────────────────────────────
    seq_ini_candidates = list(root.glob("seqinfo.ini")) + list(root.glob("*/seqinfo.ini"))
    seq_info = {}
    if seq_ini_candidates:
        seq_info = parse_seqinfo(seq_ini_candidates[0])
        print(f"📄 seqinfo: {seq_ini_candidates[0]}")
    else:
        print("⚠️  seqinfo.ini not found, using defaults")

    gt_by_frame, class_names_auto, visible_classes = load_all_class_gt_txts(
        root, show_class_filter
    )
    cn = class_names or class_names_auto

    # 類別次數統計
    class_counts = {}
    for annots in gt_by_frame.values():
        for a in annots:
            cls = a["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
    print("👉 類別出現次數:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {cls}: {cnt}")

    # ── 尋找影片 ─────────────────────────────────────────────
    vp = Path(video_path)
    if vp is None or not vp.exists():
        raise FileNotFoundError(
            f"Video not found. Pass --video_path explicitly. "
            f"(searched under: {root})"
        )
    print(f"🎬 Video: {vp}")

    # ── 開啟影片 ─────────────────────────────────────────────
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {vp}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or seq_info.get("frameRate", 30)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   {width}x{height} @ {fps:.1f}fps  total={total}f")

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # ── 逐幀處理 ─────────────────────────────────────────────
    frame_id  = 0
    drawn_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        annots = gt_by_frame.get(frame_id, [])
        if annots:
            draw_gt_on_frame(
                frame, annots,
                show_id=show_id,
                show_class=show_class,
                show_conf=show_conf,
                color_by=color_by,
                alpha=bbox_alpha,
            )
            drawn_cnt += len(annots)
        if show_legend:
            _draw_legend(frame, cn, visible_classes)
        _draw_frame_counter(frame, frame_id, total)
        writer.write(frame)
        if frame_id % 50 == 0:
            print(f"  [{frame_id}/{total}]  annotated boxes this batch: {drawn_cnt}")
            drawn_cnt = 0
        if max_frames and frame_id >= max_frames:
            print(f"  ⏹  max_frames={max_frames} reached, stopping.")
            break

    cap.release()
    writer.release()

    h264_path = out_path.with_stem(out_path.stem + "_h264")
    cmd = (
        f'ffmpeg -y -i "{out_path}" '
        f'-vcodec libx264 -pix_fmt yuv420p "{h264_path}" -loglevel error'
    )
    ret_code = os.system(cmd)
    if ret_code == 0 and h264_path.exists():
        out_path.unlink()
        h264_path.rename(out_path)
        print(f"  🔁 H.264 conversion done.")

    print(f"\n✅ Output: {out_path.resolve()}")

    # 統計各類別
    class_counter = Counter()
    total_count = 0
    for frame_annots in gt_by_frame.values():
        for ann in frame_annots:
            class_name = ann.get("class_name", "unknown")
            class_counter[class_name] += 1
            total_count += 1

    stats_lines = ["Class counts:"]
    for cname, cnt in class_counter.most_common():
        stats_lines.append(f"{cname}: {cnt}")
    stats_lines.append(f"\nTotal objects: {total_count}")

    stats_txt_path = out_path.with_suffix('.txt')
    with open(stats_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print(f"📊 Stats saved to: {stats_txt_path.resolve()}")

    return str(out_path)

def _parse_class_names(raw: List[str]) -> Optional[Dict[int, str]]:
    # 若有給則支援類似於 0:person 1:car
    result = {}
    for item in raw:
        if ":" in item:
            k, v = item.split(":", 1)
            result[int(k)] = v
    return result or None

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT annotations on video from trackeval_root."
    )

    parser.add_argument("--trackeval_root", 
                        default="dataset/中華三路_民生二路(東北角)yolo驗證用_all8/mot_annotations_all8/mot_annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1",
                        help="mot_annotation_<seq> 根目錄")
    parser.add_argument("--video_path",     
                        default="dataset/中華三路_民生二路(東北角)yolo驗證用_all8/annotations_all8/annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1",
                        help="來源影片路徑（可自動偵測）")
    parser.add_argument("--output_video",   
                        default="./runs/gt_viz/C000025_gt.mp4",
                        help="輸出影片路徑（.mp4）")

    # 類別設定
    parser.add_argument("--class_names", nargs="*", default=None,
                        help='類別名稱 e.g. "0:person 1:car"')
    parser.add_argument("--show_class_filter", nargs="*", default=None,
                        help='只顯示的類別 e.g. "person car"')

    # 顯示控制
    parser.add_argument("--color_by",   default="class", choices=["class", "id"],
                        help="bbox 顏色依據")
    parser.add_argument("--no_id",      action="store_true", help="不顯示 track_id")
    parser.add_argument("--no_class",   action="store_true", help="不顯示 class_name")
    parser.add_argument("--show_conf",  action="store_true", help="顯示 conf 分數")
    parser.add_argument("--no_legend",  action="store_true", help="不顯示圖例")
    parser.add_argument("--bbox_alpha", type=float, default=0.0,
                        help="bbox 填色透明度 0~1（0 = 只畫框線）")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最多處理幾幀（除錯用）")

    args = parser.parse_args()

    cn = _parse_class_names(args.class_names) if args.class_names else None

    

    SOURCE_VIDEO_DIR = Path("dataset/中華三路_民生二路(東北角)yolo驗證用_all8")
    run_det_eval_dir = Path("annotations_all8")

    mp4_pattern = str(SOURCE_VIDEO_DIR / run_det_eval_dir / "annotation_*" / "**" / "*.mp4")

    mp4s = sorted(
        p for p in glob.glob(mp4_pattern, recursive=True)
        if not Path(p).stem.startswith("Demo")
        if "_org.mp4" not in p
    )

    for mp4_path in mp4s:
        # 根據 .mp4 檔名推出 trackeval_root 目錄（假設與 mot_annotation_<seq> 規則一致）
        # 例如：.../annotations_all8/annotation_C000025 .../C000025 ..._xxx.mp4
        # 取annotation_xxx/.../filename.mp4，對應mot_annotations_all8/mot_annotation_xxx
        mp4_path = Path(mp4_path)
        seq_dir_name = mp4_path.parent.name.replace("annotation_", "mot_annotation_")
        seq_root = SOURCE_VIDEO_DIR / "mot_annotations_all8" / seq_dir_name

        # 輸出影片名稱
        output_video = Path(args.output_video)
        output_video = output_video.parent / f"{mp4_path.stem}_gt.mp4"

        # 若 batch 處理多影片者，請參考舊 version main 的那些 batch search 邏輯
        visualize_gt_on_video(
            trackeval_root=str(seq_root),
            output_video=str(output_video),
            video_path=str(mp4_path),
            class_names=cn,
            show_class_filter=args.show_class_filter,
            show_id=not args.no_id,
            show_class=not args.no_class,
            show_conf=args.show_conf,
            color_by=args.color_by,
            bbox_alpha=args.bbox_alpha,
            show_legend=not args.no_legend,
            max_frames=args.max_frames,
        )

if __name__ == "__main__":
    main()