"""
visualize_gt_on_video.py
========================
將 trackeval_root 內的 Ground Truth 標注畫在對應的來源影片上，
輸出帶有 GT bbox / track-id / class 標籤的標注影片。

目錄假設
--------
trackeval_root/
  gt/
    gt.txt              ← MOT 格式 GT (7 或 8 欄)
  seqinfo.ini           ← 序列基本資訊
  <seq_name>.mp4        ← 原始影片（可由 video_path 手動指定覆蓋）

GT txt 格式
-----------
frame, id, x, y, w, h, conf, [class_id], [visibility]
（col index 0-based；class_id 在第 7 欄，若無則設 -1）

使用方式
--------
# 最簡單：僅給 trackeval_root，自動尋找影片
python visualize_gt_on_video.py \\
    --trackeval_root ./dataset/mot_annotation_C000025 \\
    --output_video   ./runs/gt_viz/C000025_gt.mp4

# 手動指定影片與類別名稱對照表
python visualize_gt_on_video.py \\
    --trackeval_root ./dataset/mot_annotation_C000025 \\
    --video_path     ./dataset/C000025.mp4 \\
    --class_names    0:person 1:car 2:scooter 3:bus 4:truck 5:small_truck \\
    --output_video   ./runs/gt_viz/C000025_gt.mp4 \\
    --show_class_filter person car     # 只畫這兩類
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

# 統計各類別及總的物件標記次數
from collections import Counter

import glob

from safetrajai.visualization.gt_overlay import CLASS_COLORS, draw_gt_on_frame, parse_seqinfo

# ── 預設類別名稱（與 CLASS_NAMES 保持一致）─────────────────────────────────
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    1: "person",
    2: "car",
    3: "scooter",
    4: "small_truck",
    5: "bus",
    6: "truck",
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────


def load_gt_txt(
    gt_path: Path,
    class_filter: Optional[List[str]] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[int, List[dict]]:
    """
    讀取 MOT 格式 GT txt，依幀號建索引。

    Returns
    -------
    dict[frame_id -> list[{id, x, y, w, h, conf, class_id, class_name, visibility}]]
    """
    cn = class_names or DEFAULT_CLASS_NAMES
    filter_set = set(class_filter) if class_filter else None

    gt_by_frame: Dict[int, List[dict]] = defaultdict(list)
    if not gt_path.exists():
        print(f"⚠️  GT file not found: {gt_path}")
        return gt_by_frame

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id  = int(parts[0])
            track_id  = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf      = float(parts[6]) if len(parts) > 6 else 1.0
            class_id  = int(float(parts[7])) if len(parts) > 7 else -1
            visibility= float(parts[8]) if len(parts) > 8 else 1.0

            # conf=0 → ignored region，跳過
            if conf == 0:
                continue

            class_name = cn.get(class_id, f"cls_{class_id}")
            if filter_set and class_name not in filter_set:
                continue

            gt_by_frame[frame_id].append({
                "id":         track_id,
                "x":          x,
                "y":          y,
                "w":          w,
                "h":          h,
                "conf":       conf,
                "class_id":   class_id,
                "class_name": class_name,
                "visibility": visibility,
            })

    total = sum(len(v) for v in gt_by_frame.values())
    print(f"✅ GT loaded: {len(gt_by_frame)} frames, {total} annotations  ({gt_path})")
    return gt_by_frame


def find_video(trackeval_root: Path) -> Optional[Path]:
    """在 trackeval_root 下自動尋找 .mp4（排除 Demo）。"""
    for p in sorted(trackeval_root.glob("*.mp4")):
        if "demo" not in p.stem.lower():
            return p
    # 往上一層找
    for p in sorted(trackeval_root.parent.glob("*.mp4")):
        if "demo" not in p.stem.lower():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 核心繪製函式
# ─────────────────────────────────────────────────────────────────────────────


def _draw_frame_counter(frame: np.ndarray, frame_id: int, total: int) -> None:
    """左上角畫幀號。"""
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
    """右上角畫類別圖例。"""
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


# ─────────────────────────────────────────────────────────────────────────────
# 主要輸出函式
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    核心函式：將 GT 標注畫在影片上並輸出。

    Parameters
    ----------
    trackeval_root    : mot_annotation_<seq> 根目錄
    output_video      : 輸出影片路徑（.mp4）
    video_path        : 來源影片（None → 自動尋找）
    class_names       : {class_id: class_name}（None → 使用預設）
    show_class_filter : 只顯示這些類別（None → 全部）
    show_id           : 顯示 track_id
    show_class        : 顯示 class_name
    show_conf         : 顯示 conf
    color_by          : "class" 或 "id"
    bbox_alpha        : bbox 填色透明度
    show_legend       : 右上角顯示圖例
    max_frames        : 最多處理幾幀（None → 全部）

    Returns
    -------
    str : 輸出影片路徑
    """
    root = Path(trackeval_root)
    cn   = class_names or DEFAULT_CLASS_NAMES

    # ── 解析 seqinfo ─────────────────────────────────────────
    seq_ini_candidates = list(root.glob("seqinfo.ini")) + list(root.glob("*/seqinfo.ini"))
    seq_info = {}
    if seq_ini_candidates:
        seq_info = parse_seqinfo(seq_ini_candidates[0])
        print(f"📄 seqinfo: {seq_ini_candidates[0]}")
    else:
        print("⚠️  seqinfo.ini not found, using defaults")

    # ── 尋找 GT txt ──────────────────────────────────────────
    gt_candidates = list(root.glob("gt/gt.txt")) + list(root.glob("**/gt.txt"))
    if not gt_candidates:
        raise FileNotFoundError(f"GT file not found under: {root}")
    gt_path = gt_candidates[0]

    gt_by_frame = load_gt_txt(gt_path, show_class_filter, cn)
    if not gt_by_frame:
        print("⚠️  No GT annotations loaded (check class_filter?)")

    # 收集實際出現的類別（供圖例使用）
    visible_classes: set = set()
    for annots in gt_by_frame.values():
        for a in annots:
            visible_classes.add(a["class_name"])

    # 統計出現類別的次數
    class_counts = {}
    for annots in gt_by_frame.values():
        for a in annots:
            cls = a["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
    print("👉 類別出現次數:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {cls}: {cnt}")

    # ── 尋找影片 ─────────────────────────────────────────────
    # vp_root = Path(video_path) 
    # vp = vp_root / Path(str(vp_root.name).replace("annotation_", "")+".mp4")

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

    # ── 建立輸出目錄與 VideoWriter ───────────────────────────
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

    # ── ffmpeg 轉 H.264 ──────────────────────────────────────
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

    # 準備輸出文字內容
    stats_lines = ["Class counts:"]
    for cname, cnt in class_counter.most_common():
        stats_lines.append(f"{cname}: {cnt}")
    stats_lines.append(f"\nTotal objects: {total_count}")

    # 決定輸出文字檔路徑，與影片同名但為 .txt
    stats_txt_path = out_path.with_suffix('.txt')
    with open(stats_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print(f"📊 Stats saved to: {stats_txt_path.resolve()}")

    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_class_names(raw: List[str]) -> Dict[int, str]:
    """解析 "0:person 1:car" 形式的字串列表。"""
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
                        default="dataset/eval/cctv_all8/mot_annotations_all8/mot_annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1",
                        help="mot_annotation_<seq> 根目錄")
    parser.add_argument("--video_path",     
                        default="dataset/eval/cctv_all8/annotations_all8/annotation_C000025 中華三路 民生二路(東北角)_20251209145630_15s_1",
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

    

    SOURCE_VIDEO_DIR = Path("dataset/eval/cctv_all8")
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