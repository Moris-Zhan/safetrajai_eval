# mot_converter.py
# -*- coding: utf-8 -*-
"""
Ultralytics 追蹤標籤 (*.txt) -> MOT tracker txt (7 cols) 轉換工具。

處理以下常見問題：
- 同一 frame 整批重複貼兩次（exact duplicate lines）
- 同 frame 同 track_id 出現多個 bbox（duplicate ID conflicts）
- GT gt.txt 同 frame 重複 ID
"""

import re
from pathlib import Path
import numpy as np
import cv2

from utils.track_eval.utils import (
    labels_dir_fingerprint,
    get_video_wh,
    yolo_xywhn_to_xywh_px,
)


# =========================================================
# 標籤目錄檢查
# =========================================================
def _warn_if_looks_like_dataset_gt_labels(labels_dir: str):
    """
    啟發式警告：若 labels_dir 路徑看起來像 dataset GT（labels/val 或 labels/train），
    就提醒使用者避免誤用 GT 當作 tracker prediction。
    """
    p = Path(labels_dir).as_posix().lower()
    if (
        "/labels/val" in p or "/labels/train" in p
        or "\\labels\\val" in labels_dir.lower()
        or "\\labels\\train" in labels_dir.lower()
    ):
        print("\n⚠️ WARNING: Your labels_dir looks like dataset GT (labels/val or labels/train).")
        print("   If you use this as tracker predictions, TrackEval will be near-perfect (fake).")
        print("   Use Ultralytics tracking output labels under runs/track/.../labels instead.\n")


def scan_ultralytics_label_classes(trk_label_dir: str, max_files=999999):
    """掃描標籤目錄，統計各類別 (0-based) 出現次數。"""
    label_dir = Path(trk_label_dir)
    files = sorted(label_dir.glob("*.txt"))
    cls_counter = {}
    bad_lines = 0
    scanned = 0

    for p in files:
        scanned += 1
        if scanned > max_files:
            break

        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            # 預期格式: cls cx cy w h track_id [conf]
            if len(parts) < 6:
                bad_lines += 1
                continue
            try:
                yolo_cls0 = int(float(parts[0]))
                cls_counter[yolo_cls0] = cls_counter.get(yolo_cls0, 0) + 1
            except Exception:
                bad_lines += 1

    return cls_counter, bad_lines


# =========================================================
# 主轉換函式：Ultralytics tracking labels -> MOT txt
# =========================================================
def ul_track_labels_to_mot(
    trk_label_dir: str,
    out_txt: str,
    video_path: str,
    keep_cls_ids=None,           # 0-based YOLO cls ids，例如 [1] 只保留 car
    tid_one_based=True,
    float_fmt="{:.6f}",
    # ---- 穩健性選項 ----
    drop_exact_dup_lines=True,   # 先去掉完全重複的 line（處理整批重複貼兩次的問題）
    fix_duplicate_ids=True,      # 同 frame 同 tid 多個 bbox → 保留一個原 tid，其餘改新 id
    choose_keep="best_conf",     # "best_conf" or "largest_area"
    verbose=True,
):
    """
    Ultralytics tracking labels (*.txt) -> MOT tracker txt (7 cols):
      frame, id, x, y, w, h, conf

    預期 Ultralytics 標籤格式：
      cls cx cy w h conf track_id

    穩健性處理：
    - 同 frame 完全相同的 line 去重（處理整塊重複的情況）
    - 同 frame 相同 ID 有多個 bbox 時，重新指派新唯一 ID
    """

    _warn_if_looks_like_dataset_gt_labels(trk_label_dir)

    if verbose:
        print("\n=== Labels dir fingerprint ===")
        fp = labels_dir_fingerprint(trk_label_dir)
        print("labels_dir:", trk_label_dir)
        print("labels_fingerprint:", fp)

        cls_counter, bad_lines = scan_ultralytics_label_classes(trk_label_dir)
        print("\n=== Labels class distribution (from labels/*.txt) ===")
        if cls_counter:
            top = sorted(cls_counter.items(), key=lambda x: -x[1])
            print("cls_counter:", top[:30], "(show top30)")
        else:
            print("❌ No valid label lines found. bad_lines =", bad_lines)

    out_path = Path(out_txt)
    W, H = get_video_wh(video_path)

    keep_set = None
    if keep_cls_ids is not None:
        keep_set = set(int(x) for x in keep_cls_ids)
        if verbose:
            print("keep_cls_ids (0-based):", sorted(list(keep_set)))

    label_dir = Path(trk_label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Track labels dir not found: {label_dir}")

    files = sorted(label_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No txt labels found under: {label_dir}")

    # 支援 _f#### 命名模式（例如 frame0001.txt → frame index 1）
    pat_f = re.compile(r"_f(\d+)\.txt$", re.IGNORECASE)
    use_f_pattern = any(pat_f.search(p.name) for p in files)

    if verbose:
        print("\n=== Frame indexing strategy ===")
        print("use _f#### pattern:", use_f_pattern)
        print("label files:", len(files), "example:", files[0].name)

    # frame -> list[(cls, tid_out, conf, x, y, w, h, raw_line_key)]
    per_frame = {}
    max_tid_seen = 0

    # 統計
    total_lines = 0
    kept_lines = 0
    exact_dups_dropped = 0
    parse_bad = 0

    for idx, p in enumerate(files):
        # 決定 frame index
        if use_f_pattern:
            m = pat_f.search(p.name)
            if not m:
                continue
            frame0 = int(m.group(1))
            frame = frame0 + 1
        else:
            frame = idx + 1

        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            continue

        seen_line_keys = set()  # 同 frame 內去重用
        for line in lines:
            if not line.strip():
                continue
            total_lines += 1

            parts = line.split()
            # 預期 >= 7 欄: cls cx cy w h conf tid
            if len(parts) < 7:
                parse_bad += 1
                continue

            try:
                yolo_cls0 = int(float(parts[0]))
                if keep_set is not None and yolo_cls0 not in keep_set:
                    continue

                cx, cy, ww, hh = map(float, parts[1:5])
                conf = float(parts[5])
                tid = int(float(parts[6]))
            except Exception:
                parse_bad += 1
                continue

            # 以 round 後的數值建立 line key，避免浮點文本差異造成誤判
            line_key = (
                yolo_cls0,
                round(cx, 6), round(cy, 6),
                round(ww, 6), round(hh, 6),
                round(conf, 6),
                tid,
            )

            if drop_exact_dup_lines:
                if line_key in seen_line_keys:
                    exact_dups_dropped += 1
                    continue
                seen_line_keys.add(line_key)

            x, y, bw, bh = yolo_xywhn_to_xywh_px(cx, cy, ww, hh, W, H)
            tid_out = tid + 1 if tid_one_based else tid
            max_tid_seen = max(max_tid_seen, tid_out)

            per_frame.setdefault(frame, []).append(
                (yolo_cls0, tid_out, conf, x, y, bw, bh, line_key)
            )
            kept_lines += 1

    if not per_frame:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        print("⚠️ No detections parsed after filtering. Output empty.")
        return

    # 修正同 frame 重複 ID（exact-dup 去除後仍存在的情況）
    new_tid = max_tid_seen + 1
    dup_id_conflicts = 0
    out_lines = []

    for frame in sorted(per_frame.keys()):
        dets = per_frame[frame]

        # 依 tid_out 分群
        by_tid = {}
        for (cls0, tid_out, conf, x, y, bw, bh, lk) in dets:
            by_tid.setdefault(tid_out, []).append((cls0, conf, x, y, bw, bh, lk))

        for tid_out, lst in by_tid.items():
            if len(lst) == 1 or not fix_duplicate_ids:
                # 直接寫出
                for (cls0, conf, x, y, bw, bh, lk) in lst:
                    out_lines.append(
                        f"{frame},{tid_out},"
                        f"{float_fmt.format(x)},{float_fmt.format(y)},"
                        f"{float_fmt.format(bw)},{float_fmt.format(bh)},"
                        f"{conf:.6f}"
                    )
                continue

            # 同 frame 多個 bbox 共用相同 tid_out
            dup_id_conflicts += (len(lst) - 1)

            # 決定保留哪一個維持原 id
            if choose_keep == "largest_area":
                lst_sorted = sorted(lst, key=lambda t: (t[3] * t[4], t[1]), reverse=True)  # area, conf
            else:
                # best_conf（預設）
                lst_sorted = sorted(lst, key=lambda t: (t[1], t[3] * t[4]), reverse=True)  # conf, area

            # 第一個保留原 id
            cls0, conf, x, y, bw, bh, lk = lst_sorted[0]
            out_lines.append(
                f"{frame},{tid_out},"
                f"{float_fmt.format(x)},{float_fmt.format(y)},"
                f"{float_fmt.format(bw)},{float_fmt.format(bh)},"
                f"{conf:.6f}"
            )

            # 其餘指派新 id，避免 TrackEval 崩潰且不丟失偵測結果
            for cls0, conf, x, y, bw, bh, lk in lst_sorted[1:]:
                out_lines.append(
                    f"{frame},{new_tid},"
                    f"{float_fmt.format(x)},{float_fmt.format(y)},"
                    f"{float_fmt.format(bw)},{float_fmt.format(bh)},"
                    f"{conf:.6f}"
                )
                new_tid += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""),
        encoding="utf-8",
    )

    # 摘要
    print(f"✅ MOT tracker txt saved: {out_path.resolve()} (lines={len(out_lines)})")
    if verbose:
        print(f"  total label lines read      : {total_lines}")
        print(f"  kept lines (after cls filter): {kept_lines}")
        print(f"  exact dup lines dropped     : {exact_dups_dropped}")
        print(f"  parse_bad lines             : {parse_bad}")
        print(f"  duplicate-id conflicts fixed: {dup_id_conflicts}")
        print(f"  max tid seen                : {max_tid_seen}")
        print(f"  next new tid start          : {max_tid_seen+1}")

    if len(out_lines) == 0:
        print("⚠️ tracker txt is empty → keep_cls_ids 選錯或 labels 沒那個類。")


# =========================================================
# GT gt.txt 唯一 ID 修正
# =========================================================
def fix_mot_gt_unique_ids(
    src_gt_txt: Path,
    dst_gt_txt: Path,
    keep_policy: str = "largest_area",  # "largest_area" or "first"
    verbose: bool = True,
):
    """
    修正 GT 使得每個 frame 內每個 ID 最多出現一次。

    MOT gt 格式（常見）：
      frame, id, x, y, w, h, conf, class, vis

    策略：
      - 以 frame 分群，同 id 出現多次時保留一個（依 keep_policy），
        其餘重新指派新 id，新 id 從 max(現有非負 id)+1 開始。
    """
    src_gt_txt = Path(src_gt_txt)
    dst_gt_txt = Path(dst_gt_txt)
    lines = src_gt_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        dst_gt_txt.parent.mkdir(parents=True, exist_ok=True)
        dst_gt_txt.write_text("", encoding="utf-8")
        return

    parsed = []
    max_id = -1
    bad = 0

    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 6:
            bad += 1
            continue
        try:
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except Exception:
            bad += 1
            continue

        if tid >= 0:
            max_id = max(max_id, tid)

        area = max(0.0, w) * max(0.0, h)
        parsed.append((frame, tid, area, parts))

    # 分群: frame -> id -> list[record]
    by_frame = {}
    for rec in parsed:
        frame, tid, area, parts = rec
        by_frame.setdefault(frame, {}).setdefault(tid, []).append(rec)

    new_id = max_id + 1
    changes = 0
    out_rows = []

    for frame in sorted(by_frame.keys()):
        id_map = by_frame[frame]
        for tid, recs in id_map.items():
            if len(recs) == 1:
                out_rows.append(recs[0][3])
                continue

            # 選擇保留哪筆
            if keep_policy == "first":
                keeper = recs[0]
                rest = recs[1:]
            else:
                # largest_area
                recs_sorted = sorted(recs, key=lambda r: r[2], reverse=True)
                keeper = recs_sorted[0]
                rest = recs_sorted[1:]

            out_rows.append(keeper[3])

            # 其餘重新指派 id
            for r in rest:
                parts2 = r[3].copy()
                parts2[1] = str(new_id)
                new_id += 1
                changes += 1
                out_rows.append(parts2)

    # 依 frame, id 排序
    def _key(p):
        try:
            return (int(float(p[0])), int(float(p[1])))
        except Exception:
            return (10**18, 10**18)

    out_rows = sorted(out_rows, key=_key)

    dst_gt_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_gt_txt.write_text(
        "\n".join([",".join(r) for r in out_rows]) + "\n",
        encoding="utf-8",
    )

    if verbose:
        print(f"✅ GT unique-id fix written: {dst_gt_txt.resolve()}")
        print(f"  bad_lines_skipped: {bad}")
        print(f"  reassignments    : {changes}")
        print(f"  new_id_end       : {new_id}")

# =========================================================
# GT gt.txt 判斷 座標 落在多邊形
# =========================================================
def keep_mot_gt_valid_ids(
    gt_txt: Path,
    polygon: np.ndarray,
    center_mode: str = "bbox_center",  # "bbox_center" or "top_center"
    verbose: bool = True,
):
    """
    過濾 MOT GT 檔案，只保留中心點落在多邊形區域內的 bbox。

    MOT gt 格式（常見）：
      frame, id, x, y, w, h, conf, class, vis
      其中 x, y 為左上角座標，w, h 為寬高。

    參數：
      gt_txt      : GT 檔案路徑（in-place 覆寫）
      polygon     : np.ndarray, shape (N, 2), dtype float32，多邊形頂點座標
      center_mode : 中心點計算方式
                    "bbox_center" -> (x + w/2, y + h/2)
                    "top_center"  -> (x + w/2, y)
      verbose     : 是否印出統計資訊

    cv2.pointPolygonTest 回傳值：
      > 0  : 點在多邊形內部  → 保留
      == 0 : 點在邊界上      → 保留
      < 0  : 點在多邊形外部  → 過濾掉
    """
    # gt_txt = Path(gt_txt)
    lines = gt_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    if not lines:
        return

    poly = polygon.reshape((-1, 1, 2)).astype(np.float32)

    parsed_parts = []
    bad = 0

    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 6:
            bad += 1
            continue
        try:
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except Exception:
            bad += 1
            continue
        parsed_parts.append(parts)

    kept = 0
    filtered = 0
    out_rows = []

    for parts in parsed_parts:
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])

        if center_mode == "top_center":
            cx, cy = x + w / 2.0, y
        else:  # bbox_center
            cx, cy = x + w / 2.0, y + h / 2.0

        result = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)

        if result >= 0:  # 內部或邊界 → 保留
            out_rows.append(parts)
            kept += 1
        else:
            filtered += 1

    # 依 frame, id 排序
    def _key(p):
        try:
            return (int(float(p[0])), int(float(p[1])))
        except Exception:
            return (10**18, 10**18)

    out_rows.sort(key=_key)

    # 新增一個 new_gt_txt，放在與 gt_txt 相同目錄下
    # new_gt_txt = gt_txt.parent / ("new_" + gt_txt.name)
    # new_gt_txt.write_text(
    #     "\n".join([",".join(r) for r in out_rows]) + "\n",
    #     encoding="utf-8",
    # )
     
    # 這是重新寫入 (overwrite) gt_txt
    gt_txt.write_text(
        "\n".join([",".join(r) for r in out_rows]) + "\n",
        encoding="utf-8",
    )

    if verbose:
        print(f"✅ GT valid-id filter written: {gt_txt.resolve()}")
        print(f"  orign num         : {len(lines)}")
        print(f"  bad_lines_skipped : {bad}")
        print(f"  kept              : {kept}")
        print(f"  filtered_out      : {filtered}")
        print(f"  center_mode       : {center_mode}")        