# trackeval_runner.py
# -*- coding: utf-8 -*-
"""
TrackEval 執行相關工具：
- GT shadow layout 準備（flat -> MOT Challenge 格式）
- seqinfo.ini 生成
- GT 格式驗證
- TrackEval 評估執行（run_trackeval_mot / run_one_class）
"""

from pickle import FALSE
import re
import shutil
from pathlib import Path
from typing import List
import numpy as np

import cv2
import trackeval

# from utils.track_eval.utils import config_loader
from utils.track_eval.utils import (
    _assert_exists,
    file_md5,
    get_video_wh,
    print_file_probe,
)
from utils.track_eval.mot_converter import (
    ul_track_labels_to_mot,
    fix_mot_gt_unique_ids,
    keep_mot_gt_valid_ids
)

class_name_to_class_id = {
    'person': 1,
    'car': 2,
    'scooter': 3,
    'small_truck': 4,
    'bus': 5,
    'truck': 6,
}

class_id_to_class_name = {v: k for k, v in class_name_to_class_id.items()}



# =========================================================
# GT 重複 ID 檢查（除錯用）
# =========================================================
def dump_gt_lines_for_frame_id(gt_txt: Path, frame: int, tid: int, max_lines=50):
    """印出 GT 中特定 frame + id 的所有對應行（除錯用）。"""
    gt_txt = Path(gt_txt)
    lines = gt_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    hit = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            f = int(float(parts[0]))
            i = int(float(parts[1]))
        except Exception:
            continue
        if f == frame and i == tid:
            hit.append(ln)
            if len(hit) >= max_lines:
                break
    print(f"\n=== GT lines for frame={frame}, id={tid} (count={len(hit)}) ===")
    for h in hit:
        print(h)


def check_gt_duplicate_ids_per_frame(gt_txt: Path, verbose_topk=20, verbose=False):
    """檢查 GT 是否有同 frame 重複 ID，回傳 True 表示無重複。"""
    gt_txt = Path(gt_txt)
    lines = gt_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    dup_list = []
    by_frame = {}

    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
        except Exception:
            continue
        by_frame.setdefault(frame, {}).setdefault(tid, 0)
        by_frame[frame][tid] += 1

    for f, m in by_frame.items():
        for tid, c in m.items():
            if c > 1:
                dup_list.append((f, tid, c))

    dup_list.sort(key=lambda x: (x[0], x[1]))
    if dup_list:
        if verbose: 
            print(f"❌ GT duplicate IDs per frame still exist: {len(dup_list)} cases")
        for item in dup_list[:verbose_topk]:
            if verbose: 
                print("  frame:", item[0], "id:", item[1], "count:", item[2])
        return False
    else:
        if verbose: 
            print("✅ GT duplicate IDs per frame: NONE")
        return True


# =========================================================
# seqinfo.ini 生成
# =========================================================
def _write_seqinfo_ini(dst_ini: Path, seq_name: str, video_path: str):
    """
    根據影片資訊生成最小化的 MOT seqinfo.ini，
    供 TrackEval / MOTChallenge2DBox 使用。
    """
    W, H = get_video_wh(video_path)
    cap = cv2.VideoCapture(str(video_path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if nframes <= 0:
        # 無法取得 frame count 時退回 0，TrackEval 會警告但仍可執行
        nframes = 0

    text = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=img1\n"
        f"frameRate=30\n"
        f"seqLength={nframes}\n"
        f"imWidth={W}\n"
        f"imHeight={H}\n"
        "imExt=.jpg\n"
    )
    dst_ini.parent.mkdir(parents=True, exist_ok=True)
    dst_ini.write_text(text, encoding="utf-8")


# =========================================================
# GT shadow layout 準備
# =========================================================
def prepare_mot_gt_shadow(
    trackeval_root: str,
    benchmark_name: str,
    split_name: str,
    seq_name: str,
    video_path: str,
    det_corner: None | List,
    verbose: bool = True,
):
    """
    將 flat GT layout 轉換成 TrackEval 需要的 MOT Challenge 巢狀目錄結構。

    Flat GT layout（你目前的格式）：
      <trackeval_root>/gt/gt.txt
      <trackeval_root>/seqinfo.ini   OR  <trackeval_root>/gt/seqinfo.ini  (optional)

    TrackEval 需要的格式：
      <trackeval_root>/gt/mot_challenge/<benchmark>-<split>/<seq>/gt/gt.txt
      <trackeval_root>/gt/mot_challenge/<benchmark>-<split>/<seq>/seqinfo.ini

    額外處理：
    - 修正 GT 中同 frame 重複 ID（fix_mot_gt_unique_ids）
    - 若 seqinfo.ini 不存在，從影片自動生成
    """
    root = Path(trackeval_root).resolve()
    split_folder = f"{benchmark_name}-{split_name}"

    # 來源 GT（flat）full points
    src_gt_txt = root / "gt" / "gt.txt"
    if not src_gt_txt.exists():
        raise FileNotFoundError(f"Flat GT not found: {src_gt_txt}")

    # 來源 seqinfo.ini（可選）
    src_seq_ini = root / "seqinfo.ini"
    if not src_seq_ini.exists():
        alt = root / "gt" / "seqinfo.ini"
        if alt.exists():
            src_seq_ini = alt
        else:
            src_seq_ini = None  # 後面自動生成

    # 目標 shadow layout
    tgt_base = root / "gt" / "mot_challenge" / split_folder / seq_name
    tgt_gt_dir = tgt_base / "gt"
    tgt_gt_dir.mkdir(parents=True, exist_ok=True)

    tgt_gt_txt = tgt_gt_dir / "gt.txt"          # inside poly points
    tgt_seq_ini = tgt_base / "seqinfo.ini"

    def _copy_if_needed(src: Path, dst: Path):
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return
        shutil.copy2(src, dst)

    _copy_if_needed(src_gt_txt, tgt_gt_txt)

    # 修正 GT 重複 ID
    fix_mot_gt_unique_ids(tgt_gt_txt, tgt_gt_txt, keep_policy="largest_area", verbose=verbose)
    
    # 透過 video_path 取得影片解析度
    W, H = get_video_wh(video_path)

    # 根據 det_corner 決定 polygon
    if det_corner is not None:
        polygon = det_corner
    else:
        polygon = np.array([
            [0, 0],
            [W, 0],
            [W, H],
            [0, H],
        ], dtype=np.float32)

    # 去除不在 polygon 內之GT
    keep_mot_gt_valid_ids(
        gt_txt=tgt_gt_txt,
        polygon=polygon,
        center_mode="bbox_center",
        verbose=verbose,
    )

    # 驗證修正結果
    ok = check_gt_duplicate_ids_per_frame(tgt_gt_txt, verbose_topk=30 if verbose else 0)
    if not ok:
        raise RuntimeError(
            f"GT still has duplicate IDs per frame after fixing: {tgt_gt_txt}"
        )

    if src_seq_ini is not None:
        _copy_if_needed(src_seq_ini, tgt_seq_ini)
    else:
        # 自動生成 seqinfo.ini
        _write_seqinfo_ini(tgt_seq_ini, seq_name=seq_name, video_path=video_path)

    # 依據MOT格式，tgt_gt_txt第8欄(第7個index, 0-based)為category id
    from collections import Counter

    category_counts = Counter()
    try:
        with open(tgt_gt_txt, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                cols = line.strip().split(",")
                if len(cols) < 8:
                    continue
                cat = cols[7]
                category_counts[cat] += 1
    except Exception as e:
        print(f"讀取{tgt_gt_txt}時發生錯誤: {e}")

    if verbose and category_counts:
        BLUE = "\033[94m"
        RESET = "\033[0m"
        print(f"{BLUE}{tgt_gt_txt.parent.parent.stem} GT 類別物件數:{RESET}")
        for cat, count in category_counts.items():
            print(f"{BLUE}  {class_id_to_class_name[int(cat)]}: {count} objects{RESET}")

    return tgt_gt_txt, tgt_seq_ini


# =========================================================
# TrackEval 路徑工具
# =========================================================
def build_trackeval_paths(trackeval_root, benchmark_name, split_name, seq_name, tracker_name):
    """回傳 TrackEval 常用路徑的字典（single source of truth）。"""
    root = Path(trackeval_root)
    split_folder = f"{benchmark_name}-{split_name}"

    gt_base = root / "gt" / "mot_challenge" / split_folder
    trk_base = root / "trackers" / "mot_challenge" / split_folder

    gt_txt = gt_base / seq_name / "gt" / "gt.txt"
    seq_ini = gt_base / seq_name / "seqinfo.ini"
    trk_txt = trk_base / tracker_name / "data" / f"{seq_name}.txt"

    return {
        "root": root,
        "split_folder": split_folder,
        "gt_base": gt_base,
        "trk_base": trk_base,
        "gt_txt": gt_txt,
        "seq_ini": seq_ini,
        "trk_txt": trk_txt,
    }


def verify_gt_basic(gt_txt: Path, seq_ini: Path):
    """基本 GT 格式驗證：確認檔案存在、行數足夠、欄數正確。"""
    _assert_exists(gt_txt, "GT gt.txt")
    _assert_exists(seq_ini, "GT seqinfo.ini")

    gt_lines = gt_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(gt_lines) < 10:
        raise RuntimeError(
            f"❌ GT too small (lines={len(gt_lines)}). You may be editing wrong gt.txt."
        )

    ini = seq_ini.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"seqLength\s*=\s*(\d+)", ini)
    if m:
        print("seqLength =", int(m.group(1)))
    else:
        print("⚠️ seqinfo.ini missing seqLength")

    gt_cols = len(gt_lines[0].split(","))
    print("GT cols =", gt_cols)
    if gt_cols not in (9, 10):
        print("⚠️ GT cols not in (9,10). Please confirm GT format for MotChallenge2DBox.")
    print("GT head:", gt_lines[0])


# =========================================================
# TrackEval 執行
# =========================================================
def run_trackeval_mot(
    cfg,
    trackeval_root: str,
    benchmark_name: str,
    split_name: str,
    seq_name: str,
    tracker_names_to_eval: list,
    out_root: str,
    video_path: str,
    classes_to_eval=None,
    allow_empty_tracker_file=True,
    verbose: bool = True,
    DEBUG: bool = False,
    ):
    """
    執行 TrackEval 評估（HOTA / CLEAR / Identity）。

    流程：
    1. prepare_mot_gt_shadow: 準備 GT shadow layout + 修正重複 ID
    2. 驗證 tracker txt 存在且非空
    3. 呼叫 trackeval.Evaluator.evaluate()

    Returns:
        dict: TrackEval 結果，若無有效 tracker 則回傳 None。
    """
    if classes_to_eval is None:
        classes_to_eval = ["person", "car", "scooter", "small_truck", "bus", "truck"]
    classes_to_eval = list(classes_to_eval)

    root = Path(trackeval_root)
    split_folder = f"{benchmark_name}-{split_name}"

    tgt_gt_txt, tgt_seq_ini = prepare_mot_gt_shadow(
        trackeval_root, benchmark_name, split_name, seq_name, video_path=video_path,
        det_corner=cfg.det_corner,
        verbose=verbose,
    )

    # 檢查 shadow GT 重複 ID（已在 prepare_mot_gt_shadow 內執行，此處僅在 DEBUG 時補充）
    if DEBUG:
        print("=== SHADOW GT USED BY TrackEval ===")
        check_gt_duplicate_ids_per_frame(tgt_gt_txt, verbose_topk=50)
        dump_gt_lines_for_frame_id(tgt_gt_txt, frame=2, tid=30, max_lines=20)

    # runs/track_eval/mot_annotations_all8/mot_annotation_<seq>/gt/mot_challenge/CCTV-val
    # 外層(gt)
    gt_base = root / "gt" / "mot_challenge" / split_folder

    # runs/track_eval/mot_annotations_all8/mot_annotation_<seq>/trackers/mot_challenge/CCTV-val
    # 外層(trackers)
    trk_base = root / "trackers" / "mot_challenge" / split_folder

    # runs/track_eval/mot_annotations_all8/mot_annotation_<seq>/gt/mot_challenge/CCTV-val/<seq>/gt/gt.txt
    # 內層
    gt_txt = gt_base / seq_name / "gt" / "gt.txt"
    seq_ini = gt_base / seq_name / "seqinfo.ini"

    if DEBUG:
        print("\n=== TrackEval base paths ===")
        print("GT_BASE :", gt_base)
        print("TRK_BASE:", trk_base)
        print("seq_name:", seq_name)
        print("classes_to_eval:", classes_to_eval)
        print("trackers_to_eval:", tracker_names_to_eval)

    _assert_exists(gt_txt, "GT gt.txt")
    _assert_exists(seq_ini, "GT seqinfo.ini")    

    # 篩選有效的 tracker：
    valid_trackers = []
    for tn in tracker_names_to_eval:
        trk_txt = trk_base / tn / "data" / f"{seq_name}.txt"
        if DEBUG:
            print("\n--- Tracker check:", tn, "---")
            print("TRK:", trk_txt)

        # 若檔案不存在，略過此 tracker
        if not trk_txt.exists():
            if DEBUG:
                print("❌ tracker txt missing -> skip:", tn)
            continue

        size = trk_txt.stat().st_size
        # 若檔案為空且不允許空 tracker file，略過
        if size == 0 and not allow_empty_tracker_file:
            if DEBUG:
                print("❌ tracker txt empty -> skip:", tn)
            continue

        if DEBUG:
            # 檔案存在且符合大小條件，列為有效並輸出前3行與md5
            print("md5:", file_md5(trk_txt), "bytes:", size)
            head = "\n".join(trk_txt.read_text(encoding="utf-8", errors="ignore").splitlines()[:3])
            print("head:\n", head)
        valid_trackers.append(tn)

    if not valid_trackers:
        if DEBUG:
            print("⚠️ No valid tracker files found for this run.")
        return None

    eval_config = trackeval.Evaluator.get_default_eval_config()
    # eval_config.update({
    #     "USE_PARALLEL": False,
    #     "PRINT_RESULTS": True,
    #     "OUTPUT_SUMMARY": True,
    #     "OUTPUT_DETAILED": False,
    #     "OUTPUT_FOLDER": out_root,
    # })
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": False,
        "OUTPUT_FOLDER": out_root,
    })
    evaluator = trackeval.Evaluator(eval_config)

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_base),
        "TRACKERS_FOLDER": str(trk_base),
        "SKIP_SPLIT_FOL": True,
        "BENCHMARK": benchmark_name,
        "SPLIT_TO_EVAL": split_name,
        "SEQ_INFO": {seq_name: None},
        "TRACKERS_TO_EVAL": valid_trackers,
        "CLASSES_TO_EVAL": classes_to_eval,
        "PRINT_CONFIG": True,
    })

    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    ds = dataset_list[0]
    if DEBUG:
        print("TrackEval dataset class list:", getattr(ds, "class_list", None))
        print("TrackEval dataset class_name_to_class_id:", getattr(ds, "class_name_to_class_id", None))
        print("Requested CLASSES_TO_EVAL:", classes_to_eval)

    results = evaluator.evaluate(dataset_list, metrics_list)
    return results


# def run_one_class(
#     trackeval_root,
#     benchmark_name,
#     split_name,
#     seq_name,
#     base_tracker_name,
#     ul_track_labels_dir,
#     video_path,
#     out_root,
#     cls_id,
#     cls_name,
#     cfg=None,
#     verbose: bool = True,
#     debug: bool = False,
# ):
#     """
#     針對單一類別執行完整流程：
#     1. Ultralytics tracking labels -> MOT txt（ul_track_labels_to_mot）
#     2. run_trackeval_mot

#     Returns:
#         (results, tracker_variant, note)
#     """
#     paths = build_trackeval_paths(
#         trackeval_root, benchmark_name, split_name, seq_name, base_tracker_name
#     )

#     tracker_variant = f"{base_tracker_name}__{cls_name}"
#     trk_txt = paths["trk_base"] / tracker_variant / "data" / f"{seq_name}.txt"

#     print(f"\n=== BUILD TRK for class {cls_name} (cls_id={cls_id}) ===")
#     ul_track_labels_to_mot(
#         trk_label_dir=ul_track_labels_dir,
#         out_txt=str(trk_txt),
#         video_path=video_path,
#         keep_cls_ids=[cls_id],
#         tid_one_based=True,
#         drop_exact_dup_lines=True,   # ✅ 先把整包重複去掉
#         fix_duplicate_ids=True,      # ✅ 再修同 frame 同 ID
#         choose_keep="best_conf",     # ✅ 保留最高 conf 的那筆當原 ID
#         verbose=verbose,
#     )
#     note = ""
#     if trk_txt.exists() and trk_txt.stat().st_size == 0:
#         note = "tracker txt empty (no det for this class in tracking labels)"

#     results = run_trackeval_mot(
#         cfg,
#         trackeval_root=str(Path(trackeval_root)),
#         benchmark_name=benchmark_name,
#         split_name=split_name,
#         seq_name=seq_name,
#         tracker_names_to_eval=[tracker_variant],
#         out_root=out_root,
#         video_path=video_path,
#         classes_to_eval=[cls_name],
#         allow_empty_tracker_file=True,
#         verbose=verbose,
#         DEBUG=debug,
#     )

#     if results is None:
#         if not note:
#             note = "TrackEval skipped (missing tracker file)"
#         return None, tracker_variant, note

#     return results, tracker_variant, note