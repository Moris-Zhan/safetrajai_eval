"""評估前的忽略區域（ignore region）前處理，見 CONTEXT.md「忽略區域」、docs/adr/0005。

採 UA-DETRAC 式語意，逐幀、逐類別：
1. GT 框中心點在 ROI 外者標為 ignore。
2. 預測與該類別全部 GT（含 ignore）以 IoU 做一對一最佳匹配，IoU < 門檻的配對作廢。
3. 匹配到有效 GT（ROI 內）的預測一律保留；其餘預測中，匹配到 ignore GT 或中心點在 ROI 外者移除。
4. 移除全部 ignore GT。
因此 FP 只會來自 ROI 內；跨越 ROI 邊界、匹配到 ROI 內 GT 的預測仍計為 TP。

只處理 stage2 輸出目錄中的 GT shadow 與 tracker 副本。tracker 原始副本第一次處理時
備份到 <root>/trackers_raw/，之後一律從備份重新計算，重跑 stage2 結果不變。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from safetrajai.classes import CLASS_NAMES
from safetrajai.roi import is_center_in_roi

IOU_THRESHOLD = 0.5

_CLASS_ID_BY_NAME = {name: cls_id for cls_id, name in CLASS_NAMES.items()}


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes 為 (N, 4) 的 left, top, w, h。"""
    a_x2 = boxes_a[:, 0] + boxes_a[:, 2]
    a_y2 = boxes_a[:, 1] + boxes_a[:, 3]
    b_x2 = boxes_b[:, 0] + boxes_b[:, 2]
    b_y2 = boxes_b[:, 1] + boxes_b[:, 3]
    inter_w = np.clip(np.minimum(a_x2[:, None], b_x2[None, :]) - np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0]), 0, None)
    inter_h = np.clip(np.minimum(a_y2[:, None], b_y2[None, :]) - np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1]), 0, None)
    inter = inter_w * inter_h
    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ignore_masks(
    gt_rows: np.ndarray,
    pred_rows: np.ndarray,
    roi: np.ndarray,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (gt_keep, pred_keep) 布林遮罩。

    gt_rows、pred_rows 為同一類別的 MOT 列，欄位 0..5 是 frame, id, left, top, w, h。
    """
    gt_ignore = np.array(
        [not is_center_in_roi(roi, r[2] + r[4] / 2, r[3] + r[5] / 2) for r in gt_rows], dtype=bool,
    )
    pred_in_roi = np.array(
        [is_center_in_roi(roi, r[2] + r[4] / 2, r[3] + r[5] / 2) for r in pred_rows], dtype=bool,
    )
    matched_valid = np.zeros(len(pred_rows), dtype=bool)
    matched_ignore = np.zeros(len(pred_rows), dtype=bool)

    if len(gt_rows) and len(pred_rows):
        for frame in np.intersect1d(gt_rows[:, 0], pred_rows[:, 0]):
            gt_idx = np.flatnonzero(gt_rows[:, 0] == frame)
            pred_idx = np.flatnonzero(pred_rows[:, 0] == frame)
            iou = _iou_matrix(gt_rows[gt_idx, 2:6], pred_rows[pred_idx, 2:6])
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] < iou_threshold:
                    continue
                if gt_ignore[gt_idx[r]]:
                    matched_ignore[pred_idx[c]] = True
                else:
                    matched_valid[pred_idx[c]] = True

    pred_keep = matched_valid | (pred_in_roi & ~matched_ignore)
    return ~gt_ignore, pred_keep


def _read_rows(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()] if path.exists() else []
    values = np.array([[float(v) for v in ln.split(",")[:6]] for ln in lines], dtype=np.float64).reshape(-1, 6)
    return lines, values


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")


def apply_ignore_region(
    trackeval_root: Path,
    split_folder: str,
    seq_name: str,
    tracker_names: list[str],
    roi: np.ndarray,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, tuple[int, int]]:
    """對 stage2 輸出目錄中的 GT shadow 與指定 tracker 副本套用忽略區域，原地改寫。

    tracker 名稱格式為 <tracker>__<class>。回傳 {class_name: (ignore GT 數, 移除的預測數)}。
    """
    root = Path(trackeval_root)
    gt_txt = root / "gt" / "mot_challenge" / split_folder / seq_name / "gt" / "gt.txt"
    gt_lines, gt_values = _read_rows(gt_txt)
    gt_classes = np.array([int(float(ln.split(",")[7])) for ln in gt_lines], dtype=int)

    gt_keep_all = np.array(
        [is_center_in_roi(roi, r[2] + r[4] / 2, r[3] + r[5] / 2) for r in gt_values], dtype=bool,
    )
    stats: dict[str, tuple[int, int]] = {}

    for tracker_name in tracker_names:
        class_name = tracker_name.rsplit("__", 1)[1]
        mot_class_id = _CLASS_ID_BY_NAME[class_name] + 1

        rel = Path("mot_challenge") / split_folder / tracker_name / "data" / f"{seq_name}.txt"
        trk_txt = root / "trackers" / rel
        raw_txt = root / "trackers_raw" / rel
        if not raw_txt.exists():
            if not trk_txt.exists():
                continue
            raw_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trk_txt, raw_txt)

        pred_lines, pred_values = _read_rows(raw_txt)
        class_mask = gt_classes == mot_class_id
        _, pred_keep = ignore_masks(gt_values[class_mask], pred_values, roi, iou_threshold)
        _write_lines(trk_txt, [ln for ln, keep in zip(pred_lines, pred_keep) if keep])
        stats[class_name] = (int((~gt_keep_all[class_mask]).sum()), int((~pred_keep).sum()))

    _write_lines(gt_txt, [ln for ln, keep in zip(gt_lines, gt_keep_all) if keep])
    return stats
