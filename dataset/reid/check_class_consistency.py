#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 身分跨類別一致性稽核（Tier 3）。

對每個 CCTV 來源身分逐張裁切重新呼叫本專案既有 YOLO 偵測器，比對偵測結果的最高信心
類別是否等於資料夾所屬類別，用來找出「整個身分可能歸錯類別」或「身分內因追蹤器 ID
switch 混入其他物體」的候選項目。只標記、不刪除／搬移任何原始影像檔案，對應
openspec/changes/reid-crop-identity-audit/specs/reid/identity-class-consistency/spec.md。

DSEC 來源身分（資料夾名稱前綴 dsecdet_）一律略過評分——reid-training-data-pipeline
的 pilot 已證實本專案偵測器在 DSEC 視角對 bus 類別系統性誤判，用同一顆偵測器驗證
同視角資料沒有意義，見 design.md Decision 2。

用法：
    python dataset/reid/check_class_consistency.py
    python dataset/reid/check_class_consistency.py --classes person small_truck
    python dataset/reid/check_class_consistency.py --config config/botsort_noreid.yaml \
        --out dataset/reid/class_consistency_flags.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

# ── 專案根目錄加入 path（比照 module/grid_search/*.py 的慣例：本檔同樣位於 repo 根目錄
#    下兩層 dataset/reid/，直接執行 `python dataset/reid/check_class_consistency.py` 時
#    sys.path[0] 是 dataset/reid/ 而非 repo 根目錄，需手動補上才能 import evaluation/module）──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config_loader import load_config
from module.detection.detector import Detector
from module.tracker.config import CLASS_NAMES, load_track_config

CLASSES = ["bus", "car", "person", "scooter", "small_truck", "truck"]
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
DSEC_PREFIX = "dsecdet_"


def discover_identities(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_dir())


def list_images(identity_dir: Path) -> list[Path]:
    return sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def is_dsec_identity(identity_dir: Path) -> bool:
    return identity_dir.name.startswith(DSEC_PREFIX)


def predicted_class_for_crop(detector: Detector, crop, iou: float, imgsz: int):
    """對單張裁切推論，回傳信心最高偵測框的類別名稱與信心值；沒有框通過門檻則回傳 (None, 0.0)。"""
    boxes = detector.detect(crop, iou=iou, imgsz=imgsz)
    if len(boxes) == 0:
        return None, 0.0
    best_idx = int(boxes.conf.argmax())
    cls_id = int(boxes.cls[best_idx])
    conf = float(boxes.conf[best_idx])
    return CLASS_NAMES.get(cls_id, f"class_{cls_id}"), conf


def check_identity(identity_dir: Path, assigned_class: str, detector: Detector, iou: float, imgsz: int) -> list[dict]:
    rows = []
    for img_path in list_images(identity_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        predicted_class, conf = predicted_class_for_crop(detector, img, iou, imgsz)
        agree = predicted_class == assigned_class
        rows.append({
            "class": assigned_class,
            "identity": identity_dir.name,
            "image": str(img_path),
            "predicted_class": predicted_class or "",
            "confidence": round(conf, 4),
            "agree": agree,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    parser.add_argument("--config", default="config/botsort_noreid.yaml")
    parser.add_argument("--out", default="dataset/reid/class_consistency_flags.csv")
    args = parser.parse_args()

    # 必須先載入 track config 才能取得正確的 CLASS_NAMES／class_conf_thresh，
    # 順序比照 track_eval.py 的 CLI 入口（load_track_config 先於 Detector 建立）。
    load_track_config(args.config)
    cfg = load_config(args.config)
    detector = Detector(cfg.det_model_path)

    crop_root = Path(args.crop_root)
    all_rows: list[dict] = []
    identity_summaries: list[dict] = []

    for cls in args.classes:
        class_dir = crop_root / cls
        if not class_dir.is_dir():
            print(f"skip {cls}: {class_dir} not found")
            continue
        identity_dirs = discover_identities(class_dir)
        n_dsec = 0
        for identity_dir in identity_dirs:
            if is_dsec_identity(identity_dir):
                n_dsec += 1
                identity_summaries.append({
                    "class": cls, "identity": identity_dir.name,
                    "n_images": len(list_images(identity_dir)),
                    "agreement_rate": "", "skipped_reason": "dsec_source",
                })
                continue
            rows = check_identity(identity_dir, cls, detector, cfg.iou, cfg.imgsz)
            all_rows.extend(rows)
            n = len(rows)
            n_agree = sum(1 for r in rows if r["agree"])
            identity_summaries.append({
                "class": cls, "identity": identity_dir.name, "n_images": n,
                "agreement_rate": round(n_agree / n, 4) if n else "",
                "skipped_reason": "",
            })
        print(f"  {cls}: {len(identity_dirs)} identities scanned ({n_dsec} DSEC skipped)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "identity", "image", "predicted_class", "confidence", "agree"])
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_path.with_name(out_path.stem + "_identities.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "identity", "n_images", "agreement_rate", "skipped_reason"])
        writer.writeheader()
        scored = [r for r in identity_summaries if r["agreement_rate"] != ""]
        skipped = [r for r in identity_summaries if r["agreement_rate"] == ""]
        scored.sort(key=lambda r: r["agreement_rate"])
        writer.writerows(scored + skipped)

    n_mismatch = sum(1 for r in all_rows if not r["agree"])
    print(
        f"\ndone: scanned {len(all_rows)} images across {len(identity_summaries)} identities, "
        f"{n_mismatch} image-level mismatches -> {out_path}\n"
        f"identity summary (sorted by agreement_rate) -> {summary_path}"
    )
    print("只標記，未刪除／搬移任何原始檔案。")


if __name__ == "__main__":
    main()
