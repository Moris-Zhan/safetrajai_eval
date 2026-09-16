#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 身分跨類別一致性稽核（Tier 3，embedding 版）。

對每個 CCTV 來源身分逐張裁切抽取特徵向量（embedding，透過 encoders.py::ReID 包裝既有
偵測權重 models/yolo/yolo11x_dennis.pt 當通用特徵抽取器），與各類別中心點（該類別全部
非略過身分的平均 embedding）比較距離，取距離最近的類別當作該影像的判定類別，記錄與資料夾
所屬類別是否一致。只標記、不刪除／搬移任何原始影像檔案，對應
openspec/changes/reid-crop-identity-audit/specs/reid/identity-class-consistency/spec.md。

本檔原採「重跑偵測器比對類別」設計；因偵測器對已裁切、脫離場景脈絡的小圖定位能力不足
（person 類別 78% 裁切完全偵測不到任何框）而改採此版，理由見 design.md Decision 1（修訂版）。

DSEC 來源身分（資料夾名稱前綴 dsecdet_）一律略過——不計分、也不參與任何類別中心點的計算。

用法：
    python dataset/reid/check_class_consistency.py
    python dataset/reid/check_class_consistency.py --classes person small_truck
    python dataset/reid/check_class_consistency.py --reid-model models/yolo/yolo11x_dennis.pt \
        --out dataset/reid/class_consistency_flags.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# ── 專案根目錄加入 path（比照 module/grid_search/*.py 的慣例：本檔同樣位於 repo 根目錄
#    下兩層 dataset/reid/，直接執行 `python dataset/reid/check_class_consistency.py` 時
#    sys.path[0] 是 dataset/reid/ 而非 repo 根目錄，需手動補上才能 import module）──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module.reid.encoders import ReID

CLASSES = ["bus", "car", "person", "scooter", "small_truck", "truck"]
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
DSEC_PREFIX = "dsecdet_"
REID_MODEL_PATH = "models/yolo/yolo11x_dennis.pt"


def discover_identities(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_dir())


def list_images(identity_dir: Path) -> list[Path]:
    return sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def is_dsec_identity(identity_dir: Path) -> bool:
    return identity_dir.name.startswith(DSEC_PREFIX)


def embed_crop(reid: ReID, crop: np.ndarray) -> np.ndarray:
    """對單張已裁切影像抽取 L2-normalized embedding。

    ReID.__call__(img, dets) 設計上吃「完整畫面 + bbox」，內部呼叫 save_one_box 裁切；
    這裡把整張裁切圖當成 img，dets 給一個涵蓋全圖的 bbox（xywh），save_one_box 預設的
    gain=1.02/pad=10 會被裁切邊界夾回原圖範圍，等於把整張裁切圖原封不動傳回去。
    """
    h, w = crop.shape[:2]
    dets = np.array([[w / 2.0, h / 2.0, w, h]], dtype=np.float32)
    feats = reid(crop, dets)
    vec = np.asarray(feats[0], dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_class_centroids(crop_root: Path, classes: list[str], reid: ReID):
    """兩件事一次做完：(1) 每個類別的中心點；(2) 每個身分逐張影像的 embedding 快取
    （避免中心點算完後，逐影像判定階段要重複抽一次特徵）。

    中心點＝該類別全部非 DSEC 身分的「身分平均 embedding」再平均（不是全部影像直接平均），
    避免張數多的身分主導中心點。
    """
    per_class_identity_means: dict[str, list[np.ndarray]] = {c: [] for c in classes}
    embedding_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for cls in classes:
        class_dir = crop_root / cls
        if not class_dir.is_dir():
            continue
        for identity_dir in discover_identities(class_dir):
            if is_dsec_identity(identity_dir):
                continue
            per_image: dict[str, np.ndarray] = {}
            for img_path in list_images(identity_dir):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                per_image[str(img_path)] = embed_crop(reid, img)
            embedding_cache[(cls, identity_dir.name)] = per_image
            if per_image:
                identity_mean = np.mean(list(per_image.values()), axis=0)
                per_class_identity_means[cls].append(identity_mean)

    centroids = {
        cls: np.mean(means, axis=0)
        for cls, means in per_class_identity_means.items()
        if means
    }
    return centroids, embedding_cache


def nearest_class(embedding: np.ndarray, centroids: dict[str, np.ndarray]) -> str | None:
    best_cls, best_dist = None, None
    for cls, centroid in centroids.items():
        dist = float(np.linalg.norm(embedding - centroid))
        if best_dist is None or dist < best_dist:
            best_cls, best_dist = cls, dist
    return best_cls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    parser.add_argument("--reid-model", default=REID_MODEL_PATH)
    parser.add_argument("--out", default="dataset/reid/class_consistency_flags.csv")
    args = parser.parse_args()

    crop_root = Path(args.crop_root)
    reid = ReID(args.reid_model)

    print("Pass 1/2: 抽 embedding、計算各類別中心點...")
    centroids, embedding_cache = compute_class_centroids(crop_root, args.classes, reid)
    for cls in args.classes:
        print(f"  {cls}: 中心點 {'OK' if cls in centroids else '無有效身分，略過'}")

    print("Pass 2/2: 逐影像判定...")
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
            per_image = embedding_cache.get((cls, identity_dir.name), {})
            rows = []
            for img_path_str, embedding in per_image.items():
                predicted_class = nearest_class(embedding, centroids)
                dist_to_own = float(np.linalg.norm(embedding - centroids[cls])) if cls in centroids else None
                agree = predicted_class == cls
                rows.append({
                    "class": cls, "identity": identity_dir.name, "image": img_path_str,
                    "predicted_class": predicted_class or "",
                    "distance": round(dist_to_own, 4) if dist_to_own is not None else "",
                    "agree": agree,
                })
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
        writer = csv.DictWriter(f, fieldnames=["class", "identity", "image", "predicted_class", "distance", "agree"])
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
