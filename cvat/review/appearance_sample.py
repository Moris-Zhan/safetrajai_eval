#!/usr/bin/env python3
"""
身分去重（dedup_clusters.py）沒辦法合併的獨立 track，用純外觀（DINOv2 特徵）
分成 K 群，每群抽 1 條代表給人工看，把覆核量從 N 條壓到 K 條。

注意：群內其餘 track **不是同一個實體物件**，只是外觀相近；代表 track 的判決
沒辦法安全套用到整群，只能當作「這個外觀類型大致上常不常出事」的參考，群內
沒被抽到代表的 track 仍然是未知風險，不會被自動修正（跟 dedup_clusters.py
的身分去重是兩種不同保證層級，使用者已在 2026-11-16 確認接受這個取捨）。

用法（conda env track_eval，需要 cv2/numpy/torch/timm/scikit-learn）：
    conda run -n track_eval python cvat/review/appearance_sample.py \
        --label small_truck --track-ids-csv runs/cvat_review/small_truck_singles.csv \
        --k 14 --out runs/cvat_review/small_truck_appearance_sample.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from safetrajai.reid.dinov2_features import letterbox, MODEL_NAME, INPUT_SIZE  # noqa: E402


def embed_one_crop(model, mean, std, device, track: dict, manifest: list[dict], images_root: Path) -> np.ndarray | None:
    real = [s for s in track["shapes"] if not s["outside"]]
    if not real:
        return None
    shape = real[len(real) // 2]
    row = manifest[shape["frame"]]
    img_path = images_root / row["seq"] / f"{int(row['frame_idx']):06d}.jpg"
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    x1, y1, x2, y2 = (max(0, int(round(v))) for v in shape["points"])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    import torch
    batch = np.stack([letterbox(crop, INPUT_SIZE)])
    tensor = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div(255.0)
    with torch.no_grad():
        feat = model((tensor - mean) / std)
    return torch.nn.functional.normalize(feat, dim=1).cpu().numpy()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path,
                         default=PROJECT_ROOT / "runs" / "cvat_review" / "pilot_v1_annotations.json")
    parser.add_argument("--manifest", type=Path,
                         default=PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv")
    parser.add_argument("--images-root", type=Path, default=PROJECT_ROOT / "dataset" / "detection" / "images")
    parser.add_argument("--label", required=True)
    parser.add_argument("--track-ids-csv", type=Path, required=True, help="dedup_clusters.py 產生的 singles CSV")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.track_ids_csv.open(newline="", encoding="utf-8") as f:
        target_ids = [int(r["track_id"]) for r in csv.DictReader(f)]

    import json
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    tracks = {t["track_id"]: t for t in payload["tracks"] if t["label"] == args.label}

    import timm
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE).eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    ids, vecs = [], []
    for i, tid in enumerate(target_ids):
        v = embed_one_crop(model, mean, std, device, tracks[tid], manifest, args.images_root)
        if v is not None:
            ids.append(tid)
            vecs.append(v)
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(target_ids)}")
    matrix = np.stack(vecs)
    print(f"embedded {len(ids)}/{len(target_ids)} tracks")

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10).fit(matrix)

    rows = []
    for cluster_id in range(args.k):
        members = [ids[i] for i in range(len(ids)) if km.labels_[i] == cluster_id]
        if not members:
            continue
        centroid = km.cluster_centers_[cluster_id]
        member_vecs = matrix[[i for i in range(len(ids)) if km.labels_[i] == cluster_id]]
        dists = np.linalg.norm(member_vecs - centroid, axis=1)
        representative = members[int(np.argmin(dists))]
        rows.append({"cluster_id": cluster_id, "size": len(members),
                      "representative_track_id": representative, "member_track_ids": ";".join(str(m) for m in members)})
    rows.sort(key=lambda r: -r["size"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "size", "representative_track_id", "member_track_ids"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} appearance clusters (k={args.k}) -> {args.out}")


if __name__ == "__main__":
    main()
