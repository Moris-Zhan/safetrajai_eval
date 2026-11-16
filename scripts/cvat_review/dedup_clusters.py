#!/usr/bin/env python3
"""
把某個類別裡「同一個實體物件被拆成好幾條破碎 track」的狀況找出來合併成一群，
讓人工覆核一群只要看 1 條代表，而不是每條都看——沿用 2026-11-13 CVAT 覆核時
驗證過的方法（同鏡頭＋同位置＋同尺寸＋幾乎靜止＋DINOv2 外觀高度相似，兩個案例
分別是 C000256 的固定紅綠燈、C000033 的固定非車輛結構，見
Markdown/training_eval_audit.md）。

只分群，不自動改標籤／刪除：群集結果要先給人工看過代表 track、確認真的是同一個
實體物件，才能套用到整群（見 scripts/cvat_review/apply_corrections.py）。

判定條件（缺一不可，任何一項不符合就不成立群集，整群打散回單條）：
  1. 同一鏡頭。
  2. track 平均框中心兩兩距離 < --center-dist-px。
  3. 框尺寸（sqrt(w*h) 中位數）比值 < --size-ratio。
  4. 群內任何一條有 >=2 個真實關鍵幀的 track，其框尺寸正規化位移都必須
     < --max-displacement（代表幾乎靜止；只要有一條明顯在動，整群判定失敗，
     打散回單條——避免把「剛好路過同一個路口」的不同真實車輛誤判成同一物體）。
  5. 群內兩兩 DINOv2 外觀相似度平均 >= --min-appearance-similarity。

用法（conda env track_eval，需要 cv2/numpy/torch/timm）：
    conda run -n track_eval python scripts/cvat_review/dedup_clusters.py \
        --label small_truck --exclude-ids 7697,9932,... \
        --clusters-out runs/cvat_review/small_truck_clusters.csv \
        --singles-out runs/cvat_review/small_truck_singles.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from safetrajai.reid.dinov2_features import letterbox, MODEL_NAME, INPUT_SIZE  # noqa: E402


def track_stats(track: dict, manifest: list[dict]) -> dict | None:
    real = [s for s in track["shapes"] if not s["outside"]]
    if not real:
        return None
    pts = []
    for s in real:
        x1, y1, x2, y2 = s["points"]
        size = math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
        pts.append(((x1 + x2) / 2, (y1 + y2) / 2, max(size, 1.0)))
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    median_size = sorted(p[2] for p in pts)[len(pts) // 2]
    max_disp = None
    if len(pts) >= 2:
        max_disp = max(math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) for p in pts) / median_size
    camera = manifest[real[0]["frame"]]["camera"]
    return {"cx": cx, "cy": cy, "size": median_size, "max_disp": max_disp,
            "camera": camera, "n_real": len(real), "first_frame": real[0]["frame"]}


def union_find_clusters(ids: list[int], stats: dict[int, dict], center_dist_px: float, size_ratio: float) -> list[list[int]]:
    parent = {tid: tid for tid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_camera: dict[str, list[int]] = {}
    for tid in ids:
        by_camera.setdefault(stats[tid]["camera"], []).append(tid)

    for cam_ids in by_camera.values():
        for i, a in enumerate(cam_ids):
            sa = stats[a]
            for b in cam_ids[i + 1:]:
                sb = stats[b]
                dist = math.hypot(sa["cx"] - sb["cx"], sa["cy"] - sb["cy"])
                ratio = max(sa["size"], sb["size"]) / min(sa["size"], sb["size"])
                if dist < center_dist_px and ratio < size_ratio:
                    union(a, b)

    groups: dict[int, list[int]] = {}
    for tid in ids:
        groups.setdefault(find(tid), []).append(tid)
    return list(groups.values())


def embed_one_crop(model, mean, std, device, track: dict, manifest: list[dict], images_root: Path) -> np.ndarray | None:
    real = [s for s in track["shapes"] if not s["outside"]]
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
    vec = torch.nn.functional.normalize(feat, dim=1).cpu().numpy()[0]
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path,
                         default=PROJECT_ROOT / "runs" / "cvat_review" / "pilot_v1_annotations.json")
    parser.add_argument("--manifest", type=Path,
                         default=PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv")
    parser.add_argument("--images-root", type=Path, default=PROJECT_ROOT / "dataset" / "detection" / "images")
    parser.add_argument("--label", required=True)
    parser.add_argument("--exclude-ids", default="", help="已經人工覆核過的 track id，不納入分群，逗號分隔")
    parser.add_argument("--center-dist-px", type=float, default=25.0)
    parser.add_argument("--size-ratio", type=float, default=1.3)
    parser.add_argument("--max-displacement", type=float, default=0.5)
    parser.add_argument("--min-appearance-similarity", type=float, default=0.75)
    parser.add_argument("--clusters-out", type=Path, required=True)
    parser.add_argument("--singles-out", type=Path, required=True)
    args = parser.parse_args()

    exclude_ids = {int(x) for x in args.exclude_ids.split(",") if x}

    import json
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    tracks = {t["track_id"]: t for t in payload["tracks"] if t["label"] == args.label and t["track_id"] not in exclude_ids}
    stats = {}
    for tid, t in tracks.items():
        s = track_stats(t, manifest)
        if s is not None:
            stats[tid] = s
    print(f"{args.label}: {len(tracks)} tracks under review, {len(stats)} with at least 1 real keyframe")

    groups = union_find_clusters(list(stats.keys()), stats, args.center_dist_px, args.size_ratio)
    candidate_groups = [g for g in groups if len(g) >= 2]
    single_ids = [g[0] for g in groups if len(g) == 1]
    print(f"location/size candidate clusters (size>=2): {len(candidate_groups)}, singles: {len(single_ids)}")

    # motion veto: any member with real displacement kills the cluster
    valid_groups = []
    killed_by_motion = 0
    for g in candidate_groups:
        if any(stats[tid]["max_disp"] is not None and stats[tid]["max_disp"] >= args.max_displacement for tid in g):
            killed_by_motion += 1
            single_ids.extend(g)
            continue
        valid_groups.append(g)
    print(f"clusters killed by motion veto: {killed_by_motion} (members returned to singles)")

    import timm
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE).eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    cluster_rows = []
    killed_by_appearance = 0
    for i, g in enumerate(valid_groups):
        vecs = [embed_one_crop(model, mean, std, device, tracks[tid], manifest, args.images_root) for tid in g]
        vecs = [v for v in vecs if v is not None]
        if len(vecs) < 2:
            single_ids.extend(g)
            continue
        matrix = np.stack(vecs)
        sims = matrix @ matrix.T
        iu = np.triu_indices(len(vecs), k=1)
        mean_sim = float(sims[iu].mean())
        if mean_sim < args.min_appearance_similarity:
            killed_by_appearance += 1
            single_ids.extend(g)
            continue
        representative = max(g, key=lambda tid: stats[tid]["n_real"])
        cluster_rows.append({
            "cluster_id": i, "size": len(g), "camera": stats[g[0]]["camera"],
            "center_x": round(stats[g[0]]["cx"], 1), "center_y": round(stats[g[0]]["cy"], 1),
            "median_box_size": round(stats[g[0]]["size"], 1), "mean_appearance_similarity": round(mean_sim, 4),
            "representative_track_id": representative, "member_track_ids": ";".join(str(t) for t in g),
        })
        if (i + 1) % 20 == 0:
            print(f"  clustered {i + 1}/{len(valid_groups)}")

    print(f"clusters killed by appearance check: {killed_by_appearance}")
    cluster_rows.sort(key=lambda r: -r["size"])

    args.clusters_out.parent.mkdir(parents=True, exist_ok=True)
    with args.clusters_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "size", "camera", "center_x", "center_y",
                                                "median_box_size", "mean_appearance_similarity",
                                                "representative_track_id", "member_track_ids"])
        writer.writeheader()
        writer.writerows(cluster_rows)

    with args.singles_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "camera", "n_real_keyframes"])
        writer.writeheader()
        for tid in sorted(set(single_ids)):
            writer.writerow({"track_id": tid, "camera": stats[tid]["camera"], "n_real_keyframes": stats[tid]["n_real"]})

    n_in_clusters = sum(r["size"] for r in cluster_rows)
    print(f"confirmed clusters: {len(cluster_rows)} (covering {n_in_clusters} tracks) -> {args.clusters_out}")
    print(f"singles needing individual review: {len(set(single_ids))} -> {args.singles_out}")
    print(f"total review burden if you check 1 representative per cluster + all singles: "
          f"{len(cluster_rows) + len(set(single_ids))}")


if __name__ == "__main__":
    main()
