#!/usr/bin/env python3
"""
用 DINOv2 外觀特徵（沿用 scripts/reid_data/detect_contamination.py 的做法，見
utils/modules/safetrajai/reid/dinov2_features.py）掃描 CVAT task 某個類別的其餘
track，找出外觀跟已確認正確的參考 track 差異很大的候選，供人工覆核排序優先序——
不自動改標籤：這支只分對類別內「看起來奇怪」的 track，不知道正確類別是什麼。

每條 track 取最多 --samples-per-track 個真實關鍵幀（均勻間隔），裁切後算
DINOv2 嵌入向量平均成一個 track 向量，再跟每條參考 track 的向量比對 cosine
相似度，取最大值；最大相似度低的 track 排在候選名單前面。

用法（conda env track_eval，需要 cv2/numpy/torch/timm）：
    conda run -n track_eval python cvat/review/reid_outlier_scan.py \
        --label small_truck --reference-ids 7697,9932,... --exclude-ids 3329,3871,...
        --out runs/cvat_review/small_truck_outliers.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from safetrajai.reid.dinov2_features import letterbox, MODEL_NAME, INPUT_SIZE  # noqa: E402


def load_data(annotations_path: Path, manifest_path: Path):
    import json
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    return payload["tracks"], manifest


def sample_crops(track: dict, manifest: list[dict], images_root: Path, samples_per_track: int):
    real = [s for s in track["shapes"] if not s["outside"]]
    if not real:
        return []
    if len(real) <= samples_per_track:
        picks = real
    else:
        idxs = np.linspace(0, len(real) - 1, samples_per_track).round().astype(int)
        picks = [real[i] for i in sorted(set(idxs))]
    crops = []
    for shape in picks:
        row = manifest[shape["frame"]]
        img_path = images_root / row["seq"] / f"{int(row['frame_idx']):06d}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        x1, y1, x2, y2 = (max(0, int(round(v))) for v in shape["points"])
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def embed_batch(model, mean, std, device, crops: list[np.ndarray]):
    import torch
    batch = np.stack([letterbox(c, INPUT_SIZE) for c in crops])
    tensor = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div(255.0)
    with torch.no_grad():
        feats = model((tensor - mean) / std)
    return torch.nn.functional.normalize(feats, dim=1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path,
                         default=PROJECT_ROOT / "runs" / "cvat_review" / "pilot_v1_annotations.json")
    parser.add_argument("--manifest", type=Path,
                         default=PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv")
    parser.add_argument("--images-root", type=Path, default=PROJECT_ROOT / "dataset" / "detection" / "images")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference-ids", required=True, help="已確認正確的 track id，逗號分隔")
    parser.add_argument("--exclude-ids", default="", help="不納入掃描的 track id（例如剛修正過的），逗號分隔")
    parser.add_argument("--samples-per-track", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference_ids = {int(x) for x in args.reference_ids.split(",") if x}
    exclude_ids = {int(x) for x in args.exclude_ids.split(",") if x}

    import timm
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE).eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    tracks, manifest = load_data(args.annotations, args.manifest)
    by_label = {t["track_id"]: t for t in tracks if t["label"] == args.label}

    missing = reference_ids - set(by_label)
    if missing:
        raise SystemExit(f"參考 track 不在目前標註裡（可能已被刪除/改標）：{sorted(missing)}")

    def track_vector(track_id: int) -> np.ndarray | None:
        crops = sample_crops(by_label[track_id], manifest, args.images_root, args.samples_per_track)
        if not crops:
            return None
        feats = embed_batch(model, mean, std, device, crops)
        mean_vec = feats.mean(axis=0)
        return mean_vec / max(np.linalg.norm(mean_vec), 1e-12)

    ref_vectors = {tid: track_vector(tid) for tid in sorted(reference_ids)}
    ref_vectors = {tid: v for tid, v in ref_vectors.items() if v is not None}
    ref_matrix = np.stack(list(ref_vectors.values()))
    ref_ids_list = list(ref_vectors.keys())
    print(f"reference tracks embedded: {len(ref_ids_list)}")

    target_ids = sorted(set(by_label) - reference_ids - exclude_ids)
    rows = []
    for i, tid in enumerate(target_ids):
        vec = track_vector(tid)
        if vec is None:
            continue
        sims = ref_matrix @ vec
        best_idx = int(np.argmax(sims))
        real_n = len([s for s in by_label[tid]["shapes"] if not s["outside"]])
        first_pos = next(s["frame"] for s in by_label[tid]["shapes"] if not s["outside"])
        rows.append({
            "track_id": tid, "camera": manifest[first_pos]["camera"], "n_real_keyframes": real_n,
            "max_similarity": round(float(sims[best_idx]), 4), "nearest_reference_id": ref_ids_list[best_idx],
        })
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(target_ids)}")

    rows.sort(key=lambda r: r["max_similarity"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "camera", "n_real_keyframes", "max_similarity", "nearest_reference_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    print("lowest similarity (most anomalous) 15:")
    for r in rows[:15]:
        print(" ", r)


if __name__ == "__main__":
    main()
