#!/usr/bin/env python3
"""
Draw the appearance-stratified acceptance sample for the user's second web task.

Per class, every track currently labeled with that class is embedded (DINOv2 features of the
middle-frame crop) and split into K clusters of similar-looking tracks. One random member of each
cluster becomes a question; its weight is the cluster size. The user answers only these questions
and the page reports a size-weighted disagreement rate per class (a stratified estimate of the
residual error; the representative is random, not the one nearest the centroid). A member the user
already answered in an earlier sample is kept as its cluster's representative when one exists.

Output: runs/cvat_review/acceptance_sample.json (read by serve_audit.py --sample) and
        runs/cvat_review/acceptance_embeddings.npz (cache, so reruns skip the feature extraction)

Usage (conda env track_eval, needs cv2/torch/timm/scikit-learn):
    conda run -n track_eval python cvat/review/build_acceptance_sample.py --clusters 20
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
MANIFEST = PROJECT_ROOT / "dataset" / "detection" / "splits" / "pilot_v1" / "manifest.csv"
IMAGES_ROOT = PROJECT_ROOT / "dataset" / "detection" / "images"
CLASSES = ("person", "car", "scooter", "small_truck", "bus", "truck")
BATCH = 64


def middle_shape(track: dict) -> Optional[dict]:
    real = [s for s in track["shapes"] if not s["outside"]]
    return real[len(real) // 2] if real else None


def pick_representatives(members_by_cluster: Dict[int, List[int]], answered: set, seed: int) -> Dict[int, int]:
    """One track per cluster: an already-answered member if any, otherwise a random member."""
    rng = random.Random(seed)
    chosen: Dict[int, int] = {}
    for cluster in sorted(members_by_cluster):
        members = sorted(members_by_cluster[cluster])
        pool = [m for m in members if m in answered] or members
        chosen[cluster] = rng.choice(pool)
    return chosen


def build_items(label: str, track_ids: Sequence[int], cluster_labels: Sequence[int], answered: set,
                seed: int) -> List[Dict[str, object]]:
    members: Dict[int, List[int]] = {}
    for track_id, cluster in zip(track_ids, cluster_labels):
        members.setdefault(int(cluster), []).append(int(track_id))
    chosen = pick_representatives(members, answered, seed)
    return [{"item_id": f"acceptance-{label}-{chosen[cluster]}", "kind": f"acceptance_{label}",
             "track_id": chosen[cluster], "old_label": label, "claimed_final": label, "reason": "",
             "cluster": cluster, "weight": len(members[cluster])} for cluster in sorted(chosen)]


def embed_tracks(tracks: List[dict], manifest: List[dict]) -> Dict[int, np.ndarray]:
    import cv2
    import timm
    import torch
    from safetrajai.reid.dinov2_features import MODEL_NAME, INPUT_SIZE, letterbox

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE).eval().to(device)
    mean = torch.tensor(model.pretrained_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"], device=device).view(1, 3, 1, 1)

    jobs = sorted(((s["frame"], t["track_id"], s["points"]) for t in tracks if (s := middle_shape(t))))
    vectors: Dict[int, np.ndarray] = {}
    crops: List[np.ndarray] = []
    ids: List[int] = []

    def flush() -> None:
        if not crops:
            return
        batch = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float().div(255.0)
        with torch.no_grad():
            feats = torch.nn.functional.normalize(model((batch - mean) / std), dim=1).cpu().numpy()
        vectors.update(zip(ids, feats))
        crops.clear()
        ids.clear()

    image, image_frame = None, -1
    for frame, track_id, points in jobs:
        if frame != image_frame:
            row = manifest[frame]
            image, image_frame = cv2.imread(str(IMAGES_ROOT / row["seq"] / f"{int(row['frame_idx']):06d}.jpg")), frame
        if image is None:
            continue
        x1, y1, x2, y2 = (max(0, int(round(v))) for v in points)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(letterbox(crop, INPUT_SIZE))
        ids.append(track_id)
        if len(crops) == BATCH:
            flush()
    flush()
    return vectors


def load_answered(path: Path) -> set:
    answered = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                answered.add(int(json.loads(line)["item_id"].rsplit("-", 1)[1]))
    return answered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", type=int, default=20, help="clusters (= questions) per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--annotations", type=Path, default=REVIEW_DIR / "pilot_v1_annotations.json")
    parser.add_argument("--responses", type=Path, default=REVIEW_DIR / "acceptance_responses.jsonl")
    parser.add_argument("--cache", type=Path, default=REVIEW_DIR / "acceptance_embeddings.npz")
    parser.add_argument("--out", type=Path, default=REVIEW_DIR / "acceptance_sample.json")
    args = parser.parse_args()

    from sklearn.cluster import KMeans

    tracks = [t for t in json.loads(args.annotations.read_text(encoding="utf-8"))["tracks"] if middle_shape(t)]
    if args.cache.exists():
        cached = np.load(args.cache)
        vectors = dict(zip(cached["ids"].tolist(), cached["vectors"]))
    else:
        with MANIFEST.open(encoding="utf-8") as handle:
            vectors = embed_tracks(tracks, list(csv.DictReader(handle)))
        np.savez(args.cache, ids=np.array(list(vectors)), vectors=np.stack(list(vectors.values())))

    answered = load_answered(args.responses)
    items: List[Dict[str, object]] = []
    for label in CLASSES:
        ids = sorted(t["track_id"] for t in tracks if t["label"] == label and t["track_id"] in vectors)
        k = min(args.clusters, len(ids))
        fitted = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit(np.stack([vectors[i] for i in ids]))
        class_items = build_items(label, ids, fitted.labels_, answered, args.seed)
        print(f"{label}: {len(ids)} tracks -> {len(class_items)} questions "
              f"(largest clusters {sorted((int(i['weight']) for i in class_items), reverse=True)[:5]})")
        items.extend(class_items)
    args.out.write_text(json.dumps({"seed": args.seed, "clusters_per_class": args.clusters, "items": items},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
