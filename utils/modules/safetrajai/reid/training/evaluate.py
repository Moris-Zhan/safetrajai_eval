"""val 檢索評估（design D4）：每個身分以固定種子取 1 張作 query，其餘全部（跨類別）作 gallery。

val 身分都在同一序列內，分數偏樂觀；只用於挑 epoch 與比較訓練設定，不作為追蹤啟用 Re-ID 的依據。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from safetrajai.reid.training.data import CropPack, ManifestDataset, build_transform, identity_labels
from safetrajai.reid.training.model import ReIDModel, build_from_checkpoint, embed_batch


@torch.no_grad()
def extract_features(model: ReIDModel, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    feats = []
    for images, _, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            feats.append(embed_batch(model, images.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, model.embed_dim), dtype=np.float32)


def split_query_gallery(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)
    queries = [int(rng.choice(idx)) for _, idx in sorted(by_label.items()) if len(idx) > 1]
    gallery = np.setdiff1d(np.arange(len(labels)), queries)
    return np.array(queries, dtype=int), gallery


def _average_precision(matches: np.ndarray) -> float:
    hits = np.flatnonzero(matches)
    return float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))) if len(hits) else 0.0


def retrieval_metrics(features: np.ndarray, labels: np.ndarray, classes: list[str], seed: int) -> dict:
    """features are L2-normalized; returns overall and per-class mAP / Rank-1 over the queries."""
    labels = np.asarray(labels)
    queries, gallery = split_query_gallery(labels, seed)
    per_query = []
    for start in range(0, len(queries), 512):
        chunk = queries[start:start + 512]
        sims = features[chunk] @ features[gallery].T
        order = np.argsort(-sims, axis=1)
        for q, ranked in zip(chunk, order):
            matches = labels[gallery][ranked] == labels[q]
            per_query.append((classes[q], _average_precision(matches), float(matches[0])))

    def summarize(items: list[tuple[str, float, float]]) -> dict:
        if not items:
            return {"queries": 0, "mAP": None, "rank1": None}
        return {"queries": len(items), "mAP": round(float(np.mean([i[1] for i in items])), 4),
                "rank1": round(float(np.mean([i[2] for i in items])), 4)}

    per_class = {cls: summarize([i for i in per_query if i[0] == cls]) for cls in sorted(set(classes))}
    return {"overall": summarize(per_query), "per_class": per_class}


def evaluate_checkpoint(weights: Path, rows: list[dict], pack: CropPack, seed: int,
                        batch_size: int = 256, num_workers: int = 8) -> dict:
    """Loads a safetrajai-reid-v1 weight file (weights_only) and scores it on the given manifest rows."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_from_checkpoint(checkpoint, device)
    dataset = ManifestDataset(rows, pack, identity_labels(rows), checkpoint["input_size"],
                              build_transform(checkpoint["mean"], checkpoint["std"], train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features = extract_features(model, loader, device)
    metrics = retrieval_metrics(features, np.array(dataset.labels), [r["class"] for r in rows], seed)
    return {"weights": str(weights), "weights_manifest_version": checkpoint["manifest_version"],
            "arch": checkpoint["arch"], "identities": len(set(dataset.labels)), "images": len(rows), **metrics}
