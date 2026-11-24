#!/usr/bin/env python3
"""
Extend the user's manual person-box answers to look-alike boxes (appearance matching).

Every candidate box is embedded (DINOv2, padded crop). For each box without a manual answer the
nearest manually answered box is found; its verdict is copied when the cosine similarity reaches
a threshold. The threshold is calibrated, not guessed: each manually answered box is predicted
from the other answered boxes (leave-one-out) and the smallest similarity at which the copied
verdict agrees with the user at least --min-agreement of the time is used. Boxes below it stay
pending for manual review. "unsure" answers are never copied.

Output: runs/cvat_review/person_embeddings.npz (cache) and runs/cvat_review/person_verdicts.json
  {threshold, loo_agreement, coverage, boxes: {"f12-0": {verdict, source: manual|propagated|pending, sim}}}

Usage (conda env track_eval, needs cv2/torch/timm):
    python scripts/detection_data/propagate_person_answers.py [--min-agreement 0.95]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "runs" / "cvat_review"
VIDEOS = PROJECT_ROOT / "dataset" / "raw" / "videos"
PAD_FRACTION = 0.3
BATCH = 64
COPYABLE = ("person", "none", "scooter", "car", "small_truck", "bus", "truck")
LEGACY_VERDICTS = {"not_person": "none"}


def box_ids(candidates: List[dict]) -> List[str]:
    return [f"f{n}-{i}" for n, c in enumerate(candidates) for i in range(len(c["persons"]))]


def manual_verdicts(responses: Path) -> Dict[str, str]:
    latest: Dict[str, dict] = {}
    if responses.exists():
        for line in responses.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["frame_id"]] = record
    return {f"{frame}-{index}": LEGACY_VERDICTS.get(verdict, verdict)
            for frame, record in latest.items() for index, verdict in record["boxes"].items()}


def calibrate_threshold(similarity: np.ndarray, labels: List[str], min_agreement: float) -> Tuple[Optional[float], float]:
    """Leave-one-out: smallest top-1 similarity whose copied verdict agrees >= min_agreement of the time."""
    sim = similarity.copy()
    np.fill_diagonal(sim, -1.0)
    nearest = sim.argmax(axis=1)
    top = sim[np.arange(len(labels)), nearest]
    agree = np.array([labels[j] == labels[i] for i, j in enumerate(nearest)])
    for threshold in sorted(set(np.round(top, 4))):
        kept = top >= threshold
        if kept.sum() >= 10 and agree[kept].mean() >= min_agreement:
            return float(threshold), float(agree[kept].mean())
    return None, 0.0


def embed_boxes(candidates: List[dict]) -> Dict[str, np.ndarray]:
    import cv2
    import timm
    import torch
    from safetrajai.reid.dinov2_features import MODEL_NAME, INPUT_SIZE, letterbox

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE).eval().to(device)
    mean = torch.tensor(model.pretrained_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"], device=device).view(1, 3, 1, 1)
    vectors: Dict[str, np.ndarray] = {}
    crops: List[np.ndarray] = []
    ids: List[str] = []

    def flush() -> None:
        if crops:
            batch = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float().div(255.0)
            with torch.no_grad():
                feats = torch.nn.functional.normalize(model((batch - mean) / std), dim=1).cpu().numpy()
            vectors.update(zip(ids, feats))
            crops.clear()
            ids.clear()

    for n, candidate in enumerate(candidates):
        capture = cv2.VideoCapture(str(VIDEOS / candidate["camera"] / candidate["video"]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate["frame_idx"])
        ok, image = capture.read()
        capture.release()
        if not ok:
            continue
        height, width = image.shape[:2]
        for i, (x1, y1, x2, y2, _) in enumerate(candidate["persons"]):
            padx, pady = (x2 - x1) * PAD_FRACTION, (y2 - y1) * PAD_FRACTION
            crop = image[max(0, int(y1 - pady)):min(height, int(y2 + pady)), max(0, int(x1 - padx)):min(width, int(x2 + padx))]
            if crop.size == 0:
                continue
            crops.append(letterbox(crop, INPUT_SIZE))
            ids.append(f"f{n}-{i}")
            if len(crops) == BATCH:
                flush()
    flush()
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-agreement", type=float, default=0.95)
    parser.add_argument("--candidates", type=Path, default=REVIEW_DIR / "person_candidates.json")
    parser.add_argument("--responses", type=Path, default=REVIEW_DIR / "person_review_responses.jsonl")
    parser.add_argument("--cache", type=Path, default=REVIEW_DIR / "person_embeddings.npz")
    parser.add_argument("--out", type=Path, default=REVIEW_DIR / "person_verdicts.json")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if args.cache.exists():
        cached = np.load(args.cache)
        vectors = dict(zip(cached["ids"].tolist(), cached["vectors"]))
    else:
        vectors = embed_boxes(candidates)
        np.savez(args.cache, ids=np.array(list(vectors)), vectors=np.stack(list(vectors.values())))
    print(f"embedded {len(vectors)} boxes")

    manual = manual_verdicts(args.responses)
    labeled = sorted(i for i, v in manual.items() if v in COPYABLE and i in vectors)
    result: Dict[str, dict] = {}
    for box in box_ids(candidates):
        if box in manual:
            result[box] = {"verdict": manual[box], "source": "manual", "sim": 1.0}
        else:
            result[box] = {"verdict": None, "source": "pending", "sim": None}
    threshold: Optional[float] = None
    agreement = 0.0
    if len(labeled) >= 20 and len({manual[i] for i in labeled}) >= 2:
        matrix = np.stack([vectors[i] for i in labeled])
        threshold, agreement = calibrate_threshold(matrix @ matrix.T, [manual[i] for i in labeled], args.min_agreement)
        print(f"{len(labeled)} labeled boxes; leave-one-out threshold {threshold} (agreement {agreement:.3f})")
        if threshold is not None:
            for box, entry in result.items():
                if entry["source"] == "pending" and box in vectors:
                    sims = matrix @ vectors[box]
                    j = int(sims.argmax())
                    if sims[j] >= threshold:
                        result[box] = {"verdict": manual[labeled[j]], "source": "propagated", "sim": float(sims[j])}
    else:
        print(f"need at least 20 answered boxes covering both verdicts (have {len(labeled)}); nothing propagated")
    counts = {s: sum(1 for e in result.values() if e["source"] == s) for s in ("manual", "propagated", "pending")}
    args.out.write_text(json.dumps({"threshold": threshold, "loo_agreement": agreement, "min_agreement": args.min_agreement,
                                    "counts": counts, "boxes": result}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(counts, "->", args.out)


if __name__ == "__main__":
    main()
