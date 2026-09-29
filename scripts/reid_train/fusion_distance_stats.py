#!/usr/bin/env python3
"""
量測 Re-ID 外觀距離的分布，定出融合門檻的候選值（spec: reid-fusion，design D4）。

輸入：
  - --tune-root   調參子集（預設 dataset/eval/cctv_all8_subsets/tune/）；只接受 C000025 的 145630、150634 時段，
                  出現其他序列（例如保留序列 152639）時以錯誤停止
  - --weights     Re-ID 權重（safetrajai-reid-v1 格式），以 Custom_ReID 與追蹤時相同的裁切與前處理抽特徵
  - --version     資料集版本；val 的分布只作對照
  - --config      讀取各類別的 norm_factor（硬負樣本半徑＝2×norm_factor）
輸出（--out）：JSON，含各類別調參序列與 val 的正／負樣本外觀距離百分位數，以及候選的
  appearance_thresh（寬／嚴）與 appearance_veto_sim（1 − P99／1 − P99.9）。
裁切只在記憶體中處理，不寫出任何影像或訓練資料。

用法：
    python scripts/reid_train/fusion_distance_stats.py --out runs/reid_accept/fusion_distance_stats.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from safetrajai.evaluation.dataset_layout import discover_source_sequences, source_gt_dir
from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.encoders import Custom_ReID
from safetrajai.reid.fusion_stats import (
    any_negative_pairs, candidates, check_tuning_sequences, class_stats, distances, read_mot_gt, summarize,
)
from safetrajai.reid.training.data import (
    CropPack, ManifestDataset, build_transform, default_pack_dir, identity_labels, read_manifest,
)
from safetrajai.reid.training.evaluate import extract_features
from safetrajai.reid.training.model import build_from_checkpoint


def tune_features(encoder: Custom_ReID, video: Path, boxes: list[dict]) -> np.ndarray:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        by_frame[b["frame"]].append(i)
    features = np.zeros((len(boxes), encoder.config["embed_dim"]), dtype=np.float32)
    cap = cv2.VideoCapture(str(video))
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        idx = by_frame.get(frame_id)
        if not idx:
            continue
        dets = np.array([[boxes[i]["cx"], boxes[i]["cy"], boxes[i]["w"], boxes[i]["h"], 0.0] for i in idx])
        features[idx] = np.stack(encoder(frame, dets))
    cap.release()
    missing = sorted(set(by_frame) - set(range(1, frame_id + 1)))
    if missing:
        raise ValueError(f"{video.name}：GT 幀 {missing[:5]}… 超出影片幀數 {frame_id}")
    return features


def val_stats(weights: Path, version: str, pack_dir: Path, seed: int) -> dict:
    rows = [r for r in read_manifest(PROJECT_ROOT / "dataset" / "reid" / "splits" / version / "manifest.csv")
            if r["split"] == "val"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_from_checkpoint(checkpoint, device)
    dataset = ManifestDataset(rows, CropPack(pack_dir), identity_labels(rows), checkpoint["input_size"],
                              build_transform(checkpoint["mean"], checkpoint["std"], train=False))
    features = extract_features(model, DataLoader(dataset, batch_size=256, num_workers=8), device)
    keys = [(r["class"], r["identity"]) for r in rows]
    by_identity: dict[tuple, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        by_identity[key].append(i)
    out = {}
    for cls in sorted({r["class"] for r in rows}):
        pos = [(a, b) for key, idx in by_identity.items() if key[0] == cls for a in idx for b in idx if a < b]
        neg = any_negative_pairs([i for i, r in enumerate(rows) if r["class"] == cls], keys, seed)
        out[cls] = {"positive": summarize(distances(features, pos)), "negative": summarize(distances(features, neg)),
                    "negative_kind": "any"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "class_reid_weights" / "dinov2s_v3" / "model.pt"))
    parser.add_argument("--tune-root", default=str(PROJECT_ROOT / "dataset" / "eval" / "cctv_all8_subsets" / "tune"))
    parser.add_argument("--version", default="v3", help="資料集版本（val 分布只作對照）")
    parser.add_argument("--pack-dir", default=None, help="預設 ~/.cache/safetrajai/reid_packs/<version>/")
    parser.add_argument("--config", default=str(CONFIG_DIR / "reid_crops.yaml"), help="讀取各類別的 norm_factor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="結果 JSON 路徑")
    args = parser.parse_args()

    tune_root = Path(args.tune_root)
    videos = discover_source_sequences(tune_root, "annotations_all8")
    check_tuning_sequences([v.stem for v in videos])

    ctp = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["class_tracking"]["class_tracking_params"]
    radius_by_class = {cls: 2.0 * float(p["norm_factor"]) for cls, p in ctp.items()}

    encoder = Custom_ReID(args.weights)
    boxes, features, fps_by_seq = [], [], {}
    for video in videos:
        seq = video.stem
        seq_boxes = read_mot_gt(source_gt_dir(tune_root, "mot_annotations_all8", seq) / "gt.txt", seq)
        cap = cv2.VideoCapture(str(video))
        fps_by_seq[seq] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        features.append(tune_features(encoder, video, seq_boxes))
        boxes.extend(seq_boxes)
        print(f"[tune] {seq}: {len(seq_boxes)} GT boxes, {fps_by_seq[seq]:.2f} fps")

    tune = class_stats(np.concatenate(features), boxes, fps_by_seq, radius_by_class, args.seed)
    pack_dir = Path(args.pack_dir) if args.pack_dir else default_pack_dir(args.version)
    val = val_stats(Path(args.weights), args.version, pack_dir, args.seed)
    result = {
        "weights": args.weights, "tune_sequences": [v.stem for v in videos], "fps": fps_by_seq,
        "hard_negative_radius_px": radius_by_class, "version": args.version, "seed": args.seed,
        "per_class": {cls: {"tune": tune.get(cls), "val": val.get(cls)} for cls in sorted(set(tune) | set(val))},
        "candidates": candidates(tune),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["candidates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
