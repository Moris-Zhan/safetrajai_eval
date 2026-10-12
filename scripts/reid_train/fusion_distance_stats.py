#!/usr/bin/env python3
"""
量測 Re-ID 外觀距離的分布與外觀分離度，定出融合門檻的候選值並判定外觀分離度關卡
（spec: reid-fusion、reid-model；change align-reid-training-domain D4）。

輸入：
  - --tune-root   調參子集（預設 dataset/eval/cctv_all8_subsets/tune/）；只接受 C000025 的 145630、150634 時段，
                  出現其他序列（例如保留序列 152639）時以錯誤停止
  - --weights     Re-ID 權重（safetrajai-reid-v1 格式），以 Custom_ReID 與追蹤時相同的裁切（緊框）與前處理抽特徵；
                  --legacy-crop 改用修改前的裁切（框放大 1.02 倍、加 10px 邊），只用來量測參考
  - --version     資料集版本；val 的分布只作對照
  - --config      追蹤情境設定（預設 botsort_reid_timm.yaml）：各類別的 norm_factor（干擾物半徑＝2×norm_factor）
                  與 track_buffer（保留期限幀數，ADR 0009，不換算）；reid_crops.yaml 是依 10fps 來源設定的，
                  不能用於 C000025
  - --reference        參考的結果 JSON（dinov2s_v3 以 --legacy-crop 量測）
  - --noise-reference  兩個結果 JSON（只換種子的兩個模型，皆以緊框量測），差距即雜訊底線
輸出（--out）：JSON，含各類別調參序列與 val 的正樣本（保留期限內、≥ 1 秒）與干擾物的外觀距離百分位數、
  AUC、TPR@FPR5%、分尺寸結果，候選的 appearance_thresh（寬／嚴）與 appearance_veto_sim（1 − P99／1 − P99.9）；
  同時給 --reference 與 --noise-reference 時另含關卡判定（gate）。
裁切只在記憶體中處理，不寫出任何影像或訓練資料。

用法：
    python scripts/reid_train/fusion_distance_stats.py --out runs/reid_accept/fusion_distance_stats.json
    python scripts/reid_train/fusion_distance_stats.py --legacy-crop --out runs/reid_domain/reference.json
    python scripts/reid_train/fusion_distance_stats.py --weights models/class_reid_weights/dinov2s_v3_both/model.pt \
        --reference runs/reid_domain/reference.json \
        --noise-reference runs/reid_domain/dinov2s_v3_seed1.json runs/reid_domain/dinov2s_v3_tight.json \
        --out runs/reid_domain/dinov2s_v3_both.json
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
    any_negative_pairs, candidates, check_tuning_sequences, class_stats, distances, drop_unlabeled, noise_floor,
    read_mot_gt,
    retention_frames, separability_gate, summarize,
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


def tune_metrics(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {cls: v["tune"] for cls, v in result["per_class"].items() if v.get("tune")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "class_reid_weights" / "dinov2s_v3" / "model.pt"))
    parser.add_argument("--tune-root", default=str(PROJECT_ROOT / "dataset" / "eval" / "cctv_all8_subsets" / "tune"))
    parser.add_argument("--version", default="v3", help="資料集版本（val 分布只作對照）")
    parser.add_argument("--pack-dir", default=None, help="預設 ~/.cache/safetrajai/reid_packs/<version>/")
    parser.add_argument("--config", default=str(CONFIG_DIR / "botsort_reid_timm.yaml"),
                        help="讀取各類別的 norm_factor 與 track_buffer（保留期限幀數）")
    parser.add_argument("--legacy-crop", action="store_true", help="用修改前的裁切（量測參考用）")
    parser.add_argument("--reference", default=None, help="參考的結果 JSON")
    parser.add_argument("--noise-reference", nargs=2, default=None, metavar=("SEED_MODEL", "BASE_MODEL"),
                        help="只換種子的兩個模型的結果 JSON（緊框）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="結果 JSON 路徑")
    args = parser.parse_args()

    tune_root = Path(args.tune_root)
    videos = discover_source_sequences(tune_root, "annotations_all8")
    check_tuning_sequences([v.stem for v in videos])

    scenario = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctp = scenario["class_tracking"]["class_tracking_params"]
    radius_by_class = {cls: 2.0 * float(p["norm_factor"]) for cls, p in ctp.items()}
    retention_by_class = retention_frames(scenario)

    encoder = Custom_ReID(args.weights, legacy_crop=args.legacy_crop)
    boxes, features, fps_by_seq = [], [], {}
    skipped_unlabeled: dict[str, int] = defaultdict(int)
    for video in videos:
        seq = video.stem
        gt_path = source_gt_dir(tune_root, "mot_annotations_all8", seq) / "gt.txt"
        seq_boxes, skipped = drop_unlabeled(read_mot_gt(gt_path, seq))
        for cls, count in skipped.items():
            skipped_unlabeled[cls] += count
        cap = cv2.VideoCapture(str(video))
        fps_by_seq[seq] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        features.append(tune_features(encoder, video, seq_boxes))
        boxes.extend(seq_boxes)
        print(f"[tune] {seq}: {len(seq_boxes)} GT boxes, {fps_by_seq[seq]:.2f} fps")

    tune = class_stats(np.concatenate(features), boxes, fps_by_seq, radius_by_class, args.seed, retention_by_class)
    pack_dir = Path(args.pack_dir) if args.pack_dir else default_pack_dir(args.version)
    val = val_stats(Path(args.weights), args.version, pack_dir, args.seed)
    result = {
        "weights": args.weights, "crop": "legacy" if args.legacy_crop else "tight", "config": args.config,
        "tune_sequences": [v.stem for v in videos], "fps": fps_by_seq,
        "hard_negative_radius_px": radius_by_class, "retention_frames": retention_by_class,
        "skipped_unlabeled_boxes": dict(skipped_unlabeled),
        "version": args.version, "seed": args.seed,
        "per_class": {cls: {"tune": tune.get(cls), "val": val.get(cls)} for cls in sorted(set(tune) | set(val))},
        "candidates": candidates(tune),
    }
    if args.reference and args.noise_reference:
        reference = tune_metrics(Path(args.reference))
        noise = noise_floor(*(tune_metrics(Path(p)) for p in args.noise_reference))
        result["gate"] = {"reference": args.reference, "noise_reference": args.noise_reference, "noise_floor": noise,
                          **separability_gate(tune, reference, noise)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["candidates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
