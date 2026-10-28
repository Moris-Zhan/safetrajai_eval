"""Re-ID 訓練流程（design D1）：manifest → 本機快取 → PK 取樣 → 骨幹＋BNNeck → CE＋三元組。

輸出目錄（路徑必須含 class_reid_weights，追蹤器以此選用 Custom_ReID）：
  - model.pt            select_by 指定的 val mAP 最佳 epoch 的權重（D3 格式）；val 為空時存最後一個 epoch
  - train_record.json   資料集版本、骨幹、輸入尺寸、種子、超參數、解析度增強與干擾物取樣的設定、
                        各 epoch 的損失、原始 val 與縮小版 val 指標、實際干擾物比例、選用的 epoch

change align-reid-training-domain：select_by="reduced_val"（預設）以縮小版 val（D1）選 epoch；
downscale_prob > 0 啟用解析度增強；distractor_fraction > 0 啟用干擾物取樣（D3，需要 crop_root/_index/ 與 roi_dir）。
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from safetrajai.paths import CONFIG_DIR, PROJECT_ROOT
from safetrajai.reid.training.data import (
    CropPack, Downscale, ManifestDataset, build_transform, identity_labels, manifest_version, read_manifest, verify_pack,
)
from safetrajai.reid.training.evaluate import extract_features, retrieval_metrics
from safetrajai.reid.training.losses import reid_loss
from safetrajai.reid.training.model import ReIDModel, create_backbone, normalization, save_weights
from safetrajai.reid.training.sampler import (
    DistractorPKBatchSampler, PKBatchSampler, build_label_tracks, camera_widths, distractor_radius, find_distractors,
    load_crop_points,
)

SELECT_BY = ("reduced_val", "val")


@dataclass(frozen=True)
class TrainConfig:
    manifest: str
    pack_dir: str
    output_dir: str
    arch: str
    input_size: int = 224
    pretrained: bool = True
    epochs: int = 30
    p: int = 32
    k: int = 4
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    margin: float = 0.3
    seed: int = 0
    num_workers: int = 8
    eval_batch_size: int = 256
    max_identities: int | None = None
    downscale_prob: float = 0.0
    downscale_min: int = 8
    downscale_max: int = 40
    distractor_fraction: float = 0.0
    cooccur_min_frames: int = 10
    pair_max_frame_gap: int = 5
    crop_root: str = str(PROJECT_ROOT / "dataset" / "reid" / "crops")
    roi_dir: str = str(CONFIG_DIR / "roi")
    track_config: str = str(CONFIG_DIR / "reid" / "reid_crops.yaml")
    select_by: str = "reduced_val"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _limit_identities(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None:
        return rows
    keys = sorted({(r["class"], r["identity"]) for r in rows})
    keep = {keys[i] for i in np.random.default_rng(seed).permutation(len(keys))[:limit]}
    return [r for r in rows if (r["class"], r["identity"]) in keep]


def _lr_lambda(warmup_iters: int, total_iters: int):
    def factor(step: int) -> float:
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return factor


def selection_map(raw_map: float | None, reduced_map: float | None, select_by: str) -> float | None:
    return reduced_map if select_by == "reduced_val" else raw_map


def _norm_factors(track_config: Path) -> dict[str, float]:
    params = yaml.safe_load(Path(track_config).read_text(encoding="utf-8"))["class_tracking"]["class_tracking_params"]
    return {cls: float(p["norm_factor"]) for cls, p in params.items()}


def _distractor_sampler(cfg: TrainConfig, rows: list[dict], labels: list[int]) -> tuple[DistractorPKBatchSampler, dict]:
    points = load_crop_points(rows, Path(cfg.crop_root))
    widths = camera_widths({r["camera"] for r in rows}, Path(cfg.roi_dir))
    norm = _norm_factors(Path(cfg.track_config))
    tracks = build_label_tracks(rows, labels, points)
    radius = {label: distractor_radius(norm[row["class"]], widths[row["camera"]]) for row, label in zip(rows, labels)}
    distractors = find_distractors(tracks, radius, cfg.cooccur_min_frames, cfg.pair_max_frame_gap)
    with_any = sum(1 for others in distractors.values() if others)
    info = {
        "norm_factor": norm, "camera_width": widths,
        "radius_px": {f"{cls}@{cam}": round(distractor_radius(norm[cls], w), 1)
                      for cls in sorted({r["class"] for r in rows}) for cam, w in widths.items()},
        "identities_with_distractors": with_any,
        "identities_with_distractors_share": round(with_any / max(1, len(distractors)), 4),
    }
    return DistractorPKBatchSampler(labels, cfg.p, cfg.k, cfg.seed, distractors, cfg.distractor_fraction), info


def train(cfg: TrainConfig) -> dict:
    if cfg.select_by not in SELECT_BY:
        raise ValueError(f"select_by 必須是 {SELECT_BY} 之一：{cfg.select_by}")
    output_dir = Path(cfg.output_dir)
    if "class_reid_weights" not in output_dir.as_posix():
        raise ValueError(f"輸出目錄必須位於 class_reid_weights 之下：{output_dir}")
    if (output_dir / "model.pt").exists():
        raise FileExistsError(f"權重已存在，請改用新的輸出目錄：{output_dir / 'model.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = manifest_version(Path(cfg.manifest))
    rows = read_manifest(Path(cfg.manifest))
    pack = CropPack(Path(cfg.pack_dir))
    verify_pack(pack, rows)
    if pack.info.get("version") != version:
        raise ValueError(f"快取版本 {pack.info.get('version')} 與 manifest 版本 {version} 不同")

    train_rows = _limit_identities([r for r in rows if r["split"] == "train"], cfg.max_identities, cfg.seed)
    val_rows = _limit_identities([r for r in rows if r["split"] == "val"], cfg.max_identities, cfg.seed)
    train_labels = identity_labels(train_rows)
    val_labels = identity_labels(val_rows)

    backbone = create_backbone(cfg.arch, cfg.input_size, cfg.pretrained)
    mean, std = normalization(backbone)
    model = ReIDModel(backbone, num_classes=len(train_labels)).to(device)

    reduced = Downscale(prob=1.0, min_side=cfg.downscale_min, max_side=cfg.downscale_max)
    augment = (Downscale(cfg.downscale_prob, cfg.downscale_min, cfg.downscale_max)
               if cfg.downscale_prob > 0 else None)
    train_set = ManifestDataset(train_rows, pack, train_labels, cfg.input_size, build_transform(mean, std, train=True),
                                downscale=augment)
    distractor_info = None
    if cfg.distractor_fraction > 0:
        sampler, distractor_info = _distractor_sampler(cfg, train_rows, train_set.labels)
    else:
        sampler = PKBatchSampler(train_set.labels, cfg.p, cfg.k, cfg.seed)
    loader_kwargs = {"num_workers": cfg.num_workers, "pin_memory": device == "cuda",
                     "persistent_workers": cfg.num_workers > 0}
    train_loader = DataLoader(train_set, batch_sampler=sampler, **loader_kwargs)
    val_transform = build_transform(mean, std, train=False)
    val_set = ManifestDataset(val_rows, pack, val_labels, cfg.input_size, val_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.eval_batch_size, shuffle=False, **loader_kwargs)
    reduced_set = ManifestDataset(val_rows, pack, val_labels, cfg.input_size, val_transform,
                                  downscale=reduced, fixed=True)
    reduced_loader = DataLoader(reduced_set, batch_size=cfg.eval_batch_size, shuffle=False, **loader_kwargs)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": list(model.backbone.parameters()), "lr": cfg.backbone_lr},
         {"params": head_params, "lr": cfg.head_lr}],
        weight_decay=cfg.weight_decay,
    )
    iters = len(sampler)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(cfg.warmup_epochs * iters, cfg.epochs * iters))

    record = {
        "manifest_version": version, "manifest": str(cfg.manifest), "pack": pack.info,
        "arch": cfg.arch, "input_size": cfg.input_size, "embed_dim": model.embed_dim, "seed": cfg.seed,
        "config": asdict(cfg), "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "train_identities": len(train_labels), "train_images": len(train_rows),
        "val_identities": len(val_labels), "val_images": len(val_rows),
        "iters_per_epoch": iters, "epochs": [], "best_epoch": None, "best_val": None, "best_reduced_val": None,
        "resolution_augmentation": {"enabled": augment is not None, "prob": cfg.downscale_prob,
                                    "min_side": cfg.downscale_min, "max_side": cfg.downscale_max},
        "distractor_sampling": {"enabled": distractor_info is not None, "fraction": cfg.distractor_fraction,
                                "cooccur_min_frames": cfg.cooccur_min_frames,
                                "pair_max_frame_gap": cfg.pair_max_frame_gap, **(distractor_info or {})},
        "select_by": cfg.select_by,
        "reduced_val": {"min_side": cfg.downscale_min, "max_side": cfg.downscale_max, "seed": "crc32(crop_path)"},
    }
    record_path = output_dir / "train_record.json"
    best_map = -1.0
    val_classes = [r["class"] for r in val_rows]

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sampler.set_epoch(epoch)
        model.train()
        sums = np.zeros(3)
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                feats, logits = model(images)
            total, ce, triplet = reid_loss(feats, logits, labels, cfg.label_smoothing, cfg.margin)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            scheduler.step()
            sums += [total.item(), ce.item(), triplet.item()]
        train_seconds = time.time() - start

        metrics = (retrieval_metrics(extract_features(model, val_loader, device), np.array(val_set.labels),
                                     val_classes, cfg.seed) if val_rows else None)
        reduced_metrics = (retrieval_metrics(extract_features(model, reduced_loader, device),
                                             np.array(reduced_set.labels), val_classes, cfg.seed) if val_rows else None)
        entry = {
            "epoch": epoch, "loss": round(sums[0] / iters, 4), "ce": round(sums[1] / iters, 4),
            "triplet": round(sums[2] / iters, 4), "train_seconds": round(train_seconds, 1),
            "epoch_seconds": round(time.time() - start, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            "val": metrics,
            "reduced_val": reduced_metrics,
            "distractor": getattr(sampler, "epoch_stats", None),
        }
        record["epochs"].append(entry)
        raw_map = metrics["overall"]["mAP"] if metrics else None
        reduced_map = reduced_metrics["overall"]["mAP"] if reduced_metrics else None
        val_map = selection_map(raw_map, reduced_map, cfg.select_by)
        print(f"[epoch {epoch}/{cfg.epochs}] loss={entry['loss']} ce={entry['ce']} triplet={entry['triplet']} "
              f"val_mAP={raw_map} reduced_val_mAP={reduced_map} time={entry['epoch_seconds']}s", flush=True)

        if val_map is None or val_map > best_map:
            best_map = -1.0 if val_map is None else val_map
            save_weights(output_dir / "model.pt", model, cfg.arch, cfg.input_size, version)
            record["best_epoch"], record["best_val"], record["best_reduced_val"] = epoch, metrics, reduced_metrics
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return record
