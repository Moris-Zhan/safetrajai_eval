#!/usr/bin/env python3
"""
以資料集版本訓練 6 類共用的 Re-ID 模型（spec: reid-model，design D1）。

輸入：
  - dataset/reid/splits/<version>/manifest.csv   依 split 區分 train／val
  - 本機快取（pack_manifest.py 產生）              影像只從快取讀取，張數必須與 manifest 一致
輸出（--output，路徑必須含 class_reid_weights；model.pt 已存在時報錯）：
  - model.pt            --select-by 指定的 val mAP 最佳 epoch 的權重（state_dict＋設定，weights_only 可讀）
  - train_record.json   資料集版本、骨幹、輸入尺寸、種子、超參數、解析度增強與干擾物取樣的設定、
                        各 epoch 的損失、原始 val 與縮小版 val 指標、實際干擾物比例

用法：
    python scripts/reid_train/train_reid.py --version v2 --output models/class_reid_weights/dinov2s_v2
    python scripts/reid_train/train_reid.py --version v2 --output models/class_reid_weights/smoke_v2 \\
        --epochs 1 --max-identities 2000
    # change align-reid-training-domain：解析度增強＋干擾物取樣（干擾物取樣讀 --crop-root/_index/ 與 config/roi/）
    python scripts/reid_train/train_reid.py --version v3 --output models/class_reid_weights/dinov2s_v3_both \
        --downscale-prob 0.5 --distractor-fraction 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.training.data import default_pack_dir
from safetrajai.reid.training.model import DEFAULT_ARCH, DEFAULT_INPUT_SIZE
from safetrajai.reid.training.train import SELECT_BY, TrainConfig, train


def main() -> None:
    defaults = TrainConfig(manifest="", pack_dir="", output_dir="", arch=DEFAULT_ARCH)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="資料集版本名稱，例如 v2")
    parser.add_argument("--output", required=True, help="權重目錄，例如 models/class_reid_weights/dinov2s_v2")
    parser.add_argument("--splits-root", default=str(PROJECT_ROOT / "dataset" / "reid" / "splits"))
    parser.add_argument("--pack-dir", default=None, help="預設 ~/.cache/safetrajai/reid_packs/<version>/")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="timm 骨幹名稱")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--no-pretrained", action="store_true", help="不載入預訓練權重（測試用）")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--p", type=int, default=defaults.p, help="每批身分數")
    parser.add_argument("--k", type=int, default=defaults.k, help="每個身分張數")
    parser.add_argument("--backbone-lr", type=float, default=defaults.backbone_lr)
    parser.add_argument("--head-lr", type=float, default=defaults.head_lr)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-identities", type=int, default=None, help="train／val 各只取前 N 個身分（冒煙測試用）")
    parser.add_argument("--downscale-prob", type=float, default=defaults.downscale_prob,
                        help="解析度增強機率，0＝關閉")
    parser.add_argument("--downscale-min", type=int, default=defaults.downscale_min, help="縮小後最短邊下限（px）")
    parser.add_argument("--downscale-max", type=int, default=defaults.downscale_max, help="縮小後最短邊上限（px）")
    parser.add_argument("--distractor-fraction", type=float, default=defaults.distractor_fraction,
                        help="每批中種子＋干擾物所占比例，0＝關閉")
    parser.add_argument("--cooccur-min-frames", type=int, default=defaults.cooccur_min_frames,
                        help="干擾物與種子幀範圍至少重疊的幀數")
    parser.add_argument("--pair-max-frame-gap", type=int, default=defaults.pair_max_frame_gap,
                        help="比較框中心距離的一對裁切的最大幀差")
    parser.add_argument("--crop-root", default=defaults.crop_root, help="裁切根目錄（只讀 _index/ 的 metadata）")
    parser.add_argument("--roi-dir", default=defaults.roi_dir, help="鏡頭 ROI 目錄（取畫面寬度）")
    parser.add_argument("--track-config", default=defaults.track_config, help="讀取各類別的 norm_factor")
    parser.add_argument("--select-by", choices=SELECT_BY, default=defaults.select_by,
                        help="以縮小版 val（預設）或原始 val 的 mAP 選 epoch")
    args = parser.parse_args()

    output = Path(args.output)
    cfg = TrainConfig(
        manifest=str(Path(args.splits_root) / args.version / "manifest.csv"),
        pack_dir=str(Path(args.pack_dir) if args.pack_dir else default_pack_dir(args.version)),
        output_dir=str(output if output.is_absolute() else PROJECT_ROOT / output),
        arch=args.arch, input_size=args.input_size, pretrained=not args.no_pretrained, epochs=args.epochs,
        p=args.p, k=args.k, backbone_lr=args.backbone_lr, head_lr=args.head_lr, seed=args.seed,
        num_workers=args.workers, max_identities=args.max_identities,
        downscale_prob=args.downscale_prob, downscale_min=args.downscale_min, downscale_max=args.downscale_max,
        distractor_fraction=args.distractor_fraction, cooccur_min_frames=args.cooccur_min_frames,
        pair_max_frame_gap=args.pair_max_frame_gap, crop_root=args.crop_root, roi_dir=args.roi_dir,
        track_config=args.track_config, select_by=args.select_by,
    )
    record = train(cfg)
    print(f"[train] best epoch {record['best_epoch']}: {record['best_val'] and record['best_val']['overall']}")
    print(f"[train] weights: {cfg.output_dir}/model.pt")


if __name__ == "__main__":
    main()
