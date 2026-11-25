#!/usr/bin/env python3
"""
WiSE-FT: blend fine-tuned weights with the starting weights, theta = (1 - alpha) * base + alpha * tuned.

alpha 0 is the base model, alpha 1 the fine-tuned one. Used to check whether a fine-tune that
regresses on the evaluation camera can be rescued without retraining.

Output: a normal Ultralytics checkpoint (base checkpoint with the blended model weights).

Usage (conda env track_eval):
    python scripts/detection_data/wise_ft.py --tuned runs/finetune/ft_v2_e0/weights/best.pt --alpha 0.5 --out runs/finetune/wise/e0_a050.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = PROJECT_ROOT / "models" / "yolo" / "yolo11x_base.pt"


def blend_state(base: dict, tuned: dict, alpha: float) -> dict:
    blended = {}
    for key, base_value in base.items():
        if base_value.is_floating_point():
            blended[key] = ((1.0 - alpha) * base_value.float() + alpha * tuned[key].float()).to(base_value.dtype)
        else:
            blended[key] = tuned[key]
    return blended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    base_ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned_ckpt = torch.load(args.tuned, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"] if base_ckpt.get("model") is not None else base_ckpt["ema"]
    tuned_model = tuned_ckpt["model"] if tuned_ckpt.get("model") is not None else tuned_ckpt["ema"]
    base_model.load_state_dict(blend_state(base_model.state_dict(), tuned_model.state_dict(), args.alpha))
    base_ckpt["model"], base_ckpt["ema"] = base_model, None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_ckpt, args.out)
    print(f"wrote {args.out} (alpha={args.alpha})")


if __name__ == "__main__":
    main()
