"""骨幹＋BNNeck＋分類頭，與不依賴 pickle 類別路徑的權重格式（design D1、D3）。

權重檔是一般字典：{"format", "arch", "embed_dim", "input_size", "mean", "std", "manifest_version", "state_dict"}，
可用 torch.load(..., weights_only=True) 讀取。state_dict 只含骨幹與 BNNeck，不含分類頭。
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_FORMAT = "safetrajai-reid-v1"
DEFAULT_ARCH = "vit_small_patch14_dinov2.lvd142m"
DEFAULT_INPUT_SIZE = 224


def create_backbone(arch: str, input_size: int, pretrained: bool) -> nn.Module:
    extra = {"img_size": input_size} if arch.startswith("vit") else {}
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, **extra)


class ReIDModel(nn.Module):
    """Training returns (pre-BN feature, logits); eval returns the BNNeck feature."""

    def __init__(self, backbone: nn.Module, num_classes: int = 0):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.num_features
        self.bnneck = nn.BatchNorm1d(self.embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else None

    def forward(self, images: torch.Tensor):
        feats = self.backbone(images)
        bn_feats = self.bnneck(feats)
        if self.training and self.classifier is not None:
            return feats, self.classifier(bn_feats)
        return bn_feats


def normalization(backbone: nn.Module) -> tuple[list[float], list[float]]:
    cfg = backbone.pretrained_cfg
    return [float(v) for v in cfg["mean"]], [float(v) for v in cfg["std"]]


def save_weights(path: Path, model: ReIDModel, arch: str, input_size: int, manifest_version: str) -> dict:
    mean, std = normalization(model.backbone)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("classifier.")}
    checkpoint = {
        "format": WEIGHT_FORMAT, "arch": arch, "embed_dim": model.embed_dim, "input_size": input_size,
        "mean": mean, "std": std, "manifest_version": manifest_version, "state_dict": state_dict,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def is_weight_checkpoint(checkpoint: object) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get("format") == WEIGHT_FORMAT


def build_from_checkpoint(checkpoint: dict, device: str) -> ReIDModel:
    backbone = create_backbone(checkpoint["arch"], checkpoint["input_size"], pretrained=False)
    model = ReIDModel(backbone)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().to(device)


@torch.no_grad()
def embed_batch(model: ReIDModel, images: torch.Tensor) -> torch.Tensor:
    """images: normalized float (N, 3, H, W) on the model device -> L2-normalized (N, D)."""
    return F.normalize(model(images).float(), dim=1)
