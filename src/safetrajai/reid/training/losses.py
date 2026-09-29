"""身分分類交叉熵（label smoothing）＋ batch-hard 三元組（design D1）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    dist = torch.cdist(features.float(), features.float())
    same = labels[:, None] == labels[None, :]
    hardest_positive = (dist * same).max(dim=1).values
    hardest_negative = dist.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def reid_loss(
    features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float, margin: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, cross-entropy, triplet)."""
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)
    triplet = batch_hard_triplet_loss(features, labels, margin)
    return ce + triplet, ce, triplet
