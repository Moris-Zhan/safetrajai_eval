# ============================================================
# safetrajai/reid/yolo_reid_trainer.py
#
# 設計原則：
#   - 不建立任何 wrapper nn.Module；直接在 ultralytics YOLO
#     原生模型上掛 hook，取倒數第二層輸出做 triplet loss。
#   - projector 以獨立 nn.Module 形式存在，但儲存時以
#     model.model.reid_projector 屬性附掛回 YOLO 原生物件，
#     確保 torch.save(yolo.model) 能完整重建。
#   - 讀取時同樣直接 YOLO(path)，再取 model.reid_projector
#     還原 projector 並重掛 hook。
# ============================================================

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
# 1. Batch-Hard Triplet Loss（不變）
# ──────────────────────────────────────────────────────────────
class BatchHardTripletLoss(nn.Module):
    """每個 anchor 選最難的 positive / negative 配對。"""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        N = embeddings.size(0)

        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos
        mask_pos ^= torch.eye(N, dtype=torch.bool, device=device)

        dist_ap, dist_an = [], []
        for i in range(N):
            pos_i = dist[i][mask_pos[i]]
            neg_i = dist[i][mask_neg[i]]
            if len(pos_i) == 0 or len(neg_i) == 0:
                continue
            dist_ap.append(pos_i.max())
            dist_an.append(neg_i.min())

        if not dist_ap:
            return torch.tensor(0.0, requires_grad=True, device=device)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        return self.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))


# ──────────────────────────────────────────────────────────────
# 2. ReIDProjector — 獨立 nn.Module，可附掛到 YOLO model
# ──────────────────────────────────────────────────────────────
class ReIDProjector(nn.Module):
    """
    輕量 projector，接在 YOLO backbone 倒數第二層之後。
    feat_dim 在第一次 probe forward 後自動決定。
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# 3. ReIDHookManager
#    負責：掛 hook、池化、過 projector、L2 normalize
#    不繼承 nn.Module，純粹是訓練期的輔助管理器。
# ──────────────────────────────────────────────────────────────
class ReIDHookManager:
    """
    直接操作 ultralytics YOLO 內部模型，掛 hook 取倒數第二層特徵。

    使用方式::

        manager = ReIDHookManager(yolo_model)
        embed = manager.embed(imgs)   # (B, D)  L2-normalized
    """

    def __init__(self, yolo_model: YOLO, device: str = "cpu"):
        self.yolo    = yolo_model          # ultralytics YOLO 物件
        self.device  = device
        self._feat: torch.Tensor | None = None
        self._hook_handle = None
        self.projector: ReIDProjector | None = None

        inner = yolo_model.model           # nn.Module (DetectionModel / …)
        self.hook_layer_idx = len(inner.model) - 2
        self._register_hook(inner)

        # 自動 probe feat_dim 並建立 projector
        self._probe_and_build()

    # ── hook ────────────────────────────────────────────────────
    def _register_hook(self, inner: nn.Module):
        if self._hook_handle is not None:
            self._hook_handle.remove()
        target = inner.model[self.hook_layer_idx]
        self._hook_handle = target.register_forward_hook(self._cache_feat)
        print(f"[ReIDHookManager] Hook → layer[{self.hook_layer_idx}]: "
              f"{type(target).__name__}")

    def _cache_feat(self, module, inp, output):
        if isinstance(output, torch.Tensor):
            self._feat = output
            if output.requires_grad:
                output.retain_grad()
        else:
            self._feat = output

    # ── pool ────────────────────────────────────────────────────
    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4:      # (B, C, H, W)
            return feat.mean(dim=[2, 3])
        if feat.dim() == 3:      # (B, seq, C)
            return feat.mean(dim=1)
        if feat.dim() == 1:
            return feat.unsqueeze(0)
        return feat

    # ── probe ───────────────────────────────────────────────────
    def _probe_and_build(self):
        inner = self.yolo.model
        inner.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64).to(self.device)
            inner(dummy)
        feat_dim = self._pool(self._feat).shape[-1]

        self.projector = ReIDProjector(feat_dim).to(self.device)
        # 附掛到 yolo.model 作為屬性，torch.save 時會一起序列化
        self.yolo.model.reid_projector = self.projector
        print(f"[ReIDHookManager] feat_dim={feat_dim}，projector built & attached.")

    # ── embed（訓練 / 推論皆用此） ────────────────────────────
    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: (B, 3, H, W)  已移到正確 device
        Returns:
            (B, D)  L2-normalized embeddings
        """
        imgs = self._pad_to_stride(imgs, 32)
        self.yolo.model(imgs)              # forward → hook 填入 self._feat

        feat = self._feat
        if not feat.requires_grad:
            feat = feat.detach().requires_grad_(True)

        pooled = self._pool(feat)          # (B, C)
        embed  = self.projector(pooled)    # (B, D)
        return F.normalize(embed, p=2, dim=1)

    @staticmethod
    def _pad_to_stride(x: torch.Tensor, stride: int = 32) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    # ── 訓練模式控制 ─────────────────────────────────────────
    def train_mode(self):
        self.yolo.model.train()
        self.projector.train()

    def eval_mode(self):
        self.yolo.model.eval()
        self.projector.eval()

    # ── 凍結 / 解凍 backbone ──────────────────────────────────
    def freeze_backbone(self, freeze: bool = True):
        for p in self.yolo.model.parameters():
            p.requires_grad_(not freeze)
        # projector 永遠可訓練
        for p in self.projector.parameters():
            p.requires_grad_(True)

    # ── 釋放 hook（訓練結束後呼叫）────────────────────────────
    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None


# ──────────────────────────────────────────────────────────────
# 4. 儲存 / 載入（ultralytics 原生格式）
# ──────────────────────────────────────────────────────────────
def save_yolo_reid(yolo: YOLO, save_path: str, train_args: dict | None = None):
    """
    以 ultralytics 原生格式儲存整個 YOLO 物件。
    reid_projector 已附掛在 yolo.model 上，會一起序列化。

    wandb.watch / ReIDHookManager 等工具會在 model 上掛含 lambda 的
    forward hook，這些物件無法被 pickle。因此儲存前先暫時移除全部
    hooks，存完後再還原。

    Args:
        yolo      : ultralytics YOLO 物件（已訓練）
        save_path : 輸出路徑，建議副檔名 .pt
        train_args: 可選，訓練超參數，附掛到 yolo.model.reid_train_args
    """
    if train_args is not None:
        yolo.model.reid_train_args = train_args

    # ── 暫存並清除所有不可序列化的 hooks ────────────────────
    saved_hooks: dict[int, dict] = {}
    for module in yolo.model.modules():
        mid = id(module)
        saved_hooks[mid] = {
            "fwd":      dict(module._forward_hooks),
            "fwd_pre":  dict(module._forward_pre_hooks),
            "bwd":      dict(module._backward_hooks),
        }
        # 用 OrderedDict()（而非 {}）清空，維持跟 nn.Module 原生 hook 容器同型別，
        # 否則存檔內容會把 _forward_hooks 序列化成一般 dict，之後 register_forward_hook()
        # 嘗試對它建立 weakref 時會拋 TypeError: cannot create weak reference to 'dict' object。
        module._forward_hooks     = OrderedDict()
        module._forward_pre_hooks = OrderedDict()
        module._backward_hooks    = OrderedDict()

    try:
        yolo.save(save_path)
        print(f"✅  YOLO ReID model saved → {save_path}")
    finally:
        # ── 還原所有 hooks（無論成功與否）───────────────────
        for module in yolo.model.modules():
            mid = id(module)
            if mid in saved_hooks:
                module._forward_hooks.update(saved_hooks[mid]["fwd"])
                module._forward_pre_hooks.update(saved_hooks[mid]["fwd_pre"])
                module._backward_hooks.update(saved_hooks[mid]["bwd"])


def load_reid_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """
    重新載入 YOLO ReID 模型，並重掛 hook。

    Returns:
        (yolo, manager)  — yolo 是 ultralytics YOLO 物件，
                           manager 是 ReIDHookManager（可直接呼叫 embed）
    """
    yolo = YOLO(checkpoint_path)
    yolo.model.to(device)

    # 重建 manager（會重掛 hook）
    manager = ReIDHookManager(yolo, device=device)

    # 若 checkpoint 內已有 projector 權重，直接複寫
    if hasattr(yolo.model, "reid_projector"):
        manager.projector = yolo.model.reid_projector.to(device)
        print(f"✅  reid_projector restored from checkpoint.")

    manager.eval_mode()
    print(f"✅  YOLO ReID model loaded from {checkpoint_path}")
    return yolo, manager