import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics.pairwise import cosine_distances

# ===========================================
# 五、模型定義（與原腳本相同，num_cams 改用 total_cams）
# ===========================================

class OcclusionRepair(nn.Module):
    """遮擋特徵修復模組（殘差結構）。"""
    def __init__(self, emb_dim=512):
        super().__init__()
        self.repair = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        return x + 0.5 * self.repair(x)


class CameraEmbedding(nn.Module):
    """將 camid 映射為可學習向量，讓模型感知攝影機特性。"""
    def __init__(self, n_cams, emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(n_cams + 1, emb_dim)   # +1 保留 padding

    def forward(self, camids):
        return self.emb(camids)


class ReIDNet(nn.Module):
    """
    ResNet-50 骨幹 + 攝影機嵌入 + 遮擋修復 + 分類頭。
    n_cams 必須等於 TOTAL_CAMS（14），涵蓋兩個資料集所有攝影機。
    """
    def __init__(self, n_classes, n_cams):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone      = nn.Sequential(*list(backbone.children())[:-1])
        self.cam_emb       = CameraEmbedding(n_cams, emb_dim=32)
        self.embedding_dim = 2048 + 32
        self.metric        = nn.Linear(self.embedding_dim, 512)
        self.occlusion_repair = OcclusionRepair(emb_dim=512)
        self.classifier    = nn.Linear(512, n_classes)

    def forward(self, x, camids):
        if not torch.is_tensor(camids):
            camids = torch.as_tensor(camids, device=x.device, dtype=torch.long)
        feat   = self.backbone(x).reshape(x.size(0), -1)
        camfeat = self.cam_emb(camids) if camids.numel() > 0 else \
                  torch.zeros(x.size(0), 32, device=x.device)
        comb   = torch.cat([feat, camfeat], dim=1)
        emb    = torch.nn.functional.normalize(self.metric(comb), dim=1)
        emb_r  = self.occlusion_repair(emb)
        logits = self.classifier(emb_r)
        return emb, logits
