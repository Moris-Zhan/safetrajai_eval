import numpy as np
import torch
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.ops import xywh2xyxy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReID:
    """YOLO model as encoder for re-identification."""

    def __init__(self, model):
        """Initialize encoder for re-identification."""
        from ultralytics import YOLO

        self.model = YOLO(model)
        # 這行會觸發YOLO模型的初始化，特別是載入權重後做一次空forward推論：
        # embed參數指定輸出倒數第二層（如果是.pt模型）作為特徵向量輸出，否則用-1代表最後一層。
        # 這主要確保reid模型推論時encoder已就緒。
        self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False)  # initialize

    def __call__(self, img, dets):
        """Extract embeddings for detected objects."""
        crop_imgs = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        
        feats = self.model(crop_imgs)
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]
       


class Custom_ReID:
    """SOTA model as encoder for re-identification."""

    def __init__(self, model_path):
        """Initialize encoder for re-identification."""
   
        self.model = torch.load(model_path,
                             map_location="cuda" if torch.cuda.is_available() else "cpu", 
                             weights_only=False)
        self.model.eval()

    def __call__(self, img, dets):
        """Extract embeddings for detected objects."""
        # crop_imgs - [ ... ][0] 其實只裁出第一個 bbox；此時 crop_imgs shape 為 (H, W, 3)
        # crop_imgs = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))][0]
        # # 將 crop_imgs 處理成符合模型輸入要求的 torch.Tensor: [N, 3, 256, 128]
        # img_tensor = torch.from_numpy(crop_imgs.copy()).permute(2, 0, 1).float()  # -> (3, H, W)

        crop_imgs = [
            torch.from_numpy(save_one_box(det, img, save=False).copy())
                .permute(2, 0, 1)   # (H,W,C) → (C,H,W)
                .float()
                .div(255.0)          # ← uint8 0~255 轉 float 0~1 (Normalize 需要)
                .unsqueeze(0)        # ← 加 batch dim: (C,H,W) → (1,C,H,W)
            for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))
        ][0]

        # feats = self.model(crop_imgs, camids=[])
        feats = self.model(crop_imgs, camids=[0])


        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().detach().numpy() for f in feats]