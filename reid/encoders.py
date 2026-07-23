import numpy as np
import torch
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.ops import xywh2xyxy
import torch
import torch.nn as nn
import torch.nn.functional as F
from reid.yolo_reid_trainer import load_reid_model

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
       


class SOTA_ReID:
    """SOTA model as encoder for re-identification."""

    def __init__(self, model_path):
        """Initialize encoder for re-identification."""

        self.with_multi_cam = "vit_transreid" in model_path
   
        # "<class 'reid_models.modeling.utils.MutiInputSequential'>"
        self.model = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
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

        # AssertionError: Input image size (24*17) doesn't match model (256*128).
        if self.with_multi_cam:
            feats = self.model(
                    crop_imgs, 
                    # cam_label=-1, view_label=-1
                    cam_label=0, view_label=0
                    )
        else:
            feats = self.model(crop_imgs)


        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().detach().numpy() for f in feats]
   
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
            

# class ReID:
#     """YOLO model as encoder for re-identification."""

#     def __init__(self, model):
#         """Initialize encoder for re-identification."""

#         self.cust_model = load_reid_model(
#             checkpoint_path = model ,
#             device='cpu',
#         )
      

#     def __call__(self, img, dets):
#         """Extract embeddings for detected objects."""        
#         # 將每個偵測框 (dets) 從 xywh 格式轉為 xyxy 格式後，利用 save_one_box 把對應的區域從輸入影像 img 裁剪出來，
#         # 並收集成 box_imgs 清單。這樣 box_imgs 就包含了每個被偵測物件 (例如：行人) 的影像切片 (numpy 格式)。
#         box_imgs = [
#             torch.from_numpy(save_one_box(det, img, save=False).copy()).permute(2, 0, 1).float()
#             for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))
#         ]       
       
#         cust_feats = self.cust_model(torch.stack(box_imgs)) # torch.Size([1, 256])

#         if len(cust_feats) != dets.shape[0] and cust_feats[0].shape[0] == dets.shape[0]:
#             cust_feats = cust_feats[0]  # batched prediction with non-PyTorch backend
#         return [f.cpu().detach().numpy() for f in cust_feats] 