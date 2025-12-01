import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import sys

# sys.path.append("D:/Projects/GeoFeat/3rdparty/Depth-Anything-V2")
# from depth_anything_v2.dpt import DepthAnythingV2
# from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

# import time

# VITS_MODEL_PATH = "D:/Projects/GeoFeat/3rdparty/Depth-Anything-V2/depth_anything_v2/checkpoints/depth_anything_v2_vits.pth"
# VITB_MODEL_PATH = "D:/Projects/GeoFeat/3rdparty/Depth-Anything-V2/depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth"
# VITL_MODEL_PATH = "D:/Projects/GeoFeat/3rdparty/Depth-Anything-V2/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth"
# model_configs = {
#         "vits": {
#             "encoder": "vits",
#             "features": 64,
#             "out_channels": [48, 96, 192, 384]},
#         "vitb": {
#             "encoder": "vitb",
#             "features": 128,
#             "out_channels": [96, 192, 384, 768],
#         },
#         "vitl": {
#             "encoder": "vitl",
#             "features": 256,
#             "out_channels": [256, 512, 1024, 1024],
#         },
#         "vitg": {
#             "encoder": "vitg",
#             "features": 384,
#             "out_channels": [1536, 1536, 1536, 1536],
#         },
#     }

# # DAMv2模型, 用于提取图片深度法向量
# class DepthAnythingExtractor(nn.Module):
#     def __init__(self, encoder_type, device, input_size, process_size=(600,800)):
#         super().__init__()
#         self.net = DepthAnythingV2(**model_configs[encoder_type])
#         self.device = device
#         if encoder_type == "vits":
#             print(f"loading {VITS_MODEL_PATH}")
#             self.net.load_state_dict(torch.load(VITS_MODEL_PATH, map_location="cpu"))
#         elif encoder_type == "vitb":
#             print(f"loading {VITB_MODEL_PATH}")
#             self.net.load_state_dict(torch.load(VITB_MODEL_PATH, map_location="cpu"))
#         elif encoder_type == "vitl":
#             print(f"loading {VITL_MODEL_PATH}")
#             self.net.load_state_dict(torch.load(VITL_MODEL_PATH, map_location="cpu"))
#         else:
#             raise RuntimeError("unsupport encoder type")
#         # 将模型移动到GPU
#         self.net.to(self.device).eval()
#         # 调整尺寸 -> 归一化 -> 转为浮点类型 (该预处理与DAMv2的预处理相同)
#         self.tranform = Compose([
#                 Resize(
#                     width=input_size,
#                     height=input_size,
#                     resize_target=False,
#                     keep_aspect_ratio=True,
#                     ensure_multiple_of=14,
#                     resize_method='lower_bound',
#                     image_interpolation_method=cv2.INTER_CUBIC,
#                 ),
#                 NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 PrepareForNet(),
#             ])
#         self.process_size=process_size  # 输出图像大小
#         self.input_size=input_size      # 输入图像大小
        
#     @torch.inference_mode()
#     def infer_image(self,img):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
#         img = self.tranform({'image': img})['image']
        
#         img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
#         # 仅进行特征提取，不进行计算
#         with torch.no_grad():
#             depth = self.net.forward(img)
        
#         depth = F.interpolate(depth[:, None], self.process_size, mode="bilinear", align_corners=True)[0, 0]
        
#         return depth.cpu().numpy()
    
#     @torch.inference_mode()
#     def compute_normal_map_torch(self, depth_map, scale=1.0):
#         """
#         通过深度图计算法向量 (PyTorch 实现)

#         参数：
#             depth_map (torch.Tensor): 深度图，形状为 (H, W)
#             scale (float): 深度值的比例因子，用于调整深度图中的梯度计算

#         返回：
#             torch.Tensor: 法向量图，形状为 (H, W, 3)
#         """
#         if depth_map.ndim != 2:
#             raise ValueError("输入 depth_map 必须是二维张量。")
        
#         # 计算深度图的梯度
#         dzdx = torch.diff(depth_map, dim=1, append=depth_map[:, -1:]) * scale   # 宽的梯度
#         dzdy = torch.diff(depth_map, dim=0, append=depth_map[-1:, :]) * scale   # 高的梯度

#         # 初始化法向量图
#         H, W = depth_map.shape
#         normal_map = torch.zeros((H, W, 3), dtype=depth_map.dtype, device=depth_map.device)
#         normal_map[:, :, 0] = -dzdx  # x 分量
#         normal_map[:, :, 1] = -dzdy  # y 分量
#         normal_map[:, :, 2] = 1.0    # z 分量

#         # 归一化法向量
#         norm = torch.linalg.norm(normal_map, dim=2, keepdim=True)
#         norm = torch.where(norm == 0, torch.tensor(1.0, device=depth_map.device), norm)  # 避免除以零
#         normal_map /= norm

#         return normal_map
    
#     # 深度提取
#     @torch.inference_mode()
#     def extract(self, img):
#         depth = self.infer_image(img)
#         # 深度图归一化, 将深度值线性映射到 [0, 255] 范围
#         depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
#         # 转换为torch张量
#         depth_t=torch.from_numpy(depth).float().to(self.device)
#         # 法向量计算
#         normal_map = self.compute_normal_map_torch(depth_t,1.0)
#         return depth_t,normal_map
    
    
# if __name__=="__main__":
#     img_path=os.path.join(os.path.dirname(__file__),'../assert/ref.jpg')
#     img=cv2.imread(img_path)
#     img=cv2.resize(img,(800,608))
#     # import pdb;pdb.set_trace()
#     DAExtractor=DepthAnythingExtractor('vits',torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),256)
#     depth_t,norm=DAExtractor.extract(img)
#     norm=norm.cpu().numpy()
#     norm=(norm+1)/2*255
#     norm=norm.astype(np.uint8)
#     cv2.imwrite(os.path.join(os.path.dirname(__file__),"norm.png"),norm)
#     start=time.perf_counter()
#     for i in range(20):
#         depth_t,norm=DAExtractor.extract(img)
#     end=time.perf_counter()
#     print(f"cost {end-start} seconds")

# Add Depth-Anything-3 to path
sys.path.append("D:/Projects/GeoFeat/3rdparty/Depth-Anything-3/src")
try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"Could not import DepthAnything3. Make sure the path is correct. Error: {e}")
    pass

import time

# Mapping from old encoder types to new HF model names
MODEL_MAP = {
    "vits": "da3-small",
    "vitb": "da3-base",
    "vitl": "da3-large",
    "vitg": "da3-giant",
}

# Local checkpoint paths (Update these paths to your local weights)
# Supports .pth, .pt, .bin, and .safetensors
CHECKPOINT_MAP = {
    "da3-small": "D:/Projects/GeoFeat/3rdparty/Depth-Anything-3/checkpoints/depth_anything_v3_small.safetensors",
    "da3-base": "D:/Projects/GeoFeat/3rdparty/Depth-Anything-3/checkpoints/depth_anything_v3_base.safetensors",
    "da3-large": "D:/Projects/GeoFeat/3rdparty/Depth-Anything-3/checkpoints/depth_anything_v3_large.safetensors",
    "da3-giant": "D:/Projects/GeoFeat/3rdparty/Depth-Anything-3/checkpoints/depth_anything_v3_giant.safetensors",
}

# DAMv3模型, 用于提取图片深度法向量
class DepthAnythingExtractor(nn.Module):
    def __init__(self, encoder_type, device, input_size=518, process_size=(600,800)):
        super().__init__()
        self.device = device
        self.input_size = input_size # Used as process_res for DA3
        self.process_size = process_size # Output size (H, W)
        
        model_name = MODEL_MAP.get(encoder_type, "da3-small")
        print(f"Initializing Depth Anything 3 model: {model_name}")
        
        # Initialize model structure without loading weights from Hub
        self.net = DepthAnything3(model_name=model_name)
        
        # Load weights locally
        ckpt_path = CHECKPOINT_MAP.get(model_name)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading local checkpoint: {ckpt_path}")
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location='cpu')
                
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load with strict=False to handle potential missing auxiliary head weights
            missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]} ... (total {len(missing_keys)})")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]} ... (total {len(unexpected_keys)})")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Model initialized with random weights.")
            
        self.net.to(self.device).eval()
        
    @torch.inference_mode()
    def compute_normal_map_torch(self, depth_map, scale=1.0):
        """
        通过深度图计算法向量 (PyTorch 实现)

        参数：
            depth_map (torch.Tensor): 深度图，形状为 (H, W)
            scale (float): 深度值的比例因子，用于调整深度图中的梯度计算

        返回：
            torch.Tensor: 法向量图，形状为 (H, W, 3)
        """
        if depth_map.ndim != 2:
            raise ValueError("输入 depth_map 必须是二维张量。")
        
        # 计算深度图的梯度
        dzdx = torch.diff(depth_map, dim=1, append=depth_map[:, -1:]) * scale   # 宽的梯度
        dzdy = torch.diff(depth_map, dim=0, append=depth_map[-1:, :]) * scale   # 高的梯度

        # 初始化法向量图
        H, W = depth_map.shape
        normal_map = torch.zeros((H, W, 3), dtype=depth_map.dtype, device=depth_map.device)
        normal_map[:, :, 0] = -dzdx  # x 分量
        normal_map[:, :, 1] = -dzdy  # y 分量
        normal_map[:, :, 2] = 1.0    # z 分量

        # 归一化法向量
        norm = torch.linalg.norm(normal_map, dim=2, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1.0, device=depth_map.device), norm)  # 避免除以零
        normal_map /= norm

        return normal_map
    
    # 深度提取
    @torch.inference_mode()
    def extract(self, img, target_size=None):
        """
        Extract depth and normal from a single image.
        Args:
            img: numpy array (H, W, 3) BGR
            target_size: tuple (H, W) for output size. If None, uses self.process_size
        Returns:
            depth_t: (H, W) tensor
            normal_map: (H, W, 3) tensor
        """
        # DA3 inference expects RGB images (numpy, PIL, or path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Determine current device from model parameters
        try:
            current_device = next(self.net.parameters()).device
        except StopIteration:
            current_device = self.device

        # Run inference
        # process_res controls the internal resolution. 
        prediction = self.net.inference(
            [img_rgb], 
            process_res=self.input_size,
            process_res_method="lower_bound_resize"
        )
        
        # Get depth (N, H, W) -> (H, W)
        # DA3 returns numpy array
        depth = prediction.depth[0]
        
        # Convert to tensor and move to device
        depth_t = torch.from_numpy(depth).float().to(current_device)
        
        # Resize to target process_size
        # Interpolate expects (B, C, H, W)
        output_size = target_size if target_size is not None else self.process_size
        
        depth_t = depth_t.unsqueeze(0).unsqueeze(0)
        depth_t = F.interpolate(depth_t, output_size, mode="bilinear", align_corners=True)
        depth_t = depth_t.squeeze()
        
        # 深度图归一化, 将深度值线性映射到 [0, 255] 范围
        if depth_t.max() - depth_t.min() > 1e-6:
            depth_t = (depth_t - depth_t.min()) / (depth_t.max() - depth_t.min()) * 255.0
        
        # 法向量计算
        normal_map = self.compute_normal_map_torch(depth_t, 1.0)
        
        return depth_t, normal_map
    
    
if __name__=="__main__":
    img_path=os.path.join(os.path.dirname(__file__),'../assert/ref.jpg')
    if os.path.exists(img_path):
        img=cv2.imread(img_path)
        img=cv2.resize(img,(800,60))
        
        DAExtractor=DepthAnythingExtractor('vits',torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),256)
        depth_t,norm=DAExtractor.extract(img)
        
        norm=norm.cpu().numpy()
        norm=(norm+1)/2*255
        norm=norm.astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(__file__),"norm.png"),norm)
        
        start=time.perf_counter()
        for i in range(20):
            depth_t,norm=DAExtractor.extract(img)
        end=time.perf_counter()
        print(f"cost {end-start} seconds")
    else:
        print(f"Image not found: {img_path}")
    