import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.depth_anything_wrapper import DepthAnythingExtractor

# 计算高阶几何特征
class CurvatureComputer(nn.Module):
    """
    Compute geometric curvature features from depth maps.
    Calculates:
    - Gradients (grad_x, grad_y)
    - Principal Curvatures (k1, k2)
    - Gaussian Curvature (K)
    - Mean Curvature (H)
    - Shape Index (SI)
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels for gradients
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def compute_gradients(self, depth):
        """Compute first and second order derivatives."""
        # Pad for valid convolution
        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        
        # First order
        grad_x = F.conv2d(depth_pad, self.sobel_x)
        grad_y = F.conv2d(depth_pad, self.sobel_y)
        
        # Pad gradients for second order
        grad_x_pad = F.pad(grad_x, (1, 1, 1, 1), mode='replicate')
        grad_y_pad = F.pad(grad_y, (1, 1, 1, 1), mode='replicate')
        
        # Second order
        grad_xx = F.conv2d(grad_x_pad, self.sobel_x)
        grad_yy = F.conv2d(grad_y_pad, self.sobel_y)
        grad_xy = F.conv2d(grad_x_pad, self.sobel_y)
        
        return grad_x, grad_y, grad_xx, grad_yy, grad_xy

    def forward(self, depth):
        """
        Args:
            depth: (B, 1, H, W) depth map
        Returns:
            dict containing geometric features
        """
        B, C, H, W = depth.shape
        
        # 1. Compute derivatives
        grad_x, grad_y, grad_xx, grad_yy, grad_xy = self.compute_gradients(depth)
        
        # Compute Normal Vector
        # Surface normal n = (-dz/dx, -dz/dy, 1) normalized
        ones = torch.ones_like(grad_x)
        normal = torch.cat((-grad_x, -grad_y, ones), dim=1)
        normal = F.normalize(normal, dim=1)
        
        # 2. Compute Principal Curvatures
        # Eigenvalues of Hessian: (trace +/- sqrt(trace^2 - 4*det)) / 2
        trace = grad_xx + grad_yy
        det = grad_xx * grad_yy - grad_xy ** 2
        
        # Numerical stability
        discriminant = torch.sqrt(torch.clamp(trace**2 - 4*det, min=1e-6))
        
        k1 = (trace + discriminant) / 2.0
        k2 = (trace - discriminant) / 2.0
        
        # 3. Derived Curvatures
        gaussian_curv = k1 * k2
        mean_curv = (k1 + k2) / 2.0
        
        # 4. Shape Index
        # SI = 2/pi * arctan((k2 + k1) / (k2 - k1))
        # Avoid division by zero
        diff = k2 - k1
        diff = torch.where(torch.abs(diff) < 1e-6, torch.ones_like(diff) * 1e-6, diff)
        shape_index = (2.0 / 3.1415926) * torch.atan((k2 + k1) / diff)
        
        return {
            'normal': normal,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'k1': k1,
            'k2': k2,
            'gaussian_curv': gaussian_curv,
            'mean_curv': mean_curv,
            'shape_index': shape_index
        }

# 计算所有的几何特征
class GeometricExtractor(nn.Module):
    """
    Extracts multi-order geometric features from depth and normal maps.
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.curvature_computer = CurvatureComputer()
        
        # Initialize DepthAnythingExtractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_extractor = DepthAnythingExtractor('vits', self.device, 256)

    def forward(self, depth=None, normal=None, image=None):
        """
        Args:
            depth: (B, 1, H, W) tensor
            normal: (B, 3, H, W) tensor
            image: (B, 3, H, W) tensor or (H, W, 3) numpy array
        Returns:
            dict with all geometric features
        """
        if depth is None or normal is None:
            if image is None:
                raise ValueError("Must provide either (depth, normal) or image")
            
            # Compute depth and normal from image
            if isinstance(image, torch.Tensor):
                # Assume (B, 3, H, W)
                B, C, H, W = image.shape
                depth_list = []
                normal_list = []
                
                # Move to cpu and numpy for DA3 wrapper
                # Ensure image is in range [0, 255] and uint8
                imgs_np = image.permute(0, 2, 3, 1).detach().cpu().numpy() # (B, H, W, 3)
                
                if imgs_np.max() <= 1.0 + 1e-6:
                    imgs_np = (imgs_np * 255).astype(np.uint8)
                else:
                    imgs_np = imgs_np.astype(np.uint8)
                    
                for i in range(len(imgs_np)):
                    # Pass target size (H, W) to extract
                    target_size = (H, W)
                    d, _ = self.depth_extractor.extract(imgs_np[i], target_size=target_size)
                    depth_list.append(d)
                    # normal_list.append(n)
                
                depth = torch.stack(depth_list).unsqueeze(1) # (B, 1, H, W)
                # normal = torch.stack(normal_list).permute(0, 3, 1, 2) # (B, 3, H, W)
                
            elif isinstance(image, np.ndarray):
                # Assume single numpy image (H, W, 3)
                target_size = (image.shape[0], image.shape[1])
                d, _ = self.depth_extractor.extract(image, target_size=target_size)
                depth = d.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                # normal = n.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
            else:
                 raise ValueError("Image must be torch.Tensor or numpy.ndarray")

        # Compute 1-order (Gradients) and 2-order (Curvature) geometric features
        # curvature_dict contains: grad_x, grad_y, k1, k2, gaussian_curv, mean_curv, shape_index
        curvature_dict = self.curvature_computer(depth)
        
        geo_features = {}
        
        # 0-order
        if self.model_config.get('depth', True):
            geo_features['depth'] = depth
            
        # 1-order
        if self.model_config.get('normal', True):
            geo_features['normal'] = curvature_dict['normal']
            
        # Gradients
        if self.model_config.get('gradients', True):
            geo_features['grad_x'] = curvature_dict['grad_x']
            geo_features['grad_y'] = curvature_dict['grad_y']
            
        # Curvatures
        if self.model_config.get('curvatures', True):
            geo_features['k1'] = curvature_dict['k1']
            geo_features['k2'] = curvature_dict['k2']
            geo_features['gaussian_curv'] = curvature_dict['gaussian_curv']
            geo_features['mean_curv'] = curvature_dict['mean_curv']
            geo_features['shape_index'] = curvature_dict['shape_index']
        
        return geo_features

class GeometricAttentionFusion(nn.Module):
    """
    Adaptive fusion of geometric and texture features using attention.
    """
    def __init__(self, model_config, feature_dim=64):
        super().__init__()
        self.model_config = model_config
        
        # Calculate input dimension based on config
        geo_input_dim = 0
        # 0-order
        if self.model_config.get('depth', True): 
            geo_input_dim += 1
        # 1-order
        if self.model_config.get('normal', True): 
            geo_input_dim += 3
        # Gradients
        if self.model_config.get('gradients', True): 
            geo_input_dim += 2 # grad_x, grad_y
        # Curvatures
        if self.model_config.get('curvatures', True): 
            geo_input_dim += 5 # k1, k2, K, H, SI
            
        if geo_input_dim == 0:
            # Fallback or minimal dimension to avoid crash during init, though forward might skip
            geo_input_dim = 1 

        # Geometric Complexity Network
        self.complexity_net = nn.Sequential(
            nn.Conv2d(geo_input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Geometric Feature Encoder
        self.geo_encoder = nn.Sequential(
            nn.Conv2d(geo_input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1)
        )
        
        # Fusion projection
        self.fusion_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, texture_feat, geo_features_dict):
        """
        Args:
            texture_feat: (B, C, H, W)
            geo_features_dict: dict containing geometric maps
        """
        geo_list = []
        
        # Detect and collect non-None geometric features
        # 0-order
        if geo_features_dict.get('depth') is not None:
            geo_list.append(geo_features_dict['depth'])
            
        # 1-order
        if geo_features_dict.get('normal') is not None:
            geo_list.append(geo_features_dict['normal'])
            
        # Gradients
        if geo_features_dict.get('grad_x') is not None:
            geo_list.append(geo_features_dict['grad_x'])
        if geo_features_dict.get('grad_y') is not None:
            geo_list.append(geo_features_dict['grad_y'])
            
        # Curvatures
        if geo_features_dict.get('k1') is not None:
            geo_list.append(geo_features_dict['k1'])
        if geo_features_dict.get('k2') is not None:
            geo_list.append(geo_features_dict['k2'])
        if geo_features_dict.get('gaussian_curv') is not None:
            geo_list.append(geo_features_dict['gaussian_curv'])
        if geo_features_dict.get('mean_curv') is not None:
            geo_list.append(geo_features_dict['mean_curv'])
        if geo_features_dict.get('shape_index') is not None:
            geo_list.append(geo_features_dict['shape_index'])
        
        if not geo_list:
            # If no geometric features are available, return texture features directly
            return texture_feat, None
        
        geo_concat = torch.cat(geo_list, dim=1)
        
        # Compute complexity attention map
        complexity = self.complexity_net(geo_concat)
        
        # Encode geometric features
        geo_feat = self.geo_encoder(geo_concat)
        
        # Adaptive Fusion
        # F = A * G + (1 - A) * T
        fused_feat = complexity * geo_feat + (1 - complexity) * texture_feat
        
        # Optional final projection
        fused_feat = self.fusion_proj(fused_feat)
        
        return fused_feat, complexity

if __name__ == "__main__":
    print("=== Verifying GeometricExtractor ===")
    
    # Mock config
    model_config = {
        "geometric_features": {
            "depth": True,
            "normal": True,
            "gradients": True,
            "curvatures": True
        }
    }
    
    # Initialize
    print("Initializing GeometricExtractor...")
    try:
        extractor = GeometricExtractor(model_config)
        if torch.cuda.is_available():
            extractor = extractor.cuda()
        print("[Success] Initialized")
    except Exception as e:
        print(f"[Error] Initialization failed: {e}")
        sys.exit(1)
        
    # Dummy Input
    H, W = 480, 640
    dummy_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # Forward
    print(f"Running inference on image shape {dummy_img.shape}...")
    try:
        # GeometricExtractor.forward handles numpy image by calling depth_extractor.extract
        features = extractor(image=dummy_img)
        
        print("[Success] Inference complete")
        print("Output features:")
        for k, v in features.items():
            print(f"  {k}: {v.shape}")
            
        # Check resolution
        if features['depth'].shape[-2:] == (H, W):
            print("[Pass] Output resolution matches input")
        else:
            print(f"[Fail] Output resolution mismatch: {features['depth'].shape[-2:]} vs {(H, W)}")
            
    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        import traceback
        traceback.print_exc()