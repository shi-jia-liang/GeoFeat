import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricAttentionFusion(nn.Module):
    """
    Adaptive fusion of geometric and texture features using attention.
    """
    def __init__(self, feature_dim=64, geo_input_dim=10):
        super().__init__()
        
        # Geometric Complexity Network
        # Inputs: depth(1) + normal(3) + gradient(2) + k1(1) + k2(1) + K(1) + H(1) = 10 channels
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
        # Concatenate geometric features
        # Expected keys: depth, normal, grad_x, grad_y, k1, k2, gaussian_curv, mean_curv
        # Note: Ensure all are resized to texture_feat resolution if needed
        
        geo_list = [
            geo_features_dict['depth'],
            geo_features_dict['normal'],
            geo_features_dict['grad_x'],
            geo_features_dict['grad_y'],
            geo_features_dict['k1'],
            geo_features_dict['k2'],
            geo_features_dict['gaussian_curv'],
            geo_features_dict['mean_curv']
        ]
        
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
