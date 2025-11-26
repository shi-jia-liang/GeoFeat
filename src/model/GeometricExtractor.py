import torch
import torch.nn as nn
from .CurvatureComputer import CurvatureComputer

class GeometricExtractor(nn.Module):
    """
    Extracts and aggregates multi-order geometric features.
    Assumes depth and normal are provided (or computed externally).
    """
    def __init__(self):
        super().__init__()
        self.curvature_computer = CurvatureComputer()

    def forward(self, depth, normal):
        """
        Args:
            depth: (B, 1, H, W)
            normal: (B, 3, H, W)
        Returns:
            dict with all geometric features
        """
        # Compute curvature and gradients from depth
        geo_features = self.curvature_computer(depth)
        
        # Add depth and normal to the dict
        geo_features['depth'] = depth
        geo_features['normal'] = normal
        
        return geo_features
