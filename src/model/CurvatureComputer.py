import torch
import torch.nn as nn
import torch.nn.functional as F

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
            'grad_x': grad_x,
            'grad_y': grad_y,
            'k1': k1,
            'k2': k2,
            'gaussian_curv': gaussian_curv,
            'mean_curv': mean_curv,
            'shape_index': shape_index
        }
