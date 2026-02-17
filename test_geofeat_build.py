
import sys
import os
import torch
import torch.nn as nn
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

try:
    from src.model.GeoFeatModel import GeoFeatModel, default_conf as geo_config
    
    # Enable depth and normal geometric features to test full capability
    geo_config['geometric_features']['depth'] = True
    geo_config['geometric_features']['normal'] = True
    geo_config['geometric_features']['gradients'] = False
    geo_config['geometric_features']['curvatures'] = False
    
    model = GeoFeatModel(geo_config)
    print("Model initialized successfully.")
    
    # Test forward pass with dummy data
    B, C, H, W = 1, 3, 224, 224
    img = torch.randn(B, C, H, W)
    
    # Also need dummy geometric map if we want to exercise the boost
    # GeoFeatModel.forward takes (x) and computes geo_map internally?
    
    # Looking at GeoFeatModel.forward:
    # def forward(self, input_image, geo_map=None):
    # If geo_map is None, it computes it.
    # In 'GeoFeatModel.py':
    # x = self.backbone(x)
    # ...
    # geo_map (B, 6, H, W) is assumed input if provided, else maybe computed?
    # Wait, GeoFeatModel likely calls a method to compute geo features or expects them.
    # Let's check `forward` impl.
    
    # Actually, in the provided code, `GeoFeatModel.forward` calls `self.compute_geometric_features` if `geo_map` is None?
    # Or does it rely on external computation? 
    # Let's inspect `GeoFeatModel.forward` via read_file first or assume if it fails.
    
    start_time = time.time()
    # We pass random 'geo_map' input: (B, C_geo, H, W)
    # C_geo depends on what features are enabled. 
    # Depth(1) + Normal(3) = 4 channels minimum if using those.
    # The code expects `geo_map` to be passed if available.
    
    # Let's try passing random geo map
    geo_in = torch.randn(B, 4, H, W)
    
    out = model(img, geo_map=geo_in)
    end_time = time.time()
    
    print(f"Forward pass took {end_time - start_time:.4f}s")
    print("Output keys:", out.keys())
    
    # Check shapes
    if 'keypoints' in out:
        kpts = out['keypoints']
        print(f"Keypoints shape (batch 0): {kpts[0].shape}") # Lists of tensors
        
    if 'descriptors' in out:
        desc = out['descriptors'] 
        print(f"Descriptors shape (batch 0): {desc[0].shape}")

    if 'scores' in out:
        scores = out['scores']
        print(f"Scores shape (batch 0): {scores[0].shape}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
