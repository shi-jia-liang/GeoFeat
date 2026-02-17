
import sys
import os
import torch
import torch.nn as nn
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

try:
    from src.model.GeoFeatModel import GeoFeatModel
    
    # Mock config manually based on typical usage
    model_config = {
        'backbone': 'RepVGG',
        'upsample_type': 'bilinear',
        'pos_enc_type': 'rot_inv',      # Or 'fourier'
        'keypoint_dim': 65,             # 8x8 + 1 or similar for keypoints
        'descriptor_dim': 128,
        'geometric_features': {
            'depth': True,
            'normal': True,
            'gradients': False, 
            'curvatures': False
        },
        # Feature Booster Config
        'attention_layers': 2,
        'AFT': {
            'ffn_type': 'swin',         # Use 'swin' to test SwinFFN path
            'input_resolution': (32, 32),
            'depth_per_layer': 2,
            'num_heads': 4,
            'window_size': 8,
            'ffn_type': 'positionwiseFFN' # Inner FFN type for Swin block
        },
        'Swin': { # Still need this block because AttentionalNN reads it via safe get, but for swin ffn we need it
             'input_resolution': (32, 32),
             'depth_per_layer': 2,
             'num_heads': 4,
             'window_size': 8,
             'ffn_type': 'positionwiseFFN'
        },
        # Add required dimensions for GeoHead
        'depth_dim': 1,
        'normal_dim': 3,
        'gradient_dim': 2,
        'curvature_dim': 5,
        
        # Encoders for GeoHead (dummy values)
        'depth_encoder': [32, 128], # Must output 128 (descriptor_dim) if GeoHead does projection? 
        # Actually GeoHead outputs `geo_map` with specific channel dims?
        # No, `geo_head` returns `geo_map` which is concatenation of features.
        # It doesn't encode them into `descriptor_dim`.
        'normal_encoder': [32, 128],
        'gradient_encoder': [32, 128],
        'curvature_encoder': [32, 128],
        
        # But wait, GeoHead logic might use these lists to build MLPs ending with... what dim?
        # Let's check GeoHead later if it fails.
        
        'keypoint_encoder': [32, 64],
        'descriptor_encoder': [32, 128], # Matching descriptor_dim
        
        'last_activation': 'None',
    }

    print("Initializing GeoFeatModel...")
    model = GeoFeatModel(model_config)
    print("Model initialized successfully.")
    
    # Test forward pass with dummy data
    B, C, H, W = 2, 3, 256, 256 # Batch > 1
    # Note: 256 / 8 = 32, matches Swin input_resolution if used
    img = torch.randn(B, C, H, W)
    
    print("Running forward pass...")
    start = time.time()
    
    # Forward pass: returns (descs_refine, des_map, geo_map, keypoint_map)
    out = model(img) # geo_map None -> Computed internally?
    # Original forward1 calls geo_head(x) if geo_map is None?
    # But wait, forward1 only receives 'x'. 
    # forward: (x) -> forward1(x) -> des, geo, kpt -> forward2(des, geo, kpt)
    # Correct.

    end = time.time()
    print(f"Forward pass took {end - start:.4f}s")
    
    if isinstance(out, tuple):
        # returns descs_refine, des_map, geo_map, keypoint_map
        descs_refine, des_map, geo_map, keypoint_map = out
        print(f"Refined descriptors shape: {descs_refine.shape}")
        
    else:
        print("Output type:", type(out))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
