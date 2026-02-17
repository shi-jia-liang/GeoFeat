
import torch
import torch.nn as nn
import sys
import os
import time

# Ensure paths are correct
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('LiftFeat'))

# Mock imports that might fail
try:
    from model.GeoFeatModel import GeoFeatModel
except ImportError:
    print("Could not import GeoFeatModel directly. Trying to fix paths.")
    sys.path.append(os.path.abspath('.'))
    from src.model.GeoFeatModel import GeoFeatModel

try:
    from LiftFeat.models.model import LiftFeatSPModel
    from LiftFeat.utils.config import featureboost_config
except ImportError as e:
    print(f"Error importing LiftFeat: {e}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        param_size += buffer.nelement() * buffer.element_size()
    return param_size / 1024 / 1024

def measure_flops(model, input_shape=(1, 1, 640, 480)):
    # Simple FLOPs estimation using thop if available, else fallback to torch profiler or manual
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    
    # Try using thop
    try:
        from thop import profile
        # Use a smaller input or ensure checking carefully. 
        # Note: If model forward fails, we catch it here.
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        return macs * 2 # FLOPs ~= 2 * MACs
    except Exception as e:
        print(f"FLOPs measurement failed: {e}")
        return "N/A"

def get_precision(model):
    try:
        dtype = next(model.parameters()).dtype
        return str(dtype)
    except:
        return "Unknown"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. GeoFeat Model
    # Reconstructing config from src/config/config.py
    geofeat_config = {
        'backbone': 'RepVGG',
        'upsample_type': 'FSR', # Use FSR for fair comparison if that's the intent, or bilinear as default
        'pos_enc_type': 'rot_inv', 
        'keypoint_dim': 65,
        'descriptor_dim': 64,
        'keypoint_encoder': [128, 64, 64],
        'descriptor_encoder': [64, 64],
        'geometric_features': {
            'depth': False,
            'normal': True,    # Enabled in config.py
            'gradients': False,
            'curvatures': False
        },
        'depth_encoder': [128, 64, 64],
        'normal_encoder': [128, 64, 64],
        'gradient_encoder': [64, 32, 32],
        'use_geometric_fusion': True, # As requested in conversation
        'normal_dim': 192,
        'depth_dim': 64,
        'Attentional_layers': 3,
        'attention_layers': 3,
        'AFT': {
            'ffn_type': 'pffn'
        },
        'Swin': {
             'input_resolution': (640, 480), # Just dummy values to satisfy init
             'window_size': 7,
             'depth_per_layer': 2,
             'num_heads': 4,
             'ffn_type': 'PositionwiseFeedForward'
        },
        'last_activation': 'None',
        'l2_normalization': 'None',
        'output_dim': 64,
    }
    
    print("\n--- GeoFeat Model ---")
    try:
        geofeat_model = GeoFeatModel(geofeat_config).to(device)
        geofeat_params = count_parameters(geofeat_model)
        geofeat_size = get_model_size_mb(geofeat_model)
        geofeat_prec = get_precision(geofeat_model)
        geofeat_flops = measure_flops(geofeat_model)
        
        print(f"Parameters: {geofeat_params:,}")
        print(f"Precision: {geofeat_prec}")
        print(f"Model Size: {geofeat_size:.2f} MB")
        print(f"FLOPs (sample input 640x480): {geofeat_flops}")
        
    except Exception as e:
        print(f"Error initializing GeoFeatModel: {e}")
        import traceback
        traceback.print_exc()

    # 2. LiftFeat Model
    print("\n--- LiftFeat Model ---")
    try:
        liftfeat_model = LiftFeatSPModel(featureboost_config).to(device)
        liftfeat_params = count_parameters(liftfeat_model)
        liftfeat_size = get_model_size_mb(liftfeat_model)
        liftfeat_prec = get_precision(liftfeat_model)
        liftfeat_flops = measure_flops(liftfeat_model)
        
        print(f"Parameters: {liftfeat_params:,}")
        print(f"Precision: {liftfeat_prec}")
        print(f"Model Size: {liftfeat_size:.2f} MB")
        print(f"FLOPs (sample input 640x480): {liftfeat_flops}")
        
    except Exception as e:
        print(f"Error initializing LiftFeatSPModel: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
