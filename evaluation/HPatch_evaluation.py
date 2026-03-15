import cv2
import os
import sys
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.geofeat_wrapper import GeoFeat
from src.config.config import get_cfg_defaults

# Configuration
DATASET_ROOT = os.path.join(os.path.dirname(__file__), '../data/hpatches-sequences-release')

def benchmark_features(match_fn):
    seq_names = sorted(os.listdir(DATASET_ROOT))
    
    # Pre-calculate totals for normalization
    i_seqs = [s for s in seq_names if s.startswith('i')]
    v_seqs = [s for s in seq_names if s.startswith('v')]
    total_i_imgs = len(i_seqs) * 5
    total_v_imgs = len(v_seqs) * 5

    # Thresholds for accumulation (1, 3, 5, 7, 9 pixels)
    thresholds = [1, 3, 5, 7, 9]
    # Structure: {'i': {1: count, 3: count...}, 'v': {...}}
    errors = {
        'i': {t: 0 for t in thresholds},
        'v': {t: 0 for t in thresholds}
    }
    
    processed_counts = {'i': 0, 'v': 0}

    print(f"Starting evaluation on {len(seq_names)} sequences...")

    for seq_name in tqdm(seq_names):
        seq_type = 'i' if seq_name.startswith('i') else 'v'
        
        # Load reference image (Index 1)
        ref_path = os.path.join(DATASET_ROOT, seq_name, "1.ppm")
        ref_img = cv2.imread(ref_path)
        h, w = ref_img.shape[:2]
        
        # Define corner points for reprojection error calculation
        corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32).reshape(-1, 1, 2)

        # Loop through query images (Indices 2-6)
        for im_idx in range(2, 7):
            query_path = os.path.join(DATASET_ROOT, seq_name, f"{im_idx}.ppm")
            homo_path = os.path.join(DATASET_ROOT, seq_name, f"H_1_{im_idx}")
            
            query_img = cv2.imread(query_path)
            gt_homo = np.loadtxt(homo_path)

            # 1. Feature Matching
            # Assuming match_fn handles its own internal errors if needed, 
            # or crashes loudly so we can fix the model.
            mkpts0, mkpts1 = match_fn(ref_img, query_img)

            # 2. Homography Estimation
            pred_homo = None
            if len(mkpts0) >= 4:
                # USAC_MAGSAC is robust and fast
                pred_homo, _ = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.0)

            # 3. Error Calculation
            if pred_homo is None:
                err = float('inf')
            else:
                # Project corners using GT and Predicted Homographies
                warped_corners_gt = cv2.perspectiveTransform(corners, gt_homo)
                warped_corners_pred = cv2.perspectiveTransform(corners, pred_homo)
                
                # Calculate mean corner error
                diff = warped_corners_gt - warped_corners_pred
                err = np.mean(np.linalg.norm(diff, axis=2))

            # Accumulate results
            for t in thresholds:
                if err <= t:
                    errors[seq_type][t] += 1
            
            # Simple progress logging
            processed_counts[seq_type] += 1

    return errors, total_i_imgs, total_v_imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified HPatch Evaluation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    args = parser.parse_args()

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Load Configuration
    # We strictly expect a config_snapshot.json next to the weights.
    weights_path = os.path.abspath(args.weights)
    snapshot_path = os.path.join(os.path.dirname(weights_path), 'config_snapshot.json')
    
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Critical: No config_snapshot.json found at {snapshot_path}. Comparison requires precise config.")

    print(f"Loading config from: {snapshot_path}")
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
        # Handle cases where snapshot wraps config in 'model_config' key or is flat
        model_config = data.get('model_config', data)

    # 2. Initialize Model
    print(f"Loading weights from: {weights_path}")
    model = GeoFeat(model_config=model_config, weight_path=weights_path)

    # 3. Run Benchmark
    errs, n_i, n_v = benchmark_features(model.match_featnet)

    # 4. Report Results
    print(f"\n{'='*20} Results: {os.path.basename(weights_path)} {'='*20}")
    print(f"{'Threshold':<10} {'Illumination':<15} {'Viewpoint':<15}")
    print("-" * 40)
    
    for t in [3, 5, 7]:
        i_acc = errs['i'][t] / n_i if n_i > 0 else 0
        v_acc = errs['v'][t] / n_v if n_v > 0 else 0
        print(f"{t:<10} {i_acc:<15.2%} {v_acc:<15.2%}")
    print("=" * 60)
