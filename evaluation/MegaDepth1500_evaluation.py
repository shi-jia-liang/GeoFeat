import os
import sys
import cv2
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.geofeat_wrapper import GeoFeat
from src.config.config import get_cfg_defaults
from evaluation.eval_utils import compute_pose_error, compute_maa

# Configuration
DATASET_ROOT = os.path.join(os.path.dirname(__file__), '../data/megadepth_test_1500')
JSON_PATH = os.path.join(os.path.dirname(__file__), '../data/megadepth_1500.json')

class MegaDepthDataset:
    def __init__(self, json_path, root_dir):
        print(f"Loading dataset index from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.root_dir = root_dir
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"MegaDepth images not found at {root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Shallow copy is fine as we only read/replace images
        item = self.data[idx].copy()
        
        # Paths
        p1 = os.path.join(self.root_dir, item['pair_names'][0])
        p2 = os.path.join(self.root_dir, item['pair_names'][1])

        # Load images
        # We assume dataset integrity. If files are missing, let cv2/os error out naturally or return None.
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        
        if img1 is None: raise ValueError(f"Failed to read image: {p1}")
        if img2 is None: raise ValueError(f"Failed to read image: {p2}")

        # Resize to specs in validatation set
        h1, w1 = item['size0_hw']
        h2, w2 = item['size1_hw']
        
        img1 = cv2.resize(img1, (w1, h1))
        img2 = cv2.resize(img2, (w2, h2))

        # Add images to item dict (as numpy arrays, handled by updated eval_utils.tensor2bgr)
        item['image0'] = img1
        item['image1'] = img2
        
        # Ensure matrix types are correct for computation
        item['K0'] = np.array(item['K0'], dtype=np.float32)
        item['K1'] = np.array(item['K1'], dtype=np.float32)
        item['T_0to1'] = np.array(item['T_0to1'], dtype=np.float32)
        
        return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified MegaDepth Evaluation')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Load Configuration
    weights_path = os.path.abspath(args.weights)
    snapshot_path = os.path.join(os.path.dirname(weights_path), 'config_snapshot.json')
    
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Critical: No config_snapshot.json found at {snapshot_path}")

    print(f"Loading config from: {snapshot_path}")
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
        # Handle cases where snapshot wraps config in 'model_config' or is flat
        model_config = data.get('model_config', data)

    # 2. Initialize Model
    print(f"Loading weights from: {weights_path}")
    model = GeoFeat(model_config=model_config, weight_path=weights_path)

    # 3. Prepare Data
    try:
        dataset = MegaDepthDataset(JSON_PATH, DATASET_ROOT)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)
        
    results = []

    print(f"Starting evaluation on {len(dataset)} pairs...")
    
    # 4. Main Loop
    for i in tqdm(range(len(dataset))):
        data_item = dataset[i]
        
        try:
            # Match & Estimate Pose
            error_info = compute_pose_error(
                match_fn=model.match_featnet,
                data=data_item
            )
            results.append(error_info)
        except Exception as e:
            tqdm.write(f"Error processing pair {i}: {e}")
            # Continue to next pair to get partial results if one fails
            continue

    # 5. Report Metrics
    if not results:
        print("No results collected.")
        sys.exit(0)
        
    auc_dict, _ = compute_maa(results)
    
    print(f"\n{'='*20} Results: {os.path.basename(weights_path)} {'='*20}")
    print(f"AUC@5:  {auc_dict.get('auc@5', 0)*100:.2f}%")
    print(f"AUC@10: {auc_dict.get('auc@10', 0)*100:.2f}%")
    print(f"AUC@20: {auc_dict.get('auc@20', 0)*100:.2f}%")
    print("=" * 60)
