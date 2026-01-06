import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import json

import scipy.io as scio
import poselib

import argparse
import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from src.utils.geofeat_wrapper import GeoFeat
from src.config.config import get_cfg_defaults
from evaluation.eval_utils import *

def get_model_config(cfg):
    return {
        'backbone': cfg.MODEL.BACKBONE,
        'upsample_type': cfg.MODEL.UPSAMPLE_TYPE,
        'pos_enc_type': cfg.MODEL.POS_ENC_TYPE,
        'keypoint_encoder': list(cfg.MODEL.KEYPOINT_ENCODER),
        'keypoint_dim': int(cfg.MODEL.KEYPOINT_DIM),
        'descriptor_encoder': list(cfg.MODEL.DESCRIPTOR_ENCODER),
        'descriptor_dim': int(cfg.MODEL.DESCRIPTOR_DIM),
        'geometric_features': {
            'depth': bool(cfg.MODEL.GEOMETRIC_FEATURES.DEPTH),
            'normal': bool(cfg.MODEL.GEOMETRIC_FEATURES.NORMAL),
            'gradients': bool(cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS),
            'curvatures': bool(cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES),
        },
        'depth_encoder': list(cfg.MODEL.DEPTH_ENCODER),
        'depth_dim': int(cfg.MODEL.DEPTH_DIM),
        'normal_encoder': list(cfg.MODEL.NORMAL_ENCODER),
        'normal_dim': int(cfg.MODEL.NORMAL_DIM),
        'gradient_encoder': list(cfg.MODEL.GRADIENT_ENCODER),
        'gradient_dim': int(cfg.MODEL.GRADIENT_DIM),
        'curvature_encoder': list(cfg.MODEL.CURVATURE_ENCODER),
        'curvature_dim': int(cfg.MODEL.CURVATURE_DIM),
        'Swin': {
            'input_resolution': list(cfg.MODEL.SWIN.INPUT_RESOLUTION),
            'depth_per_layer': int(cfg.MODEL.SWIN.DEPTH_PER_LAYER),
            'num_heads': int(cfg.MODEL.SWIN.NUM_HEADS),
            'window_size': int(cfg.MODEL.SWIN.WINDOW_SIZE),
            'ffn_type': cfg.MODEL.ATTENTION.SWIN.FFN_TYPE,
        },
        'attention_layers': int(cfg.MODEL.ATTENTIONAL_LAYERS),
        'attention_type': cfg.MODEL.ATTENTION.TYPE,
        'AFT': {
            'ffn_type': cfg.MODEL.ATTENTION.AFT.FFN_TYPE,
        },
        'last_activation': cfg.MODEL.LAST_ACTIVATION,
        'l2_normalization': bool(cfg.MODEL.L2_NORMALIZATION),
        'use_coord_loss': bool(cfg.MODEL.USE_COORD_LOSS),
        'output_dim': int(cfg.MODEL.OUTPUT_DIM),
    }

from torch.utils.data import Dataset,DataLoader

parser=argparse.ArgumentParser(description='MegaDepth dataset evaluation script')
parser.add_argument('--name',type=str,default='GeoFeat',help='experiment name')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')

args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

DATASET_ROOT = os.path.join(os.path.dirname(__file__),'../datasets/megadepth_test_1500')
DATASET_JSON = os.path.join(os.path.dirname(__file__),'../datasets/megadepth_1500.json')

class MegaDepth1500(Dataset):
    """
        Streamlined MegaDepth-1500 dataloader. The camera poses & metadata are stored in a formatted json for facilitating 
        the download of the dataset and to keep the setup as simple as possible.
    """
    def __init__(self, json_file, root_dir):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])

        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']

        # Here we resize the images to max_dim = 1200, as described in the paper, and adjust the image such that it is divisible by 32
        # following the protocol of the LoFTR's Dataloader (intrinsics are corrected accordingly). 
        # For adapting this with different resolution, you would need to re-scale intrinsics below.
        image0_path = os.path.join(self.root_dir, data['pair_names'][0])
        image1_path = os.path.join(self.root_dir, data['pair_names'][1])
        
        if not os.path.exists(image0_path):
             print(f"Warning: Image not found {image0_path}")
        if not os.path.exists(image1_path):
             print(f"Warning: Image not found {image1_path}")

        image0 = cv2.resize(cv2.imread(image0_path),(w1, h1))
        image1 = cv2.resize(cv2.imread(image1_path),(w2, h2))

        # GeoFeat wrapper expects numpy array or tensor. 
        # compute_pose_error in eval_utils expects tensor and converts to BGR numpy.
        # We follow the pattern from the original script.
        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data

if __name__ == "__main__":
    weights = os.path.join(os.path.dirname(__file__), '../weights/baseline_20260101_224524/baseline_step30000.pth')
    
    print(f"Loading weights from {weights}")

    # Load config from snapshot if available for consistency
    weights_dir = os.path.dirname(weights)
    snapshot_path = os.path.join(weights_dir, 'config_snapshot.json')
    if os.path.exists(snapshot_path):
        print(f"Loading config from snapshot: {snapshot_path}")
        with open(snapshot_path, 'r') as f:
             model_config = json.load(f)['model_config']
    else:
        print("Snapshot not found, using default config")
        model_config = get_model_config(get_cfg_defaults())

    geofeat = GeoFeat(model_config=model_config, weight_path=weights)
    
    dataset = MegaDepth1500(json_file = DATASET_JSON, root_dir = DATASET_ROOT)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics = {}
    R_errs = []
    t_errs = []
    inliers = []
    
    results=[]

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for d in tqdm.tqdm(loader, desc="processing"):
        error_infos = compute_pose_error(geofeat.match_featnet, d)
        results.append(error_infos)

    print(f'\n==={cur_time}==={args.name}===')
    d_err_auc,errors=compute_maa(results)
    for s_k,s_v in d_err_auc.items():
        print(f'{s_k}: {s_v*100:.3f}')
