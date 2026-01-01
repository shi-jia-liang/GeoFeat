import cv2
import os
from tqdm import tqdm
import torch
import numpy as np
import sys
import poselib

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import argparse
import datetime
import json

from src.config.config import get_cfg_defaults

parser=argparse.ArgumentParser(description='HPatch dataset evaluation script')
parser.add_argument('--name',type=str,default='LiftFeat',help='experiment name')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

top_k = None
n_i = 52
n_v = 56

DATASET_ROOT = os.path.join(os.path.dirname(__file__),'../data/hpatches-sequences-release')

from evaluation.eval_utils import *
from src.utils.geofeat_wrapper import GeoFeat


poselib_config = {"ransac_th": 3.0, "options": {}}

class PoseLibHomographyEstimator:
    def __init__(self, conf):
        self.conf = conf

    def estimate(self, mkpts0,mkpts1):
        M, info = poselib.estimate_homography(
            mkpts0,
            mkpts1,
            {
                "max_reproj_error": self.conf["ransac_th"],
                **self.conf["options"],
            },
        )
        success = M is not None
        if not success:
            M = np.eye(3,dtype=np.float32)
            inl = np.zeros(mkpts0.shape[0],dtype=np.bool_)
        else:
            inl = info["inliers"]

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
        }

        return estimation
    
    
estimator=PoseLibHomographyEstimator(poselib_config)


def poselib_homography_estimate(mkpts0,mkpts1):
    data=estimator.estimate(mkpts0,mkpts1)
    return data


def generate_standard_image(img,target_size=(1920,1080)):
    sh,sw=img.shape[0],img.shape[1]
    rh,rw=float(target_size[1])/float(sh),float(target_size[0])/float(sw)
    ratio=min(rh,rw)
    nh,nw=int(ratio*sh),int(ratio*sw)
    ph,pw=target_size[1]-nh,target_size[0]-nw
    nimg=cv2.resize(img,(nw,nh))
    nimg=cv2.copyMakeBorder(nimg,0,ph,0,pw,cv2.BORDER_CONSTANT,value=(0,0,0))
    
    return nimg,ratio,ph,pw
    

def benchmark_features(match_fn):
    lim = [1, 9]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted(os.listdir(DATASET_ROOT))

    # actual illumination/viewpoint sequence counts for proper normalization
    total_i = sum(1 for name in seq_names if name[0] == "i")
    total_v = len(seq_names) - total_i

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    i_err_homo = {thr: 0 for thr in rng}
    v_err_homo = {thr: 0 for thr in rng}

    i_im_count = 0  # processed illumination images
    v_im_count = 0  # processed viewpoint images

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        # load reference image
        ref_img = cv2.imread(os.path.join(DATASET_ROOT, seq_name, "1.ppm"))
        ref_img_shape=ref_img.shape

        # load query images
        for im_idx in range(2, 7):
            # read ground-truth homography
            homography = np.loadtxt(os.path.join(DATASET_ROOT, seq_name, "H_1_" + str(im_idx)))
            query_img = cv2.imread(os.path.join(DATASET_ROOT, seq_name, f"{im_idx}.ppm"))
            
            try:
                mkpts_a,mkpts_b=match_fn(ref_img,query_img)
            except Exception as e:
                mkpts_a = np.zeros((0, 2), dtype=np.float32)
                mkpts_b = np.zeros((0, 2), dtype=np.float32)
                tqdm.write(f"[Seq {seq_name} img {im_idx}] match failed: {type(e).__name__}: {e}")

            pos_a = mkpts_a
            pos_a_h = np.concatenate([pos_a, np.ones([pos_a.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

            pos_b = mkpts_b

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(pos_a.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == "i":
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

            # estimate homography (guard against too-few matches or solver errors)
            gt_homo = homography
            pred_homo = None
            if mkpts_a.shape[0] >= 4:
                try:
                    pred_homo, _ = cv2.findHomography(mkpts_a,mkpts_b,cv2.USAC_MAGSAC)
                except cv2.error:
                    pred_homo = None

            if pred_homo is None:
                homo_dist = np.array([float("inf")])
            else:
                corners = np.array(
                    [
                        [0, 0],
                        [ref_img_shape[1] - 1, 0],
                        [0, ref_img_shape[0] - 1],
                        [ref_img_shape[1] - 1, ref_img_shape[0] - 1],
                    ]
                )
                real_warped_corners = homo_trans(corners, gt_homo)
                warped_corners = homo_trans(corners, pred_homo)
                homo_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

            for thr in rng:
                if seq_name[0] == "i":
                    i_err_homo[thr] += np.mean(homo_dist <= thr)
                else:
                    v_err_homo[thr] += np.mean(homo_dist <= thr)

        # update processed image counters (5 query images per sequence)
        if seq_name[0] == "i":
            i_im_count += 5
        else:
            v_im_count += 5

        # emit running MHA after each sequence finishes
        def fmt_running(thr):
            ill = i_err_homo[thr] / i_im_count * 100 if i_im_count > 0 else 0.0
            view = v_err_homo[thr] / v_im_count * 100 if v_im_count > 0 else 0.0
            return f"{ill:.2f}%-{view:.2f}%"

        tqdm.write(
            f"[Seq {seq_idx + 1}/{len(seq_names)}: {seq_name}] "
            f"MHA@3 {fmt_running(3)} | MHA@5 {fmt_running(5)} | MHA@7 {fmt_running(7)}"
        )

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return (
        i_err,
        v_err,
        i_err_homo,
        v_err_homo,
        [seq_type, n_feats, n_matches],
        total_i,
        total_v,
    )


if __name__ == "__main__":
    errors = {}

    # Load model config from the same yacs config used in training
    cfg = get_cfg_defaults()

    model_config = {
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

    weights = os.path.join(os.path.dirname(__file__), "../weights/GeoFeat_20260101_182828/GeoFeat_step2000.pth")
    model = GeoFeat(model_config=model_config, weight_path=weights)

    errors = benchmark_features(model.match_featnet)

    i_err, v_err, i_err_hom, v_err_hom, _, total_i, total_v = errors
    
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f'\n==={cur_time}==={args.name}===')
    print(f"MHA@3 MHA@5 MHA@7")
    for thr in [3, 5, 7]:
        ill_err_hom = i_err_hom[thr] / (total_i * 5)
        view_err_hom = v_err_hom[thr] / (total_v * 5)
        print(f"{ill_err_hom * 100:.2f}%-{view_err_hom * 100:.2f}%")
