import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from src.model.GeoFeatModel import GeoFeatModel

def load_image(path, resize=None):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    if resize:
        img = cv2.resize(img, resize)
    return img

def preprocess_image(img, device):
    # Convert to grayscale for model input if needed, or keep RGB depending on model
    # GeoFeatModel forward1 converts to grayscale internally: x = x.mean(dim=1, keepdim = True)
    # But it expects (B, C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor.to(device)

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def extract_features(model, img_tensor, top_k=1000, detection_threshold=0.01):
    with torch.no_grad():
        descs_refine, des_map, keypoint_map, geo_features = model(img_tensor)
        
        # Process keypoints
        score_map = torch.sigmoid(keypoint_map)
        score_map = simple_nms(score_map, nms_radius=4)
        
        B, C, H, W = score_map.shape
        score_map = score_map.reshape(B, -1)
        scores, indices = torch.topk(score_map, k=top_k, dim=1)
        
        y = torch.div(indices, W, rounding_mode='floor')
        x = indices % W
        kpts = torch.stack([x, y], dim=-1).float() # (B, K, 2)
        
        # Sample descriptors
        # descs_refine is (B, C, H/8, W/8) -> need to interpolate or sample?
        # Wait, GeoFeat output shape depends on architecture.
        # Let's assume descs_refine is dense map.
        
        # Grid sample descriptors at keypoint locations
        # Normalize coordinates to [-1, 1]
        kpts_norm = kpts.clone()
        kpts_norm[..., 0] = 2 * kpts_norm[..., 0] / (W - 1) - 1
        kpts_norm[..., 1] = 2 * kpts_norm[..., 1] / (H - 1) - 1
        kpts_norm = kpts_norm.unsqueeze(2) # (B, K, 1, 2)
        
        desc_samples = torch.nn.functional.grid_sample(
            descs_refine, kpts_norm, mode='bilinear', align_corners=True
        ) # (B, C, K, 1)
        
        desc_samples = desc_samples.squeeze(3).transpose(1, 2) # (B, K, C)
        
        # Normalize descriptors
        desc_samples = torch.nn.functional.normalize(desc_samples, p=2, dim=2)
        
        return kpts[0].cpu().numpy(), desc_samples[0].cpu().numpy(), scores[0].cpu().numpy()

def match_features(desc1, desc2):
    # Mutual Nearest Neighbor
    d1 = torch.from_numpy(desc1).cuda()
    d2 = torch.from_numpy(desc2).cuda()
    
    sim = torch.matmul(d1, d2.t())
    
    max0 = torch.max(sim, dim=1)[1]
    max1 = torch.max(sim, dim=0)[1]
    
    indices0 = torch.arange(sim.shape[0]).cuda()
    mutual = max1[max0] == indices0
    
    matches = torch.stack([indices0[mutual], max0[mutual]], dim=1)
    return matches.cpu().numpy()

def draw_matches(img1, kpts1, img2, kpts2, matches, save_path):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    
    for i in range(len(matches)):
        idx1, idx2 = matches[i]
        pt1 = (int(kpts1[idx1][0]), int(kpts1[idx1][1]))
        pt2 = (int(kpts2[idx2][0]) + w1, int(kpts2[idx2][1]))
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 2, color, -1)
        cv2.circle(canvas, pt2, 2, color, -1)
        
    cv2.imwrite(save_path, canvas)
    print(f"Saved matches to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--config', type=str, default='src/config/model_config.json')
    parser.add_argument('--output', type=str, default='matches.png')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    print("Loading model...")
    model = GeoFeatModel(model_config=args.config).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    # Handle potential key mismatch if saved with 'module.' prefix or different structure
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Load Images
    img1 = load_image(args.img1, resize=(640, 480))
    img2 = load_image(args.img2, resize=(640, 480))
    
    t1 = preprocess_image(img1, device)
    t2 = preprocess_image(img2, device)
    
    # Extract
    print("Extracting features...")
    kpts1, desc1, scores1 = extract_features(model, t1)
    kpts2, desc2, scores2 = extract_features(model, t2)
    
    # Match
    print("Matching...")
    matches = match_features(desc1, desc2)
    print(f"Found {len(matches)} matches")
    
    # Draw
    draw_matches(img1, kpts1, img2, kpts2, matches, args.output)

if __name__ == '__main__':
    main()
