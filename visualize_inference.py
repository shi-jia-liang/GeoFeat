import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

# Correct path
sys.path.append(os.getcwd())

from src.utils.geofeat_wrapper import GeoFeat
from src.model.GeoFeatModel import GeoFeatModel

def visualize_inference():
    weights_path = r'd:\Projects\GeoFeat\weights_0227\baseline_20260218_180605\baseline_best.pth'
    config_path = r'd:\Projects\GeoFeat\weights_0227\baseline_20260218_180605\config_snapshot.json'
    
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        if 'model_config' in cfg:
            model_config = cfg['model_config']
        else:
            model_config = cfg

    print(f"Loading weights from {weights_path}")
    geofeat = GeoFeat(model_config, weights_path)
    
    # Load first image from MegaDepth or random
    # Try finding an image in data/
    image_path = None
    if os.path.exists('data/megadepth_test_1500/0015/images/15100000_1523554152.jpg'): # Example checking
          image_path = 'data/megadepth_test_1500/0015/images/15100000_1523554152.jpg'
    
    if image_path is None:
        # Create synthetic image with patterns (checkerboard)
        print("Creating synthetic checkerboard image")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # draw checkerboard
        step = 64
        for y in range(0, 480, step):
            for x in range(0, 640, step):
                if ((x//step) + (y//step)) % 2 == 0:
                    img[y:y+step, x:x+step] = 255
        
        # Add some noise to make it interesting for descriptors
        noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
    else:
        print(f"Loading image: {image_path}")
        img = cv2.imread(image_path)

    # Run inference
    res = geofeat.extract(img)
    
    kpts = res['keypoints'].cpu().numpy()
    scores = res['scores'].cpu().numpy()
    descs = res['descriptors'].cpu().numpy()
    
    print(f"Keypoints detected: {len(kpts)}")
    if len(kpts) > 0:
        print(f"First 5 keypoints:\n{kpts[:5]}")
        print(f"Score stats: Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}")
        print(f"Descriptor shape: {descs.shape}")
        print(f"Descriptor stats: Min={descs.min():.4f}, Max={descs.max():.4f}, Mean={descs.mean():.4f}, Norm={np.linalg.norm(descs[0])}")

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(kpts) > 0:
        plt.scatter(kpts[:, 0], kpts[:, 1], c=scores, cmap='jet', s=5, alpha=0.5)
        plt.colorbar(label='Score')
    plt.title(f"Checkboard Keypoints (Count: {len(kpts)})")
    plt.savefig("visualize_checkboard.png")
    print("Saved visualization to visualize_checkboard.png")

if __name__ == "__main__":
    visualize_inference()
