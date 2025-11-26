import os
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image


def fix_path_from_d2net(path):
    if not path:
        return None

    path = path.replace('Undistorted_SfM/', '')
    path = path.replace('images', 'dense0/imgs')
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')

    return path



def analyze_megadepth(MegaDepth_v1_path, npz_root, save_dir=None):
    """
    分析MegaDepth数据集，统计配准图像对和标签数量，并可选择保存配对图像
    
    参数:
        MegaDepth_v1_path (str): MegaDepth数据集根目录
        npz_root (str): 包含.npz文件的目录路径
        save_dir (str): 保存配对图像的目录路径，如果为None则不保存
    """
    # 获取所有场景的npz文件
    npz_paths = glob.glob(os.path.join(npz_root, '*.npz'))
    
    # 初始化统计变量
    total_scenes = 0
    total_pairs = 0
    total_images = 0
    overlap_scores = []
    
    # 创建保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Analyzing {len(npz_paths)} scenes...")
    
    # 遍历每个场景
    for npz_path in tqdm(npz_paths):
        # 加载场景信息
        scene_info = np.load(npz_path, allow_pickle=True)
        
        # 统计信息
        total_scenes += 1
        total_pairs += len(scene_info['pair_infos'])
        total_images += len(scene_info['image_paths'])
        
        # 收集重叠分数
        for pair_idx, pair_info in enumerate(scene_info['pair_infos']):
            # 检查pair_info结构是否正确
            if len(pair_info) >= 2 and isinstance(pair_info[1], (int, float)):
                overlap_scores.append(pair_info[1])
            else:
                print(f"Warning: Invalid pair_info structure in {npz_path}, pair {pair_idx}")
            
            # # # 保存配对图像
            # if save_dir is not None:
            #     img1_path = os.path.join(MegaDepth_v1_path, scene_info['image_paths'][pair_info[0][0]])
            #     img2_path = os.path.join(MegaDepth_v1_path, scene_info['image_paths'][pair_info[0][1]])
            #     img1_path = fix_path_from_d2net(img1_path)
            #     img2_path = fix_path_from_d2net(img2_path)
            #     # 读取并保存图像
            #     try:
            #         img1 = Image.open(img1_path)
            #         img2 = Image.open(img2_path)
                    
            #         # 创建保存路径
            #         scene_id = os.path.splitext(os.path.basename(npz_path))[0]
            #         pair_id = f"{scene_id}_pair_{pair_idx:04d}"
            #         pair_dir = os.path.join(save_dir, pair_id)
            #         os.makedirs(pair_dir, exist_ok=True)
                    
            #         # 保存图像
            #         img1.save(os.path.join(pair_dir, "img1.jpg"))
            #         img2.save(os.path.join(pair_dir, "img2.jpg"))
                    
            #         # 保存配对信息
            #         with open(os.path.join(pair_dir, "info.txt"), 'w') as f:
            #             f.write(f"Image1: {img1_path}\n")
            #             f.write(f"Image2: {img2_path}\n")
            #             f.write(f"Overlap score: {pair_info[1] if len(pair_info) >= 2 else 'N/A':.3f}\n")
            #     except Exception as e:
            #         print(f"Error processing pair {total_pairs}: {e}")
    
    # 计算统计信息
    avg_pairs_per_scene = total_pairs / total_scenes
    avg_images_per_scene = total_images / total_scenes
    avg_overlap = np.mean(overlap_scores) if overlap_scores else 0  # 处理空列表情况
    
    # 打印结果
    print("\n=== MegaDepth Dataset Analysis ===")
    print(f"Total scenes: {total_scenes}")
    print(f"Total image pairs: {total_pairs}")
    print(f"Total unique images: {total_images}")
    print(f"Average pairs per scene: {avg_pairs_per_scene:.1f}")
    print(f"Average images per scene: {avg_images_per_scene:.1f}")
    print(f"Average overlap score: {avg_overlap:.3f}")

if __name__ == '__main__':
    # 设置MegaDepth数据集路径
    megadepth_root = 'D:/DataSets/MegaDepth/MegaDepth_v1'  # 替换为实际路径
    npz_root = 'D:/DataSets/MegaDepth/megadepth_indices/scene_info_0.1_0.7'  # 包含.npz文件的目录
    save_dir = './megadepth_pairs'  # 保存配对图像的目录
    
    # 运行分析
    analyze_megadepth(megadepth_root, npz_root, save_dir)
