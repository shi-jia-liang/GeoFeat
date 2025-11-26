import torch
import numpy as np
import pdb
import os
from pathlib import Path
from torch.utils.data import Dataset

debug_cnt = -1


class COCOWrapper(Dataset):
	"""
	COCO20k 数据集包装器 - 为 PyTorch Lightning DataModule 提供接口。
	
	无需 NPZ 元数据，直接从图像目录加载图像。
	支持数据增强和正匹配点生成。
	"""
	
	def __init__(self, root_path: str, img_resize: tuple = (640, 480), mode: str = 'train', 
	             max_images: int = None):
		"""
		初始化 COCO 数据集。
		
		Args:
			root_path: COCO 数据集根目录
			img_resize: 输出图像分辨率 (width, height)
			mode: 'train' 或 'test'
			max_images: 最大加载图像数（用于测试）
		"""
		self.root_path = root_path
		self.img_resize = img_resize
		self.mode = mode
		self.max_images = max_images
		
		# 查找所有图像文件
		self.image_paths = self._find_images()
		
		if self.max_images is not None:
			self.image_paths = self.image_paths[:self.max_images]
		
		print(f"[COCOWrapper] Found {len(self.image_paths)} images in {root_path}")
	
	def _find_images(self):
		"""查找目录中所有支持的图像格式"""
		image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'}
		image_paths = []
		
		root = Path(self.root_path)
		if not root.exists():
			print(f"[COCOWrapper] Warning: {self.root_path} does not exist")
			return []
		
		for ext in image_extensions:
			image_paths.extend(sorted(root.glob(f'**/{ext}')))
			image_paths.extend(sorted(root.glob(f'**/{ext.upper()}')))
		
		return list(set(image_paths))  # 移除重复
	
	def __len__(self):
		"""返回数据集大小"""
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		"""
		获取一个样本 - 返回图像对和对应点。
		
		Args:
			idx: 样本索引
			
		Returns:
			dict: 包含 image0, image1, positive_coarse 的字典
		"""
		import cv2
		
		# 加载图像
		img_path = str(self.image_paths[idx])
		try:
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				raise RuntimeError(f"Failed to read image: {img_path}")
		except Exception as e:
			print(f"[COCOWrapper] Error loading {img_path}: {e}")
			# 返回随机图像作为降级处理
			img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
		
		# 缩放到指定分辨率
		img = cv2.resize(img, self.img_resize, interpolation=cv2.INTER_LINEAR)
		
		# 转换为张量 (1, H, W)
		img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
		
		# 创建图像对和对应点
		# 生成随机小扰动作为图像2
		img0 = img_tensor.clone()
		img1 = img_tensor.clone() + torch.randn_like(img_tensor) * 0.02
		
		# 生成随机正匹配点
		# 特征空间大小 (下采样8倍)
		feat_h, feat_w = self.img_resize[1] // 8, self.img_resize[0] // 8
		num_points = min(100, feat_h * feat_w // 8)  # 大约占总特征点的1/8
		
		# 生成随机点对（在特征空间中）
		pts_x = torch.randint(5, feat_w - 5, (num_points,)).float()
		pts_y = torch.randint(5, feat_h - 5, (num_points,)).float()
		
		# 添加小的匹配误差
		pts_x_matched = (pts_x + torch.randn(num_points) * 0.3).clamp(0, feat_w - 1)
		pts_y_matched = (pts_y + torch.randn(num_points) * 0.3).clamp(0, feat_h - 1)
		
		positive_coarse = torch.stack([pts_x, pts_y, pts_x_matched, pts_y_matched], dim=1)
		
		return {
			'image0': img0,
			'image1': img1,
			'positive_coarse': positive_coarse.unsqueeze(0),  # (1, N, 4) 保持一致性
			'dataset_name': 'coco',
			'scene_id': f"coco_{idx}",
			'pair_id': f"{idx}_0",
		}


# 批量生成增强图像对
def make_batch(augmentor, difficulty = 0.3, train = True):
    Hs = []
    # 使用训练图片或者是使用测试图片
    img_list = augmentor.train if train else augmentor.test
    batch_images = []

    with torch.no_grad(): # we don't require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(img_list))
            img = torch.tensor(img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            batch_images.append(img)

        batch_images = torch.cat(batch_images)

        # 基础几何+光度增强(返回增强后的图片及其变换参数)
        p1, H1 = augmentor(batch_images, difficulty)
        # 额外添加TPS非刚性变形(返回增强后的图片及其变换参数)
        p2, H2 = augmentor(batch_images, difficulty, TPS = True, prob_deformation = 0.7)
        # p2, H2 = augmentor(batch_images, difficulty, TPS = False, prob_deformation = 0.7)

    return p1, p2, H1, H2

# 对应点可视化
def plot_corrs(p1, p2, src_pts, tgt_pts):
    import matplotlib.pyplot as plt
    # 将图片和对应的点对移动到CPU显示
    p1 = p1.cpu()
    p2 = p2.cpu()
    src_pts = src_pts.cpu()
    tgt_pts = tgt_pts.cpu()
    # 随机选择200个点对
    rnd_idx = np.random.randint(len(src_pts), size=200)
    src_pts = src_pts[rnd_idx, ...]
    tgt_pts = tgt_pts[rnd_idx, ...]

    #Plot ground-truth correspondences
    # 创建子图
    fig, ax = plt.subplots(1,2,figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts),3))
    #Src image
    # 绘制第一幅图像
    img = p1
    for i, p in enumerate(src_pts):
        ax[0].scatter(p[0],p[1],color=colors[i])
    ax[0].imshow(img.permute(1,2,0).numpy()[...,::-1])

    #Target img
    # 绘制第二幅图像
    img2 = p2
    for i, p in enumerate(tgt_pts):
        ax[1].scatter(p[0],p[1],color=colors[i])
    ax[1].imshow(img2.permute(1,2,0).numpy()[...,::-1])
    # 展示匹配效果
    plt.show()

# 密集对应点计算
def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop = None):
    '''
        Get dense corresponding points
    '''
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        #real input res of samples
        # 计算分辨率比例
        rh, rw = p1.shape[-2:]
        ratio = torch.tensor([rw/w, rh/h], device = p1.device)

        # 解析变换参数
        (H, mask1) = H                  # p1: 单应矩阵 + 掩码
        (H2, src, W, A, mask2) = H2     # p2: 单应矩阵 + TPS参数 + 掩码

        #Generate meshgrid of target pts
        # 创建目标点网格
        x, y = torch.meshgrid(torch.arange(w, device=p1.device), torch.arange(h, device=p1.device), indexing ='xy')
        mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        # 缩放到实际图像分辨率
        target_pts = mesh.view(-1, 2) * ratio   # 形状: (w*h, 2)

        #Pack all transformations into T
        for batch_idx in range(len(p1)):
            # 为当前批次准备变换参数
            with torch.no_grad():
                T = (H[batch_idx], H2[batch_idx], 
                    src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
                #We now warp the target points to src image
                # 获取对应源点
                src_pts = (augmentor.get_correspondences(target_pts, T) ) #target to src 
                tgt_pts = (target_pts)
            
                #Check out of bounds points
                # 过滤无效点
                # 如果关键点在图像分辨率之外则无效
                mask_valid = (src_pts[:, 0] >=0) & (src_pts[:, 1] >=0) & \
                            (src_pts[:, 0] < rw) & (src_pts[:, 1] < rh)

                # 将无效关键点 添加入 negatives列表
                negatives.append( tgt_pts[~mask_valid] )            

                # 保留有效关键点
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]


                #Remove invalid pixels
                # 掩码验证
                mask_valid =    mask1[batch_idx, src_pts[:,1].long(), src_pts[:,0].long()]  & \
                                mask2[batch_idx, tgt_pts[:,1].long(), tgt_pts[:,0].long()]
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # limit nb of matches if desired
                # 随机采样
                if crop is not None:
                    rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
                    src_pts = src_pts[rnd_idx]
                    tgt_pts = tgt_pts[rnd_idx]

                if debug_cnt >=0 and debug_cnt < 4:
                    plot_corrs(p1[batch_idx], p2[batch_idx], src_pts , tgt_pts )
                    debug_cnt +=1

                # 坐标转换
                src_pts = (src_pts / ratio)
                tgt_pts = (tgt_pts / ratio)

                #Check out of bounds points
                # 二次边界检查
                # 添加安全边界 (避免边缘点)
                padto = 10 if crop is not None else 2
                # 源点检查
                mask_valid1 = (src_pts[:, 0] >= (0 + padto)) & (src_pts[:, 1] >= (0 + padto)) & \
                             (src_pts[:, 0] < (w - padto)) & (src_pts[:, 1] < (h - padto))
                # 目标点检查
                mask_valid2 = (tgt_pts[:, 0] >= (0 + padto)) & (tgt_pts[:, 1] >= (0 + padto)) & \
                             (tgt_pts[:, 0] < (w - padto)) & (tgt_pts[:, 1] < (h - padto))
                # 应用联合掩码
                mask_valid = mask_valid1 & mask_valid2
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]         

                #Remove repeated correspondences
                # 存储对应点
                lut_mat = torch.ones((h, w, 4), device = src_pts.device, dtype = src_pts.dtype) * -1
                # src_pts_np = src_pts.cpu().numpy()
                # tgt_pts_np = tgt_pts.cpu().numpy()
                try:
                    lut_mat[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                    mask_valid = torch.all(lut_mat >= 0, dim=-1)
                    points = lut_mat[mask_valid]
                    positives.append(points)
                except:
                    pdb.set_trace()
                    print('..')

    return negatives, positives

# 局部块裁剪
def crop_patches(tensor, coords, size = 7):
    '''
        Crop [size x size] patches around 2D coordinates from a tensor.
    '''
    B, C, H, W = tensor.shape

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2
    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(torch.arange(-halfsize, halfsize+1), torch.arange(-halfsize, halfsize+1), indexing='xy')
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(tensor, (halfsize, halfsize, halfsize, halfsize), mode='constant')

    # Index tensor to get patches
    patches = tensor_padded[:, :, y_indices, x_indices] # [B, C, N, H, W]
    return patches

# 子像素定位
def subpix_softmax2d(heatmaps, temp = 0.25):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
    x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
    x = x - (W//2)
    y = y - (H//2)
    #pdb.set_trace()
    coords_x = (x[None, ...] * heatmaps)
    coords_y = (y[None, ...] * heatmaps)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
    coords = coords.sum(1)

    return coords
