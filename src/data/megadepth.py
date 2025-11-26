import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from .dataset_utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net


class MegaDepthDataset(Dataset):
	def __init__(self,
				 root_dir,
				 npz_path,
				 mode = 'train',
				 min_overlap_score = 0.4,
				 max_overlap_score = 1.0,
				 load_depth = True,
				 img_resize = None,
				 df = None,
				 img_padding = False,
				 depth_padding = False,
				 augment_fn = None,
				 fp16 = False,
				 **kwargs):
		"""
		Manage one scene(npz_path) of MegaDepth dataset.
		
		Args:
			root_dir (str): megadepth root directory that has `phoenix`.
			npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
			mode (str): options are ['train', 'val', 'test']
			min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
			img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
										This is useful during training with batches and testing with memory intensive algorithms.
			df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
			img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
			depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
			augment_fn (callable, optional): augments images with pre-defined visual effects.
		"""
		super().__init__()
		self.root_dir = root_dir
		self.scene_id = npz_path.split('.')[0]
		self.mode = mode
		self.load_depth = load_depth

		# prepare scene_info and pair_info
		if mode == 'test' and min_overlap_score != 0:
			logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
			min_overlap_score = 0

		# 将 pair_infos 从 scene_info 中提取出来
		self.scene_info = np.load(npz_path, allow_pickle=True)
		self.pair_infos = self.scene_info['pair_infos'].copy()
		del self.scene_info['pair_infos']
		self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] <= max_overlap_score]    # 过滤过小或者过大的 overlap_score

		# parameters for image resizing, padding and depthmap padding
		if mode == 'train':
			assert img_resize is not None and img_padding and depth_padding
		self.img_resize = img_resize
		self.img_padding = img_padding
		self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

		# for training LoFTR
		self.augment_fn = augment_fn if mode == 'train' else None
		self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)  # 存在 coarse_scale 参数时使用，否则默认0.125

		# fix paths from d2-net
		for idx in range(len(self.scene_info['image_paths'])):
			self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])

		for idx in range(len(self.scene_info['depth_paths'])):
			self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])
		
		self.df = df
		self.fp16 = fp16
		
	def __len__(self):
		return len(self.pair_infos)

	def __getitem__(self, idx):
		# pair_info:([idx0, idx1], overlap_score, central_matches)
		# (idx0, idx1) are indices of images in scene_info
		# overlap_score is importance score of the image pair
		# central_matches is Nx2 np array of matched keypoint indices in the two images # 仅供参考，训练时不使用，只有一个关键点
		(idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

		# read grayscale image and mask. (1, h, w) and (h, w)
		img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
		img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
	
		# TODO: Support augmentation & handle seeds for each worker correctly.
		image0, mask0, scale0 = read_megadepth_gray(img_name0, self.img_resize, self.df, self.img_padding, None)
			# np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
		image1, mask1, scale1 = read_megadepth_gray(img_name1, self.img_resize, self.df, self.img_padding, None)
			# np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

		# read depth. shape: (h, w)
		# 仅在需要时按需加载 depth，以节省 IO/内存
		if self.load_depth and self.mode in ['train', 'val']:
			depth0 = read_megadepth_depth(
				osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
			depth1 = read_megadepth_depth(
				osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
		else:
			# 如果不加载 depth，则返回空 tensor，调用方可以按需加载
			depth0 = depth1 = torch.tensor([])

		# read intrinsics of original size
		# 读取内参矩阵(3, 3)
		K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
		K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

		# read and compute relative poses
		# 读取并计算外参矩阵(4, 4)
		T0 = self.scene_info['poses'][idx0]
		T1 = self.scene_info['poses'][idx1]
		T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
		T_1to0 = T_0to1.inverse()

		# 半精度处理（可选）
		if self.fp16:
			image0, image1, depth0, depth1, scale0, scale1 = map(lambda x: x.half(), [image0, image1, depth0, depth1, scale0, scale1])

		# 数据包装
		data = {
			'dataset_name': 'MegaDepth',
			'image0': image0,  # (1, h, w)
			'depth0': depth0,  # (h, w)
			'image1': image1,
			'depth1': depth1,
			'T_0to1': T_0to1,  # (4, 4)
			'T_1to0': T_1to0,
			'K0': K_0,  # (3, 3)
			'K1': K_1,
			'scale0': scale0,  # [scale_w, scale_h]
			'scale1': scale1,
			'scene_id': self.scene_id,  # 场景ID
			'pair_id': idx,             # 场景对应的图像对ID
			'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
		}

		# for LoFTR training
		# mask 下采样处理
		if mask0 is not None and mask1 is not None:  # img_padding is True and both masks exist
			if self.coarse_scale:
				stacked = torch.stack([mask0, mask1], dim=0).unsqueeze(0).float()
				interp = F.interpolate(stacked,
									   scale_factor=self.coarse_scale,
									   mode='nearest',
									   recompute_scale_factor=False)[0].bool()
				ts_mask_0, ts_mask_1 = interp[0], interp[1]
			else:
				ts_mask_0, ts_mask_1 = mask0, mask1
			data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
		return data
