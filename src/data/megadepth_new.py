"""Unified MegaDepth cleaned dataset loader.

与旧版 `megadepth.MegaDepthDataset` 区别：
1. 聚合多个清理后的 npz（来自 cleaned_scene_info 目录）。
2. 支持最小 overlap / 视角过滤（基于 pair_infos 第二列 overlap_score）。
3. 默认根索引目录：megadepth_indices_new/cleaned_scene_info_0.1_0.7
4. 延迟加载：只在初始化时读取 npz 的 meta 头与 pair_infos，不提前展开图像。
5. 可选加载 depth。

返回结构与原版兼容字段：image0/image1 (灰度, float, shape (1,H,W)), depth0/1(可选), intrinsics K0/K1, T_0to1/T_1to0, scale0/1, pair_names。
"""
# from __future__ import annotations	# 提升类型检查/性能
from pathlib import Path  # 面向对象的文件系统路径操作
import numpy as np
import torch
import torch.nn.functional as F	
from torch.utils.data import Dataset
import glob
import numpy.random as npr

import os
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from .dataset_utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net

import pdb, tqdm

class MegaDepthCleanedDataset(Dataset):
	def __init__(
		self,
		root_dir: str | Path,
		npz_path,
		mode: str = 'train',
		min_overlap_score: float = 0.3,
		max_overlap_score: float = 1.0,
		load_depth: bool = True,
		img_resize = (800, 608),
		df: int = 32,
		img_padding: bool = False,
		depth_padding: bool = True,
		augment_fn = None,
		**kwargs
	):
		super().__init__()
		self.root_dir = root_dir
		self.scene_id = Path(npz_path).stem
		self.load_depth = load_depth

		self.mode = mode

		# prepare scene_info and pair_info
		# parameters for image resizing, padding and depthmap padding
		if mode == 'train':
			assert img_resize is not None  # and img_padding and depth_padding
		if mode == 'test' and min_overlap_score != 0:
			min_overlap_score = 0
		self.scene_info = np.load(npz_path, allow_pickle=True)
		self.pair_infos = self.scene_info['pair_infos'].copy()
		del self.scene_info['pair_infos']
		self.pair_infos = [
			pair_info for pair_info in self.pair_infos
			if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score
		]


		self.img_size = img_resize
		self.df = df
		self.img_padding = img_padding
		self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

		# for training LoFTR
		self.augment_fn = augment_fn if mode == 'train' else None
		self.coarse_scale = kwargs.get('coarse_scale', 0.125)
		for idx in range(len(self.scene_info['image_paths'])):
			self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])
		for idx in range(len(self.scene_info['depth_paths'])):
			self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])

	def __len__(self):
		return len(self.pair_infos)

	def __getitem__(self, idx):
		(idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]

		# read grayscale image and mask. (1, h, w) and (h, w)
		# 读取两个灰度图像
		img_name0 = Path(self.root_dir) / self.scene_info['image_paths'][idx0]
		img_name1 = Path(self.root_dir) / self.scene_info['image_paths'][idx1]
		image0, image0_t, mask0, scale0 = read_megadepth_gray(img_name0, self.img_size, self.df, self.img_padding, None)
		image1, image1_t, mask1, scale1 = read_megadepth_gray(img_name1, self.img_size, self.df, self.img_padding, None)

		if self.load_depth:
			# read depth.shape:(h, w)
			if self.mode in ['train', 'val']:
				depth0 = read_megadepth_depth(Path(self.root_dir) / self.scene_info['depth_paths'][idx0], pad_to=self.depth_max_size)
				depth1 = read_megadepth_depth(Path(self.root_dir) / self.scene_info['depth_paths'][idx1], pad_to=self.depth_max_size)
			else:
				depth0 = depth1 = torch.tensor([])
			# 读取相机内参
			K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
			K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
			# 读取相机外参
			T_0 = self.scene_info['poses'][idx0]
			T_1 = self.scene_info['poses'][idx1]
			# 图像之间的相对变换矩阵
			T_0to1 = torch.tensor(np.matmul(T_1, np.linalg.inv(T_0)), dtype=torch.float)[:4, :4]  # (4, 4)
			T_1to0 = T_0to1.inverse()

			# 返回数据包
			data = {
				'image0': image0_t,  # (1, h, w)
				'image0_np': image0,
				'depth0': depth0,  # (h, w)
				'image1': image1_t,
				'image1_np': image1,
				'depth1': depth1,
				'T_0to1': T_0to1,  # (4, 4)
				'T_1to0': T_1to0,
				'K0': K_0,  # (3, 3)
				'K1': K_1,  # (3, 3)
				'scale0': scale0,  # [scale_w, scale_h]
				'scale1': scale1,
				'dataset_name': 'MegaDepth',
				'scene_id': self.scene_id,
				'pair_id': idx,
				'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
			}

			# for LoFTR training
			if mask0 is not None and mask1 is not None:  # img_padding is True
				ts_mask_0, ts_mask_1 = mask0, mask1
				if self.coarse_scale:
					masks = torch.stack([mask0, mask1], dim=0).float().unsqueeze(0)
					ts_mask_0, ts_mask_1 = F.interpolate(
						masks,
						scale_factor=self.coarse_scale,
						mode='nearest',
						recompute_scale_factor=False
					)[0].bool()
				data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

		else:
			# read intrinsics of original size
			K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
			K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

			# read and compute relative poses
			T0 = self.scene_info['poses'][idx0]
			T1 = self.scene_info['poses'][idx1]
			T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
			T_1to0 = T_0to1.inverse()

			data = {
				'image0': image0,  # (1, h, w)
				'image1': image1,
				'T_0to1': T_0to1,  # (4, 4)
				'T_1to0': T_1to0,
				'K0': K_0,  # (3, 3)
				'K1': K_1,
				'scale0': scale0,  # [scale_w, scale_h]
				'scale1': scale1,
				'dataset_name': 'MegaDepth',
				'scene_id': self.scene_id,
				'pair_id': idx,
				'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
			}

		return data


