"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"

	MegaDepth data handling was adapted from 
	LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import torch
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt
import pdb
import cv2

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
	""" 
	Warp kpts0 from I0 to I1 with depth, K and Rt
	Also check covisibility and depth consistency.
	Depth is consistent if relative error < 0.2 (hard-coded).
	将关键点 kpts0 从图像 I0 通过深度、内参和外参变换到 I1。
	并检查可见性和深度一致性。
	深度一致性判据为相对误差 < 0.2。
	
	Args:
		kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
		depth0 (torch.Tensor): [N, H, W],
		depth1 (torch.Tensor): [N, H, W],
		T_0to1 (torch.Tensor): [N, 3, 4],
		K0 (torch.Tensor): [N, 3, 3],
		K1 (torch.Tensor): [N, 3, 3],
	Returns:
		calculable_mask (torch.Tensor): [N, L]
		warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
	"""
	# 对关键点坐标取整、裁剪
	kpts0_long = kpts0.round().long().clip(0, 2000-1)

	# 边界像素深度设为0，避免索引越界
	depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
	depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

	# Sample depth, get calculable_mask on depth != 0
	kpts0_depth = torch.stack([depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0)  # (N, L)
	nonzero_mask = kpts0_depth > 0

	# Draw cross marks on the image for each keypoint
	# for b in range(len(kpts0)):
	#     fig, ax = plt.subplots(1,2)
	#     depth_np = depth0.numpy()[b]
	#     depth_np_plot = depth_np.copy()
	#     for x, y in kpts0_long[b, nonzero_mask[b], :].numpy():
	#         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
	#     ax[0].imshow(depth_np)
	#     ax[1].imshow(depth_np_plot)

	# Unproject # 反投影到相机坐标系
	kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
	kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

	# Rigid Transform   # 刚体变换 # 通过外参矩阵变换到目标相机坐标系
	w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
	w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

	# Project # 投影到像素平面
	w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
	w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth

	# Covisible Check
	# h, w = depth1.shape[1:3]
	# covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
	#     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
	# w_kpts0_long = w_kpts0.long()
	# w_kpts0_long[~covisible_mask, :] = 0

	# w_kpts0_depth = torch.stack(
	#     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
	# )  # (N, L)
	# consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2


	valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

	return valid_mask, w_kpts0


@torch.no_grad()
def spvs_coarse(data, scale = 8):
	"""
		Supervise corresp with dense depth & camera poses
		利用稠密深度和相机位姿监督生成粗匹配点对。
		主要流程：
			1. 构建网格关键点，缩放到原图分辨率。
			2. 双向变换关键点，计算重投影误差。
			3. 选取互为一致且误差小于阈值的点对。
			4. 去除重复点对，保证网络收敛。
		参数：
			data (dict): 包含 image0, image1, depth0, depth1, T_0to1, T_1to0, K0, K1, scale0, scale1 等。
			scale (int): 网格缩放因子，默认 8。
		返回：
			corrs (list): 每个 batch 的有效点对列表，每个点对为 [x0, y0, x1, y1]。
	"""

	# 1. misc 杂项
	# 1. 获取图像和深度等信息
	device = data['image0'].device
	N, _, H0, W0 = data['image0'].shape
	_, _, H1, W1 = data['image1'].shape
	#scale = 8
	scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
	scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
	h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

	# 2. warp grids 变换网格点
	# 2. 构建 meshgrid 关键点并缩放到原图分辨率
	# create kpts in meshgrid and resize them to image resolution
	grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)    # [N, hw, 2]
	grid_pt1_i = scale1 * grid_pt1_c

	# 3. 双向变换关键点，计算重投影误差
	# warp kpts bi-directionally and check reproj error
	nonzero_m1, w_pt1_i  =  warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0']) 
	nonzero_m2, w_pt1_og =  warp_kpts(   w_pt1_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1']) 


	dist = torch.linalg.norm( grid_pt1_i - w_pt1_og, dim=-1)
	mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

	# 4. 选取互为一致且误差小于阈值的点对
	#_, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
	batched_corrs = [ torch.cat([w_pt1_i[i, mask_mutual[i]] / data['scale0'][i],
					   grid_pt1_i[i, mask_mutual[i]] / data['scale1'][i]],dim=-1) for i in range(len(mask_mutual))]

	# 5. 去除重复点对，保证网络收敛
	#Remove repeated correspondences - this is important for network convergence
	corrs = []
	for pts in batched_corrs:
		lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1
		lut_mat21 = torch.clone(lut_mat12)
		src_pts = pts[:, :2] / scale
		tgt_pts = pts[:, 2:] / scale
		try:
			lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
			mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
			points = lut_mat12[mask_valid12]

			#Target-src check
			src_pts, tgt_pts = points[:, :2], points[:, 2:]
			lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
			mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
			points = lut_mat21[mask_valid21]

			corrs.append(points)
		except:
			pdb.set_trace()
			print('..')

	#Plot for debug purposes    
	# for i in range(len(corrs)):
	#     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)

	return corrs

@torch.no_grad()
def get_correspondences(pts2, data, idx):
	"""
	给定目标图像关键点，利用深度和位姿变换，获取源图像对应点。
	参数：
		pts2 (Tensor): [L, 2]，目标图像关键点
		data (dict): 包含 image0, image1, depth0, depth1, T_0to1, T_1to0, K0, K1, scale0, scale1 等
		idx (int): batch 索引
	返回：
		corrs (Tensor): [L, 4]，每行为 [x0, y0, x1, y1] 匹配点对
	"""
	device = data['image0'].device
	N, _, H0, W0 = data['image0'].shape
	_, _, H1, W1 = data['image1'].shape

	pts2 = pts2[None, ...]

	scale0 = data['scale0'][idx, None][None, ...] if 'scale0' in data else 1
	scale1 = data['scale1'][idx, None][None, ...] if 'scale1' in data else 1

	# 目标关键点缩放到原图分辨率
	pts2 = scale1 * pts2 * 8

	# 通过 warp_kpts ，获取源图像对应点
	# warp kpts bi-directionally and check reproj error
	nonzero_m1, pts1  = warp_kpts(pts2, data['depth1'][idx][None, ...], 
							   		    data['depth0'][idx][None, ...], 
										data['T_1to0'][idx][None, ...], 
										data['K1'][idx][None, ...], 
										data['K0'][idx][None, ...]) 

	# 拼接成点对 [x0, y0, x1, y1]
	corrs = torch.cat([pts1[0, :] / data['scale0'][idx], pts2[0, :] / data['scale1'][idx]],dim=-1)

	#plot_corrs(data['image0'][idx], data['image1'][idx], corrs[:, :2], corrs[:, 2:])

	return corrs

