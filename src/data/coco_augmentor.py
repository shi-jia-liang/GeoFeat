"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
	COCO_20k image augmentor
"""

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import cv2
import kornia
import kornia.augmentation as K
from kornia.geometry.transform import get_tps_transform as findTPS
from kornia.geometry.transform import warp_points_tps, warp_image_tps

import glob
import random
import tqdm

import numpy as np
import pdb
import time

random.seed(0)
torch.manual_seed(0)


# 非刚性图像形变技术, 常用于模拟和处理图像中物体的弹性变形
def generateRandomTPS(shape, grid=(8, 6), GLOBAL_MULTIPLIER=0.3, prob=0.5):
	# 网格点生成
	h, w = shape
	sh, sw = h / grid[0], w / grid[1]
	src = torch.dstack(torch.meshgrid(torch.arange(0, h + sh, sh), torch.arange(0, w + sw, sw), indexing='ij'))

	# 随机偏移生成
	offsets = torch.rand(grid[0] + 1, grid[1] + 1, 2) - 0.5  # 生成[-0.5, 0.5)范围内的随机偏移
	offsets *= torch.tensor([sh / 2, sw / 2]).view(1, 1, 2) * min(0.97, 2.0 * GLOBAL_MULTIPLIER)  # 幅度控制
	dst = src + offsets if np.random.uniform() < prob else src  # 概率应用

	src, dst = src.view(1, -1, 2), dst.view(1, -1, 2)
	src = (src / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.  # 归一化：将坐标从[0, h]×[0, w]映射到[-1, 1]×[-1, 1]
	dst = (dst / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.  # 归一化：将坐标从[0, h]×[0, w]映射到[-1, 1]×[-1, 1]
	weights, A = findTPS(dst, src)  # 计算TPS变换参数

	return src, weights, A


# 刚性图像形变技术
def generateRandomHomography(shape, GLOBAL_MULTIPLIER=0.3):
	#Generate random in-plane rotation [-theta,+theta]
	# 创建随机旋转变换
	theta = np.radians(np.random.uniform(-30, 30))
	c, s = np.cos(theta), np.sin(theta)

	#Generate random scale in both x and y
	# 创建随机缩放变换
	scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)

	#Generate random translation shift
	# 创建随机平移变换
	# 设定坐标系原点(即将图像的坐标系原点设定在图像中心)
	tx, ty = -shape[1] / 2.0, -shape[0] / 2.0
	# 随机平移坐标系原点
	txn, tyn = np.random.normal(0, 120.0 * GLOBAL_MULTIPLIER, 2)

	# Affine coeffs
	# 仿射变换参数
	sx, sy = np.random.normal(0, 0.6 * GLOBAL_MULTIPLIER, 2)

	# Projective coeffs
	# 射影变换参数
	p1, p2 = np.random.normal(0, 0.006 * GLOBAL_MULTIPLIER, 2)

	# Build Homography from parmeterizations
	# 投影变换矩阵(单应性变换矩阵)
	H_t = np.array(((1, 0, tx), (0, 1, ty), (0, 0, 1)))  #t                         # 水平变换
	H_r = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))  #rotation,                  # 旋转变换
	H_a = np.array(((1, sy, 0), (sx, 1, 0), (0, 0, 1)))  # affine                   # 仿射变换
	H_p = np.array(((1, 0, 0), (0, 1, 0), (p1, p2, 1)))  # projective               # 射影变换
	H_s = np.array(((scale_x, 0, 0), (0, scale_y, 0), (0, 0, 1)))  #scale           # 缩放变换
	H_b = np.array(((1.0, 0, -tx + txn), (0, 1, -ty + tyn), (0, 0, 1)))  #t_back,   # 坐标系原点变换

	#H = H_t * H_r * H_a * H_p * H_s * H_b
	H = np.dot(np.dot(np.dot(np.dot(np.dot(H_b, H_s), H_p), H_a), H_r), H_t)

	return H


# COCO数据增强
class COCOAugmentor(nn.Module):
	def __init__(self, device, load_dataset=True,
				 img_dir="",
				 warp_resolution=(1200, 900),  # 输入图像统一缩放到 warp_resolution
				 out_resolution=(400, 300),  # 输出图像统一缩放到 out_resolution
				 sides_crop=0.2,  # 移除透视变换后的无效边缘 sides_crop 控制裁剪比例
				 max_num_imgs=50,  # 用于训练的最大图像数量
				 num_test_imgs=10,  # 用于测试的图像数量
				 batch_size=1,  # 批大小
				 photometric=True,  # 是否应用光度增强（颜色变换等）
				 geometric=True,  # 是否应用几何变换（单应性变换、TPS等）
				 reload_step=1_000  # 每处理多少张重新加载一次图像（防止过拟合）
				 ):
		super(COCOAugmentor, self).__init__()
		self.half = 16
		self.device = device

		self.dims = warp_resolution
		self.batch_size = batch_size
		self.out_resolution = out_resolution
		self.sides_crop = sides_crop
		self.max_num_imgs = max_num_imgs
		self.num_test_imgs = num_test_imgs
		# # 裁剪后的有效区域尺寸（warp_resolution扣除sides_crop）
		# self.dims_t = torch.tensor(
		# 	[int(self.dims[0] * (1. - self.sides_crop)) - int(self.dims[0] * self.sides_crop) - 1,
		# 	 int(self.dims[1] * (1. - self.sides_crop)) - int(self.dims[1] * self.sides_crop) - 1]).float().to(device).view(1, 1, 2)
		# 裁剪后的有效区域尺寸（warp_resolution扣除sides_crop）(new)
		self.dims_t = torch.tensor(
			[int(self.dims[0] * (1. - self.sides_crop)) - int(self.dims[0] * self.sides_crop) ,
			 int(self.dims[1] * (1. - self.sides_crop)) - int(self.dims[1] * self.sides_crop) ]).float().to(device).view(1, 1, 2)
		# 计算缩放比例
		self.dims_s = torch.tensor([self.dims_t[0, 0, 0] / out_resolution[0],
									self.dims_t[0, 0, 1] / out_resolution[1]]).float().to(device).view(1, 1, 2)

		# 图像加载配置
		self.all_imgs = glob.glob(img_dir + '/*.jpg') + glob.glob(img_dir + '/*.png')

		self.photometric = photometric  # 光度增强开关
		self.geometric = geometric  # 几何增强开关
		self.cnt = 1  # 初始图片加载次数设置为1, 并不断累加
		self.reload_step = reload_step  # 每处理多少张重新加载一次图像

		# 返回的图像少于10张
		if len(self.all_imgs) < 10:
			raise RuntimeError('Couldnt find enough images to train. Please check the path: ', img_dir)

		# 加载数据集
		if load_dataset:
			print('[COCO]: ', len(self.all_imgs), ' images for training..')
			if len(self.all_imgs) - num_test_imgs < max_num_imgs:
				raise RuntimeError('Error: test set overlaps with training set! Decrease number of test imgs')

			self.load_imgs()

			self.TPS = True

	def load_imgs(self):
		# 随机打乱
		random.shuffle(self.all_imgs)
		train = []
		# 图片的前self.max_num_imgs用于训练
		for p in tqdm.tqdm(self.all_imgs[:self.max_num_imgs], desc='loading train'):
			im = cv2.imread(p)
			halfH, halfW = im.shape[0] // 2, im.shape[1] // 2
			# 规定了宽width 一定大于 高height
			if halfH > halfW:
				im = np.rot90(im)
				halfH, halfW = halfW, halfH

			# 规定了图像缩放到 warp_resolution 尺寸
			if im.shape[0] != self.dims[1] or im.shape[1] != self.dims[0]:
				im = cv2.resize(im, self.dims)

			train.append(np.copy(im))

		# 训练图片
		self.train = train
		# 测试图片
		self.test = [
			cv2.resize(cv2.imread(p), self.dims)
			# 图片的后self.num_test_imgs用于测试
			for p in tqdm.tqdm(self.all_imgs[-self.num_test_imgs:], desc='loading test')
		]

	# 归一化, 将裁剪区域的关键点坐标归一化到[-1, 1]范围
	def norm_pts_grid(self, x):
		if len(x.size()) == 2:
			return (x.view(1, -1, 2) * self.dims_s / self.dims_t) * 2. - 1
		return (x * self.dims_s / self.dims_t) * 2. - 1

	# 反归一化, 将归一化坐标转换回裁剪区域的实际坐标
	def denorm_pts_grid(self, x):
		if len(x.size()) == 2:
			return ((x.view(1, -1, 2) + 1) / 2.) / self.dims_s * self.dims_t
		return ((x + 1) / 2.) / self.dims_s * self.dims_t

	# 在指定图像区域内生成随机关键点
	def rnd_kps(self, shape, n=256):
		h, w = shape
		kps = torch.rand(size=(3, n)).to(self.device)
		kps[0, :] *= w
		kps[1, :] *= h
		kps[2, :] = 1.0

		return kps

	# 使用单应性矩阵变换关键点坐标
	def warp_points(self, H, pts):
		scale = self.dims_s.view(-1, 2)
		offset = torch.tensor([int(self.dims[0] * self.sides_crop), int(self.dims[1] * self.sides_crop)],
							  device=pts.device).float()
		pts = pts * scale + offset
		pts = torch.vstack([pts.t(), torch.ones(1, pts.shape[0], device=pts.device)])
		warped = torch.matmul(H, pts)
		warped = warped / warped[2, ...]
		warped = warped.t()[:, :2]
		return (warped - offset) / scale

	@torch.inference_mode()
	def forward(self, x, difficulty=0.3, TPS=False, prob_deformation=0.5, test=False):
		"""
			Perform augmentation to a batch of images.

			input:
				x -> torch.Tensor(B, C, H, W): rgb images   # 输入图像张量
				difficulty -> float: level of difficulty, 0.1 is medium, 0.3 is already pretty hard # 增强的难度级别，影响几何变换的强度
				tps -> bool: Wether to apply non-rigid deformations in images   # 是否应用非刚性变形
				prob_deformation -> float: probability to apply a deformation   # 应用TPS变形的概率

			return:
				'output'    ->   torch.Tensor(B, C, H, W): rgb images                       # 增强后的图像
				Tuple:
					'H'       ->   torch.Tensor(3,3): homography matrix                     #  单应性变换矩阵，形状为(B, 3, 3)
					'mask'  ->     torch.Tensor(B, H, W): mask of valid pixels after warp
					(deformation only)
					src, weights, A are parameters from a TPS warp (all torch.Tensors)

		"""
		# 每处理多少步重新加载一次图像
		if self.cnt % self.reload_step == 0:
			self.load_imgs()

		# 几何增强开关
		if self.geometric is False:
			difficulty = 0.

		# 仅在前向传播时禁用梯度
		with torch.no_grad():
			x = (x / 255.).to(self.device)
			b, c, h, w = x.shape
			shape = (h, w)

			######## Geometric Transformations
			# 几何增强模块
			H = torch.tensor(np.array([generateRandomHomography(shape, difficulty) for b in range(self.batch_size)]),
							 dtype=torch.float32).to(self.device)
			output = kornia.geometry.transform.warp_perspective(x, H, dsize=shape, padding_mode='zeros')

			#crop % of image boundaries each side to reduce invalid pixels after warps
			# 边缘裁剪
			low_h = int(h * self.sides_crop);
			low_w = int(w * self.sides_crop)
			high_h = int(h * (1. - self.sides_crop));
			high_w = int(w * (1. - self.sides_crop))
			output = output[..., low_h:high_h, low_w:high_w]
			x = x[..., low_h:high_h, low_w:high_w]

			#apply TPS if desired:
			# 是否需要非刚体变换
			if TPS:
				src, weights, A = None, None, None
				for b in range(self.batch_size):
					b_src, b_weights, b_A = generateRandomTPS(shape, (8, 6), difficulty, prob=prob_deformation)
					b_src, b_weights, b_A = b_src.to(self.device), b_weights.to(self.device), b_A.to(self.device)

					if src is None:
						src, weights, A = b_src, b_weights, b_A
					else:
						src = torch.cat((b_src, src))
						weights = torch.cat((b_weights, weights))
						A = torch.cat((b_A, A))

				output = warp_image_tps(output, src, weights, A)

			# 分辨率处理（interpolate插值也可以用于下采样）
			output = F.interpolate(output, self.out_resolution[::-1], mode='nearest')
			x = F.interpolate(x, self.out_resolution[::-1], mode='nearest')

			mask = ~torch.all(output == 0, dim=1, keepdim=True)
			mask = mask.expand(-1, 3, -1, -1)

			# Make-up invalid regions with texture from the batch
			rv = 1 if not TPS else 2
			output_shifted = torch.roll(x, rv, 0)
			output[~mask] = output_shifted[~mask]
			mask = mask[:, 0, :, :]

			######## Photometric Transformations
			# 光度增强模块

			list_augmentation = [
				kornia.augmentation.ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.),
				kornia.augmentation.RandomEqualize(p=0.4),
				kornia.augmentation.RandomGaussianBlur(p=0.3, sigma=(2.0, 2.0), kernel_size=(7, 7))
			]

			if self.photometric is False:
				list_augmentation = []
			aug_list = kornia.augmentation.ImageSequential(*list_augmentation)

			output = aug_list(output)

			b, c, h, w = output.shape
			#Correlated Gaussian Noise
			# 相关高斯噪声
			if np.random.uniform() > 0.5 and self.photometric:
				noise = F.interpolate(torch.randn_like(output) * (10 / 255), (h // 2, w // 2))
				noise = F.interpolate(noise, (h, w), mode='bicubic')
				output = torch.clip(output + noise, 0., 1.)

			#Random shadows
			# 随机阴影
			if np.random.uniform() > 0.6 and self.photometric:
				noise = torch.rand((b, 1, h // 64, w // 64), device=self.device) * 1.3
				noise = torch.clip(noise, 0.25, 1.0)
				noise = F.interpolate(noise, (h, w), mode='bicubic')
				noise = noise.expand(-1, 3, -1, -1)
				output *= noise
				output = torch.clip(output, 0., 1.)

			self.cnt += 1

		if TPS:
			return output, (H, src, weights, A, mask)
		else:
			return output, (H, mask)

	# 目标关键点 -> 源关键点
	def get_correspondences(self, kps_target, T):
		H, H2, src, W, A = T
		# 应用逆TPS变换
		undeformed = self.denorm_pts_grid(warp_points_tps(self.norm_pts_grid(kps_target), src, W, A)).view(-1, 2)

		warped_to_src = self.warp_points(H @ torch.inverse(H2), undeformed)

		return warped_to_src
