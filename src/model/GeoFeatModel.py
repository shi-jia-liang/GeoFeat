"""
	"GeoFeat: 3D Geometry-Aware Local Feature Matching"
"""

import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F

import tqdm
import math
import cv2

import sys
from typing import List, Tuple, Optional

import torch_dct as DCT
from einops import rearrange

"""
foundational functions
"""
def simple_nms(scores, radius):
	"""Perform non maximum suppression on the heatmap using max-pooling.
	This method does not suppress contiguous points that have the same score.
	Args:
		scores: the score heatmap of size `(B, H, W)`.
		radius: an integer scalar, the radius of the NMS window.
	"""

	def max_pool(x):
		return torch.nn.functional.max_pool2d(
			x, kernel_size=radius * 2 + 1, stride=1, padding=radius
		)

	zeros = torch.zeros_like(scores)
	max_mask = scores == max_pool(scores)
	for _ in range(2):
		supp_mask = max_pool(max_mask.float()) > 0
		supp_scores = torch.where(supp_mask, zeros, scores)
		new_max_mask = supp_scores == max_pool(supp_scores)
		max_mask = max_mask | (new_max_mask & (~supp_mask))
	return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
	if k >= len(keypoints):
		return keypoints, scores
	scores, indices = torch.topk(scores, k, dim=0, sorted=True)
	return keypoints[indices], scores


def sample_k_keypoints(keypoints, scores, k):
	if k >= len(keypoints):
		return keypoints, scores
	indices = torch.multinomial(scores, k, replacement=False)
	return keypoints[indices], scores[indices]


def soft_argmax_refinement(keypoints, scores, radius: int):
	width = 2 * radius + 1
	sum_ = torch.nn.functional.avg_pool2d(
		scores[:, None], width, 1, radius, divisor_override=1
	)
	ar = torch.arange(-radius, radius + 1).to(scores)
	kernel_x = ar[None].expand(width, -1)[None, None]
	dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
	dy = torch.nn.functional.conv2d(
		scores[:, None], kernel_x.transpose(2, 3), padding=radius
	)
	dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
	refined_keypoints = []
	for i, kpts in enumerate(keypoints):
		delta = dydx[i][tuple(kpts.t())]
		refined_keypoints.append(kpts.float() + delta)
	return refined_keypoints


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
	"""Interpolate descriptors at keypoint locations"""
	b, c, h, w = descriptors.shape
	keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
	keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
	descriptors = torch.nn.functional.grid_sample(
		descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
	)
	descriptors = torch.nn.functional.normalize(
		descriptors.reshape(b, c, -1), p=2, dim=1
	)
	return descriptors

# --------- HS-FPN Modules (Ported & Adapted) ---------
class ConvModule(nn.Module):
	"""Shim for MMCV ConvModule"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, norm_cfg=True, act_cfg=True):
		super().__init__()
		layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)]
		if norm_cfg:
			layers.append(nn.BatchNorm2d(out_channels))
		if act_cfg:
			layers.append(nn.ReLU(inplace=True))
		self.block = nn.Sequential(*layers)
	def forward(self, x):
		return self.block(x)

class DctSpatialInteraction(nn.Module):
	def __init__(self, in_channels, ratio, isdct=True):
		super(DctSpatialInteraction, self).__init__()
		self.ratio = ratio
		self.isdct = isdct
		if not self.isdct:
			self.spatial1x1 = nn.Sequential(
				nn.Conv2d(in_channels, 1, kernel_size=1, bias=False) # Simplified ConvModule
			)

	def forward(self, x):
		_, _, h0, w0 = x.size()
		if not self.isdct:
			return x * torch.sigmoid(self.spatial1x1(x))
		# if DCT is None: return x # Fallback
		idct = DCT.dct_2d(x, norm='ortho') 
		weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
		weight = weight.view(1, h0, w0).expand_as(idct)             
		dct = idct * weight # filter out low-frequency features 
		dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
		return x * dct_

	def _compute_weight(self, h, w, ratio):
		h0 = int(h * ratio[0])
		w0 = int(w * ratio[1])
		weight = torch.ones((h, w), requires_grad=False)
		weight[:h0, :w0] = 0
		return weight

class DctChannelInteraction(nn.Module):
	def __init__(self, in_channels, patch, ratio, isdct=True):
		super(DctChannelInteraction, self).__init__()
		self.in_channels = in_channels
		self.h = patch[0]
		self.w = patch[1]
		self.ratio = ratio
		self.isdct = isdct
		self.channel1x1 = ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)
		self.channel2x1 = ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)
		self.relu = nn.ReLU()

	def forward(self, x):
		n, c, h, w = x.size()
		if not self.isdct: 
			amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
			aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
			channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
			return x * torch.sigmoid(self.channel2x1(channel))

		# if DCT is None: return x
		idct = DCT.dct_2d(x, norm='ortho')
		weight = self._compute_weight(h, w, self.ratio).to(x.device)
		weight = weight.view(1, h, w).expand_as(idct)             
		dct = idct * weight 
		dct_ = DCT.idct_2d(dct, norm='ortho') 

		amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
		aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))       
		amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
		aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

		channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
		return x * torch.sigmoid(self.channel2x1(channel))
        
	def _compute_weight(self, h, w, ratio):
		h0 = int(h * ratio[0])
		w0 = int(w * ratio[1])
		weight = torch.ones((h, w), requires_grad=False)
		weight[:h0, :w0] = 0
		return weight  

class HFP(nn.Module):
	def __init__(self, in_channels, ratio, patch=(8,8), isdct=True):
		super(HFP, self).__init__()
		self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct) 
		self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
		self.out =  nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
			nn.GroupNorm(32, in_channels)
		)
	def forward(self, x):
		spatial = self.spatial(x)
		channel = self.channel(x)
		return self.out(spatial + channel)

class SDP(nn.Module):
	def __init__(self, dim=256, inter_dim=None):
		super(SDP, self).__init__()
		self.inter_dim = inter_dim or dim
		self.conv_q = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 1, bias=False), nn.GroupNorm(32, self.inter_dim))
		self.conv_k = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 1, bias=False), nn.GroupNorm(32, self.inter_dim))
		self.softmax = nn.Softmax(dim=-1)
		
	def forward(self, x_low, x_high, patch_size):
		# x_low: current level (finer), x_high: upper level upsampled (coarser info)
		b_, _, h_, w_ = x_low.size()
		# patch_size corresponds to the grid size of the coarse level (h, w)
		# rearrange: b c (h p1) (w p2) -> ...
		p1 = patch_size[0] # h of coarse
		p2 = patch_size[1] # w of coarse
		# Actually, p1 should be H_low / h_coarse?
		# No, in HS-FPN code: patch_size passed is [h_coarse, w_coarse].
		# And p1=patch_size[0].
		# This seems to imply x_low is split into h_coarse x w_coarse patches.
		# Meaning each patch has size (H_low/h_coarse, W_low/w_coarse).
		# Wait, rearrange syntax: (h p1) means h chunks of size p1.
		# The prompt code: p1=patch_size[0]. 
		# If patch_size[0] is h (coarse height), then we have h chunks? 
		# If H_low = 2*h. Then we have 2 chunks of size h? Or h chunks of size 2?
		# HS-FPN: (h p1). With p1=h. This means H_low = h*h? This is unlikely.
		
		# Let's re-read HS-FPN carefully.
		# laterals[3] size h, w.
		# laterals[2] size 2h, 2w.
		# patch = [h, w].
		# SDP(x_low=lat2, x_high=up(lat3), patch).
		# q = rearrange(..., (h p1) ... p1=patch[0]) => (h p1) where p1=h.
		# This implies lat2 height (2h) = something * h.
		# => something = 2.
		# So h in rearrange is 2. p1 is h.
		# So it splits into 2x2 grid?? No, 2 vertical, 2 horizontal.
		# Wait, (b h w) c (p1 p2). h,w in rearrange output are the counts.
		# If input is (h_symbolic * p1). And p1=h_value.
		# Then h_symbolic = InputH / h_value = 2h / h = 2.
		# So we have 2x2 patches. Each patch is hxh size.
		# This means Global attention within 2x2 regions? No, if patch is hxh, that's huge.
		# It covers 1/4 of the image.
		# This makes sense for "Spatial Dependency".
		
		# So I will implement exactly as HS-FPN code.
		# if rearrange is None: return x_low # Fallback

		q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
		# q: (B*num_patches) C (H_patch*W_patch)
		q = q.transpose(1,2) 
		k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
		
		attn = torch.matmul(q, k) 
		attn = attn / np.power(self.inter_dim, 0.5)
		attn = self.softmax(attn)
		v = k.transpose(1,2)
		output = torch.matmul(attn,v)
		output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', 
						  p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
		return output + x_low

# --------- Backbone Classes ---------
# ========== StandardBackbone ==========
class StandardBackbone(nn.Module):
	def __init__(self, c1=24, c2=24, c3=64, c4=64, c5=128):
		super().__init__()
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
		self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
		self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
		self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
		self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
		self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
		self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
		self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
		self.conv5a = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
		self.conv5b = nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		x1 = self.relu(self.conv1a(x))
		x1 = self.relu(self.conv1b(x1))
		x1 = self.pool(x1)
		x2 = self.relu(self.conv2a(x1))
		x2 = self.relu(self.conv2b(x2))
		x2 = self.pool(x2)
		x3 = self.relu(self.conv3a(x2))
		x3 = self.relu(self.conv3b(x3))
		x3 = self.pool(x3)
		x4 = self.relu(self.conv4a(x3))
		x4 = self.relu(self.conv4b(x4))
		x4 = self.pool(x4)
		x5 = self.relu(self.conv5a(x4))
		x5 = self.relu(self.conv5b(x5))
		x5 = self.pool(x5)
		return x3, x4, x5


# ========== RepVGG Backbone ==========
def _fuse_conv_and_bn(conv, bn):
	"""Fuse conv and bn layer params to a single conv kernel and bias."""
	with torch.no_grad():
		# conv: nn.Conv2d, bn: nn.BatchNorm2d
		w = conv.weight
		if conv.bias is None:
			b = torch.zeros(w.size(0), device=w.device)
		else:
			b = conv.bias

		bn_w = bn.weight
		bn_b = bn.bias
		bn_rm = bn.running_mean
		bn_rv = bn.running_var
		eps = bn.epss

		std = torch.sqrt(bn_rv + eps)
		w_fold = w * (bn_w / std).reshape(-1, 1, 1, 1)
		b_fold = (b - bn_rm) / std * bn_w + bn_b
		return w_fold, b_fold

class RepVGGBlock(nn.Module):
	"""RepVGG-like block (training multi-branch, can be converted to single conv for deploy).

	This implementation keeps no activation inside so it matches calling code that applies activations.
	"""
	def __init__(self, in_channels, out_channels, stride=1, deploy=False):
		super().__init__()
		self.deploy = deploy
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride

		if deploy:
			self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
		else:
			# 3x3 branch
			self.rbr_dense = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
				nn.BatchNorm2d(out_channels)
			)

			# 1x1 branch
			self.rbr_1x1 = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False),
				nn.BatchNorm2d(out_channels)
			)

			# identity branch (if applicable)
			if out_channels == in_channels and stride == 1:
				self.rbr_identity = nn.BatchNorm2d(in_channels)
			else:
				self.rbr_identity = None

	def forward(self, x):
		if self.deploy:
			return self.rbr_reparam(x)
		out = self.rbr_dense(x) + self.rbr_1x1(x)
		if self.rbr_identity is not None:
			out += self.rbr_identity(x)
		return out

	def get_equivalent_kernel_bias(self):
		"""Return kernel and bias to create a single conv equivalent to the block."""
		if self.deploy:
			return self.rbr_reparam.weight, self.rbr_reparam.bias

		kernel3x3, bias3x3 = _fuse_conv_and_bn(self.rbr_dense[0], self.rbr_dense[1])
		kernel1x1, bias1x1 = _fuse_conv_and_bn(self.rbr_1x1[0], self.rbr_1x1[1])

		# pad 1x1 kernel to 3x3
		kernel1x1 = F.pad(kernel1x1, [1,1,1,1])

		if self.rbr_identity is not None:
			bn = self.rbr_identity
			std = torch.sqrt(bn.running_var + bn.eps)
			gamma = bn.weight
			beta = bn.bias
			running_mean = bn.running_mean
			# identity kernel and bias
			id_kernel = torch.zeros_like(kernel3x3)
			for i in range(self.out_channels):
				id_kernel[i, i, 1, 1] = (gamma[i] / std[i]).item()
			id_bias = (- running_mean / std * gamma + beta)
		else:
			id_kernel = torch.zeros_like(kernel3x3)
			id_bias = torch.zeros(self.out_channels, device=kernel3x3.device)

		kernel = kernel3x3 + kernel1x1 + id_kernel
		bias = bias3x3 + bias1x1 + id_bias
		return kernel, bias

	def switch_to_deploy(self):
		if self.deploy:
			return
		kernel, bias = self.get_equivalent_kernel_bias()
		self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, padding=1, bias=True)
		self.rbr_reparam.weight.data = kernel
		self.rbr_reparam.bias.data = bias

		# delete unneeded branches
		delattr(self, 'rbr_dense')
		delattr(self, 'rbr_1x1')
		if hasattr(self, 'rbr_identity'):
			delattr(self, 'rbr_identity')
		self.deploy = True

class RepVGGBackbone(nn.Module):
	def __init__(self, c1=24, c2=24, c3=64, c4=64, c5=128):
		super().__init__()
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.conv1a = RepVGGBlock(1, c1, stride=1, deploy=False)
		self.conv1b = RepVGGBlock(c1, c1, stride=1, deploy=False)
		self.conv2a = RepVGGBlock(c1, c2, stride=1, deploy=False)
		self.conv2b = RepVGGBlock(c2, c2, stride=1, deploy=False)
		self.conv3a = RepVGGBlock(c2, c3, stride=1, deploy=False)
		self.conv3b = RepVGGBlock(c3, c3, stride=1, deploy=False)
		self.conv4a = RepVGGBlock(c3, c4, stride=1, deploy=False)
		self.conv4b = RepVGGBlock(c4, c4, stride=1, deploy=False)
		self.conv5a = RepVGGBlock(c4, c5, stride=1, deploy=False)
		self.conv5b = RepVGGBlock(c5, c5, stride=1, deploy=False)

	def forward(self, x):
		x1 = self.relu(self.conv1a(x))
		x1 = self.relu(self.conv1b(x1))
		x1 = self.pool(x1)
		x2 = self.relu(self.conv2a(x1))
		x2 = self.relu(self.conv2b(x2))
		x2 = self.pool(x2)
		x3 = self.relu(self.conv3a(x2))
		x3 = self.relu(self.conv3b(x3))
		x3 = self.pool(x3)
		x4 = self.relu(self.conv4a(x3))
		x4 = self.relu(self.conv4b(x4))
		x4 = self.pool(x4)
		x5 = self.relu(self.conv5a(x4))
		x5 = self.relu(self.conv5b(x5))
		x5 = self.pool(x5)
		return x3, x4, x5


# --------- UpsampleLayer Classes ---------
# ========== UpsampleLayer ==========
class UpsampleLayer(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		# 定义特征提取层，减少通道数同时增加特征提取能力
		self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
		# 使用BN层
		self.bn = nn.BatchNorm2d(in_channels//2)
		# 使用LeakyReLU激活函数
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = self.leaky_relu(self.bn(self.conv(x)))

		return x

# ========== PixelShuffleUpsample ==========
class PixelShuffleUpsample(nn.Module):
	def __init__(self, in_channels, out_channels=None):
		super().__init__()
		if out_channels is None:
			out_channels = in_channels // 2
		# PixelShuffle 需要输入通道数为 out_channels * 4 (因为放大2倍)
		self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1)
		self.bn = nn.BatchNorm2d(out_channels * 4)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self, x):
		# Conv(C_in -> 4*C_out) -> BN -> ReLU -> PixelShuffle(4*C_out -> C_out)
		x = self.leaky_relu(self.pixel_shuffle(self.bn(self.conv(x))))
		return x

# --------- Positional Encoding ---------
class PositionEncoding2D(nn.Module):
	"""二维位置编码模块 (2D Positional Encoding)

	支持的编码类型 (pos_enc_type):
	  - 'none'            : 不使用位置编码
	  - 'fourier'         : 对 x 与 y 分别做多频率 sin/cos 展开 (4 * len(freqs) 通道)
	  - 'rot_inv'         : 旋转不变的径向位置编码；使用到中心距离 r 及其多频率的 sin/cos (1 + 2 * len(freqs) 通道)

	参数说明:
	  - pos_enc_type (str): 编码类型
	  - out_channels (int): 输出通道数（通过 1x1 Conv 将原始位置特征映射到此维度）
	  - freqs (Iterable[float]): 频率列表，控制 Fourier 展开尺度

	旋转不变性原理:
	  - rot_inv 仅依赖径向距离 r = sqrt((x-0.5)^2 + (y-0.5)^2)，与图像整体旋转无关
	"""
	def __init__(self, pos_enc_type: str = 'none', out_channels=None, freqs=None):
		super().__init__()
		assert pos_enc_type in ('None', 'fourier','rot_inv')
		self.pos_enc_type = pos_enc_type
		if freqs is not None:
			self._pos_freqs = list(freqs)
		else:
			self._pos_freqs = [1.0, 2.0, 4.0, 8.0]

		# determine number of input positional channels
		if self.pos_enc_type == 'fourier':
			in_ch = 4 * len(self._pos_freqs)
		elif self.pos_enc_type == 'rot_inv':
			# r 原值 + sin/cos(2π f r)
			in_ch = 1 + 2 * len(self._pos_freqs)
		else:
			in_ch = 0

		self.out_channels = out_channels
		if in_ch > 0 and out_channels is not None:
			# 如果是 concat 模式，我们不需要投影层，直接返回原始位置特征
			# 或者如果需要调整通道数，可以使用投影层
			# 这里为了灵活性，我们保留投影层，但如果 out_channels 设置为 None，则不投影
			self.pos_proj = nn.Conv2d(in_ch, out_channels, kernel_size=1)
			nn.init.normal_(self.pos_proj.weight, mean=0.0, std=1e-2)
			if self.pos_proj.bias is not None:
				nn.init.zeros_(self.pos_proj.bias)
		else:
			self.pos_proj = None

	def forward(self, x: torch.Tensor):
		"""Given feature map x (B, C, H, W), return positional features.
		If out_channels is set, projects to that dimension.
		Returns None if pos_enc_type is 'none'."""
		if self.pos_enc_type == 'None':
			return None

		B, Cx, H, W = x.shape
		ys = torch.linspace(0, 1, steps=H, device=x.device, dtype=x.dtype)
		xs = torch.linspace(0, 1, steps=W, device=x.device, dtype=x.dtype)
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

		if self.pos_enc_type == 'fourier':
			feats = []
			for f in self._pos_freqs:
				feats.append(torch.sin(2 * math.pi * f * grid_x))
				feats.append(torch.cos(2 * math.pi * f * grid_x))
			for f in self._pos_freqs:
				feats.append(torch.sin(2 * math.pi * f * grid_y))
				feats.append(torch.cos(2 * math.pi * f * grid_y))
			pos = torch.stack(feats, dim=0).unsqueeze(0).expand(B, -1, -1, -1)
		elif self.pos_enc_type == 'rot_inv':
			# 以 (0.5, 0.5) 为中心的径向距离 r
			cx, cy = 0.5, 0.5
			dx = grid_x - cx
			dy = grid_y - cy
			r = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=1e-12))  # (H,W)
			feats = [r]
			for f in self._pos_freqs:
				feats.append(torch.sin(2 * math.pi * f * r))
				feats.append(torch.cos(2 * math.pi * f * r))
			pos = torch.stack(feats, dim=0).unsqueeze(0).expand(B, -1, -1, -1)

		pos = pos.to(x.dtype).to(x.device) # type: ignore
		
		if self.pos_proj is not None:
			pos_feat = self.pos_proj(pos)
			return pos_feat
		else:
			return pos

# --------- Head Classes ---------
class BaseLayer(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False,activation=True):
		super().__init__()
		if activation:
			self.layer=nn.Sequential(
				nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
				nn.BatchNorm2d(out_channels,affine=False),
				nn.ReLU(inplace=True)
			)
		else:
			self.layer=nn.Sequential(
				nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
				nn.BatchNorm2d(out_channels,affine=False)
			)
		
	def forward(self,x):
		return self.layer(x)

class KeypointHead(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layer1=BaseLayer(in_channels,32)
		self.layer2=BaseLayer(32,32)
		self.layer3=BaseLayer(32,64)
		self.layer4=BaseLayer(64,64)
		self.layer5=BaseLayer(64,128)
		
		self.conv=nn.Conv2d(128,out_channels,kernel_size=3,stride=1,padding=1)
		self.bn=nn.BatchNorm2d(65)
		
	def forward(self,x):
		x=self.layer1(x)
		x=self.layer2(x)
		x=self.layer3(x)
		x=self.layer4(x)
		x=self.layer5(x)
		x=self.bn(self.conv(x))
		return x
	
	
class DescriptorHead(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layer=nn.Sequential(
			BaseLayer(in_channels,32),
			BaseLayer(32,32,activation=False),
			BaseLayer(32,64,activation=False),
			BaseLayer(64,out_channels,activation=False)
		)
		
	def forward(self,x):
		x=self.layer(x)
		# x=nn.functional.softmax(x,dim=1)
		return x
	
	
class HeatmapHead(nn.Module):
	def __init__(self,in_channels,mid_channels,out_channels):
		super().__init__()
		self.convHa = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
		self.bnHa = nn.BatchNorm2d(mid_channels)
		self.convHb = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.bnHb = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1)
		
	def forward(self,x):
		x = self.leaky_relu(self.bnHa(self.convHa(x)))
		x = self.leaky_relu(self.bnHb(self.convHb(x)))
		
		x = torch.sigmoid(x)
		return x

class GeoHead(nn.Module):
	"""LiftFeat-style depth/geo head with dynamic channels and projection to desc dim."""
	def __init__(self, in_channels: int, geo_cfg: dict):
		super().__init__()
		out_ch = 0
		if geo_cfg['depth']:
			out_ch += 1
		if geo_cfg['normal']:
			out_ch += 3
		if geo_cfg['gradients']:
			out_ch += 2
		if geo_cfg['curvatures']:
			out_ch += 5
		if out_ch == 0:
			out_ch = 1  # fallback to avoid empty head

		self.out_channels = out_ch
		# three-stage upsample like LiftFeat DepthHead
		self.upsampleDa = UpsampleLayer(in_channels)
		self.upsampleDb = UpsampleLayer(in_channels//2)
		self.upsampleDc = UpsampleLayer(in_channels//4)
		
		self.convDepa = nn.Conv2d(in_channels//2+in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
		self.bnDepa = nn.BatchNorm2d(in_channels//2)
		self.convDepb = nn.Conv2d(in_channels//4+in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1)
		self.bnDepb = nn.BatchNorm2d(in_channels//4)
		self.convDepc = nn.Conv2d(in_channels//8+in_channels//4, self.out_channels, kernel_size=3, stride=1, padding=1)
		self.bnDepc = nn.BatchNorm2d(self.out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self, x):
		x0 = F.interpolate(x, scale_factor=2,mode='bilinear',align_corners=False)
		x1 = self.upsampleDa(x)
		x1 = torch.cat([x0,x1],dim=1)
		x1 = self.leaky_relu(self.bnDepa(self.convDepa(x1)))
		
		x1_0 = F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=False)
		x2 = self.upsampleDb(x1)
		x2 = torch.cat([x1_0,x2],dim=1)
		x2 = self.leaky_relu(self.bnDepb(self.convDepb(x2)))
		
		x2_0 = F.interpolate(x2,scale_factor=2,mode='bilinear',align_corners=False)
		x3 = self.upsampleDc(x2)
		x3 = torch.cat([x2_0,x3],dim=1)
		x = self.leaky_relu(self.bnDepc(self.convDepc(x3)))
		
		x = F.normalize(x,p=2,dim=1)
		return x
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																   

# --------- Encoder Classes ---------
def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
	""" Multi-layer perceptron """
	n = len(channels)
	layers = []
	for i in range(1, n):
		layers.append(nn.Linear(channels[i - 1], channels[i]))
		if i < (n-1):
			if do_bn:
				layers.append(nn.BatchNorm1d(channels[i]))
			layers.append(nn.ReLU())
	return nn.Sequential(*layers)

def MLP_no_ReLU(channels: List[int], do_bn: bool = False) -> nn.Module:
	""" Multi-layer perceptron """
	n = len(channels)
	layers = []
	for i in range(1, n):
		layers.append(nn.Linear(channels[i - 1], channels[i]))
		if i < (n-1):
			if do_bn:
				layers.append(nn.BatchNorm1d(channels[i]))
	return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
	""" Encoding of geometric properties using MLP """
	def __init__(self, keypoint_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP([keypoint_dim] + layers + [feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, kpts):
		if self.use_dropout:
			return self.dropout(self.encoder(kpts))
		return self.encoder(kpts)

class DescriptorEncoder(nn.Module):
	""" Encoding of visual descriptor using MLP """
	def __init__(self, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP([feature_dim] + layers + [feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)
	
	def forward(self, descs):
		residual = descs
		if self.use_dropout:
			return residual + self.dropout(self.encoder(descs))
		return residual + self.encoder(descs)

class GeometricEncoder(nn.Module):
	""" Encoding of geometric properties using MLP """
	def __init__(self, input_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP_no_ReLU([input_dim] + layers + [feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, x):
		if self.use_dropout:
			return self.dropout(self.encoder(x))
		return self.encoder(x)

# --------- Linear Attention ---------
class AFTAttention(nn.Module):
	""" Attention-free attention """
	def __init__(self, d_model: int, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.dim = d_model
		self.query = nn.Linear(d_model, d_model)
		self.key = nn.Linear(d_model, d_model)
		self.value = nn.Linear(d_model, d_model)
		self.proj = nn.Linear(d_model, d_model)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		q = self.query(x)
		k = self.key(x)
		v = self.value(x)
		# q = torch.sigmoid(q)
		k = k.T
		k = torch.softmax(k, dim=-1)
		k = k.T
		kv = (k * v).sum(dim=-2, keepdim=True)
		x = q * kv
		x = self.proj(x)
		if self.use_dropout:
			x = self.dropout(x)
		x += residual
		# x = self.layer_norm(x)
		return x

class FeedForwardNetwork(nn.Module):
	def __init__(self, feature_dim: int, ffn_type: str = 'positionwiseFFN', expansion: int = 4, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.ffn_type = ffn_type
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

		if ffn_type == 'positionwiseFFN':
			self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
		elif ffn_type == 'swigluFFN':
			hidden = int(feature_dim * expansion)
			self.fc1 = nn.Linear(feature_dim, hidden * 2, bias=False)
			self.activation = nn.GELU()
			self.fc2 = nn.Linear(hidden, feature_dim, bias=False)
		else:
			raise ValueError(f"Unknown ffn_type: {ffn_type}")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.ffn_type == 'positionwiseFFN':
			x = self.mlp(x)
			if self.use_dropout:
				x = self.dropout(x)
		elif self.ffn_type == 'swigluFFN':
			x_proj = self.fc1(x)
			gate, value = x_proj.chunk(2, dim=-1)
			hidden = self.activation(gate) * value
			if self.use_dropout:
				hidden = self.dropout(hidden)
			x = self.fc2(hidden)
		
		return x

class LinearAttentionalLayer(nn.Module):
	def __init__(self, feature_dim: int, ffn_type: str = 'positionwiseFFN', dropout: bool = False, p: float = 0.1):
		super().__init__()
		self.norm1 = nn.LayerNorm(feature_dim)
		self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
		self.norm2 = nn.LayerNorm(feature_dim)
		self.ffn = FeedForwardNetwork(feature_dim, ffn_type=ffn_type, dropout=dropout, p=p)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.norm1(x))
		x = x + self.ffn(self.norm2(x))
		return x

# ---------------- utility: window partition / reverse ----------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
	# x: B, C, H, W
	B, C, H, W = x.shape
	assert H % window_size == 0 and W % window_size == 0
	x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
	x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nH, nW, winH, winW, C
	windows = x.view(-1, window_size, window_size, C)  # (num_windows*B), winH, winW, C
	return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
	# windows: (num_windows*B), winH, winW, C
	B_times = windows.shape[0] // ((H // window_size) * (W // window_size))
	C = windows.shape[-1]
	x = windows.view(B_times, H // window_size, W // window_size, window_size, window_size, C)
	x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, nH, winH, nW, winW
	x = x.view(B_times, C, H, W)
	return x

# ---------------- Simplified Swin Transformer ----------------
class SimpleWindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads, dropout=0.0):
		super().__init__()
		self.dim = dim
		self.window_size = window_size  # (Wh, Ww)
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3)
		self.proj = nn.Linear(dim, dim)
		self.attn_drop = nn.Dropout(dropout)
		self.proj_drop = nn.Dropout(dropout)

	def forward(self, x):
		# x: (B*num_windows, N, C)
		B_, N, C = x.shape
		qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class SwinBlock(nn.Module):
	def __init__(self, dim, num_heads, window_size=5, shift_size=0, mlp_ratio=4.0, dropout=0.0, ffn_type='positionwiseFFN'):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		
		self.norm1 = nn.LayerNorm(dim)
		self.attn = SimpleWindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads, dropout=dropout)
		
		self.norm2 = nn.LayerNorm(dim)
		self.mlp = FeedForwardNetwork(dim, ffn_type=ffn_type, expansion=int(mlp_ratio), dropout=(dropout>0), p=dropout)

	def forward(self, x, H, W):
		# x: B, N, C
		B, N, C = x.shape
		shortcut = x
		x = self.norm1(x)
		x = x.view(B, H, W, C)

		# Cyclic shift
		if self.shift_size > 0:
			shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
		else:
			shifted_x = x

		# Partition
		# window_partition expects B, C, H, W
		x_permuted = shifted_x.permute(0, 3, 1, 2) 
		x_windows = window_partition(x_permuted, self.window_size) 
		x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

		# W-MSA/SW-MSA
		attn_windows = self.attn(x_windows)

		# Merge windows
		attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
		shifted_x = window_reverse(attn_windows, self.window_size, H, W) # Returns B, C, H, W

		# Reverse cyclic shift
		if self.shift_size > 0:
			x = torch.roll(shifted_x.permute(0, 2, 3, 1), shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
			x = shifted_x.permute(0, 2, 3, 1)
		
		x = x.view(B, N, C)

		# FFN
		x = shortcut + x
		x = x + self.mlp(self.norm2(x))
		return x

class SwinAttentionalLayer(nn.Module):
	def __init__(self, feature_dim: int, input_resolution: Tuple[int,int], depth: int = 2, num_heads: int = 8, window_size: int = 7, ffn_type: str = 'positionwiseFFN', dropout: bool = False, p: float = 0.1):
		super().__init__()
		self.input_resolution = input_resolution
		self.blocks = nn.ModuleList()
		for i in range(depth):
			shift_size = 0 if (i % 2 == 0) else window_size // 2
			self.blocks.append(
				SwinBlock(
					dim=feature_dim, 
					num_heads=num_heads, 
					window_size=window_size, 
					shift_size=shift_size, 
					mlp_ratio=4.0, 
					dropout=p if dropout else 0.0, 
					ffn_type=ffn_type
				)
			)

	def forward(self, x: torch.Tensor, shape: Optional[Tuple[int,int]] = None) -> torch.Tensor:
		if shape is None:
			H, W = self.input_resolution
		else:
			H, W = shape
			
		for blk in self.blocks:
			x = blk(x, H, W)
		return x


class AFTBlock(nn.Module):
	"""AFT attention with selectable FFN (PFFN or Swin)."""

	def __init__(self, feature_dim: int, ffn_type: str, swin_cfg: Optional[dict], dropout: bool = False, p: float = 0.1):
		super().__init__()
		self.ffn_type = ffn_type.lower()
		self.norm1 = nn.LayerNorm(feature_dim)
		self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
		self.norm2 = nn.LayerNorm(feature_dim)
		if self.ffn_type in ("pffn", "positionwiseffn"):
			self.ffn = FeedForwardNetwork(feature_dim, ffn_type='positionwiseFFN', dropout=dropout, p=p)
			self.needs_shape = False
		elif self.ffn_type == "swigluffn":
			self.ffn = FeedForwardNetwork(feature_dim, ffn_type='swigluFFN', dropout=dropout, p=p)
			self.needs_shape = False
		elif self.ffn_type == "swin":
			if swin_cfg is None or not swin_cfg["input_resolution"]:
				raise ValueError("Swin FFN requires Swin model_config with input_resolution")
			self.ffn = SwinAttentionalLayer(
				feature_dim,
				input_resolution=tuple(swin_cfg["input_resolution"]),
				depth=swin_cfg["depth_per_layer"],
				num_heads=swin_cfg["num_heads"],
				window_size=swin_cfg["window_size"],
				ffn_type=swin_cfg["ffn_type"],
				dropout=dropout,
				p=p,
			)
			self.needs_shape = True
		else:
			raise ValueError(f"Unsupported FFN type for AFT: {ffn_type}")

	def forward(self, x: torch.Tensor, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
		x = x + self.attn(self.norm1(x))
		if self.needs_shape:
			if shape is None:
				raise ValueError("Swin FFN requires shape=(H,W) during forward")
			x = x + self.ffn(self.norm2(x), shape=shape)
		else:
			x = x + self.ffn(self.norm2(x))
		return x

# ---------------- Update AttentionalNN to support Swin ----------------
# You can replace your existing AttentionalNN with the extended one below (keeps AFT support)

# if cfg["attention_type"] == "AFT":
# 			self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
# 			if cfg["ffn_type"] == "PPN"
# 				self.ffn = PositionwiseFeedForward(feature_dim, dropout=dropout, p=p)
# 			elif cfg["ffn_type"] == "swigluFFN":
# 				self.ffn = swigluFeedForward(feature_dim, dropout=dropout, p=p)
# 		elif cfg["attention_type"] == "Swin":
# 			self.attn = SwinAttention(feature_dim, dropout=dropout, p=p)
# 			self.ffn = SwinFeedForward(feature_dim, dropout=dropout, p=p)

class AttentionalNN(nn.Module):
	def __init__(self, feature_dim: int, layer_num: int, model_config: dict, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.model_config = model_config
		self.attention_type = model_config["attention_type"]  # 'AFT' or 'Swin'
		self.layers = nn.ModuleList()
		if self.attention_type == 'AFT':
			aft_cfg = model_config["AFT"]
			ffn_type = aft_cfg["ffn_type"]
			swin_cfg = model_config["Swin"]
			for _ in range(layer_num):
				self.layers.append(AFTBlock(feature_dim, ffn_type=ffn_type, swin_cfg=swin_cfg, dropout=dropout, p=p))
		
		elif self.attention_type == 'Swin':
			# need to know spatial resolution and block depth / heads / window size
			input_resolution = model_config["Swin"]["input_resolution"]  # (H, W)
			if input_resolution is None:
				raise ValueError("For Swin attention, provide model_config['Swin']['input_resolution'] = (H, W)")
			self.swin_input_resolution = tuple(input_resolution)
			depth_per_layer = model_config["Swin"]["depth_per_layer"]
			num_heads = model_config["Swin"]["num_heads"]
			window_size = model_config["Swin"]["window_size"]
			ffn_type = model_config["Swin"]["ffn_type"]
			for _ in range(layer_num):
				self.layers.append(SwinAttentionalLayer(feature_dim, input_resolution=input_resolution, depth=depth_per_layer, num_heads=num_heads, window_size=window_size, ffn_type=ffn_type, dropout=dropout, p=p))
		else:
			raise ValueError(f"Unknown attention type: {self.attention_type}")

	def forward(self, desc: torch.Tensor, shape: Optional[Tuple[int,int]] = None) -> torch.Tensor:
		if self.attention_type == 'AFT':
			for layer in self.layers:
				if isinstance(layer, AFTBlock):
					desc = layer(desc, shape)
				else:
					desc = layer(desc)
			return desc
		elif self.attention_type == 'Swin':
			input_is_2d = desc.dim() == 2
			if input_is_2d:
				desc = desc.unsqueeze(0)

			if shape is None:
				B, N, C = desc.shape
				# 1. Try configured resolution
				H_cfg, W_cfg = self.swin_input_resolution
				if H_cfg * W_cfg == N:
					shape = (H_cfg, W_cfg)
				else:
					# 2. Try square fallback
					H = int(math.sqrt(N))
					W = H
					if H * W == N:
						shape = (H, W)
					else:
						# 3. Fail
						raise ValueError(f"Swin attention requires shape=(H,W) or square/configured input. N={N}, configured={self.swin_input_resolution}")

			H, W = shape
			B, N, C = desc.shape
			assert N == H * W, f"desc sequence length {N} incompatible with H*W={H*W}"
			for layer in self.layers:
				desc = layer(desc, shape)
			
			if input_is_2d:
				desc = desc.squeeze(0)
			return desc
		else:
			raise ValueError(f"Unknown attention type: {self.attention_type}")

class FeatureBooster(nn.Module):
	def __init__(self, model_config, dropout=False, p=0.1):
		super().__init__()
		self.model_config = model_config
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

		self.kenc = KeypointEncoder(self.model_config['keypoint_dim'], self.model_config['descriptor_dim'], self.model_config['keypoint_encoder'], dropout=dropout)
		self.denc = DescriptorEncoder(self.model_config['descriptor_dim'], self.model_config['descriptor_encoder'], dropout=dropout)

		# Geometric features setup
		self.geo_encoders = nn.ModuleDict()
		self.geo_specs = [] # List of (name, dim)
		
		geo_cfg = self.model_config['geometric_features']
		# Order must match GeoHead: depth -> normal -> gradients -> curvatures
		# Mapping from config key to (encoder_name, dim_key, encoder_cfg_key)
		features_map = [
			('depth', 'depth', 'depth_dim', 'depth_encoder'),
			('normal', 'normal', 'normal_dim', 'normal_encoder'),
			('gradients', 'gradients', 'gradient_dim', 'gradient_encoder'),
			('curvatures', 'curvatures', 'curvature_dim', 'curvature_encoder')
		]

		for cfg_key, name, dim_key, enc_key in features_map:
			if geo_cfg[cfg_key]:
				dim = self.model_config[dim_key]
				self.geo_encoders[name] = GeometricEncoder(
					dim, 
					self.model_config['descriptor_dim'], 
					self.model_config[enc_key], 
					dropout=dropout
				)
				self.geo_specs.append((name, dim))

		self.attn_proj = AttentionalNN(feature_dim=self.model_config['descriptor_dim'], layer_num=self.model_config['attention_layers'], model_config=self.model_config, dropout=dropout)

		# Final activation
		if self.model_config['last_activation'] == 'None':
			self.last_activation = None
		elif self.model_config['last_activation'] == 'relu':
			self.last_activation = nn.ReLU()
		elif self.model_config['last_activation'] == 'sigmoid':
			self.last_activation = nn.Sigmoid()
		elif self.model_config['last_activation'] == 'tanh':
			self.last_activation = nn.Tanh()
		else:
			raise Exception('Not supported activation "%s".' % self.model_config['last_activation'])

	def forward(self, desc, kpts, geo_v=None, shape=None):
		# Descriptor & Keypoint encoding
		desc = self.denc(desc) + self.kenc(kpts)
		if self.use_dropout:
			desc = self.dropout(desc)

		# Unified Geometric Features Fusion
		if geo_v is not None and self.geo_specs:
			start_idx = 0
			for name, dim in self.geo_specs:
				# Slice the corresponding feature vector along the feature dimension (last dim)
				# geo_v is (B, N, C_total)
				feat_v = geo_v[..., start_idx : start_idx + dim]
				# Encode and add
				desc = desc + self.geo_encoders[name](feat_v)
				if self.use_dropout:
					desc = self.dropout(desc)
				start_idx += dim

		## Cross boosting
		desc = self.attn_proj(desc, shape=shape)

		# Final MLP projection
		if self.last_activation is not None:
			desc = self.last_activation(desc)
		# L2 normalization
		if self.model_config['l2_normalization']:
			desc = F.normalize(desc, dim=-1)

		return desc
	
class GeoFeatModel(nn.Module):
	def __init__(self, model_config):
		super().__init__()
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_config = model_config
		self.geo_config = self.model_config['geometric_features']

		# backbone
		self.backbone = self.model_config['backbone']
		c1,c2,c3,c4,c5 = 24,24,64,64,128
		if self.backbone == "Standard":
			self.feature_extract = StandardBackbone(c1, c2, c3, c4, c5)
		elif self.backbone == "RepVGG":
			self.feature_extract = RepVGGBackbone(c1, c2, c3, c4, c5)
		else:
			raise ValueError(f"Unknown backbone: {self.backbone}")
		self.norm = nn.InstanceNorm2d(1)
		
		# feature upsample and fusion
		if self.model_config['upsample_type'] == 'HS-FPN':
			self.hfp_dim = c3 # Set to 64 to match c3 output
			
			# Align channels to HFP dim
			self.lat3 = nn.Conv2d(c3, self.hfp_dim, 1)
			self.lat4 = nn.Conv2d(c4, self.hfp_dim, 1)
			self.lat5 = nn.Conv2d(c5, self.hfp_dim, 1)
			
			# HFP Modules
			self.hfp3 = HFP(self.hfp_dim, ratio=(0.25, 0.25), patch=(8,8), isdct=True)
			self.hfp4 = HFP(self.hfp_dim, ratio=(0.25, 0.25), patch=(8,8), isdct=True)
			self.hfp5 = HFP(self.hfp_dim, ratio=None, isdct=False)

			# SDP Modules
			self.sdp4_5 = SDP(dim=self.hfp_dim)
			self.sdp3_4 = SDP(dim=self.hfp_dim)

		elif self.model_config['upsample_type'] == 'bilinear':
			self.upsample4 = UpsampleLayer(c4)
			self.upsample5 = UpsampleLayer(c5)
			self.conv_fusion45 = nn.Conv2d(c5//2+c4,c4,kernel_size=3,stride=1,padding=1)
			self.conv_fusion34 = nn.Conv2d(c4//2+c3,c3,kernel_size=3,stride=1,padding=1)
		elif self.model_config['upsample_type'] == 'pixelshuffle':
			self.upsample4 = PixelShuffleUpsample(c4)
			self.upsample5 = PixelShuffleUpsample(c5)
			self.conv_fusion45 = nn.Conv2d(c5//2+c4,c4,kernel_size=3,stride=1,padding=1)
			self.conv_fusion34 = nn.Conv2d(c4//2+c3,c3,kernel_size=3,stride=1,padding=1)
		else:
			raise ValueError(f"Unknown upsample type: {self.model_config['upsample_type']}")

		# Positional Encoding
		pos_enc_type = self.model_config['pos_enc_type']
		# 如果使用 concat，我们不需要投影到特定维度，或者投影到一个较小的维度
		# 这里我们假设直接 concat 原始位置编码，或者投影到一个固定的小维度
		# 为了简单起见，我们这里设置 out_channels=None，让 PositionEncoding2D 返回原始编码
		# 然后在 Head 中调整输入通道数
		self.pos_enc = PositionEncoding2D(pos_enc_type=pos_enc_type, out_channels=None)
		
		# 计算位置编码的通道数
		pos_channels = 0
		if pos_enc_type == 'fourier':
			pos_channels = 4 * 4 # default 4 freqs
		elif pos_enc_type == 'rot_inv':
			pos_channels = 1 + 2 * 4 # default 4 freqs
			
		# 调整 Head 的输入通道数
		head_in_channels = c3 + pos_channels

		# detector
		self.keypoint_dim = self.model_config['keypoint_dim']
		self.keypoint_head = KeypointHead(in_channels=head_in_channels, out_channels=self.keypoint_dim)
		# descriptor
		self.descriptor_dim = self.model_config['descriptor_dim']
		self.descriptor_head = DescriptorHead(in_channels=head_in_channels, out_channels=self.descriptor_dim)
		# geometric features
		self.geo_head = GeoHead(in_channels=head_in_channels, geo_cfg=self.geo_config)
		# # heatmap
		# self.heatmap_head = HeatmapHead(in_channels=head_in_channels,mid_channels=c3,out_channels=1)
		
		self.attn_fusion = FeatureBooster(model_config)

		self.fine_matcher =  nn.Sequential(
								nn.Linear(128, 512),
								nn.BatchNorm1d(512, affine=False),
								nn.ReLU(inplace = True),
								nn.Linear(512, 512),
								nn.BatchNorm1d(512, affine=False),
								nn.ReLU(inplace = True),
								nn.Linear(512, 512),
								nn.BatchNorm1d(512, affine=False),
								nn.ReLU(inplace = True),
								nn.Linear(512, 512),
								nn.BatchNorm1d(512, affine=False),
								nn.ReLU(inplace = True),
								nn.Linear(512, 64),
							)
	
	def fuse_multi_features(self, x3, x4, x5):
		if self.model_config['upsample_type'] == 'HS-FPN':
			# 1. Align Channels
			l3 = self.lat3(x3)
			l4 = self.lat4(x4)
			l5 = self.lat5(x5)
			
			# 2. HFP Enhancement (High Frequency)
			l5 = self.hfp5(l5)
			l4 = self.hfp4(l4)
			l3 = self.hfp3(l3)
			
			# 3. SDP Fusion (Top-Down)
			# Level 5 -> Level 4
			_, _, h5, w5 = l5.size()
			# For HS-FPN mode, we currently use bilinear for internal upsampling or specific if implemented
			# Using bilinear consistently here as per standard HS-FPN or we could add sub-config
			u5 = F.interpolate(l5, size=l4.shape[2:], mode='bilinear', align_corners=False)
			
			p4 = self.sdp4_5(l4, u5, [h5, w5])
			
			# Level 4 -> Level 3
			_, _, h4, w4 = p4.size()
			u4 = F.interpolate(p4, size=l3.shape[2:], mode='bilinear', align_corners=False)
				
			p3 = self.sdp3_4(l3, u4, [h4, w4])
			
			return p3
		else:
			# upsample x5 feature
			x5 = self.upsample5(x5)
			# align spatial size to x4 to avoid cat mismatch when input H/W not divisible by 32
			if x5.shape[-2:] != x4.shape[-2:]:
				x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=False)
			x4 = torch.cat([x4, x5], dim=1)
			x4 = self.conv_fusion45(x4)
			
			# upsample x4 feature
			x4 = self.upsample4(x4)
			# align spatial size to x3 before concat
			if x4.shape[-2:] != x3.shape[-2:]:
				x4 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
			x3 = torch.cat([x3, x4], dim=1)
			x = self.conv_fusion34(x3)
			return x
	
	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws).reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward1(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map
		"""
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)
		
		x3, x4, x5 = self.feature_extract(x)
		
		# features fusion
		x = self.fuse_multi_features(x3,x4,x5)
		
		# Positional Encoding (Concat)
		if self.pos_enc is not None:
			pos_feat = self.pos_enc(x)
			if pos_feat is not None:
				x = torch.cat([x, pos_feat], dim=1)
		
		# keypoint 
		keypoint_map = self.keypoint_head(x)
		# descriptor
		des_map = self.descriptor_head(x)
		# geometry
		geo_map = self.geo_head(x)
		# # heatmap
		# heatmap = self.heatmap_head(x)
	   
		return des_map, geo_map, keypoint_map
	
	def forward2(self, des_map, geo_map, keypoint_map):
		B, C, H, W = des_map.shape
		geo_feat = self._unfold2d(geo_map, ws=8)
		
		# Flatten spatial dimensions while preserving batch size: (B, C, H, W) -> (B, N, C)
		geo_v = geo_feat.flatten(2).transpose(1, 2)
		descs_v = des_map.flatten(2).transpose(1, 2)
		kpts_v = keypoint_map.flatten(2).transpose(1, 2)
		
		# Pass spatial shape (H, W) for Transformer-based attention (Swin)
		descs_refine = self.attn_fusion(descs_v, kpts_v, geo_v, shape=(H, W))
		
		# Flatten to (TotalPoints, C) to match legacy output format
		return descs_refine.reshape(-1, descs_refine.shape[-1])
	
	def forward(self,x):
		des_map, geo_map, keypoint_map = self.forward1(x)
		descs_refine = self.forward2(des_map, geo_map, keypoint_map)
		return descs_refine, des_map, geo_map, keypoint_map
	

if __name__ == "__main__":
	img_path=os.path.join(os.path.dirname(__file__),'../../assert/ref.jpg')
	img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
	if img is None:
		print(f"Failed to load image from {img_path}")
		exit(1)
	# Use safe resolution for Swin (multiple of 160)
	img=cv2.resize(img, (800,640))
	# import pdb;pdb.set_trace()
	img=torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()/255.0
	img=img.cuda() if torch.cuda.is_available() else img
	
	# Dummy model_config for testing
	model_config = {
		"backbone": "Standard",
		"upsample_type": "bilinear",
		"pos_enc_type": "None",
		"keypoint_dim": 65,
		"descriptor_dim": 64,
		"keypoint_encoder": [32, 64],
		"descriptor_encoder": [32, 64],
		"geometric_features": {
			"depth": True,
			"normal": True,
			"gradients": False,
			"curvatures": False
		},
		"depth_dim": 64, # 1 * 8 * 8
		"normal_dim": 192, # 3 * 8 * 8
		"gradient_dim": 128, # 2 * 8 * 8
		"curvature_dim": 320, # 5 * 8 * 8
		"depth_encoder": [64, 64],
		"normal_encoder": [64, 64],
		"gradient_encoder": [64, 64],
		"curvature_encoder": [64, 64],
		"attention_layers": 2,
		"attention_type": "AFT",
		"AFT": {"ffn_type": "positionwiseFFN"},
		"Swin": {"input_resolution": (100, 80), "depth_per_layer": 2, "num_heads": 4, "window_size": 5, "ffn_type": "positionwiseFFN"},
		"last_activation": "None",
		"l2_normalization": False,
		"use_coord_loss": False
	}

	geofeat_sp=GeoFeatModel(model_config=model_config).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	des_map, geo_map, keypoint_map = geofeat_sp.forward1(img)
	des_fine=geofeat_sp.forward2(des_map, geo_map, keypoint_map)
	print(f"Output descriptor shape: {des_fine.shape}")
