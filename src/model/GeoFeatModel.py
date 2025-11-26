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

from .GeometricExtractor import GeometricExtractor
from .GeometricAttention import GeometricAttentionFusion

# from models.model_dfb import LiftFeatModel
# from models.interpolator import InterpolateSparse2d
# from third_party.config import featureboost_config

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
            self.pos_proj = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            nn.init.normal_(self.pos_proj.weight, mean=0.0, std=1e-2)
            if self.pos_proj.bias is not None:
                nn.init.zeros_(self.pos_proj.bias)
        else:
            self.pos_proj = None

    def forward(self, x: torch.Tensor):
        """Given feature map x (B, C, H, W), return positional features projected to out_channels and
        ready to be added to x (shape: B, out_channels, H, W). Returns None if pos_enc_type is 'none'."""
        if self.pos_enc_type == 'None' or self.out_channels is None or self.pos_proj is None:
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
        pos_feat = self.pos_proj(pos)
        return pos_feat

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
		

class DepthHead(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.upsampleDa = UpsampleLayer(in_channels)
		self.upsampleDb = UpsampleLayer(in_channels//2)
		self.upsampleDc = UpsampleLayer(in_channels//4)
		
		self.convDepa = nn.Conv2d(in_channels//2+in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
		self.bnDepa = nn.BatchNorm2d(in_channels//2)
		self.convDepb = nn.Conv2d(in_channels//4+in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1)
		self.bnDepb = nn.BatchNorm2d(in_channels//4)
		# Output 4 channels: 1 for depth, 3 for normal
		self.convDepc = nn.Conv2d(in_channels//8+in_channels//4, 4, kernel_size=3, stride=1, padding=1)
		self.bnDepc = nn.BatchNorm2d(4)
		
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
		
		# Split into depth and normal
		depth = x[:, 0:1, :, :]
		normal = x[:, 1:4, :, :]
		
		# Normalize normal vector
		normal = F.normalize(normal, p=2, dim=1)
		
		# Concatenate back
		x = torch.cat([depth, normal], dim=1)
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

class NormalEncoder(nn.Module):
    """ Encoding of geometric properties using MLP """
    def __init__(self, normal_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP_no_ReLU([normal_dim] + layers + [feature_dim])
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
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        k = k.T
        k = torch.softmax(k, dim=-1)
        k = k.T
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv
        x = self.proj(x)
        if self.use_dropout:
            x = self.dropout(x)
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

# ---------------- Update AttentionalNN to support Swin ----------------
# You can replace your existing AttentionalNN with the extended one below (keeps AFT support)
class AttentionalNN(nn.Module):
    def __init__(self, feature_dim: int, layer_num: int, config: dict, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.config = config
        self.attention_type = config["attention_type"]  # 'AFT' or 'Swin'
        self.layers = nn.ModuleList()
        if self.attention_type == 'AFT':
            for _ in range(layer_num):
                ffn_type = config["AFT"]["ffn_type"] # 'positionwiseFFN' or 'swigluFFN'
                self.layers.append(LinearAttentionalLayer(feature_dim, ffn_type=ffn_type, dropout=dropout, p=p))
        elif self.attention_type == 'Swin':
            # need to know spatial resolution and block depth / heads / window size
            input_resolution = config["Swin"]["input_resolution"]  # (H, W)
            if input_resolution is None:
                raise ValueError("For Swin attention, provide config['Swin']['input_resolution'] = (H, W)")
            depth_per_layer = config["Swin"]["depth_per_layer"]
            num_heads = config["Swin"]["num_heads"]
            window_size = config["Swin"]["window_size"]
            ffn_type = config["Swin"]["ffn_type"] # 'positionwiseFFN' or 'swigluFFN'
            for _ in range(layer_num):
                self.layers.append(SwinAttentionalLayer(feature_dim, input_resolution=input_resolution, depth=depth_per_layer, num_heads=num_heads, window_size=window_size, ffn_type=ffn_type, dropout=dropout, p=p))
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def forward(self, desc: torch.Tensor, shape: Optional[Tuple[int,int]] = None) -> torch.Tensor:
        if self.attention_type == 'AFT':
            for layer in self.layers:
                desc = layer(desc)
            return desc
        elif self.attention_type == 'Swin':
            if shape is None:
                raise ValueError("Swin requires shape=(H,W) argument when calling forward")
            H, W = shape
            B, N, C = desc.shape
            assert N == H * W, f"desc sequence length {N} incompatible with H*W={H*W}"
            for layer in self.layers:
                desc = layer(desc, shape)
            return desc
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


class FeatureBooster(nn.Module):
    def __init__(self, config, dropout=False, p=0.1, use_kenc=True, use_normal=True, use_cross=True):
        super().__init__()
        self.config = config
        self.use_kenc = use_kenc
        self.use_cross = use_cross
        self.use_normal = use_normal

        if use_kenc:
            self.kenc = KeypointEncoder(self.config['keypoint_dim'], self.config['descriptor_dim'], self.config['keypoint_encoder'], dropout=dropout)

        if use_normal:
            self.nenc = NormalEncoder(self.config['normal_dim'], self.config['descriptor_dim'], self.config['normal_encoder'], dropout=dropout)

        if self.config.get('descriptor_encoder', False):
            self.denc = DescriptorEncoder(self.config['descriptor_dim'], self.config['descriptor_encoder'], dropout=dropout)
        else:
            self.denc = None

        if self.use_cross:
            self.attn_proj = AttentionalNN(
                feature_dim=self.config['descriptor_dim'], 
                layer_num=self.config['Attentional_layers'], 
                config=self.config,
                dropout=dropout,
                p=p
            )

        # self.final_proj = nn.Linear(self.config['descriptor_dim'], self.config['output_dim'])

        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

        # self.layer_norm = nn.LayerNorm(self.config['descriptor_dim'], eps=1e-6)

        if self.config['last_activation'] == 'None':
            self.last_activation = None
        elif self.config['last_activation'] == 'relu':
            self.last_activation = nn.ReLU()
        elif self.config['last_activation'] == 'sigmoid':
            self.last_activation = nn.Sigmoid()
        elif self.config['last_activation'] == 'tanh':
            self.last_activation = nn.Tanh()
        else:
            raise Exception('Not supported activation "%s".' % self.config['last_activation'])

    def forward(self, desc, kpts, normals, shape=None):
        # import pdb;pdb.set_trace()
        ## Self boosting
        # Descriptor MLP encoder
        if self.denc is not None:
            desc = self.denc(desc)
        # Geometric MLP encoder
        if self.use_kenc:
            desc = desc + self.kenc(kpts)
            if self.use_dropout:
                desc = self.dropout(desc)

        # 法向量特征 encoder
        if self.use_normal:
            desc = desc + self.nenc(normals)
            if self.use_dropout:
                desc = self.dropout(desc)
        
        ## Cross boosting
        # Multi-layer Transformer network.
        if self.use_cross:
            # desc = self.attn_proj(self.layer_norm(desc))
            desc = self.attn_proj(desc, shape=shape)

        ## Post processing
        # Final MLP projection
        # desc = self.final_proj(desc)
        if self.last_activation is not None:
            desc = self.last_activation(desc)
        # L2 normalization
        if self.config['l2_normalization']:
            desc = F.normalize(desc, dim=-1)

        return desc

################################### fixed #######################################

class GeoFeatModel(nn.Module):
	_MODEL_CONFIG = {
		"backbone": "Standard",
		"pos_enc_type": "None",
		"descriptor_dim": 64,
		"keypoint_encoder": [128,64,64],
		"normal_encoder": [128,64,64],
		"descriptor_encoder": [64,64],
		"Attentional_layers": 3,
		"attention_type": "AFT",
		"AFT": {
			"ffn_type": "swigluFFN"
		},
        "Swin": {
			"input_resolution": [75, 100],
			"depth_per_layer": 2,
			"num_heads": 8,
			"window_size": 5,
			"ffn_type": "swigluFFN"
		},
		"use_geometric_attention": True,
		"last_activation": "None",
		"l2_normalization": "None",
		"keypoint_dim": 65,
		"normal_dim": 192
	}

	def __init__(self, model_config, use_kenc=False, use_normal=True, use_cross=True):
		super().__init__()
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if model_config is not None:
			self.config = model_config
		else:
			self.config = self._MODEL_CONFIG
			
		self.backbone = self.config['backbone']
		self.norm = nn.InstanceNorm2d(1)

		c1,c2,c3,c4,c5 = 24,24,64,64,128
		if self.backbone == "Standard":
			self.feature_extract = StandardBackbone(c1, c2, c3, c4, c5)
		elif self.backbone == "RepVGG":
			self.feature_extract = RepVGGBackbone(c1, c2, c3, c4, c5)
		else:
			raise ValueError(f"Unknown backbone: {self.backbone}")
		
		self.upsample4 = UpsampleLayer(c4)
		self.upsample5 = UpsampleLayer(c5)
		self.conv_fusion45 = nn.Conv2d(c5//2+c4,c4,kernel_size=3,stride=1,padding=1)
		self.conv_fusion34 = nn.Conv2d(c4//2+c3,c3,kernel_size=3,stride=1,padding=1)

		# Positional Encoding
		pos_enc_type = self.config['pos_enc_type']
		self.pos_enc = PositionEncoding2D(pos_enc_type=pos_enc_type, out_channels=c3)

		# detector
		self.keypoint_head = KeypointHead(in_channels=c3,out_channels=65)
		# descriptor
		self.descriptor_dim = 64
		self.descriptor_head = DescriptorHead(in_channels=c3,out_channels=self.descriptor_dim)
		# # heatmap
		# self.heatmap_head = HeatmapHead(in_channels=c3,mid_channels=c3,out_channels=1)
		# depth
		self.depth_head = DepthHead(in_channels=c3)
		
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
		
		# feature_booster
		self.feature_boost = FeatureBooster(self.config, use_kenc=use_kenc, use_cross=use_cross, use_normal=use_normal)

		# Geometric Feature Extraction and Attention
		self.use_geometric_attention = self.config.get("use_geometric_attention", False)
		if self.use_geometric_attention:
			self.geometric_extractor = GeometricExtractor()
			# Input channels for attention: 
			# 1 (depth) + 3 (normal) + 2 (gradients) + 2 (principal curvatures) + 1 (mean) + 1 (gaussian) = 10
			self.geometric_attention = GeometricAttentionFusion(
				feature_dim=self.descriptor_dim,
				geo_input_dim=10
			)
	
	def fuse_multi_features(self,x3,x4,x5):
		# upsample x5 feature
		x5 = self.upsample5(x5)
		if x5.shape[-2:] != x4.shape[-2:]:
			x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=False)
		x4 = torch.cat([x4,x5],dim=1)
		x4 = self.conv_fusion45(x4)
		
		# upsample x4 feature
		x4 = self.upsample4(x4)
		if x4.shape[-2:] != x3.shape[-2:]:
			x4 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
		x3 = torch.cat([x3,x4],dim=1)
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
		
		# Positional Encoding
		pos_feat = self.pos_enc(x)
		if pos_feat is not None:
			x = x + pos_feat
		
		# keypoint 
		keypoint_map = self.keypoint_head(x)
		# descriptor
		des_map = self.descriptor_head(x)
		# # heatmap
		# heatmap = self.heatmap_head(x)
		
		# depth
		d_feats = self.depth_head(x)
	   
		return des_map, keypoint_map, d_feats
		# return des_map, keypoint_map, heatmap, d_feats
	
	def forward2(self, descs, kpts, normals):
		# normals input might be (B, 4, H, W) or (B, 3, H, W)
		if normals.shape[1] == 4:
			depth = normals[:, 0:1, :, :]
			normal_only = normals[:, 1:4, :, :]
		else:
			depth = None
			normal_only = normals

		if hasattr(self, 'use_geometric_attention') and self.use_geometric_attention and depth is not None:
			# Use new geometric attention
			geo_features = self.geometric_extractor(depth, normal_only)
			descs_refine, _ = self.geometric_attention(descs, geo_features)
			# Flatten to (B*H*W, C) to match legacy output format
			descs_refine = descs_refine.permute(0, 2, 3, 1).reshape(-1, descs_refine.shape[1])
			return descs_refine
		else:
			B, C, H, W = descs.shape
			descs_v = descs.permute(0, 2, 3, 1).reshape(-1, C)
			kpts_v = kpts.permute(0, 2, 3, 1).reshape(-1, kpts.shape[1])
			normals_feat = self._unfold2d(normal_only, ws=8)
			normals_v = normals_feat.permute(0, 2, 3, 1).reshape(-1, normals_feat.shape[1])
			descs_refine = self.feature_boost(descs_v, kpts_v, normals_v, shape=(H, W))
			return descs_refine
	
	def forward(self,x):
		M1,K1,D1=self.forward1(x)
		descs_refine=self.forward2(M1,K1,D1)
		return descs_refine,M1,K1,D1
	

if __name__ == "__main__":
	img_path=os.path.join(os.path.dirname(__file__),'../../assert/ref.jpg')
	img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
	if img is None:
		print(f"Failed to load image from {img_path}")
		exit(1)
	img=cv2.resize(img,(800,608))
	# import pdb;pdb.set_trace()
	img=torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()/255.0
	img=img.cuda() if torch.cuda.is_available() else img
	geofeat_sp=GeoFeatModel(model_config="../config/model_config.json").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	des_map, keypoint_map, d_feats=geofeat_sp.forward1(img)
	des_fine=geofeat_sp.forward2(des_map,keypoint_map,d_feats)
	print(des_map.shape)
