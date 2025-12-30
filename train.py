"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
	training script
"""

import argparse
import os
import time
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from src.config.config import get_cfg_defaults

def parse_arguments():
	parser = argparse.ArgumentParser(description="GeoFeat training script.")
	parser.add_argument('--name', type=str, default='GeoFeat', help='set process name')

	# Dataset settings are now loaded from config files
	# MegaDepth and COCO parameters are read from src/config/data/data_config.json

	# training setting
	# 保存权重
	parser.add_argument('--ckpt_save_path', type=str, default='weights',
						help='Path to save the checkpoints.')
	# 训练步数
	parser.add_argument('--n_steps', type=int, default=160_000,
						help='Number of training steps. Default is 160000.')
	# 学习率
	parser.add_argument('--lr', type=float, default=3e-4,
						help='Learning rate. Default is 0.0003.')
	# 学习率衰减
	parser.add_argument('--gamma_steplr', type=float, default=0.5,
						help='Gamma value for StepLR scheduler. Default is 0.5.')
	# 训练尺寸
	parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
						default=(800, 800), help='Training resolution as width,height. Default is (800, 800).')
	# 训练设备
	parser.add_argument('--device_num', type=str, default='cuda',
						help='Device number to use for training. Default is "cuda:0".')
	# 恢复训练
	parser.add_argument('--dry_run', action='store_true',
						help='If set, perform a dry run training with a mini-batch for sanity check.')
	# 每多少步保存权重
	parser.add_argument('--save_ckpt_every', type=int, default=1000,
						help='Save checkpoints every N steps. Default is 1000.')

	# Status 监控参数
	parser.add_argument('--log_interval', type=int, default=1000, help='控制台指标刷新步频(平均)')
	parser.add_argument('--status_interval', type=int, default=0, help='status.txt 写入步频 (0=跟随log_interval)')
	
	# 训练目录管理
	parser.add_argument('--max_runs', type=int, default=30, help='只保留最近 N 次训练目录 (按创建时间)')

	args = parser.parse_args()

	return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
import tqdm
import glob
import json
import sys
import os

from model.GeoFeatModel import GeoFeatModel
from loss.loss import GeoFeatLoss
# from model.interpolator import InterpolateSparse2d
from utils.depth_anything_wrapper import DepthAnythingExtractor
from utils.alike_wrapper import ALikeExtractor

from data.coco_augmentor import COCOAugmentor
from data import coco_wrapper
from src.data.megadepth_new import MegaDepthCleanedDataset
from src.data import megadepth_wrapper

import setproctitle

class Trainer():
	def __init__(self, 
				  data_config, model_config,
				model_name='GeoFeat',
				ckpt_save_path="./weights",
				n_steps=30_000,
				save_ckpt_every=1000,
				lr=3e-4, gamma_steplr=0.5,
				training_res=(800, 640), device_num="cuda:0",
				dry_run=False,
				log_interval=1000,
				status_interval=0,
				max_runs=20
				):
		
		# Store configs
		self.data_config = data_config
		self.model_config = model_config
		
		# Extract dataset parameters from config (assumed validated)
		use_megadepth = bool(self.data_config['use_megadepth'])
		megadepth_batch_size = int(self.data_config['megadepth_batch_size']) if use_megadepth else 0
		megadepth_root_path = self.data_config['megadepth_root_path'] if use_megadepth else None

		use_coco = bool(self.data_config['use_coco'])
		coco_batch_size = int(self.data_config['coco_batch_size']) if use_coco else 0
		coco_root_path = self.data_config['coco_root_path'] if use_coco else None
		
		print(f'MegeDepth: {use_megadepth}-{megadepth_batch_size}')
		print(f'COCO20k: {use_coco}-{coco_batch_size}')

		self.dev = torch.device(device_num if torch.cuda.is_available() else 'cpu')
		print(f'Device: {self.dev}')

		self.use_coord_loss = self.model_config.get('use_coord_loss', False)
		print(f'Coordinate loss: {self.use_coord_loss}')
		
		# training model - Pass model config to GeoFeatModel
		self.net = GeoFeatModel(model_config=self.model_config).to(self.dev)
		
		# Extract geometric config (validated in __main__)
		geo_config = self.model_config['geometric_features']

		self.loss_fn = GeoFeatLoss(
			self.dev, 
			lam_descs=1, 
			lam_fb_descs=2,       # 保持描述子权重
			lam_kpts=2, 
			lam_heatmap=1, 
			lam_fb_coordinates=1, # [重点] 大幅提升坐标回归权重，强迫学习像素级细节
			lam_gradients=10,     # [重点] 放大梯度损失，关注边缘细节
			lam_curvature=10,     # [重点] 放大曲率损失，关注形状细节
			geo_config=geo_config
		)

		# depth-anything model  # 深度估计提取器
		self.depth_net = DepthAnythingExtractor('vits', self.dev, 256)
		# self.depth_net = self.net.geometric_extractor.depth_extractor

		# alike model   # 高效的高精度局部特征提取模型
		self.alike_net = ALikeExtractor('alike-t', self.dev)

		#Setup optimizer # 设置优化器
		self.steps = n_steps
		self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
		# 每 step_size 步 削减 按 gamma 比例, 减少学习率
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10_000, gamma=gamma_steplr)

		##################### COCO INIT ##########################
		self.use_coco = use_coco
		self.coco_batch_size = coco_batch_size
		if self.use_coco:  # 启用COCO
			self.augmentor = COCOAugmentor(
				img_dir=coco_root_path,
				device=self.dev,
				load_dataset=True,
				batch_size=self.coco_batch_size,
				out_resolution=training_res,  # 输出分辨率
				warp_resolution=training_res,  # 几何变换后的输出分辨率
				sides_crop=0.1,  # 边缘裁剪比例
				max_num_imgs=3000,  # 最大训练图像数
				num_test_imgs=5,  # 测试图像数
				photometric=True,  # 启用光度增强
				geometric=True,  # 启用几何增强
				reload_step=4000  # 图像重载步频
			)
		##################### COCO END #######################

		##################### MEGADEPTH INIT ##########################
		self.use_megadepth = use_megadepth
		self.megadepth_batch_size = megadepth_batch_size
		if self.use_megadepth:
			TRAIN_BASE_PATH = f"{megadepth_root_path}/megadepth_indices_new"
			TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

			TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

			npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
			megadepth_dataset = torch.utils.data.ConcatDataset(
				[MegaDepthCleanedDataset(
					root_dir=TRAINVAL_DATA_SOURCE,
					npz_path=path,
					img_resize=training_res) 
				for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")])

			self.megadepth_dataloader = DataLoader(megadepth_dataset, batch_size=megadepth_batch_size, shuffle=True)
			self.megadepth_data_iter = iter(self.megadepth_dataloader)
		##################### MEGADEPTH INIT END #######################

		# 运行目录结构: <ckpt_save_path>/<model_name>_<timestamp>/
		os.makedirs(ckpt_save_path, exist_ok=True)
		stamp = time.strftime('%Y%m%d_%H%M%S')
		self.run_dir = os.path.join(ckpt_save_path, f'{model_name}_{stamp}')
		os.makedirs(self.run_dir, exist_ok=True)
		self.ckpt_dir = self.run_dir  # 后续可细分 ckpts/ 与 logs/
		logdir = os.path.join(self.run_dir, 'logdir')
		os.makedirs(logdir, exist_ok=True)
		
		# 清理旧的训练目录
		self._cleanup_old_runs(ckpt_save_path, model_name, max_runs=max_runs)

		self.dry_run = dry_run
		self.save_ckpt_every = save_ckpt_every
		self.ckpt_save_path = ckpt_save_path
		self.writer = SummaryWriter(logdir)
		self.model_name = model_name
		
		# status 相关初始化
		self._status_every = max(1, status_interval if status_interval > 0 else log_interval)
		self.status_path = os.path.join(self.run_dir, 'status.txt')
		
		# 初始化status文件
		try:
			with open(self.status_path, 'w', encoding='utf-8') as f:
				# extended status (detailed loss/acc columns)
				f.write('step,loss,loss_fb_descs,loss_kpts,loss_normals,loss_depths,loss_gradients,loss_curvature,loss_fb_coordinates,'
						'acc_coarse,acc_coordinates,acc_fb_coarse,acc_fb_coordinates,acc_kpt,'
						'total_pos,lr,forward_cost,skipped\n')
			print(f"Status file initialized: {self.status_path}")
		except Exception as e:
			print(f"Warning: Failed to initialize status file: {e}")

	def _cleanup_old_runs(self, base_dir: str, model_name: str, max_runs: int = 10):
		"""保留最近 max_runs 个以 model_name_ 开头的运行目录, 其余删除."""
		try:
			candidates = []
			for n in os.listdir(base_dir):
				full = os.path.join(base_dir, n)
				if os.path.isdir(full) and n.startswith(model_name + '_'):
					candidates.append((os.path.getctime(full), full))
			candidates.sort(key=lambda x: x[0])  # old -> new
			if len(candidates) > max_runs:
				import shutil
				for _, path in candidates[:len(candidates)-max_runs]:
					try:
						shutil.rmtree(path, ignore_errors=True)
						print(f'[cleanup] removed old run: {os.path.basename(path)}')
					except Exception as e:
						print(f'[cleanup][warn] failed to remove {path}: {e}')
				removed_count = len(candidates) - max_runs
				print(f'[cleanup] Removed {removed_count} old training directories, keeping {max_runs} most recent')
		except Exception as e:
			print(f'[cleanup][warn] cleanup failed: {e}')

	# 创建训练数据
	def generate_train_data(self):
		imgs1_t,imgs2_t=[],[]
		imgs1_np,imgs2_np=[],[]
		# norms0,norms1=[],[]
		positives_coarse=[]
		
		if self.use_coco:
				coco_imgs1, coco_imgs2, H1, H2 = coco_wrapper.make_batch(self.augmentor, 0.1)
				h_coarse, w_coarse = coco_imgs1[0].shape[-2] // 8, coco_imgs1[0].shape[-1] // 8
				_ , positives_coco_coarse = coco_wrapper.get_corresponding_pts(coco_imgs1, coco_imgs2, H1, H2, self.augmentor, h_coarse, w_coarse)
				coco_imgs1=coco_imgs1.mean(1,keepdim=True);coco_imgs2=coco_imgs2.mean(1,keepdim=True)
				imgs1_t.append(coco_imgs1);imgs2_t.append(coco_imgs2)
				positives_coarse += positives_coco_coarse
					
		if self.use_megadepth:
			try:
				megadepth_data=next(self.megadepth_data_iter)
			except StopIteration:
				print('End of MD DATASET')
				self.megadepth_data_iter=iter(self.megadepth_dataloader)
				megadepth_data=next(self.megadepth_data_iter)
			if megadepth_data is not None:
				for k in megadepth_data.keys():
					if isinstance(megadepth_data[k],torch.Tensor):
						megadepth_data[k]=megadepth_data[k].to(self.dev)
				megadepth_imgs1_t,megadepth_imgs2_t=megadepth_data['image0'],megadepth_data['image1']
				megadepth_imgs1_t=megadepth_imgs1_t.mean(1,keepdim=True);megadepth_imgs2_t=megadepth_imgs2_t.mean(1,keepdim=True)
				imgs1_t.append(megadepth_imgs1_t);imgs2_t.append(megadepth_imgs2_t)
				megadepth_imgs1_np,megadepth_imgs2_np=megadepth_data['image0_np'],megadepth_data['image1_np']
				for np_idx in range(megadepth_imgs1_np.shape[0]):
					img1_np,img2_np=megadepth_imgs1_np[np_idx].squeeze(0).cpu().numpy(),megadepth_imgs2_np[np_idx].squeeze(0).cpu().numpy()
					imgs1_np.append(img1_np);imgs2_np.append(img2_np)
				positives_megadepth_coarse=megadepth_wrapper.spvs_coarse(megadepth_data,8)
				positives_coarse += positives_megadepth_coarse
				
		with torch.no_grad():
			imgs1_t=torch.cat(imgs1_t,dim=0)
			imgs2_t=torch.cat(imgs2_t,dim=0)
			
		return imgs1_t,imgs2_t,imgs1_np,imgs2_np,positives_coarse
		
	# 神经网络训练
	def train(self):
		self.net.train()

		attempts = 0
		skipped = 0
		with tqdm.tqdm(total=self.steps) as pbar:
			for i in range(self.steps):
				attempts += 1
				# import pdb;pdb.set_trace()
				imgs1_t, imgs2_t, imgs1_np, imgs2_np, positives_coarse = self.generate_train_data()

				#Check if batch is corrupted with too few correspondences
				# 检查匹配点是否过少
				is_corrupted = False
				for p in positives_coarse:
					# 若匹配点对小于30个, 则认为无效
					if len(p) < 30:
						is_corrupted = True

				if is_corrupted:
					continue    # 跳过当前批次

				# import pdb;pdb.set_trace()
				start=time.perf_counter()
				#Forward pass
				
				# 初始描述子提取(forward1)
				# des_feats: 描述子特征图
				# geo_feats: 几何特征图（按配置启用的几何通道）
				# keypoint_map: 关键点响应图
				des_feats1, geo_feats1, keypoint_map1 = self.net.forward1(imgs1_t)
				des_feats2, geo_feats2, keypoint_map2 = self.net.forward1(imgs2_t)
				
				# 初始描述子特征匹配, 增强描述子匹配
				coordinates, fb_coordinates = [], []
				alike_kpts1, alike_kpts2 = [], []
				DA_normals1, DA_normals2 = [], []
				DA_depths1, DA_depths2 = [], []

				# import pdb;pdb.set_trace()

				fb_feats1, fb_feats2 = [], []
				# 通过COCO_20K数据训练
				for b in range(des_feats1.shape[0]):
					# 特征扁平化
					feat1 = des_feats1[b].permute(1, 2, 0).reshape(-1, des_feats1.shape[1])
					feat2 = des_feats2[b].permute(1, 2, 0).reshape(-1, des_feats2.shape[1])
					
					# 初始描述子特征匹配
					coordinate = self.net.fine_matcher(torch.cat([feat1, feat2], dim=-1))
					coordinates.append(coordinate)
					
					fb_feat1=self.net.forward2(feats1[b].unsqueeze(0),kpts1[b].unsqueeze(0),normals1[b].unsqueeze(0))
					fb_feat2=self.net.forward2(feats2[b].unsqueeze(0),kpts2[b].unsqueeze(0),normals2[b].unsqueeze(0))
					
					fb_coordinate=self.net.fine_matcher(torch.cat([fb_feat1,fb_feat2],dim=-1))
					fb_coordinates.append(fb_coordinate)
					
					fb_feats1.append(fb_feat1.unsqueeze(0))
					fb_feats2.append(fb_feat2.unsqueeze(0))
					
					img1, img2 = imgs1_t[b], imgs2_t[b]
					img1 = img1.permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
					img2 = img2.permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
					# ALIKE 关键点提取
					alike_kpt1 = torch.tensor(self.alike_net.extract_alike_kpts(img1), device=self.dev)
					alike_kpt2 = torch.tensor(self.alike_net.extract_alike_kpts(img2), device=self.dev)
					alike_kpts1.append(alike_kpt1)
					alike_kpts2.append(alike_kpt2)

				# import pdb;pdb.set_trace()
				# 通过MageDepth V2数据训练
				for b in range(len(imgs1_np)):
					megadepth_depth1,megadepth_norm1=self.depth_net.extract(imgs1_np[b])
					megadepth_depth2,megadepth_norm2=self.depth_net.extract(imgs2_np[b])
					DA_normals1.append(megadepth_norm1);DA_normals2.append(megadepth_norm2)
				 
				# import pdb;pdb.set_trace()
				# 特征重组 调整维度为 (B, C, H, W)
				fb_feats1=torch.cat(fb_feats1,dim=0)
				fb_feats2=torch.cat(fb_feats2,dim=0)
				fb_feats1=fb_feats1.reshape(feats1.shape[0],feats1.shape[2],feats1.shape[3],-1).permute(0,3,1,2)
				fb_feats2=fb_feats2.reshape(feats2.shape[0],feats2.shape[2],feats2.shape[3],-1).permute(0,3,1,2)
				
				coordinates=torch.cat(coordinates,dim=0)
				coordinates=coordinates.reshape(feats1.shape[0],feats1.shape[2],feats1.shape[3],-1).permute(0,3,1,2)
				
				fb_coordinates=torch.cat(fb_coordinates,dim=0)
				fb_coordinates=fb_coordinates.reshape(feats1.shape[0],feats1.shape[2],feats1.shape[3],-1).permute(0,3,1,2)
				
				end=time.perf_counter()
				print(f"\n\nforward cost {end-start:.1f} seconds")
				# 损失计算
				loss_items = []

				# import pdb;pdb.set_trace()
				loss_info = self.loss_fn(
					des_feats1, fb_feats1, keypoint_map1, geo_feats1,                 	# 图像1: 描述子 增强描述子 关键点图 几何特征
					des_feats2, fb_feats2, keypoint_map2, geo_feats2,                 	# 图像2: 描述子 增强描述子 关键点图 几何特征
					geo_feats1, geo_feats2,                                     	# 几何特征
					positives_coarse,                                         	# 粗糙匹配点
					coordinates, fb_coordinates,                               	# 精细匹配结果
					alike_kpts1, alike_kpts2,                                  	# ALIKE关键点
					DA_normals1, DA_normals2,                                  	# MegaDepth法向量
					DA_depths1, DA_depths2,                                    	# MegaDepth深度
					self.megadepth_batch_size, self.coco_batch_size)           	# 批大小参数

				loss_descs = loss_info['loss_descs']
				acc_coarse = loss_info['acc_coarse']
				loss_coordinates = loss_info['loss_coordinates']
				acc_coordinates = loss_info['acc_coordinates']
				
				loss_fb_descs = loss_info['loss_fb_descs']
				acc_fb_coarse = loss_info['acc_fb_coarse']
				loss_fb_coordinates = loss_info['loss_fb_coordinates']
				acc_fb_coordinates = loss_info['acc_fb_coordinates']
				
				loss_kpts = loss_info['loss_kpts']
				acc_kpt = loss_info['acc_kpt']
				loss_depths = loss_info['loss_depths']
				loss_normals = loss_info['loss_normals']
				loss_gradients = loss_info['loss_gradients']
				loss_curvature = loss_info['loss_curvature']
				loss_deep_supervision = loss_info.get('loss_deep_supervision', torch.tensor(0.0, device=self.dev))

				loss_items.append(loss_fb_descs.unsqueeze(0))   # 增强特征匹配损失
				loss_items.append(loss_kpts.unsqueeze(0))       # 关键点损失
				
				# Add normal loss if enabled
				if self.model_config.get('geometric_features', {}).get('normal', True):
					loss_items.append(loss_normals.unsqueeze(0))    # 法向量损失

				# Add geometric losses if enabled
				geo_config = self.model_config.get('geometric_features', {})
				if geo_config.get('depth', False):
					loss_items.append(loss_depths.unsqueeze(0))     # 深度损失
				if geo_config.get('gradients', False):
					loss_items.append(loss_gradients.unsqueeze(0))  # 梯度损失
				if geo_config.get('curvatures', False):
					loss_items.append(loss_curvature.unsqueeze(0))  # 曲率损失
				if self.use_coord_loss:
					loss_items.append(loss_fb_coordinates.unsqueeze(0))     # 坐标损失

				# nb_coarse = len(m1)
				# nb_coarse = len(fb_m1)
				loss = torch.cat(loss_items, -1).mean() # 加权平均总损失

				# Compute Backward Pass
				# 反向传播
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)   # 梯度裁剪
				self.opt.step()         # 参数更新
				self.opt.zero_grad()    # 梯度清零
				self.scheduler.step()   # 学习率调整

				# import pdb;pdb.set_trace()
				# 保存权重文件
				global_step = i + 1
				if global_step % self.save_ckpt_every == 0:
					print('saving iter ', global_step)
					ckpt_path = os.path.join(self.ckpt_dir, f'{self.model_name}_step{global_step}.pth')
					torch.save(self.net.state_dict(), ckpt_path)
				
				# 打印训练过程(损失函数)
				pbar.set_description(
						'loss: {:.4f} '
						'loss_descs: {:.3f} acc_coarse: {:.3f} '
						'loss_coordinates: {:.3f} acc_coordinates: {:.3f} '
						'loss_fb_descs: {:.3f} acc_fb_coarse: {:.3f} '
						'loss_fb_coordinates: {:.3f} acc_fb_coordinates: {:.3f} '
						'loss_kpts: {:.3f} acc_kpts: {:.3f} '
						'loss_normals: {:.3f} loss_depths: {:.3f} loss_gradients: {:.3f} loss_curvature: {:.3f} loss_ds: {:.3f}'.format(
						loss.item(),
						loss_descs.item(), acc_coarse,
						loss_coordinates.item(), acc_coordinates,
						loss_fb_descs.item(), acc_fb_coarse,
						loss_fb_coordinates.item(), acc_fb_coordinates,
						loss_kpts.item(), acc_kpt,
						loss_normals.item(), loss_depths.item(), loss_gradients.item(), loss_curvature.item(), loss_deep_supervision.item()
					)
				)
				pbar.update(1)
				# 为什么修改打印训练过程 会该改变训练时间？
				# tqdm 的 update(n) 方法只是进度条的计数器，它会让进度条前进n步。例如：
				# pbar.update(1)：进度条前进 1 步
				# pbar.update(5)：进度条前进 5 步
				# 它不会影响你的训练逻辑，只是显示上的变化。
				# 
				# Log metrics
				# log文件
				self.writer.add_scalar('Loss/total', loss.item(), global_step)
				self.writer.add_scalar('Accuracy/acc_coarse', acc_coarse, global_step)
				self.writer.add_scalar('Accuracy/acc_coordinates', acc_coordinates, global_step)
				self.writer.add_scalar('Accuracy/acc_fb_coarse', acc_fb_coarse, global_step)
				self.writer.add_scalar('Accuracy/acc_fb_coordinates', acc_fb_coordinates, global_step)
				self.writer.add_scalar('Loss/descs', loss_descs.item(), global_step)
				self.writer.add_scalar('Loss/coordinates', loss_coordinates.item(), global_step)
				self.writer.add_scalar('Loss/fb_descs', loss_fb_descs.item(), global_step)
				self.writer.add_scalar('Loss/fb_coordinates', loss_fb_coordinates.item(), global_step)
				self.writer.add_scalar('Loss/kpts', loss_kpts.item(), global_step)
				self.writer.add_scalar('Loss/normals', loss_normals.item(), global_step)
				self.writer.add_scalar('Loss/depths', loss_depths.item(), global_step)
				self.writer.add_scalar('Loss/gradients', loss_gradients.item(), global_step)
				self.writer.add_scalar('Loss/curvature', loss_curvature.item(), global_step)
				self.writer.add_scalar('Loss/deep_supervision', loss_deep_supervision.item(), global_step)
				
				# status.txt 写入
				if global_step % self._status_every == 0:
					try:
						total_pos = sum(len(p) for p in positives_coarse)
						with open(self.status_path, 'a', encoding='utf-8') as f:
							f.write(f"{global_step},{loss.item():.6f},{loss_fb_descs.item():.6f},{loss_kpts.item():.6f},{loss_normals.item():.6f},{loss_depths.item():.6f},{loss_gradients.item():.6f},{loss_curvature.item():.6f},{loss_fb_coordinates.item():.6f},"
									f"{acc_coarse:.4f},{acc_coordinates:.4f},{acc_fb_coarse:.4f},{acc_fb_coordinates:.4f},{acc_kpt:.4f},"
									f"{total_pos},{self.opt.param_groups[0]['lr']:.6e},{end-start:.3f},{skipped}\n")
					except Exception as e:
						print(f"Warning: Failed to write status: {e}")


if __name__ == '__main__':
	setproctitle.setproctitle(args.name)

	# Load configs from config.py (yacs) and adapt to Trainer expectations
	cfg = get_cfg_defaults()

	data_config = {
		'use_megadepth': bool(cfg.DATASET.USE_MEGADEPTH),
		'use_coco': bool(cfg.DATASET.USE_COCO),
		'megadepth_root_path': cfg.DATASET.MEGADEPTH_ROOT_PATH,
		'megadepth_batch_size': int(cfg.DATASET.MEGADEPTH_BATCH_SIZE),
		'coco_root_path': cfg.DATASET.COCO_ROOT_PATH,
		'coco_batch_size': int(cfg.DATASET.COCO_BATCH_SIZE),
	}

	model_config = {
		'backbone': cfg.MODEL.BACKBONE,									# 骨干网络
		'upsample_type': cfg.MODEL.UPSAMPLE_TYPE,						# 上采样类型
		'pos_enc_type': cfg.MODEL.POS_ENC_TYPE,							# 位置编码类型
		# 关键点编码器
		'keypoint_encoder': list(cfg.MODEL.KEYPOINT_ENCODER),			# 关键点编码器
		'keypoint_dim': int(cfg.MODEL.KEYPOINT_DIM),					# 关键点维度
		# 描述子编码器
		'descriptor_encoder': list(cfg.MODEL.DESCRIPTOR_ENCODER),		# 描述子编码器
		'descriptor_dim': int(cfg.MODEL.DESCRIPTOR_DIM),				# 描述子维度
		# 几何特征配置
		'geometric_features': {
			'depth': bool(cfg.MODEL.GEOMETRIC_FEATURES.DEPTH),
			'normal': bool(cfg.MODEL.GEOMETRIC_FEATURES.NORMAL),
			'gradients': bool(cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS),
			'curvatures': bool(cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES),
		},
		# 几何特征编码器
		# 深度编码器
		'depth_encoder': list(cfg.MODEL.DEPTH_ENCODER),
		'depth_dim': int(cfg.MODEL.DEPTH_DIM),
		# 法向量编码器
		'normal_encoder': list(cfg.MODEL.NORMAL_ENCODER),
		'normal_dim': int(cfg.MODEL.NORMAL_DIM),
		# 梯度编码器
		'gradient_encoder': list(cfg.MODEL.GRADIENT_ENCODER),
		'gradient_dim': int(cfg.MODEL.GRADIENT_DIM),
		# 曲率编码器
		'curvature_encoder': list(cfg.MODEL.CURVATURE_ENCODER),
		'curvature_dim': int(cfg.MODEL.CURVATURE_DIM),

		# Swin Transformer 配置
		'swin_transformer': {
			'input_resolution': list(cfg.MODEL.SWIN.INPUT_RESOLUTION),
			'depth_per_layer': int(cfg.MODEL.SWIN.DEPTH_PER_LAYER),
			'num_heads': int(cfg.MODEL.SWIN.NUM_HEADS),
			'window_size': int(cfg.MODEL.SWIN.WINDOW_SIZE),
		},
		# 注意力机制配置
		'attention_layers': int(cfg.MODEL.ATTENTIONAL_LAYERS),
		'attention_type': cfg.MODEL.ATTENTION.TYPE,
		'attention_aft_ffn': cfg.MODEL.ATTENTION.AFT.FFN_TYPE,
		

		# 细化匹配器配置
		'last_activation': cfg.MODEL.LAST_ACTIVATION,
		'l2_normalization': bool(cfg.MODEL.L2_NORMALIZATION),
		'use_coord_loss': bool(cfg.MODEL.USE_COORD_LOSS),

		# 模型输出维度
		'output_dim': int(cfg.MODEL.OUTPUT_DIM),
	}

	# Abort if all datasets are disabled
	if not data_config['use_megadepth'] and not data_config['use_coco']:
		print("Error: Both use_megadepth and use_coco are False. Enable at least one dataset to train.")
		sys.exit(1)

	# Abort if required dataset paths are missing when enabled
	if data_config['use_megadepth'] and not data_config['megadepth_root_path']:
		print("Error: MEGADEPTH_ROOT_PATH is empty in config.py while use_megadepth is True. Abort training.")
		sys.exit(1)
	if data_config['use_coco'] and not data_config['coco_root_path']:
		print("Error: COCO_ROOT_PATH is empty in config.py while use_coco is True. Abort training.")
		sys.exit(1)

	# GeoFeatModel will normalize key casing internally
	if not model_config:
		print("Error: MODEL section in config.py is empty. Abort training.")
		sys.exit(1)

	# Show loaded configs explicitly
	print("data_config:")
	print(json.dumps(data_config, indent=2))
	print("model_config:")
	print(json.dumps(model_config, indent=2))

	trainer = Trainer(
		# Configuration files
		data_config=data_config,
		model_config=model_config,

		# Training parameters from CLI args
		model_name=args.name,
		ckpt_save_path=args.ckpt_save_path,
		n_steps=args.n_steps,
		save_ckpt_every=args.save_ckpt_every,
		lr=args.lr,
		gamma_steplr=args.gamma_steplr,
		training_res=args.training_res,
		device_num=args.device_num,
		dry_run=args.dry_run,
		
		# Status monitoring parameters
		log_interval=args.log_interval,
		status_interval=args.status_interval,

		# Training directory management
		max_runs=args.max_runs
	)

	#The most fun part
	trainer.train()