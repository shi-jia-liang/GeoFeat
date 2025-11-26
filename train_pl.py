"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
	PyTorch Lightning training script
"""

import argparse
import os
import time
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import tqdm
import glob
import setproctitle

from old.GeoFeatModel_raw import GeoFeatModel
from loss.loss import GeoFeatLoss
from model.interpolator import InterpolateSparse2d
from utils.depth_anything_wrapper import DepthAnythingExtractor
from utils.alike_wrapper import ALikeExtractor

from data.coco_augmentor import COCOAugmentor
from data import coco_wrapper
from src.data.megadepth_new import MegaDepthCleanedDataset
from src.data import megadepth_wrapper


def load_config_from_json():
	"""Load data and model configurations from JSON files"""
	print("Loading configuration files...")
	
	# Load data config
	data_config_path = os.path.join(os.path.dirname(__file__), 'src/config/data/data_config.json')
	try:
		with open(data_config_path, 'r', encoding='utf-8') as f:
			full_data_config = json.load(f)
			data_config = full_data_config.get('data', {})
		print(f"✓ Loaded data config from: {data_config_path}")
	except Exception as e:
		print(f"✗ Error: Failed to load data config: {e}")
		sys.exit(1)
	
	# Load model config
	model_config_path = os.path.join(os.path.dirname(__file__), 'src/config/model/model_config.json')
	try:
		with open(model_config_path, 'r', encoding='utf-8') as f:
			full_config = json.load(f)
			model_config = full_config.get('model', {})
		print(f"✓ Loaded model config from: {model_config_path}")
	except Exception as e:
		print(f"✗ Error: Failed to load model config: {e}")
		sys.exit(1)
	
	return data_config, model_config


def parse_arguments():
	parser = argparse.ArgumentParser(description="LiftFeat PyTorch Lightning training script.")
	parser.add_argument('--name', type=str, default='LiftFeat-PL', help='set process name')

	# Training settings
	parser.add_argument('--ckpt_save_path', type=str, default='weights',
	                    help='Path to save the checkpoints.')
	parser.add_argument('--max_steps', type=int, default=160_000,
	                    help='Maximum number of training steps. Default is 160000.')
	parser.add_argument('--max_epochs', type=int, default=-1,
	                    help='Maximum number of training epochs (-1 for unlimited when using max_steps).')
	parser.add_argument('--lr', type=float, default=3e-4,
	                    help='Learning rate. Default is 0.0003.')
	parser.add_argument('--gamma_steplr', type=float, default=0.5,
	                    help='Gamma value for StepLR scheduler. Default is 0.5.')
	parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
	                    default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
	
	# PyTorch Lightning specific
	parser.add_argument('--accelerator', type=str, default='gpu',
	                    help='Accelerator type (gpu, cpu, tpu, etc.).')
	parser.add_argument('--devices', type=int, default=1,
	                    help='Number of devices to use.')
	parser.add_argument('--precision', type=str, default='32',
	                    help='Training precision (32, 16, bf16).')
	
	# Loss function
	parser.add_argument('--use_coord_loss', action='store_true', help='Enable coordinate loss')
	
	# Monitoring
	parser.add_argument('--log_interval', type=int, default=1000, help='Log every N steps')
	parser.add_argument('--save_ckpt_every', type=int, default=1000,
	                    help='Save checkpoints every N steps. Default is 1000.')
	
	# Training directory management
	parser.add_argument('--max_runs', type=int, default=30, help='Keep only N most recent training directories')
	
	# Debug
	parser.add_argument('--dry_run', action='store_true',
	                    help='If set, perform a dry run training with a mini-batch for sanity check.')
	
	# Dataset limits (for debugging)
	parser.add_argument('--npz_limit', type=int, default=None,
	                    help='Limit number of MegaDepth npz files to load (for debugging)')

	args = parser.parse_args()
	return args


class GeoFeatDataModule(pl.LightningDataModule):
	"""PyTorch Lightning DataModule for GeoFeat"""
	
	def __init__(self, data_config, training_res=(800, 608), npz_limit=None):
		super().__init__()
		self.data_config = data_config
		self.training_res = training_res
		self.npz_limit = npz_limit
		
		# Extract dataset parameters from config
		self.use_megadepth = self.data_config.get('use_megadepth', False)
		self.use_coco = self.data_config.get('use_coco', False)
		
		if self.use_megadepth:
			self.megadepth_batch_size = self.data_config.get('megadepth_batch_size', 4)
			self.megadepth_root_path = self.data_config.get('megadepth_root_path', 'D:/DataSets/MegaDepth')
		else:
			self.megadepth_batch_size = 0
			self.megadepth_root_path = None
			
		if self.use_coco:
			self.coco_batch_size = self.data_config.get('coco_batch_size', 4)
			self.coco_root_path = self.data_config.get('coco_root_path', './dataset/coco_20k')
		else:
			self.coco_batch_size = 0
			self.coco_root_path = None
		
		print(f'MegaDepth: {self.use_megadepth}-{self.megadepth_batch_size if self.use_megadepth else 0}')
		print(f'COCO20k: {self.use_coco}-{self.coco_batch_size if self.use_coco else 0}')
	
	def setup(self, stage=None):
		"""Setup datasets"""
		if stage == 'fit' or stage is None:
			# Setup MegaDepth dataset
			if self.use_megadepth:
				TRAIN_BASE_PATH = f"{self.megadepth_root_path}/megadepth_indices_new"
				TRAINVAL_DATA_SOURCE = f"{self.megadepth_root_path}/MegaDepth_v1"
				TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

				npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
				if self.npz_limit is not None and self.npz_limit > 0:
					npz_paths = npz_paths[:self.npz_limit]
					print(f"[DEBUG] Limiting to {len(npz_paths)} npz files")
				
				self.megadepth_dataset = torch.utils.data.ConcatDataset(
					[MegaDepthCleanedDataset(
						root_dir=TRAINVAL_DATA_SOURCE,
						npz_path=path,
						img_resize=self.training_res) 
					for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")])
	
	def train_dataloader(self):
		"""Return training dataloader"""
		if self.use_megadepth:
			return DataLoader(
				self.megadepth_dataset,
				batch_size=self.megadepth_batch_size,
				shuffle=True,
				num_workers=0,  # Set to 0 for Windows compatibility
				pin_memory=True
			)
		return None


class GeoFeatLightningModule(pl.LightningModule):
	"""PyTorch Lightning Module for GeoFeat"""
	
	def __init__(self, 
	             data_config=None,
	             model_config=None,
	             model_name='GeoFeat',
	             lr=3e-4,
	             gamma_steplr=0.5,
	             training_res=(800, 608),
	             use_coord_loss=False,
	             log_interval=100):
		super().__init__()
		
		# Save all hyperparameters automatically (Lightning will save to checkpoint)
		self.save_hyperparameters(ignore=['data_config', 'model_config'])
		
		# Save configs separately for custom handling
		self.save_hyperparameters({'data_config': data_config, 'model_config': model_config})
		
		# Store configs
		self.data_config = data_config or {}
		self.model_config = model_config or {}
		self.model_name = model_name
		self.lr = lr
		self.gamma_steplr = gamma_steplr
		self.training_res = training_res
		self.use_coord_loss = use_coord_loss
		self.log_interval = log_interval
		
		# Extract dataset parameters
		self.use_megadepth = self.data_config.get('use_megadepth', False)
		self.use_coco = self.data_config.get('use_coco', False)
		
		if self.use_megadepth:
			self.megadepth_batch_size = self.data_config.get('megadepth_batch_size', 4)
		else:
			self.megadepth_batch_size = 0
			
		if self.use_coco:
			self.coco_batch_size = self.data_config.get('coco_batch_size', 4)
			self.coco_root_path = self.data_config.get('coco_root_path', './dataset/coco_20k')
		else:
			self.coco_batch_size = 0
			self.coco_root_path = None
		
		# Initialize model components
		self.net = GeoFeatModel(config=self.model_config)
		self.loss_fn = GeoFeatLoss(self.device, lam_descs=1, lam_kpts=2, lam_heatmap=1)
		
		# Initialize auxiliary models (will be moved to device in setup)
		self.depth_net = None
		self.alike_net = None
		self.augmentor = None
		
		# Training metrics
		self.training_step_outputs = []
	
	def setup(self, stage=None):
		"""Setup models on correct device"""
		if stage == 'fit' or stage is None:
			# Move auxiliary models to device
			self.depth_net = DepthAnythingExtractor('vits', self.device, 256)
			self.alike_net = ALikeExtractor('alike-t', self.device)
			
			# Setup COCO augmentor
			if self.use_coco:
				self.augmentor = COCOAugmentor(
					img_dir=self.coco_root_path,
					device=self.device,
					load_dataset=True,
					batch_size=self.coco_batch_size,
					out_resolution=self.training_res,
					warp_resolution=self.training_res,
					sides_crop=0.1,
					max_num_imgs=3000,
					num_test_imgs=5,
					photometric=True,
					geometric=True,
					reload_step=4000
				)
	
	def forward(self, imgs):
		"""Forward pass"""
		return self.net.forward1(imgs)
	
	def generate_train_data(self, batch):
		"""Generate training data (similar to original train.py)"""
		imgs1_t, imgs2_t = [], []
		imgs1_np, imgs2_np = [], []
		positives_coarse = []

		# COCO data
		if self.use_coco and self.augmentor is not None:
			coco_imgs1, coco_imgs2, H1, H2 = coco_wrapper.make_batch(self.augmentor, 0.1)
			h_coarse, w_coarse = coco_imgs1[0].shape[-2] // 8, coco_imgs1[0].shape[-1] // 8
			_, positives_coco_coarse = coco_wrapper.get_corresponding_pts(
				coco_imgs1, coco_imgs2, H1, H2, self.augmentor, h_coarse, w_coarse)
			
			coco_imgs1 = coco_imgs1.mean(1, keepdim=True)
			coco_imgs2 = coco_imgs2.mean(1, keepdim=True)
			imgs1_t.append(coco_imgs1)
			imgs2_t.append(coco_imgs2)
			positives_coarse += positives_coco_coarse

		# MegaDepth data
		if self.use_megadepth and batch is not None:
			megadepth_data = batch
			for k in megadepth_data.keys():
				if isinstance(megadepth_data[k], torch.Tensor):
					megadepth_data[k] = megadepth_data[k].to(self.device)
			
			megadepth_imgs1_t = megadepth_data['image0'].mean(1, keepdim=True)
			megadepth_imgs2_t = megadepth_data['image1'].mean(1, keepdim=True)
			imgs1_t.append(megadepth_imgs1_t)
			imgs2_t.append(megadepth_imgs2_t)
			
			megadepth_imgs1_np = megadepth_data['image0_np']
			megadepth_imgs2_np = megadepth_data['image1_np']
			for np_idx in range(megadepth_imgs1_np.shape[0]):
				img1_np = megadepth_imgs1_np[np_idx].squeeze(0).cpu().numpy()
				img2_np = megadepth_imgs2_np[np_idx].squeeze(0).cpu().numpy()
				imgs1_np.append(img1_np)
				imgs2_np.append(img2_np)
			
			positives_megadepth_coarse = megadepth_wrapper.spvs_coarse(megadepth_data, 8)
			positives_coarse += positives_megadepth_coarse

		with torch.no_grad():
			imgs1_t = torch.cat(imgs1_t, dim=0)
			imgs2_t = torch.cat(imgs2_t, dim=0)

		return imgs1_t, imgs2_t, imgs1_np, imgs2_np, positives_coarse
	
	def training_step(self, batch, batch_idx):
		"""Training step"""
		# Generate training data
		imgs1_t, imgs2_t, imgs1_np, imgs2_np, positives_coarse = self.generate_train_data(batch)
		
		# Check if batch is corrupted
		is_corrupted = False
		for p in positives_coarse:
			if len(p) < 30:
				is_corrupted = True
				break
		
		if is_corrupted:
			# Return zero loss for corrupted batch
			return torch.tensor(0.0, requires_grad=True, device=self.device)
		
		# Forward pass
		feats1, kpts1, normals1 = self.net.forward1(imgs1_t)
		feats2, kpts2, normals2 = self.net.forward1(imgs2_t)
		
		# Feature matching
		coordinates, fb_coordinates = [], []
		alike_kpts1, alike_kpts2 = [], []
		DA_normals1, DA_normals2 = [], []
		fb_feats1, fb_feats2 = [], []
		
		# Process each batch item
		for b in range(feats1.shape[0]):
			feat1 = feats1[b].permute(1, 2, 0).reshape(-1, feats1.shape[1])
			feat2 = feats2[b].permute(1, 2, 0).reshape(-1, feats2.shape[1])
			
			coordinate = self.net.fine_matcher(torch.cat([feat1, feat2], dim=-1))
			coordinates.append(coordinate)
			
			fb_feat1 = self.net.forward2(feats1[b].unsqueeze(0), kpts1[b].unsqueeze(0), normals1[b].unsqueeze(0))
			fb_feat2 = self.net.forward2(feats2[b].unsqueeze(0), kpts2[b].unsqueeze(0), normals2[b].unsqueeze(0))
			fb_coordinate = self.net.fine_matcher(torch.cat([fb_feat1, fb_feat2], dim=-1))
			fb_coordinates.append(fb_coordinate)
			
			fb_feats1.append(fb_feat1.unsqueeze(0))
			fb_feats2.append(fb_feat2.unsqueeze(0))
			
			# ALIKE keypoints
			img1 = imgs1_t[b].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
			img2 = imgs2_t[b].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
			alike_kpt1 = torch.tensor(self.alike_net.extract_alike_kpts(img1), device=self.device)
			alike_kpt2 = torch.tensor(self.alike_net.extract_alike_kpts(img2), device=self.device)
			alike_kpts1.append(alike_kpt1)
			alike_kpts2.append(alike_kpt2)
		
		# Process MegaDepth normals
		for b in range(len(imgs1_np)):
			megadepth_depth1, megadepth_norm1 = self.depth_net.extract(imgs1_np[b])
			megadepth_depth2, megadepth_norm2 = self.depth_net.extract(imgs2_np[b])
			DA_normals1.append(megadepth_norm1)
			DA_normals2.append(megadepth_norm2)
		
		# Reshape features
		fb_feats1 = torch.cat(fb_feats1, dim=0)
		fb_feats2 = torch.cat(fb_feats2, dim=0)
		fb_feats1 = fb_feats1.reshape(feats1.shape[0], feats1.shape[2], feats1.shape[3], -1).permute(0, 3, 1, 2)
		fb_feats2 = fb_feats2.reshape(feats2.shape[0], feats2.shape[2], feats2.shape[3], -1).permute(0, 3, 1, 2)
		
		coordinates = torch.cat(coordinates, dim=0)
		coordinates = coordinates.reshape(feats1.shape[0], feats1.shape[2], feats1.shape[3], -1).permute(0, 3, 1, 2)
		
		fb_coordinates = torch.cat(fb_coordinates, dim=0)
		fb_coordinates = fb_coordinates.reshape(feats1.shape[0], feats1.shape[2], feats1.shape[3], -1).permute(0, 3, 1, 2)
		
		# Calculate loss
		loss_info = self.loss_fn(
			feats1, fb_feats1, kpts1, normals1,
			feats2, fb_feats2, kpts2, normals2,
			positives_coarse,
			coordinates, fb_coordinates,
			alike_kpts1, alike_kpts2,
			DA_normals1, DA_normals2,
			self.megadepth_batch_size, self.coco_batch_size
		)
		
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
		loss_normals = loss_info['loss_normals']
		
		# Combine losses
		loss_items = []
		loss_items.append(loss_fb_descs.unsqueeze(0))
		loss_items.append(loss_kpts.unsqueeze(0))
		loss_items.append(loss_normals.unsqueeze(0))
		
		if self.use_coord_loss:
			loss_items.append(loss_fb_coordinates.unsqueeze(0))
		
		loss = torch.cat(loss_items, -1).mean()
		
		# Log metrics
		self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=True)
		self.log('loss_descs', loss_descs, on_step=True, on_epoch=True)
		self.log('loss_fb_descs', loss_fb_descs, on_step=True, on_epoch=True)
		self.log('loss_kpts', loss_kpts, on_step=True, on_epoch=True)
		self.log('loss_normals', loss_normals, on_step=True, on_epoch=True)
		self.log('loss_coordinates', loss_coordinates, on_step=True, on_epoch=True)
		self.log('loss_fb_coordinates', loss_fb_coordinates, on_step=True, on_epoch=True)
		
		self.log('acc_coarse', acc_coarse, prog_bar=True, on_step=True, on_epoch=True)
		self.log('acc_coordinates', acc_coordinates, on_step=True, on_epoch=True)
		self.log('acc_fb_coarse', acc_fb_coarse, on_step=True, on_epoch=True)
		self.log('acc_fb_coordinates', acc_fb_coordinates, on_step=True, on_epoch=True)
		self.log('acc_kpt', acc_kpt, on_step=True, on_epoch=True)
		
		# Store for epoch end
		self.training_step_outputs.append({
			'loss': loss.detach(),
			'acc_coarse': acc_coarse,
			'acc_fb_coarse': acc_fb_coarse
		})
		
		return loss
	
	def on_train_epoch_end(self):
		"""Called at the end of training epoch"""
		# Clear stored outputs
		self.training_step_outputs.clear()
	
	def configure_optimizers(self):
		"""Configure optimizers and schedulers"""
		optimizer = optim.Adam(
			filter(lambda x: x.requires_grad, self.net.parameters()),
			lr=self.lr
		)
		
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=10_000,
			gamma=self.gamma_steplr
		)
		
		return {
			'optimizer': optimizer,
			'lr_scheduler': {
				'scheduler': scheduler,
				'interval': 'step',
				'frequency': 1
			}
		}


def cleanup_old_runs(base_dir: str, model_name: str, max_runs: int = 10):
	"""Keep only the most recent max_runs training directories"""
	try:
		candidates = []
		for n in os.listdir(base_dir):
			full = os.path.join(base_dir, n)
			if os.path.isdir(full) and n.startswith(model_name + '_'):
				candidates.append((os.path.getctime(full), full))
		candidates.sort(key=lambda x: x[0])
		
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


if __name__ == '__main__':
	args = parse_arguments()
	setproctitle.setproctitle(args.name)
	
	# Set matmul precision for better performance on modern GPUs
	torch.set_float32_matmul_precision('high')
	
	# Load configurations
	data_config, model_config = load_config_from_json()
	
	# Create run directory
	os.makedirs(args.ckpt_save_path, exist_ok=True)
	stamp = time.strftime('%Y%m%d_%H%M%S')
	run_dir = os.path.join(args.ckpt_save_path, f'{args.name}_{stamp}')
	os.makedirs(run_dir, exist_ok=True)
	
	# Cleanup old runs
	cleanup_old_runs(args.ckpt_save_path, args.name, max_runs=args.max_runs)
	
	# Initialize DataModule
	data_module = GeoFeatDataModule(
		data_config=data_config,
		training_res=args.training_res,
		npz_limit=args.npz_limit
	)
	
	# Initialize LightningModule
	model = GeoFeatLightningModule(
		data_config=data_config,
		model_config=model_config,
		model_name=args.name,
		lr=args.lr,
		gamma_steplr=args.gamma_steplr,
		training_res=args.training_res,
		use_coord_loss=args.use_coord_loss,
		log_interval=args.log_interval
	)
	
	# Setup callbacks
	checkpoint_callback = ModelCheckpoint(
		dirpath=run_dir,
		filename=f'{args.name}_{{step}}',
		save_top_k=-1,  # Save all checkpoints
		every_n_train_steps=args.save_ckpt_every,
		save_last=True
	)
	
	lr_monitor = LearningRateMonitor(logging_interval='step')
	
	# Setup logger
	logger = TensorBoardLogger(
		save_dir=run_dir,
		name='logdir'
	)
	
	# Initialize Trainer
	trainer = pl.Trainer(
		max_steps=args.max_steps,
		max_epochs=args.max_epochs,
		accelerator=args.accelerator,
		devices=args.devices,
		precision=args.precision,
		callbacks=[checkpoint_callback, lr_monitor],
		logger=logger,
		log_every_n_steps=args.log_interval,
		gradient_clip_val=1.0,
		accumulate_grad_batches=1,
		fast_dev_run=args.dry_run,
		enable_progress_bar=True,
		enable_model_summary=True
	)
	
	# Train
	print(f"Starting training...")
	print(f"Run directory: {run_dir}")
	print(f"Coordinate loss: {args.use_coord_loss}")
	
	trainer.fit(model, data_module)
	
	print("Training completed!")
