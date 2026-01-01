import math
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project modules importable when this file is used as a script.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.model.GeoFeatModel import GeoFeatModel  # noqa: E402
from src.model.interpolator import InterpolateSparse2d  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model: nn.Module, weight_path: Optional[str]) -> nn.Module:
	"""Load weights if a valid path is provided."""
	if not weight_path:
		print("Warning: weight_path is None, using randomly initialized GeoFeat.")
		return model

	if not os.path.isfile(weight_path):
		print(f"Warning: weight file not found: {weight_path}, using randomly initialized GeoFeat.")
		return model

	state = torch.load(weight_path, map_location="cpu")
	if isinstance(state, dict):
		if "model_state_dict" in state:
			state = state["model_state_dict"]
		elif "state_dict" in state:
			state = state["state_dict"]

	# Strip potential DataParallel prefixes for compatibility
	if any(k.startswith("module.") for k in state.keys()):
		state = {k.replace("module.", "", 1): v for k, v in state.items()}

	model_sd = model.state_dict()

	# Drop keys with shape mismatches to allow loading older checkpoints.
	filtered_state = {}
	dropped = []
	for k, v in state.items():
		if k not in model_sd:
			continue
		if model_sd[k].shape != v.shape:
			dropped.append(k)
			continue
		filtered_state[k] = v

	# Relaxed load: ignore missing/unexpected keys quietly to avoid noisy warnings
	if dropped:
		print(f"Warning: dropped {len(dropped)} keys due to shape mismatch: {dropped}")

	model.load_state_dict(filtered_state, strict=False)

	# Optional concise summary (comment out detailed lists to avoid clutter)
	model_keys = set(model_sd.keys())
	weight_keys = set(filtered_state.keys())
	missing = model_keys - weight_keys
	if missing:
		print(f"Warning: missing {len(missing)} keys in GeoFeat weights (suppressed list, enable debug if needed)")

	return model


class NonMaxSuppression(nn.Module):
	def __init__(self, rep_thr: float = 0.1, top_k: int = 4096):
		super().__init__()
		self.rep_thr = rep_thr
		self.top_k = top_k

	def forward(self, score: torch.Tensor) -> torch.Tensor:
		"""Return keypoint indices (B, N, 2) after NMS on a heatmap."""
		B, _, H, W = score.shape
		local_max = F.max_pool2d(score, kernel_size=5, stride=1, padding=2)
		pos_mask = (score == local_max) & (score > self.rep_thr)

		pos_list = []
		for b in range(B):
			coords = pos_mask[b].nonzero(as_tuple=False)[..., 1:].flip(-1)  # (N, 2) -> (x, y)
			if coords.numel() == 0:
				pos_list.append(coords)
				continue

			# Rank by score and keep top_k if requested
			scores_b = score[b, 0, coords[:, 1], coords[:, 0]]
			if self.top_k and scores_b.numel() > self.top_k:
				topk_idx = torch.topk(scores_b, self.top_k).indices
				coords = coords[topk_idx]
			pos_list.append(coords)

		max_len = max((len(c) for c in pos_list), default=0)
		out = torch.zeros((B, max_len, 2), dtype=torch.long, device=score.device)
		for b, coords in enumerate(pos_list):
			if coords.numel() == 0:
				continue
			out[b, : len(coords)] = coords
		return out


class GeoFeat:
	def __init__(
		self,
		model_config: Optional[dict] = None,
		weight_path: Optional[str] = None,
		top_k: int = 4096,
		detect_threshold: float = 0.1,
	):
		self.device = device
		self.top_k = top_k
		self.detect_threshold = detect_threshold

		self.net = GeoFeatModel(model_config).to(self.device).eval()
		self.net = load_model(self.net, weight_path)

		self.detector = NonMaxSuppression(rep_thr=detect_threshold, top_k=top_k).to(self.device)
		self.sampler = InterpolateSparse2d("bicubic").to(self.device)

	@staticmethod
	def _pad_size(length: int, divisor: int = 32) -> Tuple[int, int]:
		padded = math.ceil(length / divisor) * divisor
		return padded, padded - length

	def image_preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, list]:
		"""Pad to a multiple of 32 and convert to tensor on the correct device."""
		if image.ndim == 2:
			H, W = image.shape
		else:
			H, W, _ = image.shape

		_H, pad_h = self._pad_size(H)
		_W, pad_w = self._pad_size(W)

		if image.ndim == 2:
			image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, None, 0)
			image = image[None, ...]  # (1, H, W)
		else:
			image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, None, (0, 0, 0))
			image = image.transpose(2, 0, 1)  # (C, H, W)

		pad_info = [0, pad_h, 0, pad_w]
		tensor = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
		return tensor, pad_info

	@torch.inference_mode()
	def extract(self, image: np.ndarray) -> dict:
		image, pad_info = self.image_preprocess(image)
		_, _, _H, _W = image.shape

		descs_map, geo_feats, kpt_logits = self.net.forward1(image)
		descs_refined = self.net.forward2(descs_map, geo_feats, kpt_logits)

		descs_map = descs_refined.reshape(descs_map.shape[0], descs_map.shape[2], descs_map.shape[3], -1).permute(0, 3, 1, 2)
		descs_map = F.normalize(descs_map, p=2, dim=1)

		scores = F.softmax(kpt_logits, dim=1)[:, :64]
		heatmap = scores.permute(0, 2, 3, 1).reshape(scores.shape[0], scores.shape[2], scores.shape[3], 8, 8)
		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(scores.shape[0], 1, scores.shape[2] * 8, scores.shape[3] * 8)

		pos = self.detector(heatmap)
		kpts = pos.squeeze(0)
		mask_w = kpts[..., 0] < (_W - pad_info[-1])
		kpts = kpts[mask_w]
		mask_h = kpts[..., 1] < (_H - pad_info[1])
		kpts = kpts[mask_h]

		if kpts.numel() == 0:
			empty_desc = torch.empty((0, descs_map.shape[1]), device=self.device)
			empty_scores = torch.empty((0,), device=self.device)
			return {"descriptors": empty_desc, "keypoints": kpts, "scores": empty_scores}

		scores = self.sampler(heatmap, kpts.unsqueeze(0), _H, _W).squeeze(0).reshape(-1)
		descs = self.sampler(descs_map, kpts.unsqueeze(0), _H, _W).squeeze(0)
		descs = F.normalize(descs, p=2, dim=1)

		if self.top_k and scores.numel() > self.top_k:
			topk = torch.topk(scores, self.top_k)
			scores = topk.values
			kpts = kpts[topk.indices]
			descs = descs[topk.indices]

		return {"descriptors": descs, "keypoints": kpts, "scores": scores}

	def match_featnet(self, img1: np.ndarray, img2: np.ndarray, min_cossim: float = -1) -> Tuple[np.ndarray, np.ndarray]:
		data1 = self.extract(img1)
		data2 = self.extract(img2)

		kpts1, feats1 = data1["keypoints"], data1["descriptors"]
		kpts2, feats2 = data2["keypoints"], data2["descriptors"]

		if kpts1.numel() == 0 or kpts2.numel() == 0:
			return np.empty((0, 2)), np.empty((0, 2))

		def nn_argmax(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
			"""Return argmax indices in b for each row in a using chunked matmul."""
			n = a.shape[0]
			out = torch.empty(n, dtype=torch.long, device=a.device)
			for i in range(0, n, chunk):
				j = min(i + chunk, n)
				sims = a[i:j] @ b.t()
				_, idx = sims.max(dim=1)
				out[i:j] = idx
			return out

		match12 = nn_argmax(feats1, feats2, chunk=1024)
		match21 = nn_argmax(feats2, feats1, chunk=1024)

		idx0 = torch.arange(len(match12), device=self.device)
		mutual = match21[match12] == idx0

		if min_cossim > 0:
			def max_scores(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
				"""Return per-row max similarity using chunked matmul."""
				n = a.shape[0]
				vals = torch.empty(n, dtype=a.dtype, device=a.device)
				for i in range(0, n, chunk):
					j = min(i + chunk, n)
					sims = a[i:j] @ b.t()
					m, _ = sims.max(dim=1)
					vals[i:j] = m
				return vals

			cvals = max_scores(feats1, feats2, chunk=1024)
			good = cvals > min_cossim
			idx0 = idx0[mutual & good]
			idx1 = match12[mutual & good]
		else:
			idx0 = idx0[mutual]
			idx1 = match12[mutual]

		mkpts1 = kpts1[idx0].cpu().numpy()
		mkpts2 = kpts2[idx1].cpu().numpy()
		return mkpts1, mkpts2
