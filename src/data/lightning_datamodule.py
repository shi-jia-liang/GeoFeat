"""
GeoFeat Lightning DataModule

This module defines a PyTorch Lightning DataModule for the MegaDepth dataset
used in the GeoFeat training pipeline.

Features:
- Configurable dataset loading from data_config.json
- Support for MegaDepth and COCO20k datasets
- Automatic batch composition from multiple datasets
- Compatible with PyTorch Lightning training loops
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset
import glob
import tqdm
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union


class GeoFeatDataModule(L.LightningDataModule):
    """
    Lightning DataModule for GeoFeat training.
    
    Handles loading of MegaDepth and COCO datasets with configurable parameters.
    Supports dynamic batch composition from multiple data sources.
    
    Args:
        config_path (str): Path to data_config.json
        megadepth_root_path (str): Root path to MegaDepth dataset
        coco_root_path (str): Root path to COCO20k dataset
        training_res (tuple): Training resolution (width, height)
        megadepth_batch_size (int): Batch size for MegaDepth dataset
        coco_batch_size (int): Batch size for COCO dataset
        num_workers (int): Number of data loading workers
        use_megadepth (bool): Whether to use MegaDepth dataset
        use_coco (bool): Whether to use COCO dataset
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        megadepth_root_path: Optional[str] = None,
        coco_root_path: Optional[str] = None,
        training_res: tuple = (800, 608),
        megadepth_batch_size: int = 6,
        coco_batch_size: int = 4,
        num_workers: int = 10,
        use_megadepth: bool = True,
        use_coco: bool = True,
    ):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Dataset configuration
        self.megadepth_root_path = megadepth_root_path or self.config.get('megadepth_root_path')
        self.coco_root_path = coco_root_path or self.config.get('coco_root_path')
        self.training_res = training_res
        
        # Batch size configuration
        self.megadepth_batch_size = megadepth_batch_size
        self.coco_batch_size = coco_batch_size
        self.num_workers = num_workers
        
        # Dataset flags
        self.use_megadepth = use_megadepth
        self.use_coco = use_coco
        
        # Dataset and dataloader placeholders
        self.megadepth_dataset = None
        self.megadepth_dataloader = None
        self.megadepth_data_iter = None
        
        print(f"[DataModule] MegaDepth: {self.use_megadepth}")
        print(f"[DataModule] COCO20k: {self.use_coco}")
        print(f"[DataModule] Training Resolution: {self.training_res}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'config', 'data', 'data_config.json'
            )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"[DataModule] Config loaded from: {config_path}")
                return config.get('data', {})
            except Exception as e:
                print(f"[DataModule] Warning: Failed to load config from {config_path}: {e}")
        
        return {}
    
    def setup(self, stage: str = "fit"):
        """
        Setup datasets.
        
        Args:
            stage (str): 'fit' for training/validation, 'test' for testing
        """
        if stage == "fit":
            # Import dataset classes here to avoid circular imports
            from src.data.megadepth_new import MegaDepthCleanedDataset
            
            if self.use_megadepth:
                self._setup_megadepth(MegaDepthCleanedDataset)
    
    def _setup_megadepth(self, DatasetClass):
        """Setup MegaDepth dataset."""
        if not self.megadepth_root_path:
            print("[DataModule] MegaDepth root path not provided, skipping MegaDepth setup")
            return
        
        try:
            TRAIN_BASE_PATH = os.path.join(
                self.megadepth_root_path, 
                "megadepth_indices_new"
            )
            TRAINVAL_DATA_SOURCE = os.path.join(
                self.megadepth_root_path, 
                "MegaDepth_v1"
            )
            TRAIN_NPZ_ROOT = os.path.join(
                TRAIN_BASE_PATH, 
                "scene_info_0.1_0.7"
            )
            
            npz_paths = glob.glob(os.path.join(TRAIN_NPZ_ROOT, '*.npz'))[:]
            
            if not npz_paths:
                print(f"[DataModule] No NPZ files found in {TRAIN_NPZ_ROOT}")
                return
            
            datasets = []
            for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata"):
                try:
                    dataset = DatasetClass(
                        root_dir=TRAINVAL_DATA_SOURCE,
                        npz_path=path,
                        img_resize=self.training_res
                    )
                    datasets.append(dataset)
                except Exception as e:
                    print(f"[DataModule] Warning: Failed to load dataset from {path}: {e}")
            
            if datasets:
                self.megadepth_dataset = ConcatDataset(datasets)
                print(f"[DataModule] MegaDepth dataset loaded with {len(self.megadepth_dataset)} samples")
            else:
                print("[DataModule] No MegaDepth datasets were successfully loaded")
                
        except Exception as e:
            print(f"[DataModule] Error setting up MegaDepth: {e}")
    
    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Get training dataloader.
        
        Returns:
            DataLoader: Combined dataloader for training
        """
        if self.megadepth_dataset is None:
            raise RuntimeError("MegaDepth dataset not initialized. Call setup() first.")
        
        dataloader = DataLoader(
            self.megadepth_dataset,  # type: ignore
            batch_size=self.megadepth_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        self.megadepth_data_iter = iter(dataloader)
        return dataloader
    
    def get_next_batch(self, device: Optional[torch.device] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get next batch from the dataloader.
        
        This method handles dataset cycling and moves data to the specified device.
        
        Args:
            device (torch.device): Device to move data to
            
        Returns:
            Dict or None: Next batch data or None if no data available
        """
        if self.megadepth_data_iter is None:
            if self.megadepth_dataset is None:
                return None
            self.megadepth_data_iter = iter(DataLoader(
                self.megadepth_dataset,
                batch_size=self.megadepth_batch_size,
                shuffle=True,
                num_workers=self.num_workers
            ))
        
        try:
            batch = next(self.megadepth_data_iter)
        except StopIteration:
            print("[DataModule] End of dataset reached, restarting...")
            self.megadepth_data_iter = iter(DataLoader(
                self.megadepth_dataset,
                batch_size=self.megadepth_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False
            ))
            batch = next(self.megadepth_data_iter)
        
        # Move batch to device
        if device is not None:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        
        return batch
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader (optional)."""
        return None
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader (optional)."""
        return None
    
    @staticmethod
    def get_batch_items():
        """
        Get expected batch items from MegaDepth dataset.
        
        Returns:
            list: List of expected keys in batch dictionary
        """
        return [
            'image0',           # First image tensor
            'image1',           # Second image tensor
            'image0_np',        # First image numpy (for feature extraction)
            'image1_np',        # Second image numpy (for feature extraction)
            'depth0',           # Depth map for first image
            'depth1',           # Depth map for second image
            'T_0to1',           # Transformation matrix from image0 to image1
            'T_1to0',           # Transformation matrix from image1 to image0
            'K0',               # Camera intrinsics for first image
            'K1',               # Camera intrinsics for second image
            'scale0',           # Scale factor for first image
            'scale1',           # Scale factor for second image
        ]


class CombinedDataModule(L.LightningDataModule):
    """
    Combined Lightning DataModule for MegaDepth and COCO datasets.
    
    This module handles dynamic batch composition from multiple datasets during training.
    It supports interleaved batch loading from MegaDepth and COCO sources.
    
    Args:
        megadepth_config (dict): Configuration for MegaDepth dataset
        coco_config (dict): Configuration for COCO dataset
        training_res (tuple): Training resolution (width, height)
        num_workers (int): Number of data loading workers
    """
    
    def __init__(
        self,
        megadepth_config: Optional[Dict[str, Any]] = None,
        coco_config: Optional[Dict[str, Any]] = None,
        training_res: tuple = (800, 608),
        num_workers: int = 4,
    ):
        super().__init__()
        
        self.megadepth_config = megadepth_config or {}
        self.coco_config = coco_config or {}
        self.training_res = training_res
        self.num_workers = num_workers
        
        # Data iterators for dynamic batch composition
        self.megadepth_iter = None
        self.coco_iter = None
    
    def get_mixed_batch(
        self, 
        device: Optional[torch.device] = None,
        megadepth_ratio: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get a mixed batch from both MegaDepth and COCO datasets.
        
        Args:
            device (torch.device): Device to move data to
            megadepth_ratio (float): Ratio of MegaDepth samples in mixed batch (0.0-1.0)
            
        Returns:
            dict: Mixed batch containing data from both sources
        """
        # This is a placeholder for mixed batch creation
        # Implement based on your specific needs
        raise NotImplementedError("get_mixed_batch should be implemented based on your data pipeline")


if __name__ == "__main__":
    # Example usage
    datamodule = GeoFeatDataModule(
        megadepth_root_path="D:/DataSets/MegaDepth",
        coco_root_path="./dataset/coco_20k",
        training_res=(800, 608),
        megadepth_batch_size=6,
        coco_batch_size=4,
        use_megadepth=True,
        use_coco=False,
    )
    
    # Setup datasets
    datamodule.setup(stage="fit")
    
    # Get dataloader
    train_loader = datamodule.train_dataloader()
    
    # Test batch loading
    if train_loader is not None:
        for batch in train_loader:
            print("Batch keys:", batch.keys())
            print("Image0 shape:", batch['image0'].shape)
            print("Image1 shape:", batch['image1'].shape)
            break
