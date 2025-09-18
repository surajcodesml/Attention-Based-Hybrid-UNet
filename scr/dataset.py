"""
Dataset module for Attention-Based Hybrid U-Net OCT Layer Segmentation
This module handles data loading, preprocessing, augmentation, and mask generation.
"""

import os
import yaml
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import exposure


class OCTDataset(Dataset):
    """
    PyTorch Dataset for OCT layer segmentation.
    
    Handles loading HDF5 data, preprocessing, augmentation, and mask generation
    for retinal layer segmentation using ILM and BM annotations.
    
    For training mode, augmented versions are added to increase dataset size.
    """
    
    def __init__(
        self, 
        config_path: str, 
        mode: str = 'train',
        indices: Optional[np.ndarray] = None
    ):
        """
        Initialize the OCT dataset.
        
        Args:
            config_path: Path to the YAML configuration file
            mode: 'train', 'val', or 'test' - determines augmentation usage
            indices: Optional array of indices to use (for train/val split)
        """
        self.mode = mode
        self.config = self._load_config(config_path)
        
        # Load data
        self.images, self.ilm_coords, self.bm_coords = self._load_hdf5_data()
        
        # Apply indices if provided (for train/val split)
        if indices is not None:
            self.images = self.images[indices]
            self.ilm_coords = self.ilm_coords[indices]
            self.bm_coords = self.bm_coords[indices]
        
        # Setup augmentation pipelines
        self._setup_augmentations()
        
        print(f"Loaded {len(self.images)} samples for {mode} mode")
        print(f"Image shape: {self.images.shape}")
        print(f"Target size: {self.config['dataset']['target_size']}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_hdf5_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load images and layer annotations from HDF5 file.
        
        Returns:
            Tuple of (images, ilm_coordinates, bm_coordinates)
        """
        hdf5_path = self.config['dataset']['hdf5_path']
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load images
            images = f['images'][:]  # Shape: (N, 496, 768)
            
            # Load layer annotations
            ilm_coords = f['layers']['ILM'][:]  # Shape: (N, 768)
            bm_coords = f['layers']['BM'][:]    # Shape: (N, 768)
        
        print(f"Loaded data from {hdf5_path}")
        print(f"Original image shape: {images.shape}")
        print(f"ILM coordinates shape: {ilm_coords.shape}")
        print(f"BM coordinates shape: {bm_coords.shape}")
        
        return images, ilm_coords, bm_coords
    
    def _resize_image_and_coords(
        self, 
        image: np.ndarray, 
        ilm_coords: np.ndarray, 
        bm_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize image and adjust coordinates from (496, 768) to (480, 768).
        
        Args:
            image: Input image of shape (496, 768)
            ilm_coords: ILM y-coordinates for each x position
            bm_coords: BM y-coordinates for each x position
            
        Returns:
            Tuple of (resized_image, adjusted_ilm_coords, adjusted_bm_coords)
        """
        original_height = self.config['dataset']['original_size']['height']
        target_height = self.config['dataset']['target_size']['height']
        target_width = self.config['dataset']['target_size']['width']
        
        # Resize image
        resized_image = cv2.resize(
            image, 
            (target_width, target_height), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Scale coordinates proportionally
        scale_factor = target_height / original_height
        
        # Adjust coordinates and handle NaN values
        adjusted_ilm = ilm_coords.copy()
        adjusted_bm = bm_coords.copy()
        
        valid_ilm = ~np.isnan(adjusted_ilm)
        valid_bm = ~np.isnan(adjusted_bm)
        
        adjusted_ilm[valid_ilm] *= scale_factor
        adjusted_bm[valid_bm] *= scale_factor
        
        return resized_image, adjusted_ilm, adjusted_bm
    
    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur and CLAHE preprocessing to the image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        processed_image = image.copy()
        
        # Apply Gaussian blur
        if self.config['preprocessing']['gaussian_blur']['enabled']:
            kernel_size = self.config['preprocessing']['gaussian_blur']['kernel_size']
            sigma = self.config['preprocessing']['gaussian_blur']['sigma']
            processed_image = cv2.GaussianBlur(
                processed_image, 
                (kernel_size, kernel_size), 
                sigma
            )
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.config['preprocessing']['clahe']['enabled']:
            clip_limit = self.config['preprocessing']['clahe']['clip_limit']
            tile_grid_size = tuple(self.config['preprocessing']['clahe']['tile_grid_size'])
            
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size
            )
            processed_image = clahe.apply(processed_image.astype(np.uint8))
        
        return processed_image
    
    def _generate_segmentation_mask(
        self, 
        ilm_coords: np.ndarray, 
        bm_coords: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Generate 3-class segmentation mask from ILM and BM coordinates.
        
        Classes:
        - 0: Above ILM (background/vitreous)
        - 1: ILM to BM (retinal layers)
        - 2: Below BM (choroid/sclera)
        
        Args:
            ilm_coords: ILM y-coordinates for each x position
            bm_coords: BM y-coordinates for each x position
            height: Target height of the mask
            width: Target width of the mask
            
        Returns:
            Segmentation mask of shape (height, width)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for x in range(width):
            ilm_y = ilm_coords[x]
            bm_y = bm_coords[x]
            
            # Skip if either coordinate is NaN
            if np.isnan(ilm_y) or np.isnan(bm_y):
                continue
            
            # Convert to integer pixel coordinates and clip to image bounds
            ilm_y = int(np.clip(round(ilm_y), 0, height - 1))
            bm_y = int(np.clip(round(bm_y), 0, height - 1))
            
            # Ensure ILM is above BM (ILM should have smaller y-coordinate)
            if ilm_y > bm_y:
                ilm_y, bm_y = bm_y, ilm_y
            
            # Assign classes
            # Class 0: Above ILM (background) - remains 0
            # Class 1: ILM to BM (retinal tissue)
            if ilm_y < bm_y:
                mask[ilm_y:bm_y, x] = 1
            # Class 2: Below BM
            if bm_y < height - 1:
                mask[bm_y:, x] = 2
        
        return mask
    
    def _setup_augmentations(self):
        """Setup augmentation pipelines for training and validation."""
        target_height = self.config['dataset']['target_size']['height']
        target_width = self.config['dataset']['target_size']['width']
        
        # Get normalization parameters from config
        norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        norm_mean = norm_config.get('mean', 0.0)
        norm_std = norm_config.get('std', 1.0)
        
        # Base transforms for all modes
        base_transforms = [
            A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0),
            ToTensorV2()
        ]
        
        # Geometric augmentations (applied to both image and mask)
        geometric_augs = []
        
        if self.mode == 'train':
            geo_config = self.config['augmentation']['geometric']
            
            if geo_config['horizontal_flip']['enabled']:
                geometric_augs.append(
                    A.HorizontalFlip(p=geo_config['horizontal_flip']['probability'])
                )
            
            if geo_config['vertical_flip']['enabled']:
                geometric_augs.append(
                    A.VerticalFlip(p=geo_config['vertical_flip']['probability'])
                )
            
            if geo_config['rotation']['enabled']:
                geometric_augs.append(
                    A.Rotate(
                        limit=geo_config['rotation']['limit'],
                        p=geo_config['rotation']['probability'],
                        border_mode=cv2.BORDER_CONSTANT
                    )
                )
        
        # Photometric augmentations (applied only to image)
        photometric_augs = []
        
        if self.mode == 'train':
            photo_config = self.config['augmentation']['photometric']
            
            if photo_config['invert_image']['enabled']:
                photometric_augs.append(
                    A.InvertImg(p=photo_config['invert_image']['probability'])
                )
            
            if photo_config['random_snow']['enabled']:
                # Note: RandomSnow parameters may vary by albumentations version
                # Using basic parameters that are commonly supported
                photometric_augs.append(
                    A.RandomSnow(p=photo_config['random_snow']['probability'])
                )
            
            if photo_config['clahe']['enabled']:
                photometric_augs.append(
                    A.CLAHE(
                        clip_limit=photo_config['clahe']['clip_limit'],
                        tile_grid_size=tuple(photo_config['clahe']['tile_grid_size']),
                        p=photo_config['clahe']['probability']
                    )
                )
            
            if photo_config['blur']['enabled']:
                photometric_augs.append(
                    A.Blur(
                        blur_limit=photo_config['blur']['blur_limit'],
                        p=photo_config['blur']['probability']
                    )
                )
            
            if photo_config['coarse_dropout']['enabled']:
                photometric_augs.append(
                    A.CoarseDropout(
                        num_holes_range=tuple(photo_config['coarse_dropout']['num_holes_range']),
                        hole_height_range=tuple(photo_config['coarse_dropout']['hole_height_range']),
                        hole_width_range=tuple(photo_config['coarse_dropout']['hole_width_range']),
                        fill=photo_config['coarse_dropout']['fill'],
                        p=photo_config['coarse_dropout']['probability']
                    )
                )
            
            if photo_config['downscale']['enabled']:
                photometric_augs.append(
                    A.Downscale(
                        scale_range=tuple(photo_config['downscale']['scale_range']),
                        interpolation_pair={
                            'upscale': photo_config['downscale']['interpolation_upscale'],
                            'downscale': photo_config['downscale']['interpolation_downscale']
                        },
                        p=photo_config['downscale']['probability']
                    )
                )
            
            if photo_config['equalize']['enabled']:
                photometric_augs.append(
                    A.Equalize(
                        mode=photo_config['equalize']['mode'],
                        by_channels=photo_config['equalize']['by_channels'],
                        p=photo_config['equalize']['probability']
                    )
                )
        
        # Combine all transforms
        all_transforms = geometric_augs + photometric_augs + base_transforms
        
        # Create augmentation pipeline
        self.transform = A.Compose(
            all_transforms,
            additional_targets={'mask': 'mask'}
        )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Get raw data
        image = self.images[idx].astype(np.uint8)
        ilm_coords = self.ilm_coords[idx]
        bm_coords = self.bm_coords[idx]
        
        # Resize image and adjust coordinates
        image, ilm_coords, bm_coords = self._resize_image_and_coords(
            image, ilm_coords, bm_coords
        )
        
        # Apply preprocessing
        image = self._apply_preprocessing(image)
        
        # Generate segmentation mask
        height, width = image.shape
        mask = self._generate_segmentation_mask(
            ilm_coords, bm_coords, height, width
        )
        
        # Ensure image is 3-channel for augmentation (some augmentations require RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
        
        # Apply augmentations
        augmented = self.transform(image=image, mask=mask)
            
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
        
        # Convert back to single channel if needed (for our model)
        if image_tensor.shape[0] == 3:
            # Convert RGB back to grayscale by taking mean of channels
            image_tensor = image_tensor.mean(dim=0, keepdim=True)
        
        # Ensure correct tensor types
        image_tensor = image_tensor.float()
        mask_tensor = mask_tensor.long()
        
        return image_tensor, mask_tensor

    def visualize_augmentations(self, sample_idx: int = 0, save_path: str = "augmentation_test.png"):
        """
        Visualize different augmentations applied to a sample image and mask.
        
        Args:
            sample_idx: Index of the sample to visualize
            save_path: Path to save the visualization
        """
        if self.mode != 'train':
            print("Augmentation visualization only available for training mode")
            return
            
        # Get raw data
        image = self.images[sample_idx].astype(np.uint8)
        ilm_coords = self.ilm_coords[sample_idx]
        bm_coords = self.bm_coords[sample_idx]
        
        # Resize and preprocess
        image, ilm_coords, bm_coords = self._resize_image_and_coords(
            image, ilm_coords, bm_coords
        )
        image = self._apply_preprocessing(image)
        
        # Generate mask
        height, width = image.shape
        mask = self._generate_segmentation_mask(ilm_coords, bm_coords, height, width)
        
        # Prepare image for augmentation (convert to 3-channel)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create individual augmentation transforms for testing
        geo_config = self.config['augmentation']['geometric']
        photo_config = self.config['augmentation']['photometric']
        
        transforms_to_test = [
            ("Original", A.Compose([
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("Horizontal Flip", A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("Vertical Flip", A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("Rotation", A.Compose([
                A.Rotate(limit=geo_config['rotation']['limit'], p=1.0, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("Invert Image", A.Compose([
                A.InvertImg(p=1.0),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("CLAHE", A.Compose([
                A.CLAHE(
                    clip_limit=photo_config['clahe']['clip_limit'],
                    tile_grid_size=tuple(photo_config['clahe']['tile_grid_size']),
                    p=1.0
                ),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("Blur", A.Compose([
                A.Blur(blur_limit=photo_config['blur']['blur_limit'], p=1.0),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'})),
            ("Coarse Dropout", A.Compose([
                A.CoarseDropout(
                    num_holes_range=tuple(photo_config['coarse_dropout']['num_holes_range']),
                    hole_height_range=tuple(photo_config['coarse_dropout']['hole_height_range']),
                    hole_width_range=tuple(photo_config['coarse_dropout']['hole_width_range']),
                    fill=photo_config['coarse_dropout']['fill'],
                    p=1.0
                ),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            ], additional_targets={'mask': 'mask'}))
        ]
        
        # Create visualization
        fig, axes = plt.subplots(2, len(transforms_to_test), figsize=(24, 8))
        
        for i, (name, transform) in enumerate(transforms_to_test):
            # Apply transform
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Convert tensor to numpy if needed
            if torch.is_tensor(aug_image):
                aug_image = aug_image.squeeze().numpy()
            if torch.is_tensor(aug_mask):
                aug_mask = aug_mask.numpy()
            
            # Plot image
            axes[0, i].imshow(aug_image, cmap='gray')
            axes[0, i].set_title(f'{name}\nImage')
            axes[0, i].axis('off')
            
            # Plot mask with color mapping
            axes[1, i].imshow(aug_mask, cmap='viridis', vmin=0, vmax=2)
            axes[1, i].set_title(f'{name}\nMask')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Augmentation visualization saved to: {save_path}")


def create_data_splits(
    config_path: str, 
    train_ratio: float = 0.8, 
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[OCTDataset, OCTDataset]:
    """
    Create train and test datasets with proper splits.
    
    Args:
        config_path: Path to the configuration file
        train_ratio: Proportion of data for training
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Load config to get total number of samples
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with h5py.File(config['dataset']['hdf5_path'], 'r') as f:
        total_samples = f['images'].shape[0]
    
    # Create indices for splits
    np.random.seed(random_seed)
    indices = np.random.permutation(total_samples)
    
    train_end = int(train_ratio * total_samples)
    
    train_indices = indices[:train_end]
    test_indices = indices[train_end:]
    
    # Create datasets
    train_dataset = OCTDataset(config_path, mode='train', indices=train_indices)
    test_dataset = OCTDataset(config_path, mode='test', indices=test_indices)
    
    print(f"Data splits created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, test_dataset


def create_data_loaders(
    config_path: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for train and test sets.
    
    Args:
        config_path: Path to the configuration file
        train_ratio: Proportion of data for training
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get training configuration
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory']
    
    # Create datasets
    train_dataset, test_dataset = create_data_splits(
        config_path, train_ratio, test_ratio, random_seed
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=config['training']['drop_last']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader


