"""
Training and Evaluation Engine for Hybrid Attention U-Net

This module implements the training and evaluation loops for the 
OCT layer segmentation model, including loss functions, metrics,
and model saving/loading functionality.
"""

import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy.stats import pearsonr

from model import HybridAttentionUNet


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Computes the Dice coefficient loss which is particularly
    effective for segmentation problems with class imbalance.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            
        Returns:
            Dice loss value
        """
        # Apply softmax to predictions
        pred_softmax = torch.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = pred.size(1)
        target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        # Calculate Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean Dice loss
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function using Cross Entropy and Dice Loss.
    
    This combination helps with both accurate classification and
    good overlap between predicted and ground truth masks.
    """
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.ce_weight * ce + self.dice_weight * dice


class SegmentationMetrics:
    """
    Class to compute segmentation metrics including Dice, IoU, Precision, Recall, F1.
    """
    
    def __init__(self, num_classes: int = 3, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_samples = 0
        self.class_dice = []
        self.class_iou = []
        self.class_precision = []
        self.class_recall = []
        self.class_f1 = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
        """
        # Convert predictions to class indices
        pred_classes = torch.argmax(pred, dim=1)
        
        batch_size = pred.size(0)
        self.total_samples += batch_size
        
        # Calculate metrics for each class
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        start_class = 1 if self.ignore_background else 0
        
        for class_id in range(start_class, self.num_classes):
            # Create binary masks for current class
            pred_mask = (pred_classes == class_id).float()
            target_mask = (target == class_id).float()
            
            # Calculate True Positives, False Positives, False Negatives
            tp = (pred_mask * target_mask).sum(dim=(1, 2))
            fp = (pred_mask * (1 - target_mask)).sum(dim=(1, 2))
            fn = ((1 - pred_mask) * target_mask).sum(dim=(1, 2))
            
            # Calculate metrics
            dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
            iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
            precision = (tp + 1e-8) / (tp + fp + 1e-8)
            recall = (tp + 1e-8) / (tp + fn + 1e-8)
            f1 = (2 * precision * recall + 1e-8) / (precision + recall + 1e-8)
            
            dice_scores.append(dice.mean().item())
            iou_scores.append(iou.mean().item())
            precision_scores.append(precision.mean().item())
            recall_scores.append(recall.mean().item())
            f1_scores.append(f1.mean().item())
        
        self.class_dice.append(dice_scores)
        self.class_iou.append(iou_scores)
        self.class_precision.append(precision_scores)
        self.class_recall.append(recall_scores)
        self.class_f1.append(f1_scores)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing mean metrics
        """
        if not self.class_dice:
            return {
                'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 
                'recall': 0.0, 'f1': 0.0
            }
        
        # Convert to numpy arrays and compute means
        dice_array = np.array(self.class_dice)
        iou_array = np.array(self.class_iou)
        precision_array = np.array(self.class_precision)
        recall_array = np.array(self.class_recall)
        f1_array = np.array(self.class_f1)
        
        return {
            'dice': float(np.mean(dice_array)),
            'iou': float(np.mean(iou_array)),
            'precision': float(np.mean(precision_array)),
            'recall': float(np.mean(recall_array)),
            'f1': float(np.mean(f1_array))
        }


def mask_to_coordinates(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert segmentation mask back to ILM and BM coordinates.
    
    Args:
        mask: Segmentation mask of shape (H, W) with values {0, 1, 2}
              0: above ILM, 1: ILM to BM, 2: below BM
    
    Returns:
        Tuple of (ilm_coords, bm_coords) each of shape (W,)
    """
    height, width = mask.shape
    ilm_coords = np.full(width, np.nan)
    bm_coords = np.full(width, np.nan)
    
    for x in range(width):
        column = mask[:, x]
        
        # Find ILM boundary (transition from 0 to 1)
        ilm_indices = np.where((column[:-1] == 0) & (column[1:] == 1))[0]
        if len(ilm_indices) > 0:
            # Take the first transition point
            ilm_coords[x] = ilm_indices[0] + 1
        
        # Find BM boundary (transition from 1 to 2)
        bm_indices = np.where((column[:-1] == 1) & (column[1:] == 2))[0]
        if len(bm_indices) > 0:
            # Take the first transition point
            bm_coords[x] = bm_indices[0] + 1
    
    return ilm_coords, bm_coords


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Concordance Correlation Coefficient (CCC).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        CCC value
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(valid_mask) == 0:
        return 0.0
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        return 0.0
    
    # Calculate means
    mean_true = np.mean(y_true_valid)
    mean_pred = np.mean(y_pred_valid)
    
    # Calculate variances
    var_true = np.var(y_true_valid)
    var_pred = np.var(y_pred_valid)
    
    # Calculate covariance
    covariance = np.mean((y_true_valid - mean_true) * (y_pred_valid - mean_pred))
    
    # Calculate CCC
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator == 0:
        return 0.0
    
    ccc = numerator / denominator
    return float(ccc)


class RegressionMetrics:
    """
    Class to compute regression metrics for layer coordinate predictions.
    Converts masks to coordinates and calculates MAE, RMSE, and CCC.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.ilm_true_coords = []
        self.ilm_pred_coords = []
        self.bm_true_coords = []
        self.bm_pred_coords = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
        """
        # Convert predictions to class indices
        pred_classes = torch.argmax(pred, dim=1)
        
        # Convert to numpy
        pred_masks = pred_classes.cpu().numpy()
        target_masks = target.cpu().numpy()
        
        batch_size = pred_masks.shape[0]
        
        for i in range(batch_size):
            # Convert masks to coordinates
            ilm_true, bm_true = mask_to_coordinates(target_masks[i])
            ilm_pred, bm_pred = mask_to_coordinates(pred_masks[i])
            
            self.ilm_true_coords.append(ilm_true)
            self.ilm_pred_coords.append(ilm_pred)
            self.bm_true_coords.append(bm_true)
            self.bm_pred_coords.append(bm_pred)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final regression metrics.
        
        Returns:
            Dictionary containing MAE, RMSE, and CCC for ILM and BM
        """
        if not self.ilm_true_coords:
            return {
                'ilm_mae': 0.0, 'ilm_rmse': 0.0, 'ilm_ccc': 0.0,
                'bm_mae': 0.0, 'bm_rmse': 0.0, 'bm_ccc': 0.0,
                'overall_mae': 0.0, 'overall_rmse': 0.0, 'overall_ccc': 0.0
            }
        
        # Concatenate all coordinates
        ilm_true_all = np.concatenate(self.ilm_true_coords)
        ilm_pred_all = np.concatenate(self.ilm_pred_coords)
        bm_true_all = np.concatenate(self.bm_true_coords)
        bm_pred_all = np.concatenate(self.bm_pred_coords)
        
        # Calculate ILM metrics
        ilm_mae = self._calculate_mae(ilm_true_all, ilm_pred_all)
        ilm_rmse = self._calculate_rmse(ilm_true_all, ilm_pred_all)
        ilm_ccc = concordance_correlation_coefficient(ilm_true_all, ilm_pred_all)
        
        # Calculate BM metrics
        bm_mae = self._calculate_mae(bm_true_all, bm_pred_all)
        bm_rmse = self._calculate_rmse(bm_true_all, bm_pred_all)
        bm_ccc = concordance_correlation_coefficient(bm_true_all, bm_pred_all)
        
        # Calculate overall metrics (combining ILM and BM)
        all_true = np.concatenate([ilm_true_all, bm_true_all])
        all_pred = np.concatenate([ilm_pred_all, bm_pred_all])
        overall_mae = self._calculate_mae(all_true, all_pred)
        overall_rmse = self._calculate_rmse(all_true, all_pred)
        overall_ccc = concordance_correlation_coefficient(all_true, all_pred)
        
        return {
            'ilm_mae': ilm_mae,
            'ilm_rmse': ilm_rmse,
            'ilm_ccc': ilm_ccc,
            'bm_mae': bm_mae,
            'bm_rmse': bm_rmse,
            'bm_ccc': bm_ccc,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_ccc': overall_ccc
        }
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error ignoring NaN values."""
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if np.sum(valid_mask) == 0:
            return 0.0
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        return float(np.mean(np.abs(y_true_valid - y_pred_valid)))
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error ignoring NaN values."""
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if np.sum(valid_mask) == 0:
            return 0.0
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        return float(np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2)))


class TrainingEngine:
    """
    Training engine for Hybrid Attention U-Net.
    
    Handles training loop, test evaluation, metrics tracking, and model saving.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
        
        # Initialize optimizer (AdaBound as mentioned in paper, fallback to Adam)
        try:
            from adabound import AdaBound
            self.optimizer = AdaBound(
                self.model.parameters(),
                lr=config.get('learning_rate', 1e-3),
                final_lr=0.1
            )
        except ImportError:
            print("AdaBound not available, using Adam optimizer")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.get('learning_rate', 1e-3)
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_dice': [],
            'test_dice': [],
            'train_iou': [],
            'test_iou': [],
            'train_f1': [],
            'test_f1': [],
            'train_ilm_mae': [],
            'test_ilm_mae': [],
            'train_ilm_rmse': [],
            'test_ilm_rmse': [],
            'train_ilm_ccc': [],
            'test_ilm_ccc': [],
            'train_bm_mae': [],
            'test_bm_mae': [],
            'train_bm_rmse': [],
            'test_bm_rmse': [],
            'train_bm_ccc': [],
            'test_bm_ccc': [],
            'train_overall_mae': [],
            'test_overall_mae': [],
            'train_overall_rmse': [],
            'test_overall_rmse': [],
            'train_overall_ccc': [],
            'test_overall_ccc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_test_dice = 0.0
        self.best_model_path = None

        # Training parameters
        training_config = config.get('training', {})
        optimization_config = config.get('optimization', {})
        
        self.batch_size = training_config.get('batch_size', 1)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.gradient_clip_norm = optimization_config.get('gradient_clip_norm', None)
        self.mixed_precision = optimization_config.get('mixed_precision', False)
        
        # Initialize GradScaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Early stopping
        self.patience = training_config.get('patience', 10)
        self.early_stop_counter = 0
        
        # Create output directory
        self.output_dir = self._create_output_dir()
        
        print(f"Training engine initialized")
        print(f"Device: {device}")
        print(f"Output directory: {self.output_dir}")
    
    def _create_output_dir(self) -> str:
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"runs/hybrid_unet_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Also ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        return output_dir
    
    def train_epoch(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, segmentation_metrics_dict, regression_metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3, ignore_background=True)
        reg_metrics = RegressionMetrics()
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with optional mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps or at the end
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.mixed_precision:
                    # Gradient clipping for mixed precision
                    if self.gradient_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping for regular training
                    if self.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track metrics (use unscaled loss for reporting)
            batch_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += batch_loss
            seg_metrics.update(outputs.detach(), masks.detach())
            reg_metrics.update(outputs.detach(), masks.detach())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        seg_metrics_dict = seg_metrics.compute()
        reg_metrics_dict = reg_metrics.compute()
        
        return avg_loss, seg_metrics_dict, reg_metrics_dict
    
    def test_epoch(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Test for one epoch.
        
        Returns:
            Tuple of (average_loss, segmentation_metrics_dict, regression_metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3, ignore_background=True)
        reg_metrics = RegressionMetrics()
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                seg_metrics.update(outputs, masks)
                reg_metrics.update(outputs, masks)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / len(self.test_loader)
        seg_metrics_dict = seg_metrics.compute()
        reg_metrics_dict = reg_metrics.compute()
        
        return avg_loss, seg_metrics_dict, reg_metrics_dict
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        # Save latest model
        latest_path = os.path.join(self.output_dir, 'latest_model.pth')
        torch.save(model_state, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(model_state, best_path)
            self.best_model_path = best_path
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Model loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def save_training_plots(self):
        """Save training progress plots."""
        if not self.history['train_loss']:
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        axes[0, 0].set_title('Training and Test Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice score plot
        axes[0, 1].plot(epochs, self.history['train_dice'], 'b-', label='Train Dice', linewidth=2)
        axes[0, 1].plot(epochs, self.history['test_dice'], 'r-', label='Test Dice', linewidth=2)
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU plot
        axes[0, 2].plot(epochs, self.history['train_iou'], 'b-', label='Train IoU', linewidth=2)
        axes[0, 2].plot(epochs, self.history['test_iou'], 'r-', label='Test IoU', linewidth=2)
        axes[0, 2].set_title('Intersection over Union (IoU)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # ILM MAE plot
        axes[1, 0].plot(epochs, self.history['train_ilm_mae'], 'b-', label='Train ILM MAE', linewidth=2)
        axes[1, 0].plot(epochs, self.history['test_ilm_mae'], 'r-', label='Test ILM MAE', linewidth=2)
        axes[1, 0].set_title('ILM Mean Absolute Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE (pixels)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # BM MAE plot
        axes[1, 1].plot(epochs, self.history['train_bm_mae'], 'b-', label='Train BM MAE', linewidth=2)
        axes[1, 1].plot(epochs, self.history['test_bm_mae'], 'r-', label='Test BM MAE', linewidth=2)
        axes[1, 1].set_title('BM Mean Absolute Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE (pixels)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Overall MAE plot
        axes[1, 2].plot(epochs, self.history['train_overall_mae'], 'b-', label='Train Overall MAE', linewidth=2)
        axes[1, 2].plot(epochs, self.history['test_overall_mae'], 'r-', label='Test Overall MAE', linewidth=2)
        axes[1, 2].set_title('Overall Mean Absolute Error')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('MAE (pixels)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # ILM CCC plot
        axes[2, 0].plot(epochs, self.history['train_ilm_ccc'], 'b-', label='Train ILM CCC', linewidth=2)
        axes[2, 0].plot(epochs, self.history['test_ilm_ccc'], 'r-', label='Test ILM CCC', linewidth=2)
        axes[2, 0].set_title('ILM Concordance Correlation Coefficient')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('CCC')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # BM CCC plot
        axes[2, 1].plot(epochs, self.history['train_bm_ccc'], 'b-', label='Train BM CCC', linewidth=2)
        axes[2, 1].plot(epochs, self.history['test_bm_ccc'], 'r-', label='Test BM CCC', linewidth=2)
        axes[2, 1].set_title('BM Concordance Correlation Coefficient')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('CCC')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Overall CCC plot
        axes[2, 2].plot(epochs, self.history['train_overall_ccc'], 'b-', label='Train Overall CCC', linewidth=2)
        axes[2, 2].plot(epochs, self.history['test_overall_ccc'], 'r-', label='Test Overall CCC', linewidth=2)
        axes[2, 2].set_title('Overall Concordance Correlation Coefficient')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('CCC')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_path}")
    
    def save_training_results(self):
        """Save comprehensive training results to JSON."""
        results = {
            'config': self.config,
            'best_test_dice': self.best_test_dice,
            'best_model_path': self.best_model_path,
            'total_epochs': len(self.history['train_loss']),
            'final_metrics': {
                # Segmentation metrics
                'segmentation': {
                    'train_dice': self.history['train_dice'][-1] if self.history['train_dice'] else 0,
                    'test_dice': self.history['test_dice'][-1] if self.history['test_dice'] else 0,
                    'train_iou': self.history['train_iou'][-1] if self.history['train_iou'] else 0,
                    'test_iou': self.history['test_iou'][-1] if self.history['test_iou'] else 0,
                    'train_f1': self.history['train_f1'][-1] if self.history['train_f1'] else 0,
                    'test_f1': self.history['test_f1'][-1] if self.history['test_f1'] else 0,
                },
                # Regression metrics
                'regression': {
                    'ilm': {
                        'train_mae': self.history['train_ilm_mae'][-1] if self.history['train_ilm_mae'] else 0,
                        'test_mae': self.history['test_ilm_mae'][-1] if self.history['test_ilm_mae'] else 0,
                        'train_rmse': self.history['train_ilm_rmse'][-1] if self.history['train_ilm_rmse'] else 0,
                        'test_rmse': self.history['test_ilm_rmse'][-1] if self.history['test_ilm_rmse'] else 0,
                        'train_ccc': self.history['train_ilm_ccc'][-1] if self.history['train_ilm_ccc'] else 0,
                        'test_ccc': self.history['test_ilm_ccc'][-1] if self.history['test_ilm_ccc'] else 0,
                    },
                    'bm': {
                        'train_mae': self.history['train_bm_mae'][-1] if self.history['train_bm_mae'] else 0,
                        'test_mae': self.history['test_bm_mae'][-1] if self.history['test_bm_mae'] else 0,
                        'train_rmse': self.history['train_bm_rmse'][-1] if self.history['train_bm_rmse'] else 0,
                        'test_rmse': self.history['test_bm_rmse'][-1] if self.history['test_bm_rmse'] else 0,
                        'train_ccc': self.history['train_bm_ccc'][-1] if self.history['train_bm_ccc'] else 0,
                        'test_ccc': self.history['test_bm_ccc'][-1] if self.history['test_bm_ccc'] else 0,
                    },
                    'overall': {
                        'train_mae': self.history['train_overall_mae'][-1] if self.history['train_overall_mae'] else 0,
                        'test_mae': self.history['test_overall_mae'][-1] if self.history['test_overall_mae'] else 0,
                        'train_rmse': self.history['train_overall_rmse'][-1] if self.history['train_overall_rmse'] else 0,
                        'test_rmse': self.history['test_overall_rmse'][-1] if self.history['test_overall_rmse'] else 0,
                        'train_ccc': self.history['train_overall_ccc'][-1] if self.history['train_overall_ccc'] else 0,
                        'test_ccc': self.history['test_overall_ccc'][-1] if self.history['test_overall_ccc'] else 0,
                    }
                }
            },
            'complete_history': self.history
        }
        
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comprehensive training results saved to {results_path}")
        
        # Also save a summary file with key metrics
        summary = {
            'model_info': {
                'architecture': 'Hybrid Attention U-Net',
                'input_channels': self.config.get('input_channels', 1),
                'output_channels': self.config.get('output_channels', 3),
                'total_epochs': len(self.history['train_loss']),
                'best_epoch': self.history['test_dice'].index(max(self.history['test_dice'])) + 1 if self.history['test_dice'] else 0
            },
            'best_performance': {
                'test_dice': self.best_test_dice,
                'best_test_ilm_mae': min(self.history['test_ilm_mae']) if self.history['test_ilm_mae'] else 0,
                'best_test_bm_mae': min(self.history['test_bm_mae']) if self.history['test_bm_mae'] else 0,
                'best_test_overall_mae': min(self.history['test_overall_mae']) if self.history['test_overall_mae'] else 0,
                'best_test_ilm_ccc': max(self.history['test_ilm_ccc']) if self.history['test_ilm_ccc'] else 0,
                'best_test_bm_ccc': max(self.history['test_bm_ccc']) if self.history['test_bm_ccc'] else 0,
                'best_test_overall_ccc': max(self.history['test_overall_ccc']) if self.history['test_overall_ccc'] else 0,
            },
            'final_performance': results['final_metrics']
        }
        
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {summary_path}")
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_seg_metrics, train_reg_metrics = self.train_epoch()
            
            # Test
            test_loss, test_seg_metrics, test_reg_metrics = self.test_epoch()
            
            # Update learning rate
            self.scheduler.step(test_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history - segmentation metrics
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_dice'].append(train_seg_metrics['dice'])
            self.history['test_dice'].append(test_seg_metrics['dice'])
            self.history['train_iou'].append(train_seg_metrics['iou'])
            self.history['test_iou'].append(test_seg_metrics['iou'])
            self.history['train_f1'].append(train_seg_metrics['f1'])
            self.history['test_f1'].append(test_seg_metrics['f1'])
            
            # Update history - regression metrics
            self.history['train_ilm_mae'].append(train_reg_metrics['ilm_mae'])
            self.history['test_ilm_mae'].append(test_reg_metrics['ilm_mae'])
            self.history['train_ilm_rmse'].append(train_reg_metrics['ilm_rmse'])
            self.history['test_ilm_rmse'].append(test_reg_metrics['ilm_rmse'])
            self.history['train_ilm_ccc'].append(train_reg_metrics['ilm_ccc'])
            self.history['test_ilm_ccc'].append(test_reg_metrics['ilm_ccc'])
            self.history['train_bm_mae'].append(train_reg_metrics['bm_mae'])
            self.history['test_bm_mae'].append(test_reg_metrics['bm_mae'])
            self.history['train_bm_rmse'].append(train_reg_metrics['bm_rmse'])
            self.history['test_bm_rmse'].append(test_reg_metrics['bm_rmse'])
            self.history['train_bm_ccc'].append(train_reg_metrics['bm_ccc'])
            self.history['test_bm_ccc'].append(test_reg_metrics['bm_ccc'])
            self.history['train_overall_mae'].append(train_reg_metrics['overall_mae'])
            self.history['test_overall_mae'].append(test_reg_metrics['overall_mae'])
            self.history['train_overall_rmse'].append(train_reg_metrics['overall_rmse'])
            self.history['test_overall_rmse'].append(test_reg_metrics['overall_rmse'])
            self.history['train_overall_ccc'].append(train_reg_metrics['overall_ccc'])
            self.history['test_overall_ccc'].append(test_reg_metrics['overall_ccc'])
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Loss - Train: {train_loss:.4f} | Test: {test_loss:.4f}")
            print(f"Segmentation Metrics:")
            print(f"  Dice - Train: {train_seg_metrics['dice']:.4f} | Test: {test_seg_metrics['dice']:.4f}")
            print(f"  IoU  - Train: {train_seg_metrics['iou']:.4f} | Test: {test_seg_metrics['iou']:.4f}")
            print(f"  F1   - Train: {train_seg_metrics['f1']:.4f} | Test: {test_seg_metrics['f1']:.4f}")
            print(f"Regression Metrics:")
            print(f"  ILM MAE  - Train: {train_reg_metrics['ilm_mae']:.2f} | Test: {test_reg_metrics['ilm_mae']:.2f}")
            print(f"  ILM RMSE - Train: {train_reg_metrics['ilm_rmse']:.2f} | Test: {test_reg_metrics['ilm_rmse']:.2f}")
            print(f"  ILM CCC  - Train: {train_reg_metrics['ilm_ccc']:.4f} | Test: {test_reg_metrics['ilm_ccc']:.4f}")
            print(f"  BM MAE   - Train: {train_reg_metrics['bm_mae']:.2f} | Test: {test_reg_metrics['bm_mae']:.2f}")
            print(f"  BM RMSE  - Train: {train_reg_metrics['bm_rmse']:.2f} | Test: {test_reg_metrics['bm_rmse']:.2f}")
            print(f"  BM CCC   - Train: {train_reg_metrics['bm_ccc']:.4f} | Test: {test_reg_metrics['bm_ccc']:.4f}")
            print(f"  Overall MAE - Train: {train_reg_metrics['overall_mae']:.2f} | Test: {test_reg_metrics['overall_mae']:.2f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Check for best model (using test dice score)
            is_best = test_seg_metrics['dice'] > self.best_test_dice
            if is_best:
                self.best_test_dice = test_seg_metrics['dice']
                self.early_stop_counter = 0
                print(f"🎉 New best model! Test Dice: {self.best_test_dice:.4f}")
            else:
                self.early_stop_counter += 1
            
            # Combine metrics for saving
            combined_metrics = {**test_seg_metrics, **test_reg_metrics}
            
            # Save model
            self.save_model(epoch, combined_metrics, is_best)
            
            # Early stopping
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping after {self.patience} epochs without improvement")
                break
            
            # Save plots every 10 epochs
            if epoch % 10 == 0:
                self.save_training_plots()
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best test Dice score: {self.best_test_dice:.4f}")
        
        # Save final results
        self.save_training_plots()
        self.save_training_results()

    def inference(self, data_loader: DataLoader, save_path: str = None) -> Dict[str, float]:
        """
        Generate inference on a dataset and calculate metrics.
        
        Args:
            data_loader: DataLoader for inference
            save_path: Optional path to save inference results as JSON
            
        Returns:
            Dictionary containing loss and performance metrics
        """
        print("Starting inference...")
        self.model.eval()
        
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3, ignore_background=True)
        reg_metrics = RegressionMetrics()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Inference")
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                seg_metrics.update(outputs, masks)
                reg_metrics.update(outputs, masks)
                
                # Store predictions and targets for detailed analysis
                pred_classes = torch.argmax(outputs, dim=1)
                all_predictions.append(pred_classes.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        # Calculate final metrics
        avg_loss = total_loss / len(data_loader)
        seg_metrics_dict = seg_metrics.compute()
        reg_metrics_dict = reg_metrics.compute()
        
        # Combine all results
        inference_results = {
            'loss': avg_loss,
            'segmentation_metrics': seg_metrics_dict,
            'regression_metrics': reg_metrics_dict,
            'model_info': {
                'architecture': 'Hybrid Attention U-Net',
                'input_channels': 1,
                'output_channels': 3,
                'model_path': self.best_model_path
            }
        }
        
        # Save results if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(inference_results, f, indent=2)
            print(f"Inference results saved to {save_path}")
        
        # Print summary
        print(f"\nInference completed!")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Segmentation Metrics:")
        print(f"  Dice: {seg_metrics_dict['dice']:.4f}")
        print(f"  IoU: {seg_metrics_dict['iou']:.4f}")
        print(f"  F1: {seg_metrics_dict['f1']:.4f}")
        print(f"Regression Metrics:")
        print(f"  ILM MAE: {reg_metrics_dict['ilm_mae']:.2f}")
        print(f"  BM MAE: {reg_metrics_dict['bm_mae']:.2f}")
        print(f"  Overall MAE: {reg_metrics_dict['overall_mae']:.2f}")
        print(f"  ILM CCC: {reg_metrics_dict['ilm_ccc']:.4f}")
        print(f"  BM CCC: {reg_metrics_dict['bm_ccc']:.4f}")
        print(f"  Overall CCC: {reg_metrics_dict['overall_ccc']:.4f}")
        
        return inference_results


