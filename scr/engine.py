"""
Training and Evaluation Engine for Hybrid Attention U-Net

This module implements the training and evaluation loops for the 
OCT layer segmentation model, including loss functions, metrics,
and model saving/loading functionality.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .model import HybridAttentionUNet


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


class TrainingEngine:
    """
    Training engine for Hybrid Attention U-Net.
    
    Handles training loop, validation, metrics tracking, and model saving.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_dice = 0.0
        self.best_model_path = None
        
        # Early stopping
        self.patience = config.get('patience', 10)
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
        return output_dir
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        metrics = SegmentationMetrics(num_classes=3, ignore_background=True)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs.detach(), masks.detach())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        metrics_dict = metrics.compute()
        
        return avg_loss, metrics_dict
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        metrics = SegmentationMetrics(num_classes=3, ignore_background=True)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                metrics.update(outputs, masks)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        metrics_dict = metrics.compute()
        
        return avg_loss, metrics_dict
    
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dice score plot
        ax2.plot(epochs, self.history['train_dice'], 'b-', label='Train Dice', linewidth=2)
        ax2.plot(epochs, self.history['val_dice'], 'r-', label='Val Dice', linewidth=2)
        ax2.set_title('Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # IoU plot
        ax3.plot(epochs, self.history['train_iou'], 'b-', label='Train IoU', linewidth=2)
        ax3.plot(epochs, self.history['val_iou'], 'r-', label='Val IoU', linewidth=2)
        ax3.set_title('Intersection over Union (IoU)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('IoU')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # F1 score plot
        ax4.plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        ax4.plot(epochs, self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
        ax4.set_title('F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_path}")
    
    def save_training_results(self):
        """Save training results to JSON."""
        results = {
            'config': self.config,
            'best_val_dice': self.best_val_dice,
            'best_model_path': self.best_model_path,
            'total_epochs': len(self.history['train_loss']),
            'final_metrics': {
                'train_dice': self.history['train_dice'][-1] if self.history['train_dice'] else 0,
                'val_dice': self.history['val_dice'][-1] if self.history['val_dice'] else 0,
                'train_iou': self.history['train_iou'][-1] if self.history['train_iou'] else 0,
                'val_iou': self.history['val_iou'][-1] if self.history['val_iou'] else 0,
            },
            'history': self.history
        }
        
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to {results_path}")
    
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
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Check for best model
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.early_stop_counter = 0
                print(f"🎉 New best model! Val Dice: {self.best_val_dice:.4f}")
            else:
                self.early_stop_counter += 1
            
            # Save model
            self.save_model(epoch, val_metrics, is_best)
            
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
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")
        
        # Save final results
        self.save_training_plots()
        self.save_training_results()


if __name__ == "__main__":
    """Test the training engine."""
    print("Testing Training Engine...")
    
    # This is just a basic test - real training would use the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy model
    model = HybridAttentionUNet(in_channels=1, out_channels=3)
    
    # Test loss functions
    batch_size, height, width = 2, 480, 768
    pred = torch.randn(batch_size, 3, height, width)
    target = torch.randint(0, 3, (batch_size, height, width))
    
    # Test Dice loss
    dice_loss = DiceLoss()
    dice_value = dice_loss(pred, target)
    print(f"Dice loss: {dice_value:.4f}")
    
    # Test Combined loss
    combined_loss = CombinedLoss()
    combined_value = combined_loss(pred, target)
    print(f"Combined loss: {combined_value:.4f}")
    
    # Test metrics
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(pred, target)
    metrics_dict = metrics.compute()
    print(f"Metrics: {metrics_dict}")
    
    print("✓ Training engine test passed!")
