"""
Main training script for Hybrid Attention U-Net OCT Layer Segmentation

This script combines the dataset, model, and training engine to train
the complete OCT layer segmentation system.
"""

import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from scr.dataset import create_data_loaders
from scr.model import HybridAttentionUNet
from scr.engine import TrainingEngine


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> torch.nn.Module:
    """Create the Hybrid Attention U-Net model."""
    model_config = config.get('model', {})
    in_channels = model_config.get('input_channels', 1)
    out_channels = model_config.get('output_channels', 3)
    base_channels = model_config.get('base_channels', 64)
    
    model = HybridAttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels
    )
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hybrid Attention U-Net for OCT Layer Segmentation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:")
    print(f"  Dataset: {config['dataset']['hdf5_path']}")
    print(f"  Target size: {config['dataset']['target_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_data_loaders(args.config)
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create training engine
    print("\nInitializing training engine...")
    training_config = {
        'learning_rate': 1e-3,
        'patience': 10,
        **config.get('training', {})
    }
    
    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=training_config,
        device=device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, _ = engine.load_model(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    try:
        engine.train(args.epochs)
        print("\n🎉 Training completed successfully!")
        
        # Run inference on test set after training
        print("\nRunning final inference on test set...")
        inference_results = engine.inference(
            test_loader, 
            save_path=os.path.join(engine.output_dir, 'final_inference_results.json')
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\nTraining results saved to: {engine.output_dir}")
        if engine.best_model_path:
            print(f"Best model saved to: {engine.best_model_path}")
            print(f"Best test Dice score: {engine.best_test_dice:.4f}")


if __name__ == "__main__":
    main()