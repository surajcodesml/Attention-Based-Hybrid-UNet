"""
Inference Script for Attention-Based Hybrid U-Net OCT Layer Segmentation

This script loads a trained model and generates predictions on test data.
The predictions are saved in HDF5 format similar to the original dataset structure.
"""

import os
import sys
import json
import h5py
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src directory to path
sys.path.append('scr')

from model import HybridAttentionUNet
from dataset import create_data_splits, OCTDataset
from engine import mask_to_coordinates


class InferenceEngine:
    """
    Inference engine for generating predictions and saving to HDF5.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: torch.device = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to the trained model (.pth file)
            config_path: Path to the configuration file
            device: Device to run inference on
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        print(f"Inference engine initialized")
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model config from checkpoint or use defaults
        model_config = checkpoint.get('config', {}).get('model', {})
        in_channels = model_config.get('input_channels', 1)
        out_channels = model_config.get('output_channels', 3)
        base_channels = model_config.get('base_channels', 64)
        
        # Create model with correct parameters
        model = HybridAttentionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully")
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Model parameters: in_channels={in_channels}, out_channels={out_channels}, base_channels={base_channels}")
        
        return model
    
    def predict_on_dataset(
        self,
        dataset: OCTDataset,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on a dataset.
        
        Args:
            dataset: Dataset to run inference on
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (prediction_masks, pred_ilm_coords, pred_bm_coords, confidence_maps)
        """
        print(f"Running inference on {len(dataset)} samples...")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        prediction_masks = []
        pred_ilm_coords = []
        pred_bm_coords = []
        confidence_maps = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Generating predictions")
            
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Convert logits to probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predicted classes
                pred_classes = torch.argmax(outputs, dim=1)
                
                # Convert to numpy
                pred_masks_batch = pred_classes.cpu().numpy()
                prob_maps_batch = probabilities.cpu().numpy()
                
                # Process each sample in the batch
                for i in range(pred_masks_batch.shape[0]):
                    mask = pred_masks_batch[i]
                    prob_map = prob_maps_batch[i]
                    
                    # Convert mask to coordinates
                    ilm_coords, bm_coords = mask_to_coordinates(mask)
                    
                    # Calculate confidence (max probability for each pixel)
                    confidence = np.max(prob_map, axis=0)
                    
                    # Store results
                    prediction_masks.append(mask)
                    pred_ilm_coords.append(ilm_coords)
                    pred_bm_coords.append(bm_coords)
                    confidence_maps.append(confidence)
                
                progress_bar.set_postfix({
                    'Processed': f'{(batch_idx + 1) * batch_size}',
                    'Total': len(dataset)
                })
        
        # Convert to numpy arrays
        prediction_masks = np.array(prediction_masks)
        pred_ilm_coords = np.array(pred_ilm_coords)
        pred_bm_coords = np.array(pred_bm_coords)
        confidence_maps = np.array(confidence_maps)
        
        print(f"Inference completed!")
        print(f"Generated predictions for {len(prediction_masks)} samples")
        print(f"Prediction masks shape: {prediction_masks.shape}")
        print(f"ILM coordinates shape: {pred_ilm_coords.shape}")
        print(f"BM coordinates shape: {pred_bm_coords.shape}")
        print(f"Confidence maps shape: {confidence_maps.shape}")
        
        return prediction_masks, pred_ilm_coords, pred_bm_coords, confidence_maps
    
    def save_predictions_hdf5(
        self,
        predictions: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        original_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        save_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save predictions to HDF5 file with structure similar to original dataset.
        
        Args:
            predictions: Tuple of (prediction_masks, pred_ilm_coords, pred_bm_coords, confidence_maps)
            original_data: Tuple of (images, true_ilm_coords, true_bm_coords)
            save_path: Path to save the HDF5 file
            metadata: Optional metadata to include
        """
        prediction_masks, pred_ilm_coords, pred_bm_coords, confidence_maps = predictions
        images, true_ilm_coords, true_bm_coords = original_data
        
        print(f"Saving predictions to {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with h5py.File(save_path, 'w') as f:
            # Save original data
            f.create_dataset('images', data=images, compression='gzip')
            
            # Create groups for layers
            layers_group = f.create_group('layers')
            pred_layers_group = f.create_group('predicted_layers')
            
            # Save original layer coordinates
            layers_group.create_dataset('ILM', data=true_ilm_coords, compression='gzip')
            layers_group.create_dataset('BM', data=true_bm_coords, compression='gzip')
            
            # Save predicted layer coordinates
            pred_layers_group.create_dataset('ILM', data=pred_ilm_coords, compression='gzip')
            pred_layers_group.create_dataset('BM', data=pred_bm_coords, compression='gzip')
            
            # Save prediction masks and confidence maps
            f.create_dataset('prediction_masks', data=prediction_masks, compression='gzip')
            f.create_dataset('confidence_maps', data=confidence_maps, compression='gzip')
            
            # Add metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        meta_group.attrs[key] = value
                    else:
                        meta_group.attrs[key] = str(value)
            
            # Add dataset info
            info_group = f.create_group('dataset_info')
            info_group.attrs['num_samples'] = len(images)
            info_group.attrs['image_height'] = images.shape[1]
            info_group.attrs['image_width'] = images.shape[2]
            info_group.attrs['prediction_classes'] = 3
            info_group.attrs['creation_date'] = datetime.now().isoformat()
            info_group.attrs['model_path'] = self.model_path
        
        print(f"Predictions saved successfully!")
        print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    
    def run_inference(
        self,
        output_dir: str = "inference_results",
        test_split: bool = True,
        train_split: bool = False,
        batch_size: int = 8
    ) -> Dict[str, str]:
        """
        Run complete inference pipeline.
        
        Args:
            output_dir: Directory to save results
            test_split: Whether to run inference on test split
            train_split: Whether to run inference on train split
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create datasets
        train_dataset, test_dataset = create_data_splits(self.config_path)
        
        saved_files = {}
        
        # Run inference on test set
        if test_split:
            print("\n" + "="*50)
            print("Running inference on TEST set")
            print("="*50)
            
            # Generate predictions
            predictions = self.predict_on_dataset(test_dataset, batch_size)
            
            # Get original data
            original_images = test_dataset.images
            original_ilm = test_dataset.ilm_coords
            original_bm = test_dataset.bm_coords
            original_data = (original_images, original_ilm, original_bm)
            
            # Prepare metadata
            metadata = {
                'split': 'test',
                'dataset_size': len(test_dataset),
                'model_architecture': 'Hybrid Attention U-Net',
                'inference_timestamp': timestamp
            }
            
            # Save to HDF5
            test_save_path = os.path.join(output_dir, f'test_predictions_{timestamp}.h5')
            self.save_predictions_hdf5(predictions, original_data, test_save_path, metadata)
            saved_files['test_predictions'] = test_save_path
        
        # Run inference on train set
        if train_split:
            print("\n" + "="*50)
            print("Running inference on TRAIN set")
            print("="*50)
            
            # Generate predictions
            predictions = self.predict_on_dataset(train_dataset, batch_size)
            
            # Get original data
            original_images = train_dataset.images
            original_ilm = train_dataset.ilm_coords
            original_bm = train_dataset.bm_coords
            original_data = (original_images, original_ilm, original_bm)
            
            # Prepare metadata
            metadata = {
                'split': 'train',
                'dataset_size': len(train_dataset),
                'model_architecture': 'Hybrid Attention U-Net',
                'inference_timestamp': timestamp
            }
            
            # Save to HDF5
            train_save_path = os.path.join(output_dir, f'train_predictions_{timestamp}.h5')
            self.save_predictions_hdf5(predictions, original_data, train_save_path, metadata)
            saved_files['train_predictions'] = train_save_path
        
        # Save inference summary
        summary = {
            'inference_timestamp': timestamp,
            'model_path': self.model_path,
            'config_path': self.config_path,
            'device': str(self.device),
            'batch_size': batch_size,
            'saved_files': saved_files
        }
        
        summary_path = os.path.join(output_dir, f'inference_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        saved_files['summary'] = summary_path
        
        print(f"\n🎉 Inference completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Summary: {summary_path}")
        
        return saved_files


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Run inference with trained Hybrid Attention U-Net')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--test_only', action='store_true',
                       help='Run inference only on test set')
    parser.add_argument('--train_only', action='store_true',
                       help='Run inference only on train set')
    parser.add_argument('--both', action='store_true',
                       help='Run inference on both train and test sets')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    # Determine what to run
    if args.both:
        run_train = True
        run_test = True
    elif args.train_only:
        run_train = True
        run_test = False
    elif args.test_only:
        run_train = False
        run_test = True
    else:
        # Default: run on test set only
        run_train = False
        run_test = True
    
    # Create inference engine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_engine = InferenceEngine(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device
    )
    
    # Run inference
    saved_files = inference_engine.run_inference(
        output_dir=args.output_dir,
        test_split=run_test,
        train_split=run_train,
        batch_size=args.batch_size
    )
    
    print(f"\nFiles saved:")
    for key, path in saved_files.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()