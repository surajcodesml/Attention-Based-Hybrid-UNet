# Attention-Based Hybrid U-Net for OCT Layer Segmentation - Clean Version

This is the cleaned and updated version of the Attention-Based Hybrid U-Net implementation for OCT layer segmentation. The code has been refactored to be more efficient and focused.

## Key Changes Made

### 1. Dataset Module (`scr/dataset.py`)
- ✅ Removed test code and unnecessary validation logic
- ✅ Updated to use train/test splits only (no validation split)
- ✅ Cleaned up function signatures and documentation
- ✅ Kept all preprocessing and augmentation functionality

### 2. Training Engine (`scr/engine.py`)
- ✅ Updated to work with train/test splits instead of train/val/test
- ✅ Added comprehensive inference functionality
- ✅ Models are saved as `.pth` files with full checkpoint information
- ✅ Loss and performance metrics are saved as `.json` files
- ✅ Removed test code and cleaned up imports
- ✅ Enhanced plotting and results saving

### 3. Training Script (`train.py`)
- ✅ Updated to use the new train/test setup
- ✅ Fixed imports and model creation
- ✅ Added automatic inference run after training completion

### 4. New Inference Script (`inference.py`)
- ✅ Standalone script for loading trained models and generating predictions
- ✅ Saves predictions as HDF5 dataset with structure similar to original data
- ✅ Includes original data, predicted coordinates, masks, and confidence maps
- ✅ Command-line interface for easy usage

## Usage

### Training the Model

```bash
# Basic training with default config
python train.py

# Training with custom config and epochs
python train.py --config config.yaml --epochs 50

# Resume training from checkpoint
python train.py --resume runs/hybrid_unet_20250917_120000/best_model.pth --epochs 100
```

### Running Inference

```bash
# Run inference on test set (default)
python inference.py --model_path runs/hybrid_unet_20250917_120000/best_model.pth

# Run inference on both train and test sets
python inference.py --model_path runs/hybrid_unet_20250917_120000/best_model.pth --both

# Run inference with custom batch size and output directory
python inference.py --model_path models/best_model.pth --batch_size 16 --output_dir my_results
```

## File Structure

```
Attention-Based-Hybrid-UNet/
├── scr/
│   ├── __init__.py
│   ├── dataset.py          # Clean dataset implementation (train/test only)
│   ├── engine.py           # Training engine with inference capability
│   └── model.py            # Hybrid Attention U-Net model
├── train.py                # Main training script
├── inference.py            # Standalone inference script
├── config.yaml             # Configuration file
└── README.md               # This file
```

## Output Files

### Training Outputs
- `runs/hybrid_unet_TIMESTAMP/`
  - `best_model.pth` - Best model checkpoint
  - `latest_model.pth` - Latest model checkpoint
  - `training_plots.png` - Training progress plots
  - `training_results.json` - Complete training history and metrics
  - `training_summary.json` - Summary of key metrics
  - `final_inference_results.json` - Inference results on test set

### Inference Outputs
- `inference_results/` (or custom output directory)
  - `test_predictions_TIMESTAMP.h5` - Test set predictions in HDF5 format
  - `train_predictions_TIMESTAMP.h5` - Train set predictions (if requested)
  - `inference_summary_TIMESTAMP.json` - Inference run summary

## HDF5 Dataset Structure

The inference script saves predictions in HDF5 format with the following structure:

```
predictions.h5
├── images                    # Original images (N, H, W)
├── layers/
│   ├── ILM                  # True ILM coordinates (N, W)
│   └── BM                   # True BM coordinates (N, W)
├── predicted_layers/
│   ├── ILM                  # Predicted ILM coordinates (N, W)
│   └── BM                   # Predicted BM coordinates (N, W)
├── prediction_masks         # Segmentation masks (N, H, W)
├── confidence_maps          # Confidence maps (N, H, W)
├── metadata/                # Inference metadata
└── dataset_info/            # Dataset information
```

## Key Features

1. **Clean Architecture**: Removed unnecessary validation logic and test code
2. **Efficient Training**: Train/test split only for faster training cycles
3. **Comprehensive Metrics**: Both segmentation and regression metrics tracked
4. **Flexible Inference**: Standalone inference script with HDF5 output
5. **Easy Model Management**: Models saved as `.pth` with full checkpoints
6. **Rich Outputs**: JSON files with detailed metrics and plots

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install albumentations
pip install h5py
pip install matplotlib
pip install tqdm
pip install scipy
pip install pyyaml
pip install numpy
```

## Notes

- The model automatically saves the best checkpoint based on test set Dice score
- Training progress is plotted and saved every 10 epochs
- Early stopping is implemented to prevent overfitting
- All metrics (segmentation and regression) are tracked throughout training
- The inference script can handle both train and test set predictions
- HDF5 outputs preserve the original dataset structure for easy analysis