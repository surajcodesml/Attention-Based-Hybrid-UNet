# Attention-Based Hybrid U-Net for OCT Layer Segmentation

This repository contains a complete implementation of the attention-based hybrid U-Net architecture for retinal layer segmentation in Optical Coherence Tomography (OCT) images, based on the research paper "Advancing Ocular Imaging: A Hybrid Attention Mechanism-Based U-Net Model for Precise Segmentation of Sub-Retinal Layers in OCT Images".

## 🎯 Key Features

### ✅ Enhanced Dataset Implementation
- **Augmentation Multiplier**: Augmented images are added to the training set (not replaced), increasing dataset size by 2-3x
- **Comprehensive Preprocessing**: Gaussian blur and CLAHE for enhanced image quality
- **3-Class Segmentation**: Above ILM, ILM to BM, Below BM regions
- **Smart Resizing**: From 496×768 to 480×768 with proportional coordinate scaling
- **Robust Augmentation**: Geometric (both image+mask) and photometric (image only) transformations

### 🏗️ Model Architecture
- **Hybrid Attention Mechanisms**: 
  - Edge Attention Blocks for shallow layers (better edge detection)
  - Spatial Attention Blocks for deeper layers (better spatial context)
- **5-Layer U-Net**: Encoder-decoder with skip connections
- **Canny Edge Integration**: Edge attention enhanced with Canny edge detection
- **37.7M Parameters**: Optimized for segmentation performance

### 🚀 Training Engine
- **Combined Loss**: Cross Entropy + Dice Loss for optimal segmentation
- **Comprehensive Metrics**: Dice, IoU, Precision, Recall, F1 scores
- **Smart Training**: Early stopping, learning rate scheduling, model checkpointing
- **Visualization**: Automatic plotting of training progress and results

## 📁 Project Structure

```
Attention-Based-Hybrid-UNet/
├── config.yaml                 # Configuration file with all parameters
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
├── data/
│   └── Nemours_Jing_0805.h5   # OCT dataset
├── scr/
│   ├── __init__.py            
│   ├── dataset.py             # Dataset implementation with augmentation
│   ├── model.py               # Hybrid Attention U-Net model
│   └── engine.py              # Training and evaluation engine
└── runs/                      # Training outputs (auto-created)
    └── hybrid_unet_TIMESTAMP/
        ├── best_model.pth
        ├── latest_model.pth
        ├── training_plots.png
        └── training_results.json
```

## 🛠️ Setup and Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Attention-Based-Hybrid-UNet
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install packages
pip install torch torchvision
pip install h5py opencv-python scikit-image albumentations
pip install numpy matplotlib PyYAML tqdm
```

### 3. Prepare Dataset
Place your OCT dataset (`Nemours_Jing_0805.h5`) in the `data/` directory.

## 🚀 Usage

### Quick Start - Training
```bash
# Train with default settings (100 epochs, 2x augmentation)
python train.py

# Custom training configuration
python train.py --epochs 50 --augment_multiplier 3

# Resume from checkpoint
python train.py --resume runs/hybrid_unet_TIMESTAMP/best_model.pth
```

### Configuration
All parameters are configurable via `config.yaml`:

```yaml
# Dataset settings
dataset:
  target_size: {height: 480, width: 768}
  num_classes: 3

# Preprocessing
preprocessing:
  gaussian_blur: {enabled: true, kernel_size: 5, sigma: 1.0}
  clahe: {enabled: true, clip_limit: 2.0, tile_grid_size: [8, 8]}

# Training
training:
  batch_size: 8
  learning_rate: 0.001
  patience: 10
```

### Testing Dataset and Augmentation
```bash
# Test dataset loading and view augmentation examples
python scr/dataset.py
# This creates augmentation_test.png showing different augmentation types
```

### Model Architecture Testing
```bash
# Test model architecture
python scr/model.py
# Output: Model parameters and forward pass verification
```

## 📊 Dataset Details

### Input Data Format
- **Images**: 1266 OCT B-scans of size 496×768 pixels (grayscale)
- **Annotations**: ILM and BM layer coordinates (y-values for each x-position)
- **Format**: HDF5 file with 'images' and 'layers' datasets

### Data Splits
- **Training**: 80% (1012 original samples → 2024+ with augmentation)
- **Validation**: 10% (126 samples)  
- **Testing**: 10% (128 samples)

### Segmentation Classes
1. **Class 0**: Above ILM (vitreous/background)
2. **Class 1**: ILM to BM (retinal layers) 
3. **Class 2**: Below BM (choroid/sclera)

## 🏗️ Model Architecture Details

### Edge Attention Block
- Applied to shallow layers (levels 1-2)
- Integrates Canny edge detection with attention mechanism
- Enhances boundary detection accuracy

### Spatial Attention Block  
- Applied to deeper layers (levels 3-4)
- Focuses on spatial context and relationships
- Improves overall segmentation quality

### Network Flow
```
Input (1×480×768) 
    ↓ Encoder (5 levels)
    ↓ Bottleneck (1024 channels)
    ↓ Decoder with Attention
    ↓ Level 4: Spatial Attention
    ↓ Level 3: Spatial Attention  
    ↓ Level 2: Edge Attention
    ↓ Level 1: Edge Attention
Output (3×480×768)
```

## 📈 Training Process

### Loss Function
- **Combined Loss**: 0.5 × CrossEntropy + 0.5 × DiceLoss
- Optimized for both accurate classification and good overlap

### Optimization
- **Optimizer**: Adam (fallback from AdaBound)
- **Learning Rate**: 1e-3 with ReduceLROnPlateau scheduling
- **Early Stopping**: Patience of 10 epochs

### Metrics Tracked
- Dice Score, IoU, Precision, Recall, F1 Score
- Computed per-class and averaged (ignoring background)

## 📊 Expected Performance

Based on the research paper, the model achieves:
- **Average Dice Score**: 94.99%
- **Adjusted Rand Index**: 97.00%
- **Accuracy**: 97%

## 🔧 Advanced Usage

### Custom Dataset
To use your own OCT dataset:
1. Convert to HDF5 format with 'images' and 'layers' groups
2. Update `config.yaml` with new path and dimensions
3. Modify layer names in `dataset.py` if needed

### Model Modifications
- Adjust `base_channels` in model config for different model sizes
- Modify attention block placement in `HybridAttentionUNet`
- Add/remove augmentation types in config

### Hyperparameter Tuning
Key parameters to tune:
- `learning_rate`: Start with 1e-3, reduce if loss plateaus
- `batch_size`: Increase based on available memory
- `augment_multiplier`: 2-4x for more training data
- Loss weights in `CombinedLoss`

## 🐛 Troubleshooting

### Common Issues
1. **CUDA warnings**: Normal on CPU-only systems
2. **Memory errors**: Reduce batch_size in config
3. **Slow training**: Increase num_workers or use GPU
4. **Poor convergence**: Adjust learning_rate or check data quality

### Performance Tips
- Use GPU for faster training (model supports CUDA)
- Increase `num_workers` for faster data loading
- Monitor training plots for convergence issues

## 📝 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{karn2024advancing,
  title={Advancing Ocular Imaging: A Hybrid Attention Mechanism-Based U-Net Model for Precise Segmentation of Sub-Retinal Layers in OCT Images},
  author={Karn, Prakash Kumar and Abdulla, Waleed H.},
  journal={Bioengineering},
  volume={11},
  number={3},
  pages={240},
  year={2024},
  publisher={MDPI}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug fixes
- Feature enhancements  
- Documentation improvements
- Performance optimizations

## 📄 License

This project follows the same license as the original research paper (CC BY 4.0).

---

**Happy Training! 🚀**

For questions or issues, please check the troubleshooting section or open an issue.
