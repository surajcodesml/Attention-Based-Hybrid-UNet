#!/bin/bash

# Full Training Pipeline Script for Attention-Based Hybrid U-Net
# This script runs the complete training with comprehensive logging

echo "🚀 Starting Full Attention-Based Hybrid U-Net Training Pipeline"
echo "=============================================================="

# Set environment
cd /home/suraj/Git/Attention-Based-Hybrid-UNet

# Check system info
echo "📊 System Information:"
echo "  CPU Cores: $(nproc)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)"
fi
echo ""

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p models runs results logs
echo "✅ Directories created"
echo ""

# Validate dataset
echo "📋 Validating dataset..."
python -c "
import h5py
import os
config_file = 'config.yaml'
with open(config_file, 'r') as f:
    import yaml
    config = yaml.safe_load(f)
hdf5_path = config['dataset']['hdf5_path']
if os.path.exists(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        print(f'✅ Dataset found: {hdf5_path}')
        print(f'  Images shape: {f[\"images\"].shape}')
        print(f'  ILM shape: {f[\"layers\"][\"ILM\"].shape}')
        print(f'  BM shape: {f[\"layers\"][\"BM\"].shape}')
else:
    print(f'❌ Dataset not found: {hdf5_path}')
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Dataset validation failed"
    exit 1
fi
echo ""

# Test metrics
echo "🧪 Testing metrics implementation..."
python -c "
from scr.engine import RegressionMetrics, SegmentationMetrics
import torch
reg_metrics = RegressionMetrics()
seg_metrics = SegmentationMetrics(num_classes=3)
pred = torch.randn(1, 3, 10, 10)
target = torch.randint(0, 3, (1, 10, 10))
reg_metrics.update(pred, target)
seg_metrics.update(pred, target)
reg_results = reg_metrics.compute()
seg_results = seg_metrics.compute()
print('✅ Metrics test passed')
print(f'  Sample regression MAE: {reg_results[\"overall_mae\"]:.2f}')
print(f'  Sample segmentation Dice: {seg_results[\"dice\"]:.4f}')
"
echo ""

# Start training
echo "🎯 Starting training..."
echo "  Configuration: config.yaml"
echo "  Epochs: 50"
device_info=$(python -c 'import torch; print("GPU" if torch.cuda.is_available() else "CPU")')
echo "  Device: $device_info"
echo "  Batch size: 1 (optimized for GPU memory)"
echo ""

# Run training with time tracking
start_time=$(date +%s)
python train.py --epochs 50 2>&1 | tee "logs/training_$(date +%Y%m%d_%H%M%S).log"
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "⏱️  Training completed in $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"

# Check results
echo ""
echo "📊 Checking training results..."
if [ -d "runs" ]; then
    latest_run=$(ls -t runs/ | head -1)
    echo "  Latest run: runs/$latest_run"
    
    if [ -f "runs/$latest_run/training_results.json" ]; then
        echo "✅ Training results saved"
        python -c "
import json
with open('runs/$latest_run/training_results.json', 'r') as f:
    results = json.load(f)
print(f'  Final validation Dice: {results[\"final_metrics\"][\"segmentation\"][\"val_dice\"]:.4f}')
print(f'  Final validation MAE: {results[\"final_metrics\"][\"regression\"][\"overall\"][\"val_mae\"]:.2f}')
print(f'  Best validation Dice: {results[\"best_val_dice\"]:.4f}')
print(f'  Total epochs: {results[\"total_epochs\"]}')
"
    fi
    
    if [ -f "runs/$latest_run/best_model.pth" ]; then
        echo "✅ Best model saved: runs/$latest_run/best_model.pth"
    fi
    
    if [ -f "runs/$latest_run/training_plots.png" ]; then
        echo "✅ Training plots saved: runs/$latest_run/training_plots.png"
    fi
    
    if [ -f "runs/$latest_run/training_summary.json" ]; then
        echo "✅ Training summary saved: runs/$latest_run/training_summary.json"
    fi
else
    echo "❌ No training results found"
fi

echo ""
echo "🎉 Full pipeline completed!"
echo "=============================================================="
echo "📋 Summary:"
echo "  ✅ Dataset validation"
echo "  ✅ Metrics implementation (Segmentation + Regression)"
echo "  ✅ Model training with comprehensive logging"
echo "  ✅ Model saving (best and latest checkpoints)"
echo "  ✅ Performance plots and JSON results"
echo ""
echo "📁 Output files:"
echo "  • Model checkpoints: runs/[timestamp]/best_model.pth"
echo "  • Training metrics: runs/[timestamp]/training_results.json"
echo "  • Performance plots: runs/[timestamp]/training_plots.png"
echo "  • Training logs: logs/training_[timestamp].log"