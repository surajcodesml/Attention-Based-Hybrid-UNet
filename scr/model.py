"""
Hybrid Attention-Based U-Net Model for OCT Layer Segmentation

This module implements the attention-based hybrid U-Net architecture 
as described in the research paper for retinal layer segmentation.

Key Components:
- Edge Attention Block with Canny edge detection
- Spatial Attention Block
- 5-layer encoder-decoder U-Net architecture
- Skip connections with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple


class EdgeAttentionBlock(nn.Module):
    """
    Edge Attention Block that enhances edge features using Canny edge detection.
    
    This block combines feature maps with edge information to improve
    segmentation accuracy at layer boundaries.
    """
    
    def __init__(self, in_channels: int, gate_channels: int):
        """
        Initialize Edge Attention Block.
        
        Args:
            in_channels: Number of input channels
            gate_channels: Number of channels for the gating mechanism
        """
        super(EdgeAttentionBlock, self).__init__()
        
        self.in_channels = in_channels
        self.gate_channels = gate_channels
        
        # Convolutional layers for feature processing
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Edge enhancement layer
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def canny_edge_detection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Canny edge detection to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Edge map tensor of shape (B, 1, H, W)
        """
        batch_size = x.size(0)
        edge_maps = []
        
        # Convert to numpy and process each image in batch
        x_np = x.detach().cpu().numpy()
        
        for i in range(batch_size):
            # Take the first channel if multi-channel
            img = x_np[i, 0] if x.size(1) > 1 else x_np[i, 0]
            
            # Normalize to 0-255 range
            img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            
            # Apply Canny edge detection
            edges = cv2.Canny(img_normalized, threshold1=50, threshold2=150)
            
            # Convert back to tensor
            edge_tensor = torch.from_numpy(edges).float() / 255.0
            edge_maps.append(edge_tensor)
        
        # Stack and add channel dimension
        edge_output = torch.stack(edge_maps).unsqueeze(1)
        
        return edge_output.to(x.device)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Edge Attention Block.
        
        Args:
            x: Input feature map (skip connection)
            g: Gating signal from decoder
            
        Returns:
            Attention-enhanced feature map
        """
        # Generate edge maps from input features
        edge_map = self.canny_edge_detection(x)
        
        # Attention mechanism
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample gating signal if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        attended_x = x * psi
        
        # Combine with edge information
        edge_enhanced = torch.cat([attended_x, edge_map], dim=1)
        output = self.edge_conv(edge_enhanced)
        
        return output


class SpatialAttentionBlock(nn.Module):
    """
    Spatial Attention Block that focuses on important spatial regions.
    
    This block computes spatial attention weights to highlight
    relevant regions for segmentation.
    """
    
    def __init__(self, in_channels: int, gate_channels: int, inter_channels: int = None):
        """
        Initialize Spatial Attention Block.
        
        Args:
            in_channels: Number of input channels
            gate_channels: Number of channels for the gating mechanism
            inter_channels: Number of intermediate channels
        """
        super(SpatialAttentionBlock, self).__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        
        self.in_channels = in_channels
        self.gate_channels = gate_channels
        self.inter_channels = inter_channels
        
        # Gating signal processing
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Input feature processing
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Attention map generation
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Spatial attention refinement
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Spatial Attention Block.
        
        Args:
            x: Input feature map (skip connection)
            g: Gating signal from decoder
            
        Returns:
            Spatially attended feature map
        """
        # Process gating signal and input features
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample gating signal if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # Generate attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply spatial attention
        attended_x = x * psi
        
        # Spatial refinement
        refined_x = self.spatial_conv(attended_x)
        output = x + refined_x  # Residual connection
        
        return output


class ConvBlock(nn.Module):
    """
    Convolutional block with two convolutions, batch normalization, and ReLU.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class UpConvBlock(nn.Module):
    """
    Upsampling block with transposed convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConvBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class HybridAttentionUNet(nn.Module):
    """
    Hybrid Attention-Based U-Net for OCT layer segmentation.
    
    Features:
    - 5-layer encoder-decoder architecture
    - Edge attention blocks in shallow layers (for edge information)
    - Spatial attention blocks in deeper layers (for spatial context)
    - Skip connections with attention mechanisms
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 64):
        """
        Initialize Hybrid Attention U-Net.
        
        Args:
            in_channels: Number of input channels (1 for grayscale OCT images)
            out_channels: Number of output classes (3 for our segmentation task)
            base_channels: Base number of channels (doubles at each encoder level)
        """
        super(HybridAttentionUNet, self).__init__()
        
        # Calculate channel sizes for 5 levels
        self.channels = [base_channels * (2 ** i) for i in range(5)]  # [64, 128, 256, 512, 1024]
        
        # Encoder (Contracting path)
        self.encoder1 = ConvBlock(in_channels, self.channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock(self.channels[0], self.channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock(self.channels[1], self.channels[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = ConvBlock(self.channels[2], self.channels[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(self.channels[3], self.channels[4])
        
        # Decoder (Expanding path)
        self.upconv4 = UpConvBlock(self.channels[4], self.channels[3])
        self.decoder4 = ConvBlock(self.channels[4], self.channels[3])
        
        self.upconv3 = UpConvBlock(self.channels[3], self.channels[2])
        self.decoder3 = ConvBlock(self.channels[3], self.channels[2])
        
        self.upconv2 = UpConvBlock(self.channels[2], self.channels[1])
        self.decoder2 = ConvBlock(self.channels[2], self.channels[1])
        
        self.upconv1 = UpConvBlock(self.channels[1], self.channels[0])
        self.decoder1 = ConvBlock(self.channels[1], self.channels[0])
        
        # Attention blocks
        # Edge attention for shallow layers (better for edge detection)
        self.edge_att1 = EdgeAttentionBlock(self.channels[0], self.channels[1])
        self.edge_att2 = EdgeAttentionBlock(self.channels[1], self.channels[2])
        
        # Spatial attention for deeper layers (better for spatial context)
        self.spatial_att3 = SpatialAttentionBlock(self.channels[2], self.channels[3])
        self.spatial_att4 = SpatialAttentionBlock(self.channels[3], self.channels[4])
        
        # Final output layer
        self.final_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Hybrid Attention U-Net.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Output segmentation map of shape (B, 3, H, W)
        """
        # Encoder path
        e1 = self.encoder1(x)       # [B, 64, H, W]
        p1 = self.pool1(e1)         # [B, 64, H/2, W/2]
        
        e2 = self.encoder2(p1)      # [B, 128, H/2, W/2]
        p2 = self.pool2(e2)         # [B, 128, H/4, W/4]
        
        e3 = self.encoder3(p2)      # [B, 256, H/4, W/4]
        p3 = self.pool3(e3)         # [B, 256, H/8, W/8]
        
        e4 = self.encoder4(p3)      # [B, 512, H/8, W/8]
        p4 = self.pool4(e4)         # [B, 512, H/16, W/16]
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)  # [B, 1024, H/16, W/16]
        
        # Decoder path with attention
        # Level 4: Spatial attention for deep features
        up4 = self.upconv4(bottleneck)    # [B, 512, H/8, W/8]
        att4 = self.spatial_att4(e4, bottleneck)  # Apply spatial attention
        d4 = torch.cat([up4, att4], dim=1)  # [B, 1024, H/8, W/8]
        d4 = self.decoder4(d4)              # [B, 512, H/8, W/8]
        
        # Level 3: Spatial attention for mid-level features
        up3 = self.upconv3(d4)            # [B, 256, H/4, W/4]
        att3 = self.spatial_att3(e3, d4)  # Apply spatial attention
        d3 = torch.cat([up3, att3], dim=1)  # [B, 512, H/4, W/4]
        d3 = self.decoder3(d3)              # [B, 256, H/4, W/4]
        
        # Level 2: Edge attention for shallow features
        up2 = self.upconv2(d3)            # [B, 128, H/2, W/2]
        att2 = self.edge_att2(e2, d3)     # Apply edge attention
        d2 = torch.cat([up2, att2], dim=1)  # [B, 256, H/2, W/2]
        d2 = self.decoder2(d2)              # [B, 128, H/2, W/2]
        
        # Level 1: Edge attention for finest features
        up1 = self.upconv1(d2)            # [B, 64, H, W]
        att1 = self.edge_att1(e1, d2)     # Apply edge attention
        d1 = torch.cat([up1, att1], dim=1)  # [B, 128, H, W]
        d1 = self.decoder1(d1)              # [B, 64, H, W]
        
        # Final output
        output = self.final_conv(d1)       # [B, 3, H, W]
        
        return output


def create_model(config: dict) -> HybridAttentionUNet:
    """
    Create Hybrid Attention U-Net model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    model = HybridAttentionUNet(
        in_channels=model_config.get('input_channels', 1),
        out_channels=model_config.get('output_channels', 3),
        base_channels=model_config.get('base_channels', 64)
    )
    
    return model


if __name__ == "__main__":
    """Test the model implementation."""
    print("Testing Hybrid Attention U-Net...")
    
    # Create model
    model = HybridAttentionUNet(in_channels=1, out_channels=3, base_channels=64)
    
    # Test input
    batch_size = 2
    height, width = 480, 768
    x = torch.randn(batch_size, 1, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 3, {height}, {width})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("✓ Model test passed!")
