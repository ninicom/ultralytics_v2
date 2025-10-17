import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os

"""Spatial-Channel Fusion Block"""
class SCFBlock(nn.Module):
    """Spatial-Channel Fusion Block for feature enhancement."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Branch 1: Depthwise convolution for spatial feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # Branch 2: Pointwise convolution for channel-wise feature extraction
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # Branch 3: Batch normalization for feature fusion
        self.branch3 = nn.BatchNorm2d(channels)

        # Final output layer: Pointwise convolution for final feature fusion
        self.out = self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        """Forward pass for SCFBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        x1 = self.branch1(x)  # Spatial features
        x2 = self.branch2(x)  # Channel-wise features
        x3 = self.branch3(x)  # Normalized input
        output = nn.Concat(dim=1)([x1, x2, x3])  # Concatenate features
        output = self.out(output)  # Final fusion
        return output
    
""" Residual Block with two depthwise convolutions """
class ResBlock(nn.Module):
    """Residual Block with two depthwise convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # First depthwise convolution
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.silu = nn.SiLU()
        
        # Second depthwise convolution
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        """Forward pass for ResBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        residual = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.silu(out + residual)
    
class ContextBlock(nn.Module):
    """Context Block for capturing contextual information."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Two residual blocks for deep feature extraction
        self.res1 = ResBlock(channels)
        self.res2 = ResBlock(channels)

        # Depthwise convolutions for context enhancement
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        """Forward pass for ContextBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        x = self.res1(x)
        x1 = F.softmax(self.conv1(x), dim=2)  # Softmax along height dimension
        x2 = x + x1
        x2 = self.silu(self.conv2(x2))
        x3 = x + x2
        return self.res2(x3)
    
class DAB(nn.Module):
    """Dual Attention Block combining SCF and Context Blocks."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.scf_block = SCFBlock(channels)
        self.context_block = ContextBlock(channels)
        
        # Fusion layer to combine outputs
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channels)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        """Forward pass for DAB.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        x1 = self.scf_block(x)
        x2 = self.context_block(x)
        out = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
        out = self.silu(self.bn(self.conv(out)))
        return out
    
class LFAB(nn.Module):
    """Lightweight Feature Aggregation Block with multi-scale pooling."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Reduce to 4 channels for multi-scale pooling
        self.reduce = nn.Conv2d(channels, 4, kernel_size=1, bias=False)
        
        # Restore to original channel count
        self.restore = nn.Conv2d(4, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass for LFAB.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        x = self.reduce(x)  # Reduce to [B, 4, H, W]

        # Split into individual channels for different pooling scales
        x1 = x[:, 0:1, :, :]  # 1x1 pooling
        x2 = x[:, 1:2, :, :]  # 2x2 pooling
        x3 = x[:, 2:3, :, :]  # 3x3 pooling
        x4 = x[:, 3:4, :, :]  # 6x6 pooling

        # Apply adaptive average pooling and interpolate back to original size
        x1 = F.interpolate(F.adaptive_avg_pool2d(x1, 1), size=(H, W), mode='nearest')
        x2 = F.interpolate(F.adaptive_avg_pool2d(x2, 2), size=(H, W), mode='nearest')
        x3 = F.interpolate(F.adaptive_avg_pool2d(x3, 3), size=(H, W), mode='nearest')
        x4 = F.interpolate(F.adaptive_avg_pool2d(x4, 6), size=(H, W), mode='nearest')

        # Concatenate pooled features
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.act(self.bn(self.restore(out)))
        return out
    
class CustomModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.EEB = SCFBlock(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.EEB(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
