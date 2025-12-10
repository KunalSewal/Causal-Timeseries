"""Temporal Convolutional Network for Granger Causality."""

import torch
import torch.nn as nn
from typing import List


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with dilated convolutions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Size of convolving kernel
        stride: Stride of convolution
        dilation: Dilation factor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from end of sequence."""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNGranger(nn.Module):
    """
    Temporal Convolutional Network for Granger Causality.
    
    Uses dilated causal convolutions for efficient long-range modeling.
    
    Args:
        num_vars: Number of variables
        num_channels: List of channels per layer
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        
    Example:
        >>> model = TCNGranger(num_vars=4, num_channels=[64, 64, 64])
        >>> x = torch.randn(32, 5, 4)  # (batch, seq, features)
        >>> out = model(x)  # (batch, features)
    """
    
    def __init__(
        self,
        num_vars: int,
        num_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        
        self.num_vars = num_vars
        self.device = device
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_vars if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_vars)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, num_vars)
            
        Returns:
            predictions: (batch, num_vars)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Pass through TCN
        out = self.network(x)  # (batch, channels, seq_len)
        
        # Use last timestep
        out = out[:, :, -1]  # (batch, channels)
        
        # Prediction
        predictions = self.fc(out)
        
        return predictions


if __name__ == "__main__":
    # Test
    model = TCNGranger(num_vars=4, num_channels=[32, 32, 32])
    x = torch.randn(16, 10, 4)  # batch=16, seq=10, vars=4
    
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
