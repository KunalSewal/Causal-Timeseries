"""
Temporal Convolutional Network (TCN) for Granger Causality

TCN uses dilated causal convolutions to capture temporal patterns efficiently.
"""

import torch
import torch.nn as nn
from typing import List


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that ensures no future information leakage.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution."""
        x = self.conv(x)
        # Remove future timesteps
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    """
    Temporal block with residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNGranger(nn.Module):
    """
    Temporal Convolutional Network for Granger Causality.
    
    TCNs are often faster and more efficient than RNNs for sequence modeling.
    
    Args:
        num_vars: Number of variables
        num_channels: List of channel sizes for each layer
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    
    Example:
        >>> model = TCNGranger(num_vars=4, num_channels=[32, 64, 32], kernel_size=3)
        >>> x = torch.randn(32, 4, 10)  # (batch, vars, time)
        >>> out = model(x)  # (batch, vars)
    """
    
    def __init__(
        self,
        num_vars: int,
        num_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.num_vars = num_vars
        self.num_channels = num_channels
        self.device = device
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_vars if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Output layer
        self.fc = nn.Linear(num_channels[-1], num_vars)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, lag, num_vars)
        
        Returns:
            predictions: (batch, num_vars)
        """
        # TCN expects (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, num_vars, lag)
        
        # Apply TCN
        out = self.network(x)  # (batch, num_channels[-1], lag)
        
        # Use last timestep
        out = out[:, :, -1]  # (batch, num_channels[-1])
        
        # Predict
        predictions = self.fc(out)  # (batch, num_vars)
        
        return predictions


if __name__ == "__main__":
    # Example usage
    model = TCNGranger(num_vars=4, num_channels=[32, 64, 32])
    x = torch.randn(32, 5, 4)  # (batch, lag, vars)
    
    predictions = model(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
