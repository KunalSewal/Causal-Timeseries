"""
Attention-based Granger Causality Model

Uses temporal attention mechanism to capture long-range dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class TemporalAttention(nn.Module):
    """
    Temporal attention layer for time series.
    
    Learns which past timesteps are most important for prediction.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention over temporal dimension.
        
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        
        Returns:
            context: (batch, hidden_dim) - weighted sum
            attention_weights: (batch, seq_len) - attention scores
        """
        # Compute attention scores
        scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_dim)
        
        return context, attention_weights.squeeze(-1)


class AttentionGranger(nn.Module):
    """
    Attention-based Neural Granger Causality.
    
    Uses attention mechanism to identify important temporal patterns
    and variable interactions.
    
    Args:
        num_vars: Number of variables
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        lag: Lag order
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_vars: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lag: int = 5,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lag = lag
        self.device = device
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=num_vars,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Variable attention (learn importance of each variable)
        self.variable_attention = nn.Linear(num_vars, num_vars)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_vars)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with attention.
        
        Args:
            x: (batch, lag, num_vars)
            return_attention: Whether to return attention weights
        
        Returns:
            predictions: (batch, num_vars)
            attention_weights: (batch, lag) if return_attention=True
        """
        # Variable attention
        var_weights = torch.sigmoid(self.variable_attention(x))
        x_weighted = x * var_weights
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x_weighted)  # (batch, lag, hidden_dim)
        
        # Temporal attention
        context, attention_weights = self.temporal_attention(lstm_out)
        
        # Prediction
        predictions = self.fc(context)
        
        if return_attention:
            return predictions, attention_weights
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Get attention weights for input sequence.
        
        Useful for interpreting which timesteps are important.
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            _, attention_weights = self.forward(x, return_attention=True)
            return attention_weights.cpu().numpy()


if __name__ == "__main__":
    # Example usage
    model = AttentionGranger(num_vars=4, hidden_dim=64, lag=5)
    x = torch.randn(32, 5, 4)  # batch=32, lag=5, vars=4
    
    predictions, attention = model(x, return_attention=True)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Attention weights (first sample): {attention[0]}")
