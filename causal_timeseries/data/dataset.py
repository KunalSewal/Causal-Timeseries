"""
PyTorch Dataset for Time Series

Efficient sliding window dataset for causal discovery.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for multivariate time series with sliding windows.
    
    Args:
        data: DataFrame, numpy array, or path to CSV
        lag: Number of lagged timesteps
        horizon: Prediction horizon (default: 1)
        split: 'train', 'val', 'test', or None
        split_ratios: (train, val, test) proportions
        
    Example:
        >>> df = pd.read_csv('stocks.csv', index_col=0)
        >>> dataset = TimeSeriesDataset(df, lag=5, split='train')
        >>> X, y = dataset[0]  # X: (lag, num_vars), y: (num_vars,)
    """
    
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame, np.ndarray],
        lag: int = 5,
        horizon: int = 1,
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        self.lag = lag
        self.horizon = horizon
        
        # Load data
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data, index_col=0, parse_dates=True)
            self.data = df.values.astype(np.float32)
            self.columns = list(df.columns)
        elif isinstance(data, pd.DataFrame):
            self.data = data.values.astype(np.float32)
            self.columns = list(data.columns)
        else:
            self.data = np.array(data, dtype=np.float32)
            self.columns = [f"var_{i}" for i in range(data.shape[1])]
        
        self.num_vars = self.data.shape[1]
        total_samples = self.data.shape[0]
        
        # Apply split
        if split is not None:
            assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
            assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1"
            
            train_size = int(total_samples * split_ratios[0])
            val_size = int(total_samples * split_ratios[1])
            
            if split == 'train':
                self.data = self.data[:train_size]
            elif split == 'val':
                self.data = self.data[train_size:train_size+val_size]
            else:  # test
                self.data = self.data[train_size+val_size:]
            
            logger.info(f"{split} split: {len(self.data)} samples")
        
        # Number of sequences we can create
        self.num_sequences = len(self.data) - lag - horizon + 1
        
        if self.num_sequences <= 0:
            raise ValueError(f"Not enough data for lag={lag}, horizon={horizon}")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence and target.
        
        Returns:
            X: (lag, num_vars) - input sequence
            y: (num_vars,) - prediction target
        """
        # Input: lag timesteps
        X = self.data[idx:idx+self.lag]
        
        # Target: value at t+horizon
        y = self.data[idx+self.lag+self.horizon-1]
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def get_full_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all sequences as tensors."""
        X_list = []
        y_list = []
        
        for i in range(len(self)):
            X, y = self[i]
            X_list.append(X)
            y_list.append(y)
        
        return torch.stack(X_list), torch.stack(y_list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to DataFrame."""
        return pd.DataFrame(self.data, columns=self.columns)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    data = np.random.randn(1000, 4)
    dataset = TimeSeriesDataset(data, lag=5, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of variables: {dataset.num_vars}")
    
    X, y = dataset[0]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
