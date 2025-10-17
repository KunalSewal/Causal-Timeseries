"""
Data Loader Module

Handles loading and managing time series datasets for causal analysis.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Union, List
import pickle


def load_csv_data(
    filepath: Union[str, Path],
    timestamp_col: str = "timestamp",
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load time series data from CSV file.
    
    Args:
        filepath: Path to CSV file
        timestamp_col: Name of timestamp column
        value_cols: List of columns to load (if None, load all except timestamp)
    
    Returns:
        DataFrame with timestamp index and variable columns
    """
    df = pd.read_csv(filepath)
    
    # Set timestamp as index if present
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
    
    # Select specific columns if specified
    if value_cols is not None:
        df = df[value_cols]
    
    return df


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for multivariate time series with sliding window approach.
    
    Creates sequences of lagged observations for Granger causality analysis.
    
    Args:
        data: Either filepath (str/Path) or DataFrame/array
        lag: Number of lagged timesteps to include
        horizon: Prediction horizon (default: 1)
        split: One of 'train', 'val', 'test' or None
        split_ratios: Tuple of (train, val, test) ratios
    
    Example:
        >>> dataset = TimeSeriesDataset('data/stock_prices.csv', lag=5)
        >>> x, y = dataset[0]  # x: (lag, num_vars), y: (num_vars,)
    """
    
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame, np.ndarray],
        lag: int = 5,
        horizon: int = 1,
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        variable_names: Optional[List[str]] = None,
    ):
        self.lag = lag
        self.horizon = horizon
        self.split = split
        self.split_ratios = split_ratios
        
        # Load data
        if isinstance(data, (str, Path)):
            self.df = load_csv_data(data)
            self.data = self.df.values
            self.variable_names = list(self.df.columns)
        elif isinstance(data, pd.DataFrame):
            self.df = data
            self.data = data.values
            self.variable_names = list(data.columns)
        else:
            self.data = np.array(data)
            self.df = None
            self.variable_names = variable_names or [f"var_{i}" for i in range(self.data.shape[1])]
        
        self.num_vars = self.data.shape[1]
        self.num_samples = self.data.shape[0]
        
        # Apply train/val/test split if specified
        if split is not None:
            self._apply_split()
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _apply_split(self):
        """Split data into train/val/test sets."""
        train_ratio, val_ratio, test_ratio = self.split_ratios
        assert abs(sum(self.split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1"
        
        n = self.num_samples
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if self.split == 'train':
            self.data = self.data[:train_end]
        elif self.split == 'val':
            self.data = self.data[train_end:val_end]
        elif self.split == 'test':
            self.data = self.data[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
        
        self.num_samples = self.data.shape[0]
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged sequences for time series prediction.
        
        Returns:
            X: (num_sequences, lag, num_vars) - input sequences
            y: (num_sequences, num_vars) - target values
        """
        X, y = [], []
        
        for i in range(self.lag, self.num_samples - self.horizon + 1):
            X.append(self.data[i - self.lag:i])
            y.append(self.data[i + self.horizon - 1])
        
        return np.array(X), np.array(y)
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences[0])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and target.
        
        Returns:
            x: (lag, num_vars) tensor
            y: (num_vars,) tensor
        """
        X, y = self.sequences
        return torch.FloatTensor(X[idx]), torch.FloatTensor(y[idx])
    
    def get_full_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all sequences at once.
        
        Returns:
            X: (num_sequences, lag, num_vars)
            y: (num_sequences, num_vars)
        """
        X, y = self.sequences
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def save(self, filepath: Union[str, Path]):
        """Save dataset to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'lag': self.lag,
                'horizon': self.horizon,
                'variable_names': self.variable_names,
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TimeSeriesDataset':
        """Load dataset from pickle file."""
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
        
        return cls(
            data=saved['data'],
            lag=saved['lag'],
            horizon=saved['horizon'],
            variable_names=saved['variable_names'],
        )
    
    def get_var_index(self, var_name: str) -> int:
        """Get index of variable by name."""
        return self.variable_names.index(var_name)
    
    def get_var_name(self, var_idx: int) -> str:
        """Get name of variable by index."""
        return self.variable_names[var_idx]
    
    def __repr__(self) -> str:
        return (
            f"TimeSeriesDataset(num_vars={self.num_vars}, "
            f"num_sequences={len(self)}, lag={self.lag}, "
            f"horizon={self.horizon}, split={self.split})"
        )


class MultiDatasetLoader:
    """
    Manages multiple time series datasets with consistent preprocessing.
    
    Useful for batch experiments across different datasets.
    """
    
    def __init__(self, lag: int = 5, horizon: int = 1):
        self.lag = lag
        self.horizon = horizon
        self.datasets = {}
    
    def add_dataset(self, name: str, filepath: Union[str, Path]):
        """Add a dataset to the loader."""
        dataset = TimeSeriesDataset(filepath, lag=self.lag, horizon=self.horizon)
        self.datasets[name] = dataset
        return dataset
    
    def get_dataset(self, name: str) -> TimeSeriesDataset:
        """Get a dataset by name."""
        return self.datasets[name]
    
    def __getitem__(self, name: str) -> TimeSeriesDataset:
        return self.get_dataset(name)
    
    def __len__(self) -> int:
        return len(self.datasets)
