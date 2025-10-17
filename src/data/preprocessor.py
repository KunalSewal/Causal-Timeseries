"""
Time Series Preprocessor

Handles cleaning, normalization, and transformation of time series data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Literal, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle


class TimeSeriesPreprocessor:
    """
    Preprocessor for multivariate time series data.
    
    Features:
    - Missing value handling
    - Normalization/Standardization
    - Outlier detection and removal
    - Detrending and differencing
    
    Args:
        normalize: Whether to normalize data
        method: Normalization method ('standard', 'minmax', 'robust')
        handle_missing: How to handle missing values ('drop', 'interpolate', 'ffill', 'bfill')
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
        detrend: Whether to remove trend
        difference: Whether to apply differencing
    
    Example:
        >>> preprocessor = TimeSeriesPreprocessor(normalize=True, method='standard')
        >>> df_processed = preprocessor.fit_transform(df)
        >>> preprocessor.save('preprocessor.pkl')
    """
    
    def __init__(
        self,
        normalize: bool = True,
        method: Literal['standard', 'minmax', 'robust'] = 'standard',
        handle_missing: Literal['drop', 'interpolate', 'ffill', 'bfill'] = 'interpolate',
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0,
        detrend: bool = False,
        difference: bool = False,
    ):
        self.normalize = normalize
        self.method = method
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.detrend = detrend
        self.difference = difference
        
        # Initialize scaler
        if normalize:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        else:
            self.scaler = None
        
        self.is_fitted = False
        self.original_columns = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Input time series data
        
        Returns:
            self
        """
        df = self._to_dataframe(data)
        self.original_columns = df.columns.tolist()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers (fit statistics)
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(df.values)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform the data using fitted parameters.
        
        Args:
            data: Input time series data
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        df = self._to_dataframe(data)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # Detrend
        if self.detrend:
            df = self._detrend(df)
        
        # Difference
        if self.difference:
            df = self._difference(df)
        
        # Normalize
        if self.scaler is not None:
            values = self.scaler.transform(df.values)
            df = pd.DataFrame(values, index=df.index, columns=df.columns)
        
        return df
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data
        
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        df = self._to_dataframe(data)
        
        if self.scaler is not None:
            values = self.scaler.inverse_transform(df.values)
            df = pd.DataFrame(values, index=df.index, columns=df.columns)
        
        return df
    
    def _to_dataframe(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            return pd.DataFrame(data)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to specified method."""
        if df.isnull().sum().sum() == 0:
            return df
        
        if self.handle_missing == 'drop':
            return df.dropna()
        elif self.handle_missing == 'interpolate':
            return df.interpolate(method='linear', limit_direction='both')
        elif self.handle_missing == 'ffill':
            return df.fillna(method='ffill')
        elif self.handle_missing == 'bfill':
            return df.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown missing value handling method: {self.handle_missing}")
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        z_scores = np.abs((df - df.mean()) / df.std())
        mask = (z_scores < self.outlier_threshold).all(axis=1)
        return df[mask]
    
    def _detrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linear trend from each variable."""
        detrended = df.copy()
        for col in df.columns:
            x = np.arange(len(df))
            coeffs = np.polyfit(x, df[col], 1)
            trend = np.polyval(coeffs, x)
            detrended[col] = df[col] - trend
        return detrended
    
    def _difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply first-order differencing."""
        return df.diff().dropna()
    
    def save(self, filepath: Union[str, Path]):
        """Save preprocessor to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TimeSeriesPreprocessor':
        """Load preprocessor from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        return (
            f"TimeSeriesPreprocessor(normalize={self.normalize}, "
            f"method={self.method}, handle_missing={self.handle_missing}, "
            f"remove_outliers={self.remove_outliers}, is_fitted={self.is_fitted})"
        )


def preprocess_pipeline(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, TimeSeriesPreprocessor]:
    """
    Complete preprocessing pipeline from file to file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
        **kwargs: Arguments for TimeSeriesPreprocessor
    
    Returns:
        Processed DataFrame and fitted preprocessor
    
    Example:
        >>> df, preprocessor = preprocess_pipeline(
        ...     'data/raw/stock_prices.csv',
        ...     'data/processed/stock_processed.pkl',
        ...     normalize=True,
        ...     method='standard'
        ... )
    """
    # Load data
    df = pd.read_csv(input_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Preprocess
    preprocessor = TimeSeriesPreprocessor(**kwargs)
    df_processed = preprocessor.fit_transform(df)
    
    # Save
    df_processed.to_pickle(output_path)
    preprocessor.save(str(output_path).replace('.pkl', '_preprocessor.pkl'))
    
    print(f"Processed data saved to: {output_path}")
    print(f"Preprocessor saved to: {str(output_path).replace('.pkl', '_preprocessor.pkl')}")
    
    return df_processed, preprocessor
