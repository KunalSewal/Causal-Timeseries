"""
Time Series Preprocessing

Normalization, stationarity checks, missing value handling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Preprocess time series data for causal analysis.
    
    Features:
    - Multiple normalization methods
    - Missing value imputation
    - Stationarity transformation
    - Outlier detection
    
    Args:
        normalize: Normalization method ('standard', 'minmax', 'robust', None)
        handle_missing: Method for missing values ('ffill', 'bfill', 'interpolate', 'drop')
        difference: Apply differencing for stationarity
        
    Example:
        >>> prep = TimeSeriesPreprocessor(normalize='standard')
        >>> df_clean = prep.fit_transform(df)
    """
    
    def __init__(
        self,
        normalize: Optional[Literal['standard', 'minmax', 'robust']] = 'standard',
        handle_missing: Literal['ffill', 'bfill', 'interpolate', 'drop'] = 'ffill',
        difference: bool = False,
        clip_outliers: Optional[float] = None,
    ):
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.difference = difference
        self.clip_outliers = clip_outliers
        
        # Will be set during fit
        self.scaler = None
        self.columns = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesPreprocessor':
        """Fit preprocessor on data."""
        self.columns = list(data.columns)
        
        # Initialize scaler
        if self.normalize == 'standard':
            self.scaler = StandardScaler()
        elif self.normalize == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalize == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        # Fit scaler
        if self.scaler is not None:
            clean_data = self._handle_missing(data.copy())
            self.scaler.fit(clean_data)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {len(data)} samples, {len(self.columns)} variables")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        result = data.copy()
        
        # Handle missing values
        result = self._handle_missing(result)
        
        # Clip outliers
        if self.clip_outliers is not None:
            result = self._clip_outliers(result, self.clip_outliers)
        
        # Apply differencing
        if self.difference:
            result = result.diff().dropna()
        
        # Normalize
        if self.scaler is not None:
            values = self.scaler.transform(result)
            result = pd.DataFrame(values, index=result.index, columns=result.columns)
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalization."""
        if self.scaler is None:
            return data
        
        values = self.scaler.inverse_transform(data)
        return pd.DataFrame(values, index=data.index, columns=data.columns)
    
    def _handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        if data.isna().sum().sum() == 0:
            return data
        
        logger.info(f"Handling {data.isna().sum().sum()} missing values with '{self.handle_missing}'")
        
        if self.handle_missing == 'ffill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == 'bfill':
            return data.fillna(method='bfill').fillna(method='ffill')
        elif self.handle_missing == 'interpolate':
            return data.interpolate(method='linear').fillna(method='bfill')
        elif self.handle_missing == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown handle_missing method: {self.handle_missing}")
    
    def _clip_outliers(self, data: pd.DataFrame, n_std: float) -> pd.DataFrame:
        """Clip outliers beyond n standard deviations."""
        result = data.copy()
        
        for col in result.columns:
            mean = result[col].mean()
            std = result[col].std()
            lower = mean - n_std * std
            upper = mean + n_std * std
            result[col] = result[col].clip(lower, upper)
        
        return result


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with missing values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100, 7],
        'B': [10, np.nan, 30, 40, 50, 60, 70]
    })
    
    print("Original data:")
    print(data)
    
    prep = TimeSeriesPreprocessor(normalize='standard', clip_outliers=3.0)
    clean = prep.fit_transform(data)
    
    print("\nCleaned data:")
    print(clean)
