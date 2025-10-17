# Data Directory

This directory contains all datasets used in the Causal Timeseries project.

## Directory Structure

```
data/
├── raw/              # Original, unmodified datasets
│   ├── stock_prices.csv
│   ├── climate_data.csv
│   └── ...
└── processed/        # Cleaned and preprocessed data
    ├── processed_stock.pkl
    ├── ...
```

## Data Sources

### 1. Stock Market Data

**Files**: `stock_prices.csv`

**Source**: Yahoo Finance (via `yfinance` library)

**Variables**: 
- AAPL (Apple Inc.)
- MSFT (Microsoft Corporation)
- GOOGL (Alphabet Inc.)
- NVDA (NVIDIA Corporation)

**Time Range**: 2020-01-01 to 2024-12-31

**Frequency**: Daily closing prices

**How to Download**:
```python
import yfinance as yf
import pandas as pd

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Close']
data.to_csv('data/raw/stock_prices.csv')
```

### 2. Climate Data (Optional Example)

**Files**: `climate_data.csv`

**Source**: NOAA, NASA, or similar climate data providers

**Variables**: Temperature, Precipitation, Humidity, Wind Speed

**How to Download**: Visit [NOAA Climate Data](https://www.ncdc.noaa.gov/cdo-web/) or use their API

### 3. Synthetic Data (For Testing)

The framework includes a synthetic data generator for testing and validation:

```python
from src.data.time_series_generator import generate_synthetic_granger

# Generate synthetic data with known causal structure
data, true_graph = generate_synthetic_granger(
    num_vars=4,
    num_samples=1000,
    lag=5,
    sparsity=0.3
)
```

## Data Format Requirements

All datasets should follow this CSV format:

```csv
timestamp,var1,var2,var3,var4
2024-01-01,100.5,200.3,150.2,300.1
2024-01-02,101.2,201.5,149.8,301.3
2024-01-03,100.8,199.9,151.0,299.8
...
```

**Requirements**:
- First column: `timestamp` (datetime format)
- Subsequent columns: Variable names (numeric values)
- No missing values (or handle them in preprocessing)
- Regular time intervals (daily, hourly, etc.)

## Preprocessing Pipeline

Data preprocessing is handled by `src/data/preprocessor.py`:

1. **Load raw data** from `data/raw/`
2. **Handle missing values** (interpolation, forward fill)
3. **Normalize/Standardize** time series
4. **Create lagged features** for Granger causality
5. **Train-test split** (typically 80-20)
6. **Save processed data** to `data/processed/`

Example:
```python
from src.data.preprocessor import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor(
    normalize=True,
    method='standard',  # or 'minmax'
    handle_missing='interpolate'
)

processed_data = preprocessor.fit_transform('data/raw/stock_prices.csv')
preprocessor.save('data/processed/processed_stock.pkl')
```

## Sample Datasets

To get started quickly, you can use the provided sample script:

```bash
python scripts/download_sample_data.py --dataset stocks --tickers AAPL MSFT GOOGL NVDA
```

## Data Privacy & Ethics

- Ensure you have rights to use any data
- Follow data provider's terms of service
- Be cautious with sensitive/personal data
- Document data sources and preprocessing steps

## Citation

When using specific datasets, please cite the original sources appropriately.

---

**Note**: The `data/raw/` and `data/processed/` directories are included in `.gitignore` to avoid committing large files. You'll need to download or generate data locally.
