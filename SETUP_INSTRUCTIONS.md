# PROJECT SETUP INSTRUCTIONS

## 📦 What's Been Created

The following directory structure and files have been generated:

```
causal-timeseries/
├── README.md ✅
├── requirements.txt ✅
├── setup.py ✅
├── .gitignore ✅
│
├── data/
│   ├── README.md ✅
│   ├── raw/.gitkeep ✅
│   └── processed/.gitkeep ✅
│
└── src/
    ├── __init__.py ✅
    ├── data/
    │   ├── __init__.py ✅
    │   ├── data_loader.py ✅
    │   ├── preprocessor.py ✅
    │   └── time_series_generator.py ✅
    ├── models/
    │   ├── __init__.py ✅
    │   ├── granger_classical.py ✅
    │   ├── granger_neural.py ✅
    │   ├── attention_granger.py ✅
    │   └── tcn_granger.py ✅
    └── causal_discovery/
        ├── __init__.py ✅
        └── notears.py ✅
```

## 🚀 Next Steps

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Download Sample Data

You can download stock data using the provided helper function:

```python
import yfinance as yf
import pandas as pd

# Download stock prices
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Close']
data.to_csv('data/raw/stock_prices.csv')
```

Or generate synthetic data for testing:

```python
from src.data.time_series_generator import generate_stock_like_data

df, true_graph = generate_stock_like_data(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
    num_samples=1000
)
df.to_csv('data/raw/synthetic_stocks.csv')
```

## 📝 Files You Need to Create (Optional)

The core framework is complete! The following files would enhance the project but aren't critical:

### Remaining Source Files (Lower Priority)

1. **src/causal_discovery/pc_algorithm.py** - PC algorithm for causal discovery
2. **src/causal_discovery/dag_utils.py** - DAG validation and visualization utilities
3. **src/evaluation/__init__.py** - Evaluation module init
4. **src/evaluation/metrics.py** - Performance metrics
5. **src/evaluation/statistical_tests.py** - Statistical significance tests
6. **src/evaluation/visualization.py** - Plotting causal graphs
7. **src/utils/__init__.py** - Utils init
8. **src/utils/config.py** - Configuration management
9. **src/utils/logger.py** - Logging utilities
10. **src/utils/torch_utils.py** - PyTorch helpers

### Scripts (Can create as needed)

1. **scripts/train.py** - Main training script
2. **scripts/evaluate.py** - Evaluation script
3. **scripts/discover_graph.py** - Graph discovery script
4. **scripts/visualize_results.py** - Visualization script
5. **scripts/download_sample_data.py** - Data download helper

### Configuration Files

1. **experiments/configs/baseline.yaml**
2. **experiments/configs/lstm_granger.yaml**
3. **experiments/configs/attention_granger.yaml**

### Jupyter Notebooks

1. **notebooks/01_data_exploration.ipynb**
2. **notebooks/02_classical_granger.ipynb**
3. **notebooks/03_neural_granger.ipynb**
4. **notebooks/04_causal_graph_learning.ipynb**

### Documentation

1. **docs/theory.md** - Granger causality theory
2. **docs/architecture.md** - System architecture
3. **docs/results.md** - Experimental results
4. **docs/api.md** - API documentation

### Tests

1. **tests/test_data_loader.py**
2. **tests/test_models.py**
3. **tests/test_causal_discovery.py**

## ✅ Quick Start Guide

You can start using the framework immediately! Here's a simple example:

```python
# Generate synthetic data
from src.data.time_series_generator import generate_synthetic_granger
from src.data.data_loader import TimeSeriesDataset

df, true_graph = generate_synthetic_granger(
    num_vars=4,
    num_samples=1000,
    lag=5
)

# Save data
df.to_csv('data/raw/test_data.csv')

# Load as dataset
dataset = TimeSeriesDataset('data/raw/test_data.csv', lag=5)

# Test classical Granger
from src.models.granger_classical import VARGrangerTester

tester = VARGrangerTester(maxlag=5)
causality_matrix, p_values = tester.fit_test(df)
print("Classical Granger Causality Matrix:\n", causality_matrix)

# Test neural Granger (requires GPU or patience!)
from src.models.granger_neural import NeuralGrangerLSTM

model = NeuralGrangerLSTM(num_vars=4, hidden_dim=32, num_layers=2, lag=5)
neural_causality = model.fit(dataset, epochs=20)
print("Neural Granger Causality Matrix:\n", neural_causality)

# Learn DAG with NOTEARS
from src.causal_discovery.notears import NOTEARS

notears = NOTEARS(num_vars=4, lambda_sparse=0.1)
dag = notears.learn_dag(df.values)
print("Discovered DAG:\n", dag)

# Compare with ground truth
print("True Causal Graph:\n", true_graph)
```

## 🐛 Troubleshooting

### Import Errors
If you see import errors, make sure you've installed the package:
```powershell
pip install -e .
```

### CUDA/GPU Issues
All models default to CPU if CUDA is unavailable. To force CPU:
```python
model = NeuralGrangerLSTM(..., device='cpu')
```

### Missing Dependencies
Some optional dependencies (like pygraphviz) might fail to install. These are only needed for advanced visualization and can be skipped.

## 📚 Learning Resources

- **Granger Causality**: Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models"
- **NOTEARS**: Zheng et al. (2018). "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
- **Neural Granger**: Tank et al. (2018). "Neural Granger Causality"

## 🤝 Contributing

Feel free to extend this framework! Some ideas:
- Add more causal discovery algorithms (GES, GIES, DirectLiNGAM)
- Implement multivariate transfer entropy
- Add support for non-stationary time series
- Create interactive visualization dashboard
- Add Bayesian structural time series models

## 📧 Support

If you encounter issues or have questions:
1. Check the examples in each module's `if __name__ == "__main__"` section
2. Review the docstrings for detailed usage
3. Generate synthetic data first to validate your setup

---

**The framework is ready to use! The core functionality is complete.** 🎉
