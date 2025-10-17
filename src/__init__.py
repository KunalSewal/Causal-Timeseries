"""
Causal Timeseries Package
Neural Granger Causality Framework for Multivariate Time Series
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.data.data_loader import TimeSeriesDataset
from src.models.granger_neural import NeuralGrangerLSTM
from src.causal_discovery.notears import NOTEARS

__all__ = [
    "TimeSeriesDataset",
    "NeuralGrangerLSTM",
    "NOTEARS",
]
