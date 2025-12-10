"""
Causal Timeseries - Neural Granger Causality Framework

A production-grade framework for discovering causal relationships in multivariate
financial time series using deep learning and statistical methods.
"""

__version__ = "1.0.0"
__author__ = "Kunal Sewal"
__email__ = "kunalsewal@gmail.com"

# Lazy imports to avoid circular dependencies
__all__ = [
    "NeuralGrangerLSTM",
    "NeuralGrangerGRU",
    "AttentionGranger",
    "TCNGranger",
    "NOTEARS",
    "PCAlgorithm",
    "TimeSeriesDataset",
    "FinancialDataDownloader",
    "Trainer",
    "HyperparameterOptimizer",
    "CausalityMetrics",
    "ModelEvaluator",
]


def __getattr__(name):
    """Lazy import attributes."""
    if name in __all__:
        if name in ["NeuralGrangerLSTM", "NeuralGrangerGRU", "AttentionGranger", "TCNGranger"]:
            from causal_timeseries import models
            return getattr(models, name)
        elif name in ["NOTEARS", "PCAlgorithm"]:
            from causal_timeseries import causal_discovery
            return getattr(causal_discovery, name)
        elif name in ["TimeSeriesDataset", "FinancialDataDownloader"]:
            from causal_timeseries import data
            return getattr(data, name)
        elif name in ["Trainer", "HyperparameterOptimizer"]:
            from causal_timeseries import training
            return getattr(training, name)
        elif name in ["CausalityMetrics", "ModelEvaluator"]:
            from causal_timeseries import evaluation
            return getattr(evaluation, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
