"""
Unit Tests for Causal Timeseries Package

Tests data loading, preprocessing, models, and metrics.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Test data module
def test_time_series_preprocessor():
    """Test TimeSeriesPreprocessor normalization"""
    from causal_timeseries.data import TimeSeriesPreprocessor
    
    # Create sample data
    data = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, 5.0],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # Test standard scaling
    prep = TimeSeriesPreprocessor(normalize='standard')
    normalized = prep.fit_transform(data)
    
    assert normalized.shape == data.shape
    assert np.allclose(normalized.mean(), 0, atol=0.1)
    assert np.allclose(normalized.std(), 1, atol=0.1)


def test_time_series_dataset():
    """Test TimeSeriesDataset sliding window"""
    from causal_timeseries.data import TimeSeriesDataset
    
    # Create sample data
    data = pd.DataFrame({
        'A': np.arange(100),
        'B': np.arange(100) * 2
    })
    
    dataset = TimeSeriesDataset(data, lag=5, split='train')
    
    assert len(dataset) > 0
    X, y = dataset[0]
    assert X.shape[0] == 5  # lag
    assert X.shape[1] == 2  # num variables
    assert y.shape[0] == 2  # num variables


# Test models
def test_lstm_granger_forward():
    """Test NeuralGrangerLSTM forward pass"""
    from causal_timeseries.models import NeuralGrangerLSTM
    
    batch_size = 8
    seq_len = 5
    num_vars = 3
    
    model = NeuralGrangerLSTM(num_vars=num_vars, hidden_dim=32, num_layers=1)
    X = torch.randn(batch_size, seq_len, num_vars)
    
    output = model(X)
    
    assert output.shape == (batch_size, num_vars)
    assert not torch.isnan(output).any()


def test_gru_granger_forward():
    """Test NeuralGrangerGRU forward pass"""
    from causal_timeseries.models import NeuralGrangerGRU
    
    batch_size = 8
    seq_len = 5
    num_vars = 3
    
    model = NeuralGrangerGRU(num_vars=num_vars, hidden_dim=32, num_layers=1)
    X = torch.randn(batch_size, seq_len, num_vars)
    
    output = model(X)
    
    assert output.shape == (batch_size, num_vars)
    assert not torch.isnan(output).any()


def test_attention_granger_forward():
    """Test AttentionGranger forward pass"""
    from causal_timeseries.models import AttentionGranger
    
    batch_size = 8
    seq_len = 5
    num_vars = 3
    
    model = AttentionGranger(num_vars=num_vars, hidden_dim=32, num_heads=2)
    X = torch.randn(batch_size, seq_len, num_vars)
    
    output = model(X)
    
    assert output.shape == (batch_size, num_vars)
    assert not torch.isnan(output).any()


def test_tcn_granger_forward():
    """Test TCNGranger forward pass"""
    from causal_timeseries.models import TCNGranger
    
    batch_size = 8
    seq_len = 5
    num_vars = 3
    
    model = TCNGranger(num_vars=num_vars, num_channels=[16, 32], kernel_size=2)
    X = torch.randn(batch_size, seq_len, num_vars)
    
    output = model(X)
    
    assert output.shape == (batch_size, num_vars)
    assert not torch.isnan(output).any()


# Test evaluation metrics
def test_regression_metrics():
    """Test RegressionMetrics computation"""
    from causal_timeseries.evaluation.metrics import RegressionMetrics
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    metrics = RegressionMetrics()
    mse = metrics.mse(y_true, y_pred)
    mae = metrics.mae(y_true, y_pred)
    r2 = metrics.r2(y_true, y_pred)
    
    assert mse > 0
    assert mae > 0
    assert 0 <= r2 <= 1


def test_causality_metrics():
    """Test CausalityMetrics computation"""
    from causal_timeseries.evaluation.metrics import CausalityMetrics
    
    # True and predicted adjacency matrices
    true_adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    pred_adj = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]])
    
    metrics = CausalityMetrics()
    precision = metrics.precision(true_adj, pred_adj)
    recall = metrics.recall(true_adj, pred_adj)
    f1 = metrics.f1_score(true_adj, pred_adj)
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1


def test_statistical_tests():
    """Test StatisticalTests"""
    from causal_timeseries.evaluation.metrics import StatisticalTests
    
    errors1 = np.random.randn(100)
    errors2 = errors1 + 0.5  # Slightly worse
    
    tests = StatisticalTests()
    
    # Paired t-test
    t_stat, p_value = tests.paired_ttest(errors1, errors2)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    
    # Bootstrap CI
    ci_lower, ci_upper = tests.bootstrap_ci(errors1, n_bootstrap=100)
    assert ci_lower < ci_upper


# Test causal discovery
def test_notears_initialization():
    """Test NOTEARS initialization"""
    from causal_timeseries.causal_discovery.notears import NOTEARS
    
    num_vars = 5
    notears = NOTEARS(num_vars=num_vars, lambda_sparse=0.1)
    
    assert notears.num_vars == num_vars
    assert notears.lambda_sparse == 0.1


# Test configuration
def test_config_loading():
    """Test configuration loading"""
    from causal_timeseries.utils.config import load_config
    
    # Create temporary config
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("model:\n  hidden_dim: 64\n  num_layers: 2\n")
        config_path = f.name
    
    config = load_config(config_path)
    assert 'model' in config
    assert config['model']['hidden_dim'] == 64
    
    # Cleanup
    Path(config_path).unlink()


# Test utilities
def test_torch_utils():
    """Test torch utility functions"""
    from causal_timeseries.utils.torch_utils import set_seed, count_parameters
    from causal_timeseries.models import NeuralGrangerLSTM
    
    # Test set_seed
    set_seed(42)
    x1 = torch.randn(10)
    set_seed(42)
    x2 = torch.randn(10)
    assert torch.allclose(x1, x2)
    
    # Test count_parameters
    model = NeuralGrangerLSTM(num_vars=3, hidden_dim=32, num_layers=1)
    n_params = count_parameters(model)
    assert n_params > 0


# Integration tests
def test_end_to_end_training():
    """Test end-to-end training pipeline"""
    from causal_timeseries.data import TimeSeriesDataset
    from causal_timeseries.models import TCNGranger
    import torch.nn as nn
    import torch.optim as optim
    
    # Create synthetic data
    data = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    dataset = TimeSeriesDataset(data, lag=5, split='train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    
    # Create model
    model = TCNGranger(num_vars=3, num_channels=[8, 16], kernel_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train for 2 epochs
    model.train()
    for epoch in range(2):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Test that training worked
    assert loss.item() < 10  # Loss should be reasonable


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
