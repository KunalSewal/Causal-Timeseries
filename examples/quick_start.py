"""
Example: Quick Start with Synthetic Data

This script demonstrates the basic usage of the framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
Path("examples/outputs").mkdir(parents=True, exist_ok=True)

print("="*60)
print("Neural Granger Causality Framework - Quick Start Example")
print("="*60)

# Step 1: Generate Synthetic Data
print("\n[Step 1] Generating synthetic time series data...")
from src.data.time_series_generator import generate_stock_like_data

df, true_graph = generate_stock_like_data(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
    num_samples=1000,
    seed=42
)

print(f"Generated data shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTrue causal structure:")
print(true_graph)

# Save data
df.to_csv('data/raw/example_stocks.csv')
print("\nData saved to: data/raw/example_stocks.csv")

# Step 2: Classical Granger Causality Test
print("\n[Step 2] Running classical Granger causality test...")
from src.models.granger_classical import VARGrangerTester

tester = VARGrangerTester(maxlag=5)
classical_matrix, p_values = tester.fit_test(df)

print("\nClassical Granger Causality Matrix:")
print(classical_matrix)
print("\nP-values:")
print(p_values)

# Step 3: Neural Granger Causality (Small model for demo)
print("\n[Step 3] Training Neural Granger model...")
from src.data.data_loader import TimeSeriesDataset
from src.models.granger_neural import NeuralGrangerLSTM

dataset = TimeSeriesDataset(df, lag=5, split='train', split_ratios=(0.8, 0.1, 0.1))

print(f"Dataset: {dataset}")
print("Training (this may take a minute)...")

model = NeuralGrangerLSTM(
    num_vars=4,
    hidden_dim=32,
    num_layers=2,
    lag=5,
    device='cpu'  # Use CPU for demo
)

neural_matrix = model.fit(dataset, epochs=20, batch_size=32, lr=0.001, verbose=True)

print("\nNeural Granger Causality Matrix:")
print(neural_matrix)

# Step 4: Learn Causal DAG with NOTEARS
print("\n[Step 4] Learning causal DAG with NOTEARS...")
from src.causal_discovery.notears import NOTEARS

notears = NOTEARS(num_vars=4, lambda_sparse=0.1, max_iter=50)
estimated_dag = notears.learn_dag(df.values, verbose=True)

print("\nEstimated DAG:")
print(estimated_dag)

# Step 5: Compare Results
print("\n[Step 5] Comparison of Methods")
print("="*60)
print("\nTrue Causal Graph:")
print(true_graph)
print("\nClassical Granger (VAR):")
print(classical_matrix.astype(int))
print("\nNeural Granger (LSTM):")
print((neural_matrix > 0.3).astype(int))  # Threshold at 0.3
print("\nNOTEARS:")
print((estimated_dag != 0).astype(int))

# Step 6: Visualize
print("\n[Step 6] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Causal Discovery Results Comparison', fontsize=16)

methods = [
    ('True Graph', true_graph),
    ('Classical VAR', classical_matrix),
    ('Neural LSTM', (neural_matrix > 0.3).astype(float)),
    ('NOTEARS', (estimated_dag != 0).astype(float))
]

for idx, (title, matrix) in enumerate(methods):
    ax = axes[idx // 2, idx % 2]
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('Source Variable')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['AAPL', 'MSFT', 'GOOGL', 'NVDA'])
    ax.set_yticklabels(['AAPL', 'MSFT', 'GOOGL', 'NVDA'])
    
    # Add values
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                         ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig('examples/outputs/causality_comparison.png', dpi=300, bbox_inches='tight')
print("Saved visualization to: examples/outputs/causality_comparison.png")

# Plot time series
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
fig.suptitle('Stock Price Time Series', fontsize=16)

for idx, ticker in enumerate(['AAPL', 'MSFT', 'GOOGL', 'NVDA']):
    axes[idx].plot(df.index, df[ticker], label=ticker)
    axes[idx].set_ylabel('Price')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('examples/outputs/time_series.png', dpi=300, bbox_inches='tight')
print("Saved time series plot to: examples/outputs/time_series.png")

print("\n" + "="*60)
print("✅ Example completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Check the generated plots in examples/outputs/")
print("2. Try with your own data by modifying this script")
print("3. Experiment with different model architectures")
print("4. Explore notebooks/ for more detailed examples")
print("\nFor more information, see SETUP_INSTRUCTIONS.md")
