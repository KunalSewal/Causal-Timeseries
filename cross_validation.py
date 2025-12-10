"""
K-Fold Cross-Validation for Time Series Models

Implements time-series aware cross-validation with expanding window.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from causal_timeseries.data import TimeSeriesDataset, TimeSeriesPreprocessor
from causal_timeseries.models import NeuralGrangerLSTM, NeuralGrangerGRU, AttentionGranger, TCNGranger

print("="*70)
print("5-FOLD CROSS-VALIDATION")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('data/processed/tech_stocks_2020_2025.csv', index_col=0, parse_dates=True)
prep = TimeSeriesPreprocessor(normalize='standard')
df_normalized = prep.fit_transform(df)

n = len(df_normalized)
n_vars = df_normalized.shape[1]

print(f"✓ Data points: {n}")
print(f"✓ Variables: {n_vars}")

# Time-series cross-validation setup
n_folds = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n[2/4] Setting up {n_folds}-fold time-series CV...")
print("Using expanding window strategy (train grows, test is fixed size)")

# Calculate fold sizes
test_size = n // (n_folds + 1)
min_train_size = 2 * test_size

folds = []
for i in range(n_folds):
    train_end = min_train_size + i * test_size
    test_start = train_end
    test_end = test_start + test_size
    
    if test_end > n:
        break
    
    folds.append({
        'fold': i + 1,
        'train_indices': (0, train_end),
        'test_indices': (test_start, test_end),
        'train_size': train_end,
        'test_size': test_end - test_start
    })

print(f"\n✓ Created {len(folds)} folds:")
for fold in folds:
    print(f"  Fold {fold['fold']}: Train[0:{fold['train_indices'][1]}], Test[{fold['test_indices'][0]}:{fold['test_indices'][1]}]")

# Models to evaluate
models_config = {
    'LSTM': lambda: NeuralGrangerLSTM(n_vars, hidden_dim=64, num_layers=2, dropout=0.2),
    'GRU': lambda: NeuralGrangerGRU(n_vars, hidden_dim=64, num_layers=2, dropout=0.2),
    'Attention': lambda: AttentionGranger(n_vars, hidden_dim=64, num_heads=4),
    'TCN': lambda: TCNGranger(n_vars, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2),
}

# Cross-validation results
cv_results = {model_name: [] for model_name in models_config.keys()}

print("\n[3/4] Running cross-validation...")

for fold_idx, fold in enumerate(folds, 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx}/{len(folds)}")
    print(f"{'='*70}")
    
    # Split data for this fold
    train_df = df_normalized.iloc[:fold['train_indices'][1]]
    test_df = df_normalized.iloc[fold['test_indices'][0]:fold['test_indices'][1]]
    
    # Create datasets (using train split for both, but with different data)
    train_dataset = TimeSeriesDataset(train_df, lag=5, split='train')
    test_dataset = TimeSeriesDataset(test_df, lag=5, split='train')  # Use train to get all data
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train each model
    for model_name, model_fn in models_config.items():
        print(f"\nTraining {model_name}...")
        
        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train for limited epochs (faster CV)
        n_epochs = 20
        model.train()
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.numpy())
        
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # Compute metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - actuals.mean()) ** 2)
        
        cv_results[model_name].append({
            'fold': fold_idx,
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        })
        
        print(f"  ✓ {model_name} - MAE: {mae:.4f}, R²: {r2:.4f}")

# Aggregate results
print("\n[4/4] Aggregating cross-validation results...")

aggregated_results = []

for model_name, fold_results in cv_results.items():
    mse_scores = [f['mse'] for f in fold_results]
    mae_scores = [f['mae'] for f in fold_results]
    rmse_scores = [f['rmse'] for f in fold_results]
    r2_scores = [f['r2'] for f in fold_results]
    
    aggregated_results.append({
        'model': model_name,
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
    })

results_df = pd.DataFrame(aggregated_results)
results_df = results_df.sort_values('mae_mean')

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (Mean ± Std)")
print("="*70)
print("\nModel Rankings by MAE:")
for _, row in results_df.iterrows():
    print(f"\n{row['model']}:")
    print(f"  MSE:  {row['mse_mean']:.4f} ± {row['mse_std']:.4f}")
    print(f"  MAE:  {row['mae_mean']:.4f} ± {row['mae_std']:.4f}")
    print(f"  RMSE: {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}")
    print(f"  R²:   {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")

# Save results
output_dir = Path('experiments/results')

# Save detailed fold results
with open(output_dir / 'cv_detailed_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

# Save aggregated results
results_df.to_csv(output_dir / 'cv_aggregated_results.csv', index=False)

cv_summary = {
    'n_folds': len(folds),
    'strategy': 'expanding_window',
    'results': aggregated_results,
    'best_model': results_df.iloc[0]['model'],
    'best_mae': float(results_df.iloc[0]['mae_mean']),
    'best_mae_std': float(results_df.iloc[0]['mae_std']),
}

with open(output_dir / 'cv_summary.json', 'w') as f:
    json.dump(cv_summary, f, indent=2)

print(f"\n✓ Results saved to {output_dir}")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)

best_model = results_df.iloc[0]
print(f"""
Best Model: {best_model['model']}
MAE: {best_model['mae_mean']:.4f} ± {best_model['mae_std']:.4f}
R²: {best_model['r2_mean']:.4f} ± {best_model['r2_std']:.4f}

Cross-validation provides robust error estimates accounting for:
- Different time periods
- Temporal dependencies
- Model stability across folds

Standard deviation indicates model consistency across different data splits.
""")

print("="*70)
print("CROSS-VALIDATION COMPLETE!")
print("="*70)
