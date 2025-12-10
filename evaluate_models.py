"""
Comprehensive Model Evaluation

Evaluates trained models with statistical rigor:
- Regression metrics
- Statistical significance tests
- Confidence intervals
- Model comparison
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json

from causal_timeseries.data import TimeSeriesDataset, TimeSeriesPreprocessor
from causal_timeseries.models import NeuralGrangerLSTM, NeuralGrangerGRU, AttentionGranger, TCNGranger
from causal_timeseries.evaluation.metrics import RegressionMetrics, StatisticalTests, ModelComparator

print("="*70)
print("COMPREHENSIVE MODEL EVALUATION WITH STATISTICAL TESTS")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('data/processed/tech_stocks_2020_2025.csv', index_col=0, parse_dates=True)
prep = TimeSeriesPreprocessor(normalize='standard')
df_clean = prep.fit_transform(df)

# Create test dataset
dataset_test = TimeSeriesDataset(df_clean, lag=5, split='test')
print(f"✓ Test dataset: {len(dataset_test)} sequences")

# Get all test data
X_test, y_test = dataset_test.get_full_data()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_test = X_test.to(device)

# Load models and get predictions
print("\n[2/4] Loading trained models and generating predictions...")

models_config = {
    'LSTM': (NeuralGrangerLSTM, {'hidden_dim': 64, 'num_layers': 2}),
    'GRU': (NeuralGrangerGRU, {'hidden_dim': 64, 'num_layers': 2}),
    'Attention': (AttentionGranger, {'hidden_dim': 64, 'num_layers': 2}),
    'TCN': (TCNGranger, {'num_channels': [64, 64, 64]}),
}

predictions = {}
model_dir = Path('experiments/results/model_outputs')

for name, (model_class, config) in models_config.items():
    # Initialize model
    if 'num_channels' in config:
        model = model_class(num_vars=13, device=device, **config)
    else:
        model = model_class(num_vars=13, lag=5, device=device, **config)
    
    # Load weights
    model_path = model_dir / f'{name}_best.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            preds = model(X_test).cpu().numpy()
        
        predictions[name] = preds
        print(f"✓ {name}: {preds.shape}")
    else:
        print(f"⚠ {name}: model file not found")

y_test_np = y_test.numpy()

# Compute comprehensive metrics
print("\n[3/4] Computing comprehensive metrics...")

results = []
for name, preds in predictions.items():
    metrics = RegressionMetrics.compute_all(y_test_np, preds)
    
    # Add confidence intervals
    errors = np.abs(preds - y_test_np).flatten()
    mean_mae, ci_lower, ci_upper = StatisticalTests.bootstrap_confidence_interval(errors, n_bootstrap=5000)
    
    metrics.update({
        'model': name,
        'mae_ci_lower': ci_lower,
        'mae_ci_upper': ci_upper,
        'mae_ci_width': ci_upper - ci_lower,
    })
    
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mse')

print("\n" + "="*70)
print("DETAILED METRICS WITH CONFIDENCE INTERVALS")
print("="*70)
print(results_df[['model', 'mse', 'rmse', 'mae', 'r2', 'mae_ci_lower', 'mae_ci_upper']].to_string(index=False))

# Statistical comparison
print("\n[4/4] Statistical comparison between models...")

comparator = ModelComparator(predictions, y_test_np)
comparison_df = comparator.compare_all()

print("\n" + "="*70)
print("PAIRWISE MODEL COMPARISON (Statistical Significance)")
print("="*70)
print(comparison_df.to_string(index=False))

# Ranking with CI
ranking_df = comparator.rank_models()
print("\n" + "="*70)
print("MODEL RANKING (with 95% Confidence Intervals)")
print("="*70)
print(ranking_df.to_string(index=False))

# Calculate improvements
baseline_mse = results_df.iloc[-1]['mse']  # Worst model
best_mse = results_df.iloc[0]['mse']  # Best model
improvement_pct = ((baseline_mse - best_mse) / baseline_mse) * 100

baseline_mae = results_df.iloc[-1]['mae']
best_mae = results_df.iloc[0]['mae']
mae_improvement_pct = ((baseline_mae - best_mae) / baseline_mae) * 100

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print(f"✓ Best Model: {results_df.iloc[0]['model']}")
print(f"✓ MSE Improvement: {improvement_pct:.1f}% (vs {results_df.iloc[-1]['model']})")
print(f"✓ MAE Improvement: {mae_improvement_pct:.1f}%")
print(f"✓ Best R² Score: {results_df.iloc[0]['r2']:.4f}")
print(f"✓ Best RMSE: {results_df.iloc[0]['rmse']:.4f}")

# Check if improvements are statistically significant
best_model = results_df.iloc[0]['model']
baseline_model = results_df.iloc[-1]['model']

sig_comparison = comparison_df[
    ((comparison_df['model_1'] == best_model) & (comparison_df['model_2'] == baseline_model)) |
    ((comparison_df['model_1'] == baseline_model) & (comparison_df['model_2'] == best_model))
]

if not sig_comparison.empty:
    is_significant = sig_comparison.iloc[0]['ttest_significant']
    p_value = sig_comparison.iloc[0]['ttest_pvalue']
    print(f"\n✓ Improvement is {'STATISTICALLY SIGNIFICANT' if is_significant else 'NOT significant'}")
    print(f"  (p-value: {p_value:.4f})")

# Save results
output_dir = Path('experiments/results')
results_df.to_csv(output_dir / 'comprehensive_metrics.csv', index=False)
comparison_df.to_csv(output_dir / 'statistical_comparison.csv', index=False)
ranking_df.to_csv(output_dir / 'model_ranking.csv', index=False)

# Save summary
summary = {
    'best_model': results_df.iloc[0]['model'],
    'best_mse': float(results_df.iloc[0]['mse']),
    'best_mae': float(results_df.iloc[0]['mae']),
    'best_r2': float(results_df.iloc[0]['r2']),
    'mse_improvement_pct': float(improvement_pct),
    'mae_improvement_pct': float(mae_improvement_pct),
    'statistically_significant': bool(is_significant) if not sig_comparison.empty else None,
    'p_value': float(p_value) if not sig_comparison.empty else None,
}

with open(output_dir / 'evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Results saved to {output_dir}")
print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
