"""
Vector Autoregression (VAR) Baseline

Classical statistical approach for comparison with neural methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, meanabs
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from causal_timeseries.data import TimeSeriesPreprocessor

print("="*70)
print("CLASSICAL VAR BASELINE")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/processed/tech_stocks_2020_2025.csv', index_col=0, parse_dates=True)

# Preprocess
prep = TimeSeriesPreprocessor(normalize='standard')
df_normalized = prep.fit_transform(df)

# Split data (same as neural models)
n = len(df_normalized)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train_df = df_normalized[:train_size]
val_df = df_normalized[train_size:train_size+val_size]
test_df = df_normalized[train_size+val_size:]

print(f"✓ Train: {len(train_df)} samples")
print(f"✓ Val: {len(val_df)} samples")
print(f"✓ Test: {len(test_df)} samples")

# Train VAR model
print("\n[2/5] Training VAR model...")
print("Using Akaike Information Criterion (AIC) to select lag order")

# Fit VAR and select optimal lag
model = VAR(train_df)
lag_order_results = model.select_order(maxlags=10)
optimal_lag = lag_order_results.aic

print(f"✓ Optimal lag order (AIC): {optimal_lag}")
print(f"✓ AIC: {lag_order_results.aic}")
print(f"✓ BIC: {lag_order_results.bic}")

# Train with optimal lag
var_model = model.fit(optimal_lag)
print(f"\n✓ Trained VAR({optimal_lag}) model")
print(f"✓ Number of parameters: {var_model.params.size}")

# Model summary
print("\n" + "="*70)
print("VAR MODEL SUMMARY")
print("="*70)
print(f"Lag order: {optimal_lag}")
print(f"Number of equations: {var_model.neqs}")
print(f"Number of observations: {var_model.nobs}")
print(f"Parameters per equation: {var_model.params.shape[0]}")
print(f"Total parameters: {var_model.params.size}")

# Evaluate on test set
print("\n[3/5] Evaluating on test set...")

# Make predictions
lag_order = optimal_lag
test_data = pd.concat([train_df.iloc[-lag_order:], test_df])

predictions = []
actuals = []

# Rolling forecast
for i in range(lag_order, len(test_data)):
    history = test_data.iloc[i-lag_order:i]
    forecast = var_model.forecast(history.values, steps=1)
    predictions.append(forecast[0])
    actuals.append(test_data.iloc[i].values)

predictions = np.array(predictions)
actuals = np.array(actuals)

# Compute metrics
mse = mean_squared_error(actuals, predictions)
rmse_val = np.sqrt(mse)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print("\n" + "="*70)
print("VAR MODEL PERFORMANCE")
print("="*70)
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse_val:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R²:   {r2:.6f}")

# Load neural model results for comparison
print("\n[4/5] Comparing with neural models...")

neural_metrics = pd.read_csv('experiments/results/comprehensive_metrics.csv')

comparison = {
    'model': ['VAR'] + list(neural_metrics['model']),
    'mse': [mse] + list(neural_metrics['mse']),
    'rmse': [rmse_val] + list(neural_metrics['rmse']),
    'mae': [mae] + list(neural_metrics['mae']),
    'r2': [r2] + list(neural_metrics['r2']),
}

comparison_df = pd.DataFrame(comparison)
comparison_df = comparison_df.sort_values('mae')

print("\n" + "="*70)
print("MODEL COMPARISON: VAR vs NEURAL")
print("="*70)
print(comparison_df.to_string(index=False))

# Compute improvements
var_mse = mse
var_mae = mae

print("\n" + "="*70)
print("NEURAL IMPROVEMENTS OVER VAR BASELINE")
print("="*70)

for _, row in neural_metrics.iterrows():
    model_name = row['model']
    neural_mse = row['mse']
    neural_mae = row['mae']
    
    mse_improvement = (var_mse - neural_mse) / var_mse * 100
    mae_improvement = (var_mae - neural_mae) / var_mae * 100
    
    print(f"\n{model_name}:")
    print(f"  MSE improvement: {mse_improvement:+.1f}%")
    print(f"  MAE improvement: {mae_improvement:+.1f}%")
    
    if mse_improvement > 0:
        print(f"  ✓ {model_name} outperforms VAR baseline")
    else:
        print(f"  ✗ VAR baseline outperforms {model_name}")

# Granger causality from VAR
print("\n[5/5] Extracting Granger causality from VAR...")

# Test Granger causality
from statsmodels.tsa.stattools import grangercausalitytests

causality_results = {}
stock_names = list(df.columns)

print("\nGranger Causality Tests (VAR approach):")
print("Testing if variable X Granger-causes variable Y")

significant_relationships = []

for target in stock_names:
    for source in stock_names:
        if source == target:
            continue
        
        # Create dataframe with just source and target
        test_df = train_df[[target, source]]
        
        try:
            # Test if source Granger-causes target
            result = grangercausalitytests(test_df, maxlag=optimal_lag, verbose=False)
            
            # Get p-value for optimal lag
            p_value = result[optimal_lag][0]['ssr_ftest'][1]
            
            if p_value < 0.05:
                significant_relationships.append({
                    'source': source,
                    'target': target,
                    'p_value': float(p_value),
                    'lag': int(optimal_lag)
                })
        except:
            pass

print(f"\n✓ Found {len(significant_relationships)} significant Granger-causal relationships (p < 0.05)")

if significant_relationships:
    print("\nTop 10 Granger-Causal Relationships:")
    sorted_rels = sorted(significant_relationships, key=lambda x: x['p_value'])[:10]
    for i, rel in enumerate(sorted_rels, 1):
        print(f"  {i}. {rel['source']} → {rel['target']}: p={rel['p_value']:.4f}")

# Save results
output_dir = Path('experiments/results')

# Save VAR metrics
var_results = {
    'model': 'VAR',
    'lag_order': int(optimal_lag),
    'n_parameters': int(var_model.params.size),
    'metrics': {
        'mse': float(mse),
        'rmse': float(rmse_val),
        'mae': float(mae),
        'r2': float(r2)
    },
    'aic': float(lag_order_results.aic),
    'bic': float(lag_order_results.bic),
    'n_granger_relationships': len(significant_relationships)
}

with open(output_dir / 'var_baseline_results.json', 'w') as f:
    json.dump(var_results, f, indent=2)

# Save comparison
comparison_df.to_csv(output_dir / 'var_neural_comparison.csv', index=False)

# Save Granger relationships
with open(output_dir / 'var_granger_causality.json', 'w') as f:
    json.dump(significant_relationships, f, indent=2)

print(f"\n✓ Results saved to {output_dir}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

best_neural = neural_metrics.loc[neural_metrics['mae'].idxmin()]
best_neural_name = best_neural['model']
best_neural_mae = best_neural['mae']

improvement = (var_mae - best_neural_mae) / var_mae * 100

print(f"""
Classical VAR Baseline:
- MAE: {var_mae:.3f}
- R²: {r2:.3f}
- Lag: {optimal_lag}

Best Neural Model ({best_neural_name}):
- MAE: {best_neural_mae:.3f}
- Improvement: {improvement:.1f}%

{'✓ Neural methods significantly outperform classical VAR' if improvement > 10 else '⚠ VAR competitive with neural methods'}
""")

print("="*70)
print("VAR BASELINE COMPLETE!")
print("="*70)
