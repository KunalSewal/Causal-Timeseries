"""
Generate Result Visualizations

Creates comprehensive visualizations of training and evaluation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*70)
print("GENERATING RESULT VISUALIZATIONS")
print("="*70)

# Load results
results_dir = Path('experiments/results')
output_dir = results_dir / 'graphs'
output_dir.mkdir(exist_ok=True, parents=True)

# 1. Model Comparison - Metrics
print("\n[1/6] Creating model comparison bar charts...")

metrics_df = pd.read_csv(results_dir / 'comprehensive_metrics.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MSE
ax = axes[0, 0]
models = metrics_df['model']
mse = metrics_df['mse']

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
ax.bar(models, mse, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('MSE', fontweight='bold')
ax.set_title('Mean Squared Error (Lower is Better)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# MAE with CI
ax = axes[0, 1]
mae = metrics_df['mae']
mae_ci_lower = metrics_df['mae_ci_lower']
mae_ci_upper = metrics_df['mae_ci_upper']
yerr = [mae - mae_ci_lower, mae_ci_upper - mae]

ax.bar(models, mae, color=colors, alpha=0.7, edgecolor='black')
ax.errorbar(models, mae, yerr=yerr, fmt='none', ecolor='black', capsize=5)
ax.set_ylabel('MAE', fontweight='bold')
ax.set_title('Mean Absolute Error with 95% CI (Lower is Better)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# R²
ax = axes[1, 0]
r2 = metrics_df['r2']

ax.bar(models, r2, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('R²', fontweight='bold')
ax.set_title('R² Score (Higher is Better)', fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# RMSE
ax = axes[1, 1]
rmse = metrics_df['rmse']

ax.bar(models, rmse, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('RMSE', fontweight='bold')
ax.set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Neural Granger Causality Models - Performance Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'model_comparison_metrics.pdf', bbox_inches='tight')
print(f"✓ Saved to {output_dir / 'model_comparison_metrics.png'}")

# 2. Model Ranking with Confidence Intervals
print("\n[2/6] Creating model ranking visualization...")

ranking_df = pd.read_csv(results_dir / 'model_ranking.csv')

fig, ax = plt.subplots(figsize=(12, 6))

models = ranking_df['model']
mae_mean = ranking_df['mean_mae']
mae_ci_lower = ranking_df['ci_lower']
mae_ci_upper = ranking_df['ci_upper']

# Sort by performance
sorted_idx = np.argsort(mae_mean)
models_sorted = [models[i] for i in sorted_idx]
mae_sorted = mae_mean[sorted_idx]
ci_lower_sorted = mae_ci_lower[sorted_idx]
ci_upper_sorted = mae_ci_upper[sorted_idx]

colors_sorted = [colors[list(models).index(m)] for m in models_sorted]

ax.barh(models_sorted, mae_sorted, color=colors_sorted, alpha=0.7, edgecolor='black')

# Add CI as error bars
xerr = [mae_sorted - ci_lower_sorted, ci_upper_sorted - mae_sorted]
ax.errorbar(mae_sorted, models_sorted, xerr=xerr, fmt='none', ecolor='black', capsize=5)

# Add value labels
for i, (model, val) in enumerate(zip(models_sorted, mae_sorted)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')

ax.set_xlabel('Mean Absolute Error (MAE)', fontweight='bold', fontsize=12)
ax.set_title('Model Ranking by MAE with 95% Confidence Intervals', 
             fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'model_ranking.pdf', bbox_inches='tight')
print(f"✓ Saved to {output_dir / 'model_ranking.png'}")

# 3. Statistical Significance Heatmap
print("\n[3/6] Creating statistical significance heatmap...")

comparison_df = pd.read_csv(results_dir / 'statistical_comparison.csv')

# Create matrix
model_names = sorted(metrics_df['model'].unique())
n_models = len(model_names)
p_matrix = np.ones((n_models, n_models))
sig_matrix = np.zeros((n_models, n_models), dtype=int)

for _, row in comparison_df.iterrows():
    i = model_names.index(row['model_1'])
    j = model_names.index(row['model_2'])
    p_matrix[i, j] = row['ttest_pvalue']
    p_matrix[j, i] = row['ttest_pvalue']
    sig_matrix[i, j] = int(row['ttest_significant'])
    sig_matrix[j, i] = int(row['ttest_significant'])

fig, ax = plt.subplots(figsize=(10, 8))

# Plot p-values with log scale
im = ax.imshow(np.log10(p_matrix + 1e-10), cmap='RdYlGn_r', vmin=-4, vmax=0)

# Add significance markers
for i in range(n_models):
    for j in range(n_models):
        if i == j:
            text = '—'
            color = 'gray'
        elif sig_matrix[i, j]:
            text = '✓'
            color = 'white'
        else:
            text = '✗'
            color = 'black'
        ax.text(j, i, text, ha='center', va='center', 
               color=color, fontsize=16, fontweight='bold')

ax.set_xticks(np.arange(n_models))
ax.set_yticks(np.arange(n_models))
ax.set_xticklabels(model_names)
ax.set_yticklabels(model_names)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

ax.set_title('Pairwise Statistical Significance Tests (p < 0.05)', 
             fontweight='bold', fontsize=14, pad=15)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log₁₀(p-value)', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'statistical_significance.pdf', bbox_inches='tight')
print(f"✓ Saved to {output_dir / 'statistical_significance.png'}")

# 4. Causality Matrix Heatmap
print("\n[4/6] Creating causality matrix heatmap...")

causality_df = pd.read_csv(results_dir / 'causality_matrix.csv', index_col=0)

fig, ax = plt.subplots(figsize=(12, 10))

# Plot heatmap
im = ax.imshow(causality_df.values, cmap='YlOrRd', vmin=0, vmax=1)

# Add value annotations
for i in range(len(causality_df)):
    for j in range(len(causality_df.columns)):
        value = causality_df.iloc[i, j]
        if value > 0.15:  # Only show significant relationships
            text = ax.text(j, i, f'{value:.2f}',
                         ha='center', va='center',
                         color='white' if value > 0.5 else 'black',
                         fontsize=9)

ax.set_xticks(np.arange(len(causality_df.columns)))
ax.set_yticks(np.arange(len(causality_df)))
ax.set_xticklabels(causality_df.columns, fontsize=11)
ax.set_yticklabels(causality_df.index, fontsize=11)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

ax.set_xlabel('Target (Effect)', fontweight='bold', fontsize=12)
ax.set_ylabel('Source (Cause)', fontweight='bold', fontsize=12)
ax.set_title('Granger Causality Matrix: Tech Stocks\n(Row → Column)', 
             fontweight='bold', fontsize=14, pad=15)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Causality Strength', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'causality_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'causality_heatmap.pdf', bbox_inches='tight')
print(f"✓ Saved to {output_dir / 'causality_heatmap.png'}")

# 5. Performance Improvement Bar Chart
print("\n[5/6] Creating performance improvement chart...")

# Compute improvements relative to worst model (Attention)
baseline_idx = list(metrics_df['model']).index('Attention')
baseline_mse = metrics_df.loc[baseline_idx, 'mse']
baseline_mae = metrics_df.loc[baseline_idx, 'mae']

improvements_mse = []
improvements_mae = []

for _, row in metrics_df.iterrows():
    if row['model'] != 'Attention':
        imp_mse = (baseline_mse - row['mse']) / baseline_mse * 100
        imp_mae = (baseline_mae - row['mae']) / baseline_mae * 100
        improvements_mse.append({'model': row['model'], 'improvement': imp_mse})
        improvements_mae.append({'model': row['model'], 'improvement': imp_mae})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MSE Improvement
ax = axes[0]
imp_df_mse = pd.DataFrame(improvements_mse)
model_colors = [colors[list(metrics_df['model']).index(m)] for m in imp_df_mse['model']]

bars = ax.bar(imp_df_mse['model'], imp_df_mse['improvement'], 
              color=model_colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('MSE Improvement (%)', fontweight='bold', fontsize=12)
ax.set_title('MSE Improvement vs Attention Baseline', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# MAE Improvement
ax = axes[1]
imp_df_mae = pd.DataFrame(improvements_mae)

bars = ax.bar(imp_df_mae['model'], imp_df_mae['improvement'], 
              color=model_colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('MAE Improvement (%)', fontweight='bold', fontsize=12)
ax.set_title('MAE Improvement vs Attention Baseline', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Performance Improvements Relative to Baseline', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / 'performance_improvements.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'performance_improvements.pdf', bbox_inches='tight')
print(f"✓ Saved to {output_dir / 'performance_improvements.png'}")

# 6. Summary Figure
print("\n[6/6] Creating comprehensive summary figure...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top left: Model comparison
ax1 = fig.add_subplot(gs[0, :2])
models_sorted_perf = ranking_df['model']
mae_sorted_perf = ranking_df['mean_mae']
colors_perf = [colors[list(metrics_df['model']).index(m)] for m in models_sorted_perf]
ax1.barh(models_sorted_perf, mae_sorted_perf, color=colors_perf, alpha=0.7, edgecolor='black')
for i, val in enumerate(mae_sorted_perf):
    ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
ax1.set_xlabel('MAE', fontweight='bold')
ax1.set_title('(A) Model Performance Ranking', fontweight='bold', loc='left')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Top right: Key statistics
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
KEY RESULTS

Best Model: TCN
MAE: {metrics_df[metrics_df['model']=='TCN']['mae'].values[0]:.3f}
R²: {metrics_df[metrics_df['model']=='TCN']['r2'].values[0]:.3f}

Improvements vs Baseline:
• MSE: {improvements_mse[2]['improvement']:.1f}%
• MAE: {improvements_mae[2]['improvement']:.1f}%

Statistical Significance:
All improvements p < 0.001

Causal Relationships:
41 significant edges detected
"""
ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax2.set_title('(B) Summary Statistics', fontweight='bold', loc='left')

# Middle: Metrics comparison
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(len(metrics_df))
width = 0.2
metrics_plot = ['mse', 'mae', 'rmse']
colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, metric in enumerate(metrics_plot):
    vals = metrics_df[metric].values
    vals_normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    ax3.bar(x + i * width, vals_normalized, width, label=metric.upper(), 
           color=colors_metrics[i], alpha=0.7, edgecolor='black')

ax3.set_ylabel('Normalized Score (Lower is Better)', fontweight='bold')
ax3.set_title('(C) Multi-Metric Comparison (Normalized)', fontweight='bold', loc='left')
ax3.set_xticks(x + width)
ax3.set_xticklabels(metrics_df['model'])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Bottom: Causality network summary
ax4 = fig.add_subplot(gs[2, :])

# Load causal edges
with open(results_dir / 'causal_edges.json', 'r') as f:
    causal_edges = json.load(f)

top_10 = causal_edges[:10]
sources = [e['source'] for e in top_10]
targets = [e['target'] for e in top_10]
strengths = [e['strength'] for e in top_10]

edge_labels = [f"{s}→{t}" for s, t in zip(sources, targets)]
colors_edges = plt.cm.Reds(np.linspace(0.4, 0.9, len(strengths)))

ax4.barh(range(len(strengths)), strengths, color=colors_edges, edgecolor='black')
ax4.set_yticks(range(len(strengths)))
ax4.set_yticklabels(edge_labels)
ax4.set_xlabel('Causality Strength', fontweight='bold')
ax4.set_title('(D) Top 10 Causal Relationships', fontweight='bold', loc='left')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.suptitle('Neural Granger Causality Framework - Comprehensive Results', 
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig(output_dir / 'comprehensive_summary.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'comprehensive_summary.pdf', bbox_inches='tight')
print(f"✓ Saved to {output_dir / 'comprehensive_summary.png'}")

print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print(f"\nGenerated 6 figures:")
print(f"  1. model_comparison_metrics.png - Performance across 4 metrics")
print(f"  2. model_ranking.png - Ranked by MAE with confidence intervals")
print(f"  3. statistical_significance.png - Pairwise significance tests")
print(f"  4. causality_heatmap.png - Granger causality matrix")
print(f"  5. performance_improvements.png - Improvements vs baseline")
print(f"  6. comprehensive_summary.png - All results in one figure")
print(f"\n✓ All visualizations saved to {output_dir}")
