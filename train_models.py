"""
Train Neural Granger Models on Real Stock Data

This script trains LSTM, GRU, Attention, and TCN models on real financial data
and generates comprehensive results for the project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from causal_timeseries.data import TimeSeriesDataset, TimeSeriesPreprocessor
from causal_timeseries.models import NeuralGrangerLSTM, NeuralGrangerGRU, AttentionGranger, TCNGranger

# Configuration
CONFIG = {
    'data_file': 'data/processed/tech_stocks_2020_2025.csv',
    'lag': 5,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'early_stopping_patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'experiments/results',
}

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # R2 score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return avg_loss, mse, mae, r2

def train_model(model_class, model_name, train_loader, val_loader, config):
    """Train a single model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    device = config['device']
    num_vars = train_loader.dataset.num_vars
    
    # Initialize model
    if model_class == NeuralGrangerLSTM or model_class == NeuralGrangerGRU:
        model = model_class(
            num_vars=num_vars,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            lag=config['lag'],
            dropout=config['dropout'],
            device=device
        )
    elif model_class == AttentionGranger:
        model = model_class(
            num_vars=num_vars,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            lag=config['lag'],
            dropout=config['dropout'],
            device=device
        )
    else:  # TCN
        model = model_class(
            num_vars=num_vars,
            num_channels=[config['hidden_dim']] * 3,
            dropout=config['dropout'],
            device=device
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_mae': [], 'val_r2': []}
    
    # Training loop
    pbar = tqdm(range(config['epochs']), desc=f"Training {model_name}")
    
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_r2': f'{val_r2:.4f}'
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            output_dir = Path(config['output_dir']) / 'model_outputs'
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / f'{model_name}_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print(f"\nBest Validation Loss: {best_val_loss:.6f}")
    print(f"Final Val MSE: {history['val_mse'][-1]:.6f}")
    print(f"Final Val MAE: {history['val_mae'][-1]:.6f}")
    print(f"Final Val R²: {history['val_r2'][-1]:.6f}")
    
    return model, history

def main():
    print("\n" + "="*70)
    print("TRAINING NEURAL GRANGER MODELS ON REAL FINANCIAL DATA")
    print("="*70)
    
    print(f"\nDevice: {CONFIG['device'].upper()}")
    
    # Load and preprocess data
    print(f"\n[1/5] Loading data from {CONFIG['data_file']}")
    df = pd.read_csv(CONFIG['data_file'], index_col=0, parse_dates=True)
    print(f"✓ Loaded {df.shape[0]} days × {df.shape[1]} stocks")
    print(f"  Stocks: {', '.join(df.columns)}")
    
    # Preprocess
    print("\n[2/5] Preprocessing...")
    prep = TimeSeriesPreprocessor(normalize='standard')
    df_clean = prep.fit_transform(df)
    print(f"✓ Normalized data: mean={df_clean.mean().mean():.2e}, std={df_clean.std().mean():.2f}")
    
    # Create datasets
    print("\n[3/5] Creating datasets...")
    dataset_train = TimeSeriesDataset(df_clean, lag=CONFIG['lag'], split='train')
    dataset_val = TimeSeriesDataset(df_clean, lag=CONFIG['lag'], split='val')
    dataset_test = TimeSeriesDataset(df_clean, lag=CONFIG['lag'], split='test')
    
    train_loader = DataLoader(dataset_train, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(dataset_test, batch_size=CONFIG['batch_size'])
    
    print(f"✓ Train: {len(dataset_train)} sequences")
    print(f"✓ Val: {len(dataset_val)} sequences")
    print(f"✓ Test: {len(dataset_test)} sequences")
    
    # Train models
    print("\n[4/5] Training models...")
    
    models_to_train = [
        (NeuralGrangerLSTM, 'LSTM'),
        (NeuralGrangerGRU, 'GRU'),
        (AttentionGranger, 'Attention'),
        (TCNGranger, 'TCN'),
    ]
    
    results = {}
    
    for model_class, model_name in models_to_train:
        model, history = train_model(model_class, model_name, train_loader, val_loader, CONFIG)
        
        # Test set evaluation
        test_loss, test_mse, test_mae, test_r2 = evaluate(model, test_loader, nn.MSELoss(), CONFIG['device'])
        
        results[model_name] = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'best_val_loss': min(history['val_loss']),
            'final_val_r2': history['val_r2'][-1],
            'epochs_trained': len(history['train_loss']),
        }
        
        # Save history
        output_dir = Path(CONFIG['output_dir']) / 'logs'
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(output_dir / f'{model_name}_history.csv', index=False)
    
    # Generate results report
    print("\n[5/5] Generating results report...")
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('test_mse')
    
    print("\n" + "="*70)
    print("FINAL RESULTS - MODEL COMPARISON")
    print("="*70)
    print(results_df.to_string())
    
    # Calculate improvement vs baseline
    baseline_mse = results_df['test_mse'].max()
    best_mse = results_df['test_mse'].min()
    improvement = ((baseline_mse - best_mse) / baseline_mse) * 100
    
    print(f"\n✓ Best model: {results_df.index[0]}")
    print(f"✓ MSE improvement: {improvement:.1f}%")
    print(f"✓ Best R² score: {results_df['test_r2'].max():.4f}")
    
    # Save results
    output_dir = Path(CONFIG['output_dir'])
    results_df.to_csv(output_dir / 'model_comparison.csv')
    
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
