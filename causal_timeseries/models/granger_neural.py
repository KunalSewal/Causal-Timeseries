"""
Neural Granger Causality Models

Deep learning approaches for Granger causality using LSTM and GRU networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm


class NeuralGrangerLSTM(nn.Module):
    """
    Neural Granger Causality Model using LSTM networks.
    
    Core idea: Train two models for each variable Y:
    1. Model_restricted: Predicts Y using only past values of Y
    2. Model_full: Predicts Y using past values of ALL variables
    
    If Model_full >> Model_restricted → other variables Granger-cause Y
    
    Args:
        num_vars: Number of variables in the time series
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        lag: Number of lagged timesteps
        dropout: Dropout probability
    
    Example:
        >>> model = NeuralGrangerLSTM(num_vars=4, hidden_dim=64, num_layers=2, lag=5)
        >>> causality_matrix = model.fit(dataset, epochs=50)
        >>> print("Causality matrix:", causality_matrix)
    """
    
    def __init__(
        self,
        num_vars: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lag: int = 5,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lag = lag
        self.dropout = dropout
        self.device = device
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=num_vars,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, num_vars)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, lag, num_vars)
        
        Returns:
            Predictions of shape (batch, num_vars)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, lag, hidden_dim)
        
        # Use last timestep's output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Predict next values
        predictions = self.fc(last_out)  # (batch, num_vars)
        
        return predictions
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print progress
        
        Returns:
            Dictionary with training history
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        history = {'train_loss': [], 'val_loss': []}
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            # Training
            self.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_losses = []
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        predictions = self(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")
        
        return history
    
    def compute_granger_causality(
        self,
        dataset,
        target_idx: int,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> np.ndarray:
        """
        Compute Granger causality for a target variable.
        
        Args:
            dataset: TimeSeriesDataset
            target_idx: Index of target variable
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
        
        Returns:
            Causality scores for each variable (num_vars,)
        """
        X, y = dataset.get_full_data()
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Split train/val
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 1. Train restricted model (only target's past)
        X_train_restricted = X_train[:, :, target_idx:target_idx+1]
        X_val_restricted = X_val[:, :, target_idx:target_idx+1]
        
        model_restricted = self._create_restricted_model(1)
        restricted_loss = self._train_and_evaluate(
            model_restricted,
            X_train_restricted, y_train[:, target_idx],
            X_val_restricted, y_val[:, target_idx],
            epochs, batch_size, lr
        )
        
        # 2. Train full model (all variables' past)
        model_full = self._create_restricted_model(self.num_vars)
        full_loss = self._train_and_evaluate(
            model_full,
            X_train, y_train[:, target_idx],
            X_val, y_val[:, target_idx],
            epochs, batch_size, lr
        )
        
        # 3. Compute causality scores
        # Higher score = stronger causality
        causality_scores = np.zeros(self.num_vars)
        
        # For each variable, compute contribution
        for var_idx in range(self.num_vars):
            if var_idx == target_idx:
                causality_scores[var_idx] = 0.0
            else:
                # Improvement from adding this variable
                improvement = (restricted_loss - full_loss) / (restricted_loss + 1e-8)
                causality_scores[var_idx] = max(0.0, improvement)
        
        return causality_scores
    
    def _create_restricted_model(self, input_size: int):
        """Create a model with specified input size."""
        model = nn.Sequential(
            nn.LSTM(input_size, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0),
        )
        return model.to(self.device)
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> float:
        """Train a model and return validation loss."""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass through LSTM
                if isinstance(model, nn.Sequential):
                    lstm_out, _ = model[0](X_batch)
                    predictions = lstm_out[:, -1, 0]  # Use last timestep, first output
                else:
                    predictions = model(X_batch)
                    if len(predictions.shape) > 1:
                        predictions = predictions.squeeze()
                
                # Ensure shapes match
                if y_batch.dim() > 1:
                    y_batch = y_batch.squeeze()
                
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            if isinstance(model, nn.Sequential):
                lstm_out, _ = model[0](X_val)
                predictions = lstm_out[:, -1, 0]
            else:
                predictions = model(X_val)
                if len(predictions.shape) > 1:
                    predictions = predictions.squeeze()
            
            if y_val.dim() > 1:
                y_val = y_val.squeeze()
            
            val_loss = criterion(predictions, y_val).item()
        
        return val_loss
    
    def fit(
        self,
        dataset,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fit the model and compute causality matrix.
        
        Args:
            dataset: TimeSeriesDataset
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress
        
        Returns:
            Causality matrix (num_vars, num_vars)
        """
        causality_matrix = np.zeros((self.num_vars, self.num_vars))
        
        for target_idx in range(self.num_vars):
            if verbose:
                print(f"\nComputing causality for variable {target_idx}...")
            
            causality_scores = self.compute_granger_causality(
                dataset, target_idx, epochs, batch_size, lr
            )
            causality_matrix[:, target_idx] = causality_scores
        
        return causality_matrix


class NeuralGrangerGRU(NeuralGrangerLSTM):
    """
    Neural Granger Causality using GRU instead of LSTM.
    
    GRU is simpler and often trains faster than LSTM.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace LSTM with GRU
        self.lstm = nn.GRU(
            input_size=self.num_vars,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )
        
        self.to(self.device)


if __name__ == "__main__":
    # Example usage
    print("Testing Neural Granger Causality...")
    
    from src.data.data_loader import TimeSeriesDataset
    from src.data.time_series_generator import generate_synthetic_granger
    
    # Generate synthetic data
    df, true_graph = generate_synthetic_granger(num_vars=4, num_samples=500)
    
    # Create dataset
    dataset = TimeSeriesDataset(df, lag=5)
    
    # Train model
    model = NeuralGrangerLSTM(num_vars=4, hidden_dim=32, num_layers=2, lag=5)
    causality_matrix = model.fit(dataset, epochs=20, verbose=True)
    
    print("\nEstimated Causality Matrix:")
    print(causality_matrix)
    print("\nTrue Causal Graph:")
    print(true_graph)
