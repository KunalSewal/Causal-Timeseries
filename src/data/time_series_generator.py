"""
Synthetic Time Series Generator

Generate synthetic multivariate time series data with known causal structure
for testing and validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import networkx as nx


class SyntheticDataGenerator:
    """
    Generate synthetic time series with specified causal structure.
    
    Uses Vector Autoregressive (VAR) process with controlled causality.
    
    Args:
        num_vars: Number of variables
        num_samples: Number of time steps
        lag: Maximum lag order
        causal_graph: Adjacency matrix (num_vars, num_vars) specifying causal structure
        noise_std: Standard deviation of noise
        nonlinear: Whether to add nonlinear transformations
        seed: Random seed for reproducibility
    
    Example:
        >>> # Create 4 variables with known causal structure
        >>> causal_graph = np.array([
        ...     [0, 1, 0, 0],  # var0 -> var1
        ...     [0, 0, 1, 0],  # var1 -> var2
        ...     [0, 0, 0, 1],  # var2 -> var3
        ...     [0, 0, 0, 0],  # var3 causes nothing
        ... ])
        >>> gen = SyntheticDataGenerator(4, 1000, lag=5, causal_graph=causal_graph)
        >>> data, true_graph = gen.generate()
    """
    
    def __init__(
        self,
        num_vars: int,
        num_samples: int,
        lag: int = 5,
        causal_graph: Optional[np.ndarray] = None,
        noise_std: float = 0.1,
        nonlinear: bool = False,
        seed: Optional[int] = None,
    ):
        self.num_vars = num_vars
        self.num_samples = num_samples
        self.lag = lag
        self.noise_std = noise_std
        self.nonlinear = nonlinear
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate or use provided causal graph
        if causal_graph is None:
            self.causal_graph = self._generate_random_dag()
        else:
            self.causal_graph = causal_graph
            assert causal_graph.shape == (num_vars, num_vars), "Invalid causal graph shape"
        
        # Generate coefficients for VAR model
        self.coefficients = self._generate_coefficients()
    
    def _generate_random_dag(self, sparsity: float = 0.3) -> np.ndarray:
        """
        Generate a random Directed Acyclic Graph (DAG).
        
        Args:
            sparsity: Probability of edge existence
        
        Returns:
            Adjacency matrix where A[i,j] = 1 if i -> j
        """
        # Start with random graph
        adj = np.random.rand(self.num_vars, self.num_vars)
        adj = (adj < sparsity).astype(float)
        
        # Make it a DAG by zeroing upper triangle (ensures acyclicity)
        adj = np.tril(adj, k=-1)
        
        # Random permutation to make it less structured
        perm = np.random.permutation(self.num_vars)
        adj = adj[perm][:, perm]
        
        return adj
    
    def _generate_coefficients(self) -> np.ndarray:
        """
        Generate VAR coefficients based on causal graph.
        
        Returns:
            Coefficients array of shape (num_vars, num_vars, lag)
        """
        coeffs = np.zeros((self.num_vars, self.num_vars, self.lag))
        
        for i in range(self.num_vars):
            for j in range(self.num_vars):
                if self.causal_graph[j, i] > 0:  # j causes i
                    # Generate coefficients with exponential decay over lags
                    for lag_idx in range(self.lag):
                        decay = np.exp(-lag_idx / 2)
                        coeffs[i, j, lag_idx] = np.random.uniform(0.3, 0.8) * decay
        
        return coeffs
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic time series data.
        
        Returns:
            data: (num_samples, num_vars) array of time series
            true_graph: (num_vars, num_vars) true causal adjacency matrix
        """
        # Initialize data array
        data = np.zeros((self.num_samples, self.num_vars))
        
        # Initialize with random values
        data[:self.lag] = np.random.randn(self.lag, self.num_vars)
        
        # Generate time series using VAR process
        for t in range(self.lag, self.num_samples):
            for i in range(self.num_vars):
                # Autoregressive component
                for lag_idx in range(self.lag):
                    for j in range(self.num_vars):
                        coeff = self.coefficients[i, j, lag_idx]
                        past_value = data[t - lag_idx - 1, j]
                        
                        if self.nonlinear:
                            # Add nonlinear transformation
                            past_value = np.tanh(past_value)
                        
                        data[t, i] += coeff * past_value
                
                # Add noise
                data[t, i] += np.random.randn() * self.noise_std
        
        return data, self.causal_graph
    
    def to_dataframe(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert generated data to DataFrame."""
        if variable_names is None:
            variable_names = [f"var_{i}" for i in range(self.num_vars)]
        
        return pd.DataFrame(data, columns=variable_names)
    
    def visualize_graph(self, save_path: Optional[str] = None):
        """Visualize the true causal graph."""
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        for i in range(self.num_vars):
            G.add_node(i)
        
        for i in range(self.num_vars):
            for j in range(self.num_vars):
                if self.causal_graph[i, j] > 0:
                    G.add_edge(i, j, weight=self.causal_graph[i, j])
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(
            G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1000,
            font_size=16,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2
        )
        plt.title("True Causal Graph", fontsize=18)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def generate_synthetic_granger(
    num_vars: int = 4,
    num_samples: int = 1000,
    lag: int = 5,
    sparsity: float = 0.3,
    noise_std: float = 0.1,
    nonlinear: bool = False,
    seed: Optional[int] = 42,
    variable_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Quick function to generate synthetic data with Granger causality.
    
    Args:
        num_vars: Number of variables
        num_samples: Number of time steps
        lag: Lag order
        sparsity: Sparsity of causal graph (probability of edge)
        noise_std: Noise level
        nonlinear: Add nonlinear transformations
        seed: Random seed
        variable_names: Names for variables
    
    Returns:
        DataFrame with synthetic data
        True causal adjacency matrix
    
    Example:
        >>> df, true_graph = generate_synthetic_granger(
        ...     num_vars=4,
        ...     num_samples=1000,
        ...     lag=5
        ... )
    """
    generator = SyntheticDataGenerator(
        num_vars=num_vars,
        num_samples=num_samples,
        lag=lag,
        noise_std=noise_std,
        nonlinear=nonlinear,
        seed=seed,
    )
    
    # Override with specified sparsity
    generator.causal_graph = generator._generate_random_dag(sparsity=sparsity)
    generator.coefficients = generator._generate_coefficients()
    
    data, true_graph = generator.generate()
    df = generator.to_dataframe(data, variable_names=variable_names)
    
    return df, true_graph


def generate_stock_like_data(
    tickers: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
    num_samples: int = 1000,
    initial_prices: Optional[List[float]] = None,
    **kwargs
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate stock-price-like synthetic data.
    
    Applies exponential transformation to make data look like stock prices.
    
    Args:
        tickers: Stock ticker names
        num_samples: Number of days
        initial_prices: Initial stock prices
        **kwargs: Additional arguments for generate_synthetic_granger
    
    Returns:
        DataFrame with synthetic stock prices
        True causal adjacency matrix
    """
    if initial_prices is None:
        initial_prices = [100 + i * 50 for i in range(len(tickers))]
    
    # Generate base data
    df, true_graph = generate_synthetic_granger(
        num_vars=len(tickers),
        num_samples=num_samples,
        variable_names=tickers,
        **kwargs
    )
    
    # Transform to look like stock prices
    for i, (ticker, initial_price) in enumerate(zip(tickers, initial_prices)):
        # Cumulative sum to create random walk
        returns = df[ticker].values * 0.02  # Scale down
        prices = initial_price * np.exp(np.cumsum(returns))
        df[ticker] = prices
    
    # Add timestamp index
    dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='D')
    df.index = dates
    df.index.name = 'timestamp'
    
    return df, true_graph


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data...")
    
    df, true_graph = generate_synthetic_granger(
        num_vars=4,
        num_samples=1000,
        lag=5,
        seed=42
    )
    
    print(f"Generated data shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nTrue causal graph:\n{true_graph}")
    
    # Generate stock-like data
    stock_df, stock_graph = generate_stock_like_data()
    print(f"\nStock-like data shape: {stock_df.shape}")
    print(f"\nFirst few rows:\n{stock_df.head()}")
