"""
NOTEARS Algorithm

Non-combinatorial Optimization for Learning DAGs with Continuous Optimization.

Reference: Zheng et al., "DAGs with NO TEARS" (NeurIPS 2018)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class NOTEARS:
    """
    NOTEARS: Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning.
    
    Learns causal DAG from observational data using continuous optimization.
    
    Key Innovation: Converts discrete graph search into continuous optimization
    by formulating an acyclicity constraint that can be differentiated.
    
    Args:
        num_vars: Number of variables
        lambda_sparse: Sparsity regularization strength
        lambda_acyclic: Acyclicity constraint strength
        max_iter: Maximum optimization iterations
        h_tol: Tolerance for acyclicity constraint
        w_threshold: Threshold for pruning weak edges
    
    Example:
        >>> notears = NOTEARS(num_vars=4, lambda_sparse=0.1)
        >>> adj_matrix = notears.learn_dag(data)
        >>> print("Discovered causal graph:\\n", adj_matrix)
    """
    
    def __init__(
        self,
        num_vars: int,
        lambda_sparse: float = 0.1,
        lambda_acyclic: float = 1.0,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        w_threshold: float = 0.3,
        lr: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.num_vars = num_vars
        self.lambda_sparse = lambda_sparse
        self.lambda_acyclic = lambda_acyclic
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold
        self.lr = lr
        self.device = device
        
        # Initialize adjacency matrix
        self.W = None
    
    def learn_dag(
        self,
        X: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Learn DAG structure from data.
        
        Args:
            X: Data matrix of shape (num_samples, num_vars, num_timesteps)
               or (num_samples, num_vars)
            verbose: Print progress
        
        Returns:
            Adjacency matrix W of shape (num_vars, num_vars)
            W[i,j] != 0 means i -> j
        """
        # Handle input shape
        if X.ndim == 3:
            X = X[:, :, -1]  # Use last timestep
        
        X_torch = torch.FloatTensor(X).to(self.device)
        
        # Initialize adjacency matrix W
        self.W = torch.randn(self.num_vars, self.num_vars, requires_grad=True, device=self.device)
        self.W.data *= 0.1  # Small initialization
        
        optimizer = torch.optim.Adam([self.W], lr=self.lr)
        
        # Augmented Lagrangian parameters
        rho = 1.0
        alpha = 0.0
        h = np.inf
        
        iterator = tqdm(range(self.max_iter), desc="NOTEARS") if verbose else range(self.max_iter)
        
        for iter_num in iterator:
            # Compute loss
            loss, h_val = self._compute_loss(X_torch, rho, alpha)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update augmented Lagrangian parameters
            h = h_val.item()
            
            if h > 0.25 * h:
                rho *= 10
            else:
                alpha += rho * h
            
            if verbose and iter_num % 10 == 0:
                print(f"Iter {iter_num}: Loss = {loss.item():.4f}, h = {h:.6f}")
            
            # Check convergence
            if h <= self.h_tol:
                if verbose:
                    print(f"Converged at iteration {iter_num}")
                break
        
        # Threshold and return
        W_est = self.W.detach().cpu().numpy()
        W_est = self._threshold_matrix(W_est)
        
        return W_est
    
    def _compute_loss(
        self,
        X: torch.Tensor,
        rho: float,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NOTEARS loss function.
        
        Loss = score(W) + lambda_sparse * ||W||_1 + augmented_lagrangian
        """
        # Score function: least squares fit
        score = self._score_function(X, self.W)
        
        # Sparsity penalty
        sparsity = self.lambda_sparse * torch.sum(torch.abs(self.W))
        
        # Acyclicity constraint
        h = self._acyclicity_constraint(self.W)
        
        # Augmented Lagrangian
        augmented = alpha * h + (rho / 2) * h * h
        
        loss = score + sparsity + augmented
        
        return loss, h
    
    def _score_function(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute score function (prediction error).
        
        Score = ||X - X @ W||^2
        """
        X_pred = X @ W
        score = torch.sum((X - X_pred) ** 2) / X.shape[0]
        return score
    
    def _acyclicity_constraint(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute acyclicity constraint h(W).
        
        h(W) = trace(exp(W ⊙ W)) - d
        
        Key property: h(W) = 0 if and only if W represents a DAG
        """
        d = self.num_vars
        W_squared = W * W
        
        # Matrix exponential via power series
        M = torch.eye(d, device=self.device) + W_squared / d
        
        # Compute trace(M^d) efficiently
        M_power = M
        for _ in range(d - 1):
            M_power = M_power @ M
        
        h = torch.trace(M_power) - d
        
        return h
    
    def _threshold_matrix(self, W: np.ndarray) -> np.ndarray:
        """
        Threshold weak edges to enforce sparsity.
        """
        W_thresh = W.copy()
        W_thresh[np.abs(W_thresh) < self.w_threshold] = 0
        return W_thresh
    
    def get_edges(self, W: Optional[np.ndarray] = None) -> list:
        """
        Get list of edges from adjacency matrix.
        
        Returns:
            List of tuples (i, j, weight) representing edges i -> j
        """
        if W is None:
            if self.W is None:
                raise RuntimeError("Must call learn_dag first")
            W = self.W.detach().cpu().numpy()
        
        edges = []
        for i in range(self.num_vars):
            for j in range(self.num_vars):
                if W[i, j] != 0:
                    edges.append((i, j, W[i, j]))
        
        return edges


if __name__ == "__main__":
    # Example usage
    print("Testing NOTEARS algorithm...")
    
    from src.data.time_series_generator import generate_synthetic_granger
    
    # Generate data with known DAG
    df, true_graph = generate_synthetic_granger(
        num_vars=4,
        num_samples=500,
        lag=5,
        seed=42
    )
    
    # Learn DAG with NOTEARS
    notears = NOTEARS(num_vars=4, lambda_sparse=0.1)
    estimated_graph = notears.learn_dag(df.values)
    
    print("\nTrue Causal Graph:")
    print(true_graph)
    print("\nEstimated Causal Graph:")
    print(estimated_graph)
    
    edges = notears.get_edges(estimated_graph)
    print(f"\nDiscovered {len(edges)} edges:")
    for i, j, weight in edges:
        print(f"  {i} -> {j} (weight: {weight:.3f})")
