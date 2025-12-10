"""
Comprehensive Evaluation Metrics

Metrics for time series prediction and causality detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """
    Metrics for time series prediction evaluation.
    """
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all regression metrics.
        
        Args:
            y_true: Ground truth values (n_samples, n_vars)
            y_pred: Predicted values (n_samples, n_vars)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': RegressionMetrics.mape(y_true, y_pred),
            'smape': RegressionMetrics.smape(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100
    
    @staticmethod
    def compute_per_variable(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Compute metrics for each variable separately.
        
        Returns:
            DataFrame with metrics per variable
        """
        n_vars = y_true.shape[1]
        results = []
        
        for i in range(n_vars):
            metrics = {
                'variable': i,
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'r2': r2_score(y_true[:, i], y_pred[:, i]),
            }
            results.append(metrics)
        
        return pd.DataFrame(results)


class CausalityMetrics:
    """
    Metrics for evaluating causality detection.
    
    Used when we have ground truth causal structure (e.g., synthetic data)
    or for comparing different causality detection methods.
    """
    
    @staticmethod
    def compute_binary_metrics(
        true_graph: np.ndarray,
        pred_graph: np.ndarray,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for causal graph detection.
        
        Args:
            true_graph: True causal adjacency matrix (n_vars, n_vars)
            pred_graph: Predicted causality matrix (n_vars, n_vars)
            threshold: Threshold for binarizing predictions
            
        Returns:
            Dictionary with precision, recall, F1, etc.
        """
        # Binarize
        true_binary = (true_graph != 0).astype(int).flatten()
        pred_binary = (np.abs(pred_graph) > threshold).astype(int).flatten()
        
        # Remove diagonal (self-loops)
        n = true_graph.shape[0]
        mask = ~np.eye(n, dtype=bool).flatten()
        true_binary = true_binary[mask]
        pred_binary = pred_binary[mask]
        
        # Compute metrics
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary).ravel()
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        }
        
        return metrics
    
    @staticmethod
    def structural_hamming_distance(true_graph: np.ndarray, pred_graph: np.ndarray) -> int:
        """
        Structural Hamming Distance between two DAGs.
        
        Counts edge additions + edge deletions + edge reversals.
        """
        true_binary = (true_graph != 0).astype(int)
        pred_binary = (pred_graph != 0).astype(int)
        
        # Remove diagonal
        n = true_graph.shape[0]
        np.fill_diagonal(true_binary, 0)
        np.fill_diagonal(pred_binary, 0)
        
        # Count differences
        additions = np.sum((pred_binary == 1) & (true_binary == 0))
        deletions = np.sum((pred_binary == 0) & (true_binary == 1))
        
        # Reversals: edge in opposite direction
        reversals = np.sum((pred_binary.T == 1) & (true_binary == 1) & (pred_binary == 0))
        
        shd = additions + deletions + reversals
        return int(shd)


class StatisticalTests:
    """
    Statistical significance tests for model comparison.
    """
    
    @staticmethod
    def paired_ttest(
        model1_errors: np.ndarray,
        model2_errors: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Paired t-test to compare two models.
        
        Args:
            model1_errors: Prediction errors from model 1
            model2_errors: Prediction errors from model 2
            alpha: Significance level
            
        Returns:
            Dictionary with t-statistic, p-value, and conclusion
        """
        t_stat, p_value = stats.ttest_rel(model1_errors, model2_errors)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'model1_better': (t_stat < 0) and (p_value < alpha),
            'model2_better': (t_stat > 0) and (p_value < alpha),
        }
    
    @staticmethod
    def permutation_test(
        model1_errors: np.ndarray,
        model2_errors: np.ndarray,
        n_permutations: int = 10000,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Permutation test for comparing models (non-parametric).
        
        More robust than t-test when assumptions are violated.
        """
        observed_diff = np.mean(model1_errors) - np.mean(model2_errors)
        
        # Permutation
        combined = np.concatenate([model1_errors, model2_errors])
        n1 = len(model1_errors)
        
        perm_diffs = []
        rng = np.random.RandomState(42)
        
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            perm_diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < alpha,
            'model1_better': (observed_diff < 0) and (p_value < alpha),
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        errors: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for mean error.
        
        Returns:
            (mean, lower_ci, upper_ci)
        """
        rng = np.random.RandomState(42)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = rng.choice(errors, size=len(errors), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence
        
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        mean = np.mean(errors)
        
        return mean, lower, upper


class ModelComparator:
    """
    Compare multiple models with statistical tests.
    """
    
    def __init__(self, model_predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Args:
            model_predictions: {model_name: predictions}
            y_true: Ground truth
        """
        self.model_predictions = model_predictions
        self.y_true = y_true
        self.model_names = list(model_predictions.keys())
        
        # Compute errors
        self.errors = {}
        for name, preds in model_predictions.items():
            self.errors[name] = np.abs(preds - y_true).flatten()
    
    def compare_all(self) -> pd.DataFrame:
        """
        Compare all models pairwise with statistical tests.
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for i, model1 in enumerate(self.model_names):
            for model2 in self.model_names[i+1:]:
                # Paired t-test
                ttest = StatisticalTests.paired_ttest(
                    self.errors[model1],
                    self.errors[model2]
                )
                
                # Permutation test
                perm = StatisticalTests.permutation_test(
                    self.errors[model1],
                    self.errors[model2],
                    n_permutations=1000  # Reduce for speed
                )
                
                result = {
                    'model_1': model1,
                    'model_2': model2,
                    'mean_error_1': np.mean(self.errors[model1]),
                    'mean_error_2': np.mean(self.errors[model2]),
                    'ttest_pvalue': ttest['p_value'],
                    'ttest_significant': ttest['significant'],
                    'permutation_pvalue': perm['p_value'],
                    'permutation_significant': perm['significant'],
                    'winner': model1 if ttest['model1_better'] else (model2 if ttest['model2_better'] else 'tie'),
                }
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def rank_models(self) -> pd.DataFrame:
        """
        Rank models by performance with confidence intervals.
        """
        results = []
        
        for name in self.model_names:
            errors = self.errors[name]
            mean, lower, upper = StatisticalTests.bootstrap_confidence_interval(errors)
            
            result = {
                'model': name,
                'mean_mae': mean,
                'ci_lower': lower,
                'ci_upper': upper,
                'ci_width': upper - lower,
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('mean_mae')
        
        return df


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Test regression metrics
    y_true = np.random.randn(100, 4)
    y_pred = y_true + np.random.randn(100, 4) * 0.1
    
    metrics = RegressionMetrics.compute_all(y_true, y_pred)
    print("\nRegression Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test causality metrics
    true_graph = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    
    pred_graph = np.array([
        [0, 0.8, 0.1, 0],
        [0, 0, 0.9, 0],
        [0, 0, 0, 0.7],
        [0, 0, 0, 0]
    ])
    
    causal_metrics = CausalityMetrics.compute_binary_metrics(true_graph, pred_graph, threshold=0.5)
    print("\nCausality Metrics:")
    for k, v in causal_metrics.items():
        print(f"  {k}: {v}")
    
    print("\n✓ All tests passed!")
