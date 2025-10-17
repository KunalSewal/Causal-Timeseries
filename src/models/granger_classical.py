"""
Classical Granger Causality Tests

Implements traditional statistical Granger causality using VAR models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats


class VARGrangerTester:
    """
    Vector Autoregression (VAR) based Granger causality tester.
    
    This is the classical baseline for Granger causality analysis.
    Assumes linear relationships between variables.
    
    Args:
        maxlag: Maximum lag to test
        ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
    
    Example:
        >>> tester = VARGrangerTester(maxlag=5)
        >>> causality_matrix, p_values = tester.fit_test(data)
        >>> print("NVDA causes MSFT:", causality_matrix[nvda_idx, msft_idx])
    """
    
    def __init__(
        self,
        maxlag: int = 5,
        ic: str = 'aic',
    ):
        self.maxlag = maxlag
        self.ic = ic
        self.model = None
        self.results = None
        self.variable_names = None
    
    def fit(self, data: pd.DataFrame) -> 'VARGrangerTester':
        """
        Fit VAR model to data.
        
        Args:
            data: DataFrame with time series in columns
        
        Returns:
            self
        """
        self.variable_names = list(data.columns)
        self.model = VAR(data)
        
        # Fit and select lag order based on information criterion
        self.results = self.model.fit(maxlags=self.maxlag, ic=self.ic)
        
        print(f"Selected lag order: {self.results.k_ar} (based on {self.ic.upper()})")
        return self
    
    def test_granger_causality(
        self,
        data: pd.DataFrame,
        max_lag: Optional[int] = None,
        test: str = 'ssr_ftest',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test Granger causality for all pairs of variables.
        
        Args:
            data: DataFrame with time series
            max_lag: Maximum lag to test (uses self.maxlag if None)
            test: Statistical test to use ('ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest')
        
        Returns:
            causality_matrix: (num_vars, num_vars) where [i,j] indicates if i Granger-causes j
            p_values: (num_vars, num_vars) array of p-values
        """
        if max_lag is None:
            max_lag = self.maxlag
        
        self.variable_names = list(data.columns)
        num_vars = len(self.variable_names)
        
        causality_matrix = np.zeros((num_vars, num_vars))
        p_values = np.ones((num_vars, num_vars))
        
        # Test each pair
        for i in range(num_vars):
            for j in range(num_vars):
                if i == j:
                    continue
                
                cause_var = self.variable_names[i]
                effect_var = self.variable_names[j]
                
                try:
                    # Granger test: does cause_var Granger-cause effect_var?
                    result = grangercausalitytests(
                        data[[effect_var, cause_var]],
                        maxlag=max_lag,
                        verbose=False
                    )
                    
                    # Get minimum p-value across all lags
                    min_p_value = min([result[lag][0][test][1] for lag in range(1, max_lag + 1)])
                    p_values[i, j] = min_p_value
                    
                    # Significant if p < 0.05
                    causality_matrix[i, j] = 1 if min_p_value < 0.05 else 0
                    
                except Exception as e:
                    print(f"Warning: Could not test {cause_var} -> {effect_var}: {e}")
                    p_values[i, j] = 1.0
        
        return causality_matrix, p_values
    
    def fit_test(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Fit VAR and test causality in one step."""
        self.fit(data)
        return self.test_granger_causality(data, **kwargs)
    
    def forecast(self, steps: int = 1) -> pd.DataFrame:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps ahead to forecast
        
        Returns:
            DataFrame with forecasted values
        """
        if self.results is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        forecast = self.results.forecast(self.results.endog[-self.results.k_ar:], steps=steps)
        return pd.DataFrame(forecast, columns=self.variable_names)
    
    def get_coefficients(self) -> np.ndarray:
        """Get VAR coefficients."""
        if self.results is None:
            raise RuntimeError("Model must be fitted before getting coefficients")
        return self.results.params
    
    def summary(self):
        """Print model summary."""
        if self.results is None:
            raise RuntimeError("Model must be fitted before summary")
        return self.results.summary()


class GrangerCausalityTest:
    """
    Wrapper class for individual pairwise Granger causality tests.
    
    Useful for testing specific variable pairs with detailed statistics.
    """
    
    @staticmethod
    def test_pair(
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str,
        maxlag: int = 5,
        test: str = 'ssr_ftest',
        alpha: float = 0.05,
    ) -> Dict:
        """
        Test if cause_var Granger-causes effect_var.
        
        Args:
            data: DataFrame with time series
            cause_var: Name of potential cause variable
            effect_var: Name of potential effect variable
            maxlag: Maximum lag to test
            test: Statistical test type
            alpha: Significance level
        
        Returns:
            Dictionary with test results
        
        Example:
            >>> result = GrangerCausalityTest.test_pair(
            ...     data, 'NVDA', 'MSFT', maxlag=5
            ... )
            >>> print(f"P-value: {result['p_value']}")
            >>> print(f"Causal: {result['is_causal']}")
        """
        try:
            result = grangercausalitytests(
                data[[effect_var, cause_var]],
                maxlag=maxlag,
                verbose=False
            )
            
            # Extract statistics for each lag
            lag_results = []
            for lag in range(1, maxlag + 1):
                test_stat = result[lag][0][test][0]
                p_value = result[lag][0][test][1]
                lag_results.append({
                    'lag': lag,
                    'statistic': test_stat,
                    'p_value': p_value,
                    'is_significant': p_value < alpha
                })
            
            # Get best lag (minimum p-value)
            best_lag_result = min(lag_results, key=lambda x: x['p_value'])
            
            return {
                'cause': cause_var,
                'effect': effect_var,
                'is_causal': best_lag_result['is_significant'],
                'best_lag': best_lag_result['lag'],
                'p_value': best_lag_result['p_value'],
                'test_statistic': best_lag_result['statistic'],
                'all_lags': lag_results,
                'test_type': test,
                'alpha': alpha,
            }
        
        except Exception as e:
            return {
                'cause': cause_var,
                'effect': effect_var,
                'is_causal': False,
                'error': str(e),
            }
    
    @staticmethod
    def test_all_pairs(
        data: pd.DataFrame,
        maxlag: int = 5,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Test Granger causality for all pairs and return as DataFrame.
        
        Args:
            data: DataFrame with time series
            maxlag: Maximum lag to test
            alpha: Significance level
        
        Returns:
            DataFrame with test results for each pair
        """
        variable_names = list(data.columns)
        results = []
        
        for cause in variable_names:
            for effect in variable_names:
                if cause != effect:
                    result = GrangerCausalityTest.test_pair(
                        data, cause, effect, maxlag, alpha=alpha
                    )
                    results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_causality_strength(
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str,
        lag: int,
    ) -> float:
        """
        Compute the strength of causality using prediction improvement.
        
        Measures how much the prediction error decreases when including
        the cause variable.
        
        Returns:
            Causality strength (0 to 1, higher = stronger causality)
        """
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import LinearRegression
        
        # Create lagged features
        X_restricted = []  # Only effect's past
        X_full = []        # Effect's past + cause's past
        y = []
        
        for t in range(lag, len(data)):
            # Target: current effect value
            y.append(data[effect_var].iloc[t])
            
            # Restricted: only effect's past
            effect_past = data[effect_var].iloc[t-lag:t].values
            X_restricted.append(effect_past)
            
            # Full: effect's past + cause's past
            cause_past = data[cause_var].iloc[t-lag:t].values
            X_full.append(np.concatenate([effect_past, cause_past]))
        
        X_restricted = np.array(X_restricted)
        X_full = np.array(X_full)
        y = np.array(y)
        
        # Split train/test
        split = int(0.8 * len(y))
        
        # Train restricted model
        model_restricted = LinearRegression()
        model_restricted.fit(X_restricted[:split], y[:split])
        pred_restricted = model_restricted.predict(X_restricted[split:])
        mse_restricted = mean_squared_error(y[split:], pred_restricted)
        
        # Train full model
        model_full = LinearRegression()
        model_full.fit(X_full[:split], y[:split])
        pred_full = model_full.predict(X_full[split:])
        mse_full = mean_squared_error(y[split:], pred_full)
        
        # Causality strength: relative improvement
        if mse_restricted == 0:
            return 0.0
        
        strength = (mse_restricted - mse_full) / mse_restricted
        return max(0.0, min(1.0, strength))  # Clamp to [0, 1]


if __name__ == "__main__":
    # Example usage
    print("Testing classical Granger causality...")
    
    # Generate synthetic data
    from src.data.time_series_generator import generate_synthetic_granger
    
    df, true_graph = generate_synthetic_granger(num_vars=4, num_samples=500)
    
    # Test with VAR
    tester = VARGrangerTester(maxlag=5)
    causality_matrix, p_values = tester.fit_test(df)
    
    print("\nEstimated Causality Matrix:")
    print(causality_matrix)
    print("\nP-values:")
    print(p_values)
    print("\nTrue Causal Graph:")
    print(true_graph)
