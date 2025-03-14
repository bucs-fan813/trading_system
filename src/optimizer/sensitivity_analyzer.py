# trading_system/src/optimizer/sensitivity_analyzer.py

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import mlflow

# Configure module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SensitivityAnalyzer:
    """
    Conducts sensitivity analysis by perturbing the optimized parameters and 
    re‑evaluating strategy performance.
    """

    def __init__(
        self,
        strategy_optimizer,  # Instance of StrategyOptimizer
        base_params: Dict[str, Any],
        numeric_perturbation: float = 0.1,  # Default 10% perturbation
        num_samples: int = 30,
    ):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            strategy_optimizer: Instance of StrategyOptimizer to reuse setup and evaluation.
            base_params (Dict[str, Any]): Baseline (optimized) parameters.
            numeric_perturbation (float): Perturbation factor for numeric parameters (default: 0.1 or 10%).
            num_samples (int): Number of parameter perturbation samples.
        """
        self.optimizer = strategy_optimizer
        self.base_params = base_params
        self.search_space = strategy_optimizer.search_space
        self.numeric_perturbation = numeric_perturbation
        self.num_samples = num_samples
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_perturbations(self) -> List[Dict[str, Any]]:
        """
        Generate perturbed parameter samples based on the base parameters.
        
        Returns:
            List[Dict[str, Any]]: List of perturbed parameter sets.
        """
        samples = []
        for _ in range(self.num_samples):
            sample_params = {}
            for key, base_value in self.base_params.items():
                if isinstance(base_value, (int, float)):
                    # Apply perturbation to numeric parameters
                    perturb = base_value * self.numeric_perturbation
                    new_val = np.random.uniform(base_value - perturb, base_value + perturb)
                    # Preserve integer type if the base value is integer
                    sample_params[key] = int(round(new_val)) if isinstance(base_value, int) else new_val
                else:
                    # Keep non-numeric parameters as they are
                    sample_params[key] = base_value
            samples.append(sample_params)
        return samples

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run sensitivity analysis by generating parameter perturbations and evaluating
        the strategy performance using the optimizer's walk-forward cross-validation.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                sensitivity_results: DataFrame summarizing sensitivity analysis results.
                parameter_impact: DataFrame showing the impact of each parameter on performance.
        """
        # Generate parameter perturbations
        perturbed_samples = self._generate_perturbations()
        
        # Evaluate each parameter set
        sensitivity_results = []
        for sample_idx, params in enumerate(perturbed_samples):
            self.logger.info(f"Evaluating sample {sample_idx+1}/{self.num_samples}")
            
            with mlflow.start_run(nested=True, run_name=f"Sensitivity_sample_{sample_idx+1}"):
                # Evaluate the perturbed parameters.
                ticker_metrics, weighted_metrics, overall_hm = self.optimizer.walk_forward_cv(params)				
                
                # Create a result row with parameters and key metrics.
                row = {'sample': sample_idx}
                for param_name, param_value in params.items():
                    row[f"param_{param_name}"] = param_value
                for metric_name, metric_value in weighted_metrics.items():
                    row[f"metric_{metric_name}"] = metric_value
                row['overall_harmonic_mean'] = overall_hm

                # Log this trial’s parameters and metrics.
                mlflow.log_params({f"param_{k}": v for k, v in params.items()})
                mlflow.log_metrics({f"metric_{k}": v for k, v in weighted_metrics.items()})
                mlflow.log_metric("overall_harmonic_mean", overall_hm)
                
            sensitivity_results.append(row)
            
        # Create sensitivity results DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Analyze parameter impact (correlation between parameters and key metrics)
        param_columns = [col for col in sensitivity_df.columns if col.startswith('param_')]
        metric_columns = [col for col in sensitivity_df.columns if col.startswith('metric_')]
        
        param_impact = []
        for param_col in param_columns:
            param_name = param_col.replace('param_', '')
            row = {'parameter': param_name}
            
            # Calculate correlation between this parameter and each metric
            for metric_col in metric_columns:
                metric_name = metric_col.replace('metric_', '')
                correlation = sensitivity_df[param_col].corr(sensitivity_df[metric_col])
                row[metric_name] = correlation
                
            # Add correlation with overall harmonic mean
            row['overall_harmonic_mean'] = sensitivity_df[param_col].corr(sensitivity_df['overall_harmonic_mean'])
            
            param_impact.append(row)
            
        # Create parameter impact DataFrame
        param_impact_df = pd.DataFrame(param_impact).set_index('parameter')
        
        return sensitivity_df, param_impact_df