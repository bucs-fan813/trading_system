# trading_system/src/optimizer/strategy_optimizer.py

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Tuple
import mlflow
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval													 

																 
from src.optimizer.performance_evaluator import PerformanceEvaluator

# Configure module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class StrategyOptimizer:
    """
    Universal optimizer using walk-forward analysis (time-series cross-validation) 
    combined with hyperparameter search using Hyperopt to fine-tune trading strategy 
    parameters.	   
    """

    def __init__(
        self,
        strategy_class: Type,
        db_config: Any,
        search_space: Dict[str, Any],
        tickers: List[str],
        start_date: str,
        end_date: str,
        ticker_weights: Optional[Dict[str, float]] = None,
        initial_position: int = 0,
        cv_folds: int = 3,
        risk_thresholds: Optional[Dict[str, float]] = None,
        max_evals: int = 50,
    ):
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_class (Type): Trading strategy class (should implement required methods).
            db_config (Any): Database configuration.
            search_space (Dict[str, Any]): Hyperparameter search space (compatible with Hyperopt).
            tickers (List[str]): List of ticker symbols.
            start_date (str): Backtesting start date (format 'YYYY-MM-DD').
            end_date (str): Backtesting end date (format 'YYYY-MM-DD').
            ticker_weights (Optional[Dict[str, float]]): Weights for each ticker. If None, equal weights are used.
            initial_position (int): Starting trading position.
            cv_folds (int): Number of folds for walk-forward cross-validation.
            risk_thresholds (Optional[Dict[str, float]]): Dictionary of risk thresholds.
            max_evals (int): Maximum number of evaluations for hyperparameter search.
        """ 
        self.strategy_class = strategy_class
        self.db_config = db_config
        self.search_space = search_space
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize ticker weights.
        self.ticker_weights = ticker_weights
        if self.ticker_weights is None:
													   
            self.ticker_weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        else:
            # Normalize to sum to 1.
            total_weight = sum(self.ticker_weights.values())
            if total_weight != 1.0:
                self.ticker_weights = {ticker: weight / total_weight for ticker, weight in self.ticker_weights.items()}
																						
			
			# Ensure all tickers have a weight, default to 0 for missing tickers																	
            for ticker in tickers:
                if ticker not in self.ticker_weights:
                    self.ticker_weights[ticker] = 0.0
        
        self.initial_position = initial_position
        self.cv_folds = cv_folds
        self.max_evals = max_evals
        self.risk_thresholds = risk_thresholds or {
            'max_drawdown': -0.3,          # Maximum allowed drawdown (e.g., -30%).
            'drawdown_duration': 30,       # Maximum allowed consecutive days in drawdown.
            'ulcer_index': 0.15,           # Maximum allowed ulcer index.
            'annualized_volatility': 0.4   # Annualized volatility threshold.
        }
        self.logger = logging.getLogger(self.__class__.__name__)
		
									  
        self._evaluation_cache = {}

    def _generate_signals_batch(
        self, 
        strategy_instance: Any, 
        fold_start: datetime, 
        fold_end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for all tickers at once for better efficiency.
        
        Args:
            strategy_instance (Any): Instance of the strategy class.
            fold_start (datetime): Start date of the evaluation window.
            fold_end (datetime): End date of the evaluation window.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping ticker symbols to their signal DataFrames.
        """
        try:
													  
            df_signals = strategy_instance.generate_signals(
                ticker=self.tickers,
                start_date=fold_start.strftime("%Y-%m-%d"),
                end_date=fold_end.strftime("%Y-%m-%d"),
                initial_position=self.initial_position,
                latest_only=False
            )
            							  
            if isinstance(df_signals.index, pd.MultiIndex) and df_signals.index.nlevels > 1:				 
                signals_dict = {ticker: df.reset_index(level=0, drop=True) for ticker, df in df_signals.groupby(level=0)}								 
            else:											 
                if len(self.tickers) == 1:
                    signals_dict = {self.tickers[0]: df_signals}
                else:
                    self.logger.warning("Strategy returned a single DataFrame but multiple tickers were requested.")
                    signals_dict = {}
                    											
            for ticker, df in list(signals_dict.items()):
                if df.empty:				
                    self.logger.warning("No signal data for ticker %s from %s to %s.", ticker, fold_start.date(), fold_end.date())
                    signals_dict.pop(ticker)
                    continue

                # Fallback: if a 'position' column is missing, try to use 'signal' as a proxy.																										  
                if 'position' not in df.columns:
                    df['position'] = df.get('signal', 0)
                if 'close' not in df.columns:
                    self.logger.warning("Ticker %s is missing 'close' price data.", ticker)
                    signals_dict.pop(ticker)
            
            return signals_dict
                    
        except Exception as e:
            self.logger.error("Error generating signals: %s", str(e))
            return {}

    def walk_forward_cv(
        self, 
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float]:
        """
        Perform walk-forward cross-validation using the specified parameters.
        
        Args:
            params (Dict[str, Any]): Strategy parameters.
            
        Returns:
            Tuple:
                ticker_metrics (Dict[str, Dict[str, float]]): Performance metrics per ticker.
                weighted_metrics (Dict[str, float]): Weighted average metrics across tickers.
                overall_hm (float): Overall harmonic mean.
        """
		# Check cache first				   
        params_key = str(sorted(params.items()))
        if params_key in self._evaluation_cache:
            return self._evaluation_cache[params_key]
            
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        total_days = (end_dt - start_dt).days
        fold_length = total_days // self.cv_folds

		# Dictionary to accumulate signals for each ticker across folds															   
        all_signals = {ticker: [] for ticker in self.tickers}

        for fold in range(self.cv_folds):
            fold_start = start_dt + pd.Timedelta(days=fold * fold_length)
            fold_end = (start_dt + pd.Timedelta(days=(fold + 1) * fold_length)
                        if fold < self.cv_folds - 1 else end_dt)
            self.logger.info("Fold %d from %s to %s", fold + 1, fold_start.date(), fold_end.date())

			# Create a new strategy instance for the current fold													 
            strategy_instance = self.strategy_class(self.db_config, params)
			
			# Generate signals for all tickers at once										  
            signals_dict = self._generate_signals_batch(strategy_instance, fold_start, fold_end)
            
			# Accumulate signals for each ticker									
            for ticker, signals in signals_dict.items():
                if not signals.empty:
                    all_signals[ticker].append(signals)
                    
		# Combine signals across folds							  
        combined_signals = {}
        for ticker, signals_list in all_signals.items():
            if signals_list:
                combined_signals[ticker] = pd.concat(signals_list).sort_index()
                
		# Calculate metrics for all tickers								   
        if not combined_signals:
            return {}, {}, 0.0
            
        ticker_metrics, weighted_metrics = PerformanceEvaluator.compute_multi_ticker_metrics(
            combined_signals, self.ticker_weights
        )
        
        overall_hm = weighted_metrics.get('harmonic_mean', 0.0)
        
		# Cache the results				   
        self._evaluation_cache[params_key] = (ticker_metrics, weighted_metrics, overall_hm)
        
        return ticker_metrics, weighted_metrics, overall_hm

    def _objective_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hyperopt objective function that evaluates strategy performance using walk‑forward CV.
        
        Args:
            params (Dict[str, Any]): Strategy parameters.
            
        Returns:
            Dict[str, Any]: Dictionary with keys 'loss', 'status', 'metrics' for Hyperopt.
        """
        with mlflow.start_run(nested=True, run_name="Hyperopt_Trial"):
            mlflow.set_tag("strategy_class", self.strategy_class.__name__)
            mlflow.log_params(params)
            ticker_metrics, weighted_metrics, overall_hm = self.walk_forward_cv(params)
		
			# Calculate penalty based on weighted metrics												   
            penalty = 0
			
			# Add penalties based on risk metric violations in weighted me
            if weighted_metrics.get('max_drawdown', 0) < self.risk_thresholds['max_drawdown']:
                penalty += abs(weighted_metrics['max_drawdown'] - self.risk_thresholds['max_drawdown'])
				
            if weighted_metrics.get('drawdown_duration', 0) > self.risk_thresholds['drawdown_duration']:
                penalty += (weighted_metrics['drawdown_duration'] - self.risk_thresholds['drawdown_duration']) * 0.01
				
            if weighted_metrics.get('ulcer_index', 0) > self.risk_thresholds['ulcer_index']:
                penalty += (weighted_metrics['ulcer_index'] - self.risk_thresholds['ulcer_index'])
				
            if weighted_metrics.get('annualized_volatility', 0) > self.risk_thresholds['annualized_volatility']:
                penalty += (weighted_metrics['annualized_volatility'] - self.risk_thresholds['annualized_volatility'])
            
            objective_score = overall_hm - penalty
            mlflow.log_metric("overall_harmonic_mean", overall_hm)
            mlflow.log_metric("penalty", penalty)
            mlflow.log_metric("objective_score", objective_score)
											  
            return {
                'loss': -objective_score, 
                'status': STATUS_OK, 
                'ticker_metrics': ticker_metrics,
                'weighted_metrics': weighted_metrics
            }

    def run_optimization(self) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        Execute the hyperparameter search, returning the best parameters and performance reports.
        
        Returns:
            Tuple:
                best_params (Dict[str, Any]): The best parameters discovered.
                ticker_performance (pd.DataFrame): Performance metrics for each ticker.
                param_history (pd.DataFrame): History of parameter evaluations.
        """
        with mlflow.start_run(run_name="StrategyOptimization"):
            trials = Trials()
            best = fmin(
                fn=self._objective_function,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials
            )
											
            best_params = space_eval(self.search_space, best)
            mlflow.set_tag("best_strategy_class", self.strategy_class.__name__)
            mlflow.log_params(best_params)
            
            # Evaluate strategy with the best parameters over the full period.
            ticker_metrics, weighted_metrics, _ = self.walk_forward_cv(best_params)
            mlflow.log_metrics({"final_overall_harmonic_mean": weighted_metrics.get('harmonic_mean', 0.0)})
													  				   
			# Try to retrieve company info for all tickers
            try:
                strategy_instance = self.strategy_class(self.db_config, best_params)
                company_info = strategy_instance.get_company_info(self.tickers)
                if isinstance(company_info, pd.Series):
                    company_info = company_info.to_frame().T
            except Exception as e:
                self.logger.error("Error retrieving company info: %s", str(e))
                company_info = pd.DataFrame()
			# Build detailed ticker performance report									  
            report_rows = []
            for ticker, metrics in ticker_metrics.items():
                row = {'ticker': ticker, 'weight': self.ticker_weights.get(ticker, 0)}
																	 
                row.update(metrics)
                if (not company_info.empty) and (ticker in company_info.index):
                    info = company_info.loc[ticker]
                    for col in ['industry', 'industrykey', 'industrydisp', 'sector', 'sectorkey', 'sectordisp']:
                        row[col] = info.get(col, None)
				  
                report_rows.append(row)
                
            weighted_row = {'ticker': 'WEIGHTED_AVG', 'weight': 1.0}
            weighted_row.update(weighted_metrics)
            report_rows.append(weighted_row)
            
            ticker_performance = pd.DataFrame(report_rows).set_index('ticker')
            
            param_history = []
            for trial_idx, trial in enumerate(trials.trials):
                if trial['result']['status'] == 'ok':
                    row = {'trial': trial_idx}
                    params_trial = space_eval(self.search_space, trial['misc']['vals'])
                    row.update(params_trial)
                    row.update({
                        'harmonic_mean': trial['result']['weighted_metrics'].get('harmonic_mean', 0),
                        'objective_score': -trial['result']['loss'],
                    })
                    param_history.append(row)
                    
            param_history_df = pd.DataFrame(param_history)
            
            param_history_df.to_csv("param_history.csv", index=False)
            mlflow.log_artifact("param_history.csv")
            
            return best_params, ticker_performance, param_history_df
        
    def get_evaluation_cache(self) -> Dict[str, Tuple]:
        """
        Get the cached evaluation results, which can be reused by other components
        like the SensitivityAnalyzer.
        
        Returns:
            Dict[str, Tuple]: The evaluation cache mapping parameter strings to evaluation results.
        """
        return self._evaluation_cache