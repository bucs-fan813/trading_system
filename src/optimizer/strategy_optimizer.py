# trading_system/src/optimizer/strategy_optimizer.py

"""
Strategy optimization module using walk-forward cross-validation and portfolio metrics.

This module optimizes trading strategy parameters by evaluating performance
on a portfolio level across walk-forward folds. It uses Hyperopt for search
and MLflow for tracking.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Tuple
import mlflow
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import hashlib
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback # Import traceback

from src.optimizer.performance_evaluator import PerformanceEvaluator, MetricsDict

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StrategyOptimizer:
    """
    Optimizes trading strategy parameters using walk-forward cross-validation
    and portfolio-based performance evaluation.

    Evaluates parameters by:
    1. Generating signals across multiple tickers for different time folds.
    2. Calculating portfolio performance metrics for each fold using PerformanceEvaluator.
    3. Averaging portfolio metrics across folds.
    4. Applying risk penalties based on portfolio metrics.
    5. Using Hyperopt to find parameters maximizing a risk-adjusted objective (e.g., harmonic mean).
    """

    def __init__(
        self,
        strategy_class: Type,
        db_config: Any, # Use appropriate type hint if possible, e.g., DatabaseConfig
        search_space: Dict[str, Any],
        tickers: List[str],
        start_date: str,
        end_date: str,
        # ticker_weights removed - allocation is now handled by PerformanceEvaluator
        initial_position: int = 0, # Keep for strategy if needed, but not used for portfolio eval directly
        cv_folds: int = 3,
        risk_thresholds: Optional[Dict[str, float]] = None,
        optimization_metric: str = 'harmonic_mean', # Metric to optimize (from portfolio metrics)
        max_evals: int = 50,
        run_name: str = "StrategyOptimization",
        n_jobs: int = -1
    ):
        """
        Initialize the strategy optimizer.

        Args:
            strategy_class (Type): Trading strategy class. Must implement `generate_signals`.
            db_config (Any): Database configuration object.
            search_space (Dict[str, Any]): Hyperparameter search space (Hyperopt format).
            tickers (List[str]): List of ticker symbols for the portfolio.
            start_date (str): Backtesting start date ('YYYY-MM-DD').
            end_date (str): Backtesting end date ('YYYY-MM-DD').
            initial_position (int): Starting position passed to the strategy (may not be relevant for portfolio).
            cv_folds (int): Number of folds for walk-forward cross-validation.
            risk_thresholds (Optional[Dict[str, float]]): Thresholds for portfolio risk metrics
                (e.g., {'max_drawdown': -0.2, 'annualized_volatility': 0.3}).
            optimization_metric (str): The portfolio metric to maximize (default: 'harmonic_mean').
            max_evals (int): Maximum evaluations for hyperparameter search.
            run_name (str): Name for the MLflow run.
            n_jobs (int): Parallel jobs for fold evaluation (-1 uses all CPU cores).
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty.")

        self.strategy_class = strategy_class
        self.db_config = db_config
        self.search_space = search_space
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date.")

        self.run_name = run_name
        self.initial_position = initial_position # Passed to strategy
        self.cv_folds = max(1, cv_folds) # Ensure at least 1 fold
        self.max_evals = max_evals
        self.optimization_metric = optimization_metric

        # Default risk thresholds for portfolio metrics
        self.risk_thresholds = risk_thresholds or {
            'max_drawdown': -0.30,          # Max portfolio drawdown
            'annualized_volatility': 0.40,  # Max portfolio volatility
            'drawdown_duration': 90,        # Max portfolio drawdown duration (days)
            # 'ulcer_index': 0.15,          # Can add others if needed
        }
        if 'max_drawdown' in self.risk_thresholds and self.risk_thresholds['max_drawdown'] > 0:
             logger.warning("max_drawdown threshold should be negative. Adjusting.")
             self.risk_thresholds['max_drawdown'] = -abs(self.risk_thresholds['max_drawdown'])


        # Parallel jobs
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, cpu_count())

        # Caching for evaluated parameter sets
        self._evaluation_cache: Dict[str, Tuple[MetricsDict, Dict[str, MetricsDict]]] = {} # Cache stores (portfolio_metrics, per_ticker_metrics)

        # Precompute fold ranges
        self._fold_ranges = self._compute_fold_ranges()

        logger.info(f"Initialized StrategyOptimizer for {strategy_class.__name__} with {len(tickers)} tickers.")
        logger.info(f"Period: {start_date} to {end_date}, Folds: {cv_folds}, Max Evals: {max_evals}")
        logger.info(f"Optimizing for portfolio metric: {self.optimization_metric}")
        logger.info(f"Risk Thresholds: {self.risk_thresholds}")


    def _compute_fold_ranges(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Computes date ranges for each walk-forward fold."""
        total_duration = self.end_date - self.start_date
        if total_duration.days < self.cv_folds:
             logger.warning(f"Total duration ({total_duration.days} days) is less than cv_folds ({self.cv_folds}). Reducing folds to duration.")
             self.cv_folds = max(1, total_duration.days)

        fold_length = total_duration / self.cv_folds

        fold_ranges = []
        current_start = self.start_date
        for fold in range(self.cv_folds):
            current_end = current_start + fold_length
            # Ensure the last fold ends exactly on the end_date
            if fold == self.cv_folds - 1:
                current_end = self.end_date
            # Make sure start and end are distinct times if fold_length is very small
            if current_end <= current_start:
                 current_end = current_start + pd.Timedelta(days=1) # Min 1 day fold if possible
                 if current_end > self.end_date: # Ensure we don't exceed end date
                      current_end = self.end_date
                      if current_start >= current_end and fold > 0: # Avoid duplicate last fold
                           fold_ranges.pop() # Remove previous if this one collapses
                           break

            fold_ranges.append((current_start, current_end))
            current_start = current_end # Start of next fold is end of current

        # Filter out any potentially invalid ranges (start >= end)
        fold_ranges = [(s, e) for s, e in fold_ranges if e > s]

        if not fold_ranges:
             logger.warning("Could not create valid fold ranges. Using single fold for the entire period.")
             return [(self.start_date, self.end_date)]

        logger.debug(f"Computed Fold Ranges: {fold_ranges}")
        return fold_ranges

    def _params_to_key(self, params: Dict[str, Any]) -> str:
        """Convert parameters to a stable cache key."""
        try:
            # Sort keys for stability, handle non-serializable items if necessary
            serialized = json.dumps(params, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()
        except TypeError as e:
             logger.error(f"Error creating cache key for params {params}: {e}. Using fallback hash.")
             # Fallback: hash the string representation (less reliable)
             return hashlib.md5(str(params).encode()).hexdigest()


    def _generate_signals_batch(
        self,
        strategy_instance: Any,
        fold_start: pd.Timestamp,
        fold_end: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for all tickers for a given fold period.

        Args:
            strategy_instance (Any): Instantiated strategy object.
            fold_start (pd.Timestamp): Start date for the fold.
            fold_end (pd.Timestamp): End date for the fold.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping tickers to their signal DataFrames.
                                     Returns empty dict on failure.
        """
        try:
            logger.debug(f"Generating signals for fold: {fold_start.date()} to {fold_end.date()}")
            df_signals_raw = strategy_instance.generate_signals(
                ticker=self.tickers, # Pass the list of tickers
                start_date=fold_start.strftime("%Y-%m-%d"),
                end_date=fold_end.strftime("%Y-%m-%d"),
                initial_position=self.initial_position, # Pass along if strategy uses it
                latest_only=False # We need the full history for backtesting
            )

            print("\n\n")

            print("line 203")
            print(self.tickers)
            print("start: " + fold_start.strftime("%Y-%m-%d"))
            print("end: " + fold_end.strftime("%Y-%m-%d"))
            print(df_signals_raw)

            print("\n\n")

            signals_dict = {}
            if isinstance(df_signals_raw, pd.DataFrame) and not df_signals_raw.empty:
                # Handle multi-index output (index level 0 is ticker)
                if isinstance(df_signals_raw.index, pd.MultiIndex):
                    # Check if level 0 name is 'ticker' or similar, or assume it is if unnamed
                    ticker_level_name = df_signals_raw.index.names[0] if df_signals_raw.index.names[0] else None
                    if ticker_level_name:
                         signals_dict = {
                             ticker: df.reset_index(level=ticker_level_name, drop=True)
                             for ticker, df in df_signals_raw.groupby(level=ticker_level_name)
                         }
                    else: # Assume level 0 is ticker if unnamed
                         signals_dict = {
                             ticker: df.reset_index(level=0, drop=True)
                             for ticker, df in df_signals_raw.groupby(level=0)
                         }

                # Handle single DataFrame with 'ticker' column
                elif 'ticker' in df_signals_raw.columns:
                    signals_dict = {
                        ticker: group.drop(columns=['ticker']).set_index(group.index.name or 'date') # Ensure index is set
                        for ticker, group in df_signals_raw.groupby('ticker')
                    }
                 # Handle single DataFrame for a single ticker scenario
                elif len(self.tickers) == 1 and self.tickers[0] not in signals_dict:
                    signals_dict = {self.tickers[0]: df_signals_raw}

                else:
                     logger.warning(f"Unexpected DataFrame format from strategy. Assuming single ticker if only one requested.")
                     if len(self.tickers) == 1:
                          signals_dict = {self.tickers[0]: df_signals_raw}
                     else:
                          logger.error("Cannot determine ticker mapping from strategy output for multiple tickers.")
                          return {}

            elif isinstance(df_signals_raw, dict):
                 # If strategy already returns a dict
                 signals_dict = df_signals_raw

            else:
                logger.warning(f"Strategy returned empty or unexpected data type for fold {fold_start.date()} to {fold_end.date()}")
                return {}

            # --- Validation and Filtering ---
            final_signals = {}
            required_cols = ['signal', 'close'] # Essential for portfolio evaluation
            optional_cols = ['position'] # Needed for single asset metrics (informational)

            for ticker, df in signals_dict.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.debug(f"No signal data for ticker {ticker} in fold.")
                    continue
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                        logger.debug(f"Converted index for {ticker} to DatetimeIndex.")
                    except Exception:
                        logger.warning(f"Index for {ticker} is not datetime and could not be converted. Skipping.")
                        continue

                # Ensure required columns are present
                missing_required = [col for col in required_cols if col not in df.columns]
                if missing_required:
                    logger.warning(f"Ticker {ticker} missing required columns: {missing_required}. Skipping.")
                    continue

                # Check for 'position' needed for single-asset metrics reporting
                if 'position' not in df.columns:
                    logger.info(f"Ticker {ticker} missing optional 'position' column. Will skip single-asset metric calculation for it.")
                    # Add a dummy 'position' column filled with NaN or 0 if needed downstream? No, let evaluator handle missing col.

                # Ensure data is within the fold range (strategy might return extra lookback data)
                df_filtered = df.loc[fold_start:fold_end].copy()
                if df_filtered.empty:
                    logger.debug(f"No data for ticker {ticker} within the fold range {fold_start.date()}-{fold_end.date()}.")
                    continue

                final_signals[ticker] = df_filtered

            if not final_signals:
                 logger.warning(f"No valid signals generated across all tickers for fold {fold_start.date()} to {fold_end.date()}.")

            return final_signals

        except Exception as e:
            logger.error(f"Error generating signals batch for fold {fold_start.date()}-{fold_end.date()}: {e}")
            logger.error(traceback.format_exc()) # Log full traceback
            return {}

    def _evaluate_fold(self, params: Dict[str, Any], fold_idx: int) -> Dict[str, Any]:
        """
        Evaluate strategy performance for a single fold using portfolio metrics.

        Args:
            params (Dict[str, Any]): Strategy parameters for this evaluation.
            fold_idx (int): The index of the fold to evaluate.

        Returns:
            Dict[str, Any]: Results dict including status, fold index,
                            portfolio metrics, and individual ticker metrics.
        """
        if fold_idx >= len(self._fold_ranges):
             logger.error(f"Invalid fold index {fold_idx} requested.")
             return {"status": "error", "fold": fold_idx, "error": "Invalid fold index"}

        fold_start, fold_end = self._fold_ranges[fold_idx]
        fold_label = f"{fold_start.date()}_to_{fold_end.date()}"
        logger.info(f"Evaluating Fold {fold_idx + 1}/{self.cv_folds} ({fold_label}) with params: {params}")

        try:
            # 1. Instantiate strategy
            strategy_instance = self.strategy_class(self.db_config, params)

            print("\n\n")
            print(params)
            print("\n\n")

            # 2. Generate signals for the fold
            signals_dict = self._generate_signals_batch(strategy_instance, fold_start, fold_end)


            # --- DEBUG LOGGING FOR SIGNALS RECEIVED BY OPTIMIZER ---
            if not signals_dict:
                logger.warning(f"Fold {fold_idx + 1}: No valid signals dictionary generated by _generate_signals_batch.")
                # ...(existing logic for empty signals_dict)...
            else:
                total_buys_fold = 0
                total_sells_fold = 0
                total_zeros_fold = 0
                total_rows_fold = 0
                num_tickers_with_signals = 0
                for ticker, df_signals in signals_dict.items():
                    if isinstance(df_signals, pd.DataFrame) and not df_signals.empty:
                        if 'signal' in df_signals.columns:
                            total_buys_fold += (df_signals['signal'] == 1).sum()
                            total_sells_fold += (df_signals['signal'] == -1).sum()
                            total_zeros_fold += (df_signals['signal'] == 0).sum()
                            total_rows_fold += len(df_signals)
                            num_tickers_with_signals += 1
                        else:
                            logger.warning(f"Fold {fold_idx + 1}, Ticker {ticker}: Received DataFrame missing 'signal' column.")
                    # else: logger.debug(f"Fold {fold_idx + 1}, Ticker {ticker}: Received empty or non-DataFrame signal data.") # Can be verbose

                logger.info( # Use INFO for fold summary
                    f"Fold {fold_idx + 1} ({fold_label}): Evaluator received signals for {num_tickers_with_signals}/{len(signals_dict)} tickers. "
                    f"Across Tickers -> Total Buys: {total_buys_fold}, Total Sells: {total_sells_fold}, Total Holds/Exits(0): {total_zeros_fold}. "
                    f"Total Signal Rows: {total_rows_fold}"
                )
                if total_rows_fold == 0:
                     logger.warning(f"Fold {fold_idx + 1}: Although signals_dict was generated, it contained ZERO signal rows across all tickers.")
            # --- END DEBUG LOGGING ---




            if not signals_dict:
                logger.warning(f"Fold {fold_idx + 1}: No valid signals generated.")
                # Return zero metrics to avoid breaking aggregation, but mark status?
                # Let's return default zero metrics.
                zero_portfolio_metrics = {**{k: 0.0 for k in PerformanceEvaluator._get_metric_names()}, 'signal_accuracy': 0.0}
                return {
                    "status": "ok", # Still 'ok' for aggregation, but metrics are zero
                    "fold": fold_idx,
                    "portfolio_metrics": zero_portfolio_metrics,
                    "ticker_metrics": {}, # No ticker metrics if no signals
                    "notes": "No valid signals generated"
                }

            # 3. Compute Portfolio Metrics
            portfolio_metrics = PerformanceEvaluator.compute_portfolio_metrics(
                signals_dict,
                # allocation_rule='equal_weight_active' # Default in evaluator
            )

            # 4. (Optional) Compute Individual Ticker Metrics for reporting
            # Use the deprecated function carefully, only if 'position' column exists
            ticker_metrics_individual = {}
            try:
                 # Check if any signal df has 'position' before calling
                 if any('position' in df.columns for df in signals_dict.values()):
                      ticker_metrics_individual, _ = PerformanceEvaluator.compute_multi_ticker_metrics(
                           signals_dict
                      )
                 else:
                      logger.info(f"Fold {fold_idx + 1}: Skipping individual ticker metrics (no 'position' column found).")
            except Exception as e_ticker:
                 logger.warning(f"Fold {fold_idx + 1}: Failed to compute individual ticker metrics: {e_ticker}")


            return {
                "status": "ok",
                "fold": fold_idx,
                "portfolio_metrics": portfolio_metrics,
                "ticker_metrics": ticker_metrics_individual, # Store individual metrics
            }

        except Exception as e:
            logger.error(f"Error evaluating fold {fold_idx + 1} ({fold_label}): {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "fold": fold_idx, "error": str(e)}

    def walk_forward_cv(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Optional[MetricsDict], Optional[Dict[str, MetricsDict]]]:
        """
        Perform walk-forward cross-validation for a given set of parameters.

        Averages portfolio metrics across all successful folds. Also averages
        individual ticker metrics if available.

        Args:
            params (Dict[str, Any]): Strategy parameters to evaluate.

        Returns:
            Tuple[Optional[MetricsDict], Optional[Dict[str, MetricsDict]]]:
                - Average portfolio metrics across folds (or None if all folds failed).
                - Average individual ticker metrics across folds (or None if unavailable/failed).
        """
        params_key = self._params_to_key(params)
        if params_key in self._evaluation_cache:
            logger.debug(f"Cache hit for params: {params}")
            return self._evaluation_cache[params_key]

        logger.info(f"Running Walk-Forward CV for params: {params}")

        # Evaluate folds in parallel or sequentially
        if self.cv_folds > 1 and self.n_jobs > 1:
            try:
                 # Use partial to pass fixed 'params' argument
                fold_eval_func = partial(self._evaluate_fold, params)
                # Determine number of processes
                num_processes = min(self.n_jobs, self.cv_folds)
                logger.info(f"Using {num_processes} parallel processes for {self.cv_folds} folds.")
                with Pool(processes=num_processes) as pool:
                     fold_results = pool.map(fold_eval_func, range(self.cv_folds))
            except Exception as pool_error:
                 logger.error(f"Multiprocessing pool failed: {pool_error}. Falling back to sequential execution.")
                 logger.error(traceback.format_exc())
                 fold_results = [self._evaluate_fold(params, i) for i in range(self.cv_folds)]

        else:
            logger.info(f"Running {self.cv_folds} folds sequentially.")
            fold_results = [self._evaluate_fold(params, i) for i in range(self.cv_folds)]

        # Filter successful fold results
        valid_results = [r for r in fold_results if r.get("status") == "ok"]

        if not valid_results:
            logger.warning(f"No successful folds for params: {params}. Cannot calculate average metrics.")
            self._evaluation_cache[params_key] = (None, None)
            return None, None

        # --- Aggregate Portfolio Metrics ---
        all_portfolio_metrics = [r["portfolio_metrics"] for r in valid_results]
        # Average portfolio metrics across folds
        avg_portfolio_metrics = pd.DataFrame(all_portfolio_metrics).mean().to_dict()
        # Recalculate harmonic mean based on averaged positive metrics
        avg_portfolio_metrics['harmonic_mean'] = PerformanceEvaluator.harmonic_mean(avg_portfolio_metrics)


        # --- Aggregate Individual Ticker Metrics (if available) ---
        all_ticker_metrics_by_fold = [r.get("ticker_metrics", {}) for r in valid_results]
        aggregated_ticker_metrics: Dict[str, List[MetricsDict]] = {}

        for fold_ticker_metrics in all_ticker_metrics_by_fold:
            for ticker, metrics in fold_ticker_metrics.items():
                 if ticker not in aggregated_ticker_metrics:
                      aggregated_ticker_metrics[ticker] = []
                 aggregated_ticker_metrics[ticker].append(metrics)

        avg_ticker_metrics: Dict[str, MetricsDict] = {}
        if aggregated_ticker_metrics:
             for ticker, metrics_list in aggregated_ticker_metrics.items():
                  if metrics_list:
                       # Average each metric across folds for this ticker
                       avg_metrics_for_ticker = pd.DataFrame(metrics_list).mean().to_dict()
                       # Recalculate harmonic mean for the averaged ticker metrics
                       avg_metrics_for_ticker['harmonic_mean'] = PerformanceEvaluator.harmonic_mean(avg_metrics_for_ticker)
                       avg_ticker_metrics[ticker] = avg_metrics_for_ticker
        else:
             avg_ticker_metrics = None # Explicitly set to None if no ticker data


        # Cache and return results
        result = (avg_portfolio_metrics, avg_ticker_metrics)
        self._evaluation_cache[params_key] = result
        logger.info(f"Finished CV for params: {params}. Avg Portfolio Harmonic Mean: {avg_portfolio_metrics.get('harmonic_mean', 'N/A'):.4f}")
        return result


    def _calculate_risk_penalty(self, portfolio_metrics: MetricsDict) -> float:
        """
        Calculate a penalty score based on portfolio risk threshold violations.
        Higher penalty means more/worse violations.

        Args:
            portfolio_metrics (MetricsDict): The averaged portfolio metrics.

        Returns:
            float: Risk penalty score (>= 0).
        """
        if not portfolio_metrics: # Should not happen if called correctly, but safety check
             return 100.0 # High penalty if no metrics

        penalty = 0.0
        violated_thresholds = []

        # Max Drawdown (is negative, threshold is negative)
        max_dd_thresh = self.risk_thresholds.get('max_drawdown', -1.0) # Default to -100% if missing
        max_dd = portfolio_metrics.get('max_drawdown', 0.0)
        if max_dd < max_dd_thresh:
            penalty += abs(max_dd - max_dd_thresh) * 2 # Penalize drawdown more heavily? Adjust multiplier.
            violated_thresholds.append(f"Max Drawdown ({max_dd:.2%} < {max_dd_thresh:.2%})")

        # Annualized Volatility
        vol_thresh = self.risk_thresholds.get('annualized_volatility', 1.0) # Default to 100% if missing
        vol = portfolio_metrics.get('annualized_volatility', 0.0)
        if vol > vol_thresh:
            penalty += (vol - vol_thresh)
            violated_thresholds.append(f"Volatility ({vol:.2%} > {vol_thresh:.2%})")

        # Drawdown Duration
        dd_dur_thresh = self.risk_thresholds.get('drawdown_duration', 365 * 10) # Default 10 years if missing
        dd_dur = portfolio_metrics.get('drawdown_duration', 0.0)
        if dd_dur > dd_dur_thresh:
             # Scale duration penalty to be comparable to % penalties
             penalty += (dd_dur - dd_dur_thresh) / 252.0 # e.g., penalty of ~0.004 per extra day
             violated_thresholds.append(f"Drawdown Duration ({dd_dur:.0f} > {dd_dur_thresh:.0f} days)")

        # Ulcer Index (if threshold provided)
        ulcer_thresh = self.risk_thresholds.get('ulcer_index')
        if ulcer_thresh is not None:
             ulcer = portfolio_metrics.get('ulcer_index', 0.0)
             if ulcer > ulcer_thresh:
                  penalty += (ulcer - ulcer_thresh) * 2 # Scale ulcer index penalty?
                  violated_thresholds.append(f"Ulcer Index ({ulcer:.3f} > {ulcer_thresh:.3f})")

        if violated_thresholds:
             logger.debug(f"Risk penalty applied: {penalty:.4f}. Violations: {', '.join(violated_thresholds)}")
        # else:
        #      logger.debug("No risk threshold violations.")

        return penalty

    def _objective_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function for Hyperopt minimization.

        Evaluates parameters using walk_forward_cv, calculates risk penalty,
        and returns a score to be minimized (negative of reward minus penalty).

        Args:
            params (Dict[str, Any]): Parameters suggested by Hyperopt.

        Returns:
            Dict[str, Any]: Dictionary expected by Hyperopt, including 'loss' and 'status'.
        """
        # Start nested MLflow run for this trial
        with mlflow.start_run(nested=True, run_name=f"Trial_{params_to_short_str(params)}"):
            mlflow.log_params(params)
            mlflow.set_tag("strategy_class", self.strategy_class.__name__)

            # Evaluate parameters using walk-forward CV
            avg_portfolio_metrics, avg_ticker_metrics = self.walk_forward_cv(params)

            # Handle evaluation failure
            if avg_portfolio_metrics is None:
                logger.warning(f"Evaluation failed for params: {params}. Assigning high loss.")
                mlflow.log_metric("objective_score", -100.0) # Log failure score
                return {
                    'loss': 100.0,  # High loss indicates failure
                    'status': STATUS_OK, # Must return STATUS_OK for hyperopt
                    'params': params,
                    'portfolio_metrics': None,
                    'ticker_metrics': None,
                    'notes': 'Walk-forward CV failed'
                }

            # Get the primary optimization metric value
            primary_metric_value = avg_portfolio_metrics.get(self.optimization_metric, 0.0)

            # Calculate risk penalty based on portfolio metrics
            penalty = self._calculate_risk_penalty(avg_portfolio_metrics)

            # Objective score: Higher is better (metric value minus penalty)
            objective_score = primary_metric_value - penalty

            # Hyperopt minimizes 'loss', so we use the negative of the objective score
            loss = -objective_score

            # Log results to MLflow
            mlflow.log_metric(f"avg_portfolio_{self.optimization_metric}", primary_metric_value)
            mlflow.log_metric("risk_penalty", penalty)
            mlflow.log_metric("objective_score", objective_score) # Log the raw score (higher is better)
            mlflow.log_metrics({f"avg_portfolio_{k}": v for k, v in avg_portfolio_metrics.items() if k != self.optimization_metric})

            # Log average ticker metrics (optional, can be verbose)
            # if avg_ticker_metrics:
            #     for ticker, metrics in avg_ticker_metrics.items():
            #         mlflow.log_metrics({f"avg_ticker_{ticker}_{k}": v for k, v in metrics.items()}, step=0) # Step 0 or omit

            return {
                'loss': loss,
                'status': STATUS_OK,
                'params': params, # Include params for easier access in trials
                'portfolio_metrics': avg_portfolio_metrics, # Attach full metrics
                'ticker_metrics': avg_ticker_metrics, # Attach ticker metrics
                'objective_score': objective_score, # Attach score
                'risk_penalty': penalty # Attach penalty
            }

    def run_optimization(self) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        Execute the hyperparameter optimization process.

        Uses Hyperopt (TPE algorithm) to search the parameter space defined
        in `search_space`, guided by the `_objective_function`.

        Returns:
            Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
                - best_params: Dictionary of the best parameters found.
                - portfolio_performance_report: DataFrame summarizing portfolio metrics for best params.
                - param_history_report: DataFrame detailing each evaluation trial.
        """
        main_run_name = f"{self.run_name}_{self.strategy_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=main_run_name) as run:
            mlflow.log_param("strategy_class", self.strategy_class.__name__)
            mlflow.log_param("start_date", self.start_date.strftime("%Y-%m-%d"))
            mlflow.log_param("end_date", self.end_date.strftime("%Y-%m-%d"))
            mlflow.log_param("tickers", ",".join(self.tickers)) # Log tickers as comma-sep string
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("max_evals", self.max_evals)
            mlflow.log_param("optimization_metric", self.optimization_metric)
            mlflow.log_param("risk_thresholds", json.dumps(self.risk_thresholds)) # Log thresholds as JSON string


            logger.info(f"Starting optimization run: {main_run_name}")
            logger.info(f"Search Space: {self.search_space}")

            trials = Trials()
            try:
                best_hyperparams = fmin(
                    fn=self._objective_function,
                    space=self.search_space,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    trials=trials,
                    show_progressbar=True,
                    rstate=np.random.default_rng(42) # For reproducibility
                )
            except Exception as e:
                 logger.error(f"Hyperopt optimization failed: {e}")
                 logger.error(traceback.format_exc())
                 # Try to find the best trial manually from completed trials if any
                 if trials.best_trial and 'result' in trials.best_trial and 'loss' in trials.best_trial['result']:
                      logger.warning("Optimization interrupted. Using best trial found so far.")
                      best_hyperparams = trials.best_trial['misc']['vals']
                 else:
                      logger.error("No usable trials completed. Returning empty results.")
                      return {}, pd.DataFrame(), pd.DataFrame()


            # Convert best hyperparameters found by fmin back to original format/types
            best_params = space_eval(self.search_space, best_hyperparams)
            logger.info(f"Optimization complete. Best parameters found: {best_params}")

            # Log best parameters and final metrics from the best trial
            best_trial = trials.best_trial
            if best_trial and 'result' in best_trial and best_trial['result'].get('status') == STATUS_OK:
                 mlflow.log_params({f"best_{k}": v for k,v in best_params.items()})
                 mlflow.log_metric("best_objective_score", best_trial['result']['objective_score'])
                 mlflow.log_metric("best_risk_penalty", best_trial['result']['risk_penalty'])
                 if best_trial['result']['portfolio_metrics']:
                      mlflow.log_metrics({f"best_{k}": v for k, v in best_trial['result']['portfolio_metrics'].items()})

                 # --- Generate Final Reports ---
                 # Portfolio Performance Report (for best params)
                 best_portfolio_metrics = best_trial['result']['portfolio_metrics']
                 portfolio_report_df = pd.DataFrame([best_portfolio_metrics]) if best_portfolio_metrics else pd.DataFrame()
                 # Optional: Add parameters to the report
                 if not portfolio_report_df.empty:
                      for p_name, p_val in best_params.items():
                           try:
                               portfolio_report_df[f'param_{p_name}'] = p_val
                           except Exception as e:
                               converted_val = ', '.join(map(str, p_val))
                               portfolio_report_df[f'param_{p_name}'] = converted_val


                 # Parameter History Report (all trials)
                 param_history = []
                 for i, trial in enumerate(trials.trials):
                     result = trial.get('result', {})
                     if result.get('status') == STATUS_OK: # Process only successful trials
                         row = {'trial': i}
                         # Get parameters for this trial
                         trial_params = result.get('params', {}) # Get params attached during objective func
                         if not trial_params: # Fallback if params weren't attached
                              vals = {k: v[0] for k, v in trial['misc']['vals'].items() if v} # Handle empty lists in vals
                              trial_params = space_eval(self.search_space, vals)
                         row.update(trial_params)

                         # Add portfolio metrics
                         pf_metrics = result.get('portfolio_metrics')
                         if pf_metrics:
                             row.update({f"portfolio_{k}": v for k, v in pf_metrics.items()})

                         # Add scores
                         row['objective_score'] = result.get('objective_score', np.nan)
                         row['risk_penalty'] = result.get('risk_penalty', np.nan)
                         row['loss'] = result.get('loss', np.nan) # Hyperopt loss (-objective_score)

                         param_history.append(row)

                 param_history_df = pd.DataFrame(param_history)


                 # Save reports and log as artifacts
                 try:
                      report_suffix = f"{self.strategy_class.__name__}"
                      portfolio_csv = f"optimization_artefact_{report_suffix}_portfolio_performance.csv"
                      history_csv = f"optimization_artefact_{report_suffix}_param_history.csv"
                      params_json = f"optimization_artefact_{report_suffix}_bestparams.json"

                      portfolio_report_df.to_csv(portfolio_csv, index=False)
                      param_history_df.to_csv(history_csv, index=False)

                      with open(params_json, "w") as f:
                           json.dump(best_params, f, indent=4)

                      mlflow.log_artifact(portfolio_csv)
                      mlflow.log_artifact(history_csv)
                      mlflow.log_artifact(params_json)
                      logger.info(f"Saved reports: {portfolio_csv}, {history_csv}, {params_json}")

                 except Exception as io_error:
                      logger.error(f"Error saving optimization artifacts: {io_error}")

                 return best_params, portfolio_report_df, param_history_df

            else:
                 logger.error("Best trial data not found or trial failed. Cannot generate final reports.")
                 return best_params, pd.DataFrame(), pd.DataFrame() # Return best params found, but empty reports


    def get_evaluation_cache(self) -> Dict[str, Tuple[Optional[MetricsDict], Optional[Dict[str, MetricsDict]]]]:
        """
        Retrieve the cache of evaluated parameters and their results.

        Returns:
            Dict[str, Tuple[Optional[MetricsDict], Optional[Dict[str, MetricsDict]]]]:
                Cache mapping parameter hash to (portfolio_metrics, ticker_metrics) tuple.
        """
        return self._evaluation_cache

# Helper function for cleaner MLflow run names
def params_to_short_str(params: Dict[str, Any], max_len: int = 50) -> str:
    """Creates a short string representation of parameters for labels/names."""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            items.append(f"{k}={v:.2f}")
        else:
            items.append(f"{k}={v}")
    full_str = "_".join(items)
    return (full_str[:max_len] + '...') if len(full_str) > max_len else full_str