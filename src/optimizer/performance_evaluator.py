# trading_system/src/optimizer/performance_evaluator.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Configure module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PerformanceEvaluator:
    @staticmethod
    def compute_metrics(df: pd.DataFrame, annualization_factor: int = 252) -> Dict[str, float]:
        """
        Recalculate returns and compute key performance metrics based on a signals DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing trading signals and pricing.
            annualization_factor (int): The factor used to annualize performance (default 252).
            
        Returns:
            Dict[str, float]: A dictionary containing the computed performance metrics.
        """
        df = df.copy()
        # Calculate daily returns and compute strategy returns based on prior-day positions.
        df['daily_return'] = df['close'].pct_change().fillna(0)
        df['strategy_return'] = df['daily_return'] * df['position'].shift(1).fillna(0)
        
        # Cumulative and annualized returns.
        cum_return = np.prod(1 + df['strategy_return']) - 1
        n = len(df)
        ann_return = ((1 + cum_return) ** (annualization_factor / n)) - 1 if n > 0 else 0

        # Calculate the Sharpe Ratio (assuming 0 risk-free rate).
        mean_ret = df['strategy_return'].mean()
        std_ret = df['strategy_return'].std()
        sharpe = (mean_ret / std_ret * np.sqrt(annualization_factor)) if std_ret != 0 else 0

        # Calculate the Sortino Ratio.
        negative_returns = df.loc[df['strategy_return'] < 0, 'strategy_return']
        downside_std = negative_returns.std() if not negative_returns.empty else 1e-6
        sortino = (mean_ret / downside_std * np.sqrt(annualization_factor)) if downside_std != 0 else 0

        # Maximum Drawdown.
        cum_returns_series = (1 + df['strategy_return']).cumprod()
        rolling_max = cum_returns_series.cummax()
        drawdown = (cum_returns_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Vectorized Drawdown Duration.
        in_drawdown = cum_returns_series < rolling_max
        down_arr = in_drawdown.to_numpy().astype(int)
        padded = np.concatenate(([0], down_arr, [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        drawdown_duration = int((ends - starts).max() if starts.size else 0)

        # Calmar Ratio.
        calmar = (ann_return / abs(max_drawdown)) if max_drawdown != 0 else 0

        # Omega Ratio (using a threshold of 0).
        gain = df.loc[df['strategy_return'] > 0, 'strategy_return'].sum()
        loss = abs(df.loc[df['strategy_return'] < 0, 'strategy_return'].sum())
        omega = (gain / loss) if loss != 0 else 10

        # Win Rate.
        win_rate = (df['strategy_return'] > 0).mean()

        # Recovery Factor.
        recovery_factor = (cum_return / abs(max_drawdown)) if max_drawdown != 0 else 0

        # Annualized Volatility.
        ann_vol = std_ret * np.sqrt(annualization_factor)

        # Ulcer Index.
        ulcer_index = np.sqrt((drawdown ** 2).mean())

        metrics = {
            'cumulative_return': cum_return,
            'annualized_return': ann_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            'win_rate': win_rate,
            'recovery_factor': recovery_factor,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'annualized_volatility': ann_vol,
            'ulcer_index': ulcer_index
        }
        return metrics

    @staticmethod
    def compute_multi_ticker_metrics(
        signals_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        annualization_factor: int = 252
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Compute performance metrics for multiple tickers and their weighted average.
        
        Args:
            signals_dict (Dict[str, pd.DataFrame]): Dictionary mapping ticker symbols to signal DataFrames.
            weights (Optional[Dict[str, float]]): Dictionary mapping ticker symbols to their weights.
                If None, equal weights are used.
            annualization_factor (int): Factor used to annualize performance.
            
        Returns:
            Tuple containing:
                Dict[str, Dict[str, float]]: Dictionary mapping ticker symbols to their performance metrics.
                Dict[str, float]: Dictionary containing the weighted average metrics.
        """
        if not signals_dict:
            return {}, {}
            
        # Compute individual ticker metrics
        ticker_metrics = {}
        for ticker, df in signals_dict.items():
            if df.empty:
                continue
            ticker_metrics[ticker] = PerformanceEvaluator.compute_metrics(df, annualization_factor)
            
        if not ticker_metrics:
            return {}, {}
            
        # Handle weights
        if weights is None:
            # Equal weights
            ticker_weights = {ticker: 1.0 / len(ticker_metrics) for ticker in ticker_metrics}
        else:
            # Filter weights to include only tickers present in ticker_metrics
            ticker_weights = {ticker: weights.get(ticker, 0) for ticker in ticker_metrics}
            # Normalize weights to sum to 1
            total_weight = sum(ticker_weights.values())
            if total_weight != 0:
                ticker_weights = {ticker: weight / total_weight for ticker, weight in ticker_weights.items()}
            else:
                # Fallback to equal weights if sum is 0
                ticker_weights = {ticker: 1.0 / len(ticker_metrics) for ticker in ticker_metrics}
                
        # Compute weighted average metrics
        weighted_metrics = {key: 0.0 for key in next(iter(ticker_metrics.values())).keys()}
        for ticker, metrics in ticker_metrics.items():
            weight = ticker_weights.get(ticker, 0)
            for key, value in metrics.items():
                weighted_metrics[key] += value * weight
                
        # Add harmonic mean to individual ticker metrics
        for ticker, metrics in ticker_metrics.items():
            metrics['harmonic_mean'] = PerformanceEvaluator.harmonic_mean(metrics)
        
        # Add harmonic mean to weighted metrics
        weighted_metrics['harmonic_mean'] = PerformanceEvaluator.harmonic_mean(weighted_metrics)
            
        return ticker_metrics, weighted_metrics

    @staticmethod
    def harmonic_mean(metrics: Dict[str, float]) -> float:
        """
        Compute the harmonic mean of selected performance metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of performance metrics.
        
        Returns:
            float: The harmonic mean of the specified performance metrics.
        """
        required_keys = [
            'sharpe_ratio', 
            'sortino_ratio', 
            'calmar_ratio', 
            'omega_ratio', 
            'win_rate', 
            'recovery_factor'
        ]
        values = []
        for key in required_keys:
            v = metrics.get(key)
            if v is None or v <= 0:
                return 0
            values.append(v)
        return len(values) / np.sum([1.0 / v for v in values])