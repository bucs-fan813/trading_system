"""
Enhanced Know Sure Thing (KST) Momentum Strategy

This module implements the KST strategy with integrated risk management,
optimized vectorized calculations, and flexible backtesting/forecasting capabilities.

Mathematical Details:
    The KST indicator is computed as a weighted sum of several:
        KST = Σ (w_i * SMA(ROC(close, period_i), sma_period_i))
    where:
        ROC(close, period) = (close(t) - close(t - period)) / close(t - period) * 100,
        SMA(ROC, period) is the simple moving average (over a window) of the ROC,
        and w_i is the weight for each ROC component.
    A signal line is the SMA of the KST indicator over a specified period:
        signal_line = SMA(KST, signal_period)
    A bullish crossover (the KST crossing above its signal line) triggers a buy signal, while a bearish crossover
    triggers an exit (or a sell signal if short positions are permitted). In long‑only mode, bearish crossovers result in an exit.

Risk management is applied via the external RiskManager component. It adjusts entry prices for slippage and
transaction costs, sets stop-loss and take-profit levels, identifies exit events, and computes the risk–adjusted
returns, making it suitable for downstream performance optimizers.

This strategy supports both full backtesting (via start and end dates) and forecasting (using the latest available data).
It also supports processing of multiple tickers in a vectorized fashion for enhanced performance.

Args:
    ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
    start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
    end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
    initial_position (int): Starting trading position (0 for no position, 1 for long, -1 for short).
    latest_only (bool): If True, returns only the final row (per ticker when multi-ticker) for forecasting purposes.

Outputs:
    A DataFrame containing:
      - Price data: 'close', 'high', 'low'
      - Computed indicators: 'kst', 'signal_line', 'signal_strength'
      - Raw trading signal ('signal') and risk‑managed position ('position')
      - Realized trade return ('return'), cumulative return ('cumulative_return'), and risk management action ('exit_type')
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig

class KSTStrategy(BaseStrategy):
    """
    Know Sure Thing (KST) strategy using multiple ROC components with risk management.

    Hyperparameters (provided via `params`, with defaults):
        - roc_periods: List of integers for ROC lookback periods (default: [10, 15, 20, 30]).
        - sma_periods: List of integers for smoothing (SMA) periods corresponding to each ROC (default: [10, 10, 10, 15]).
        - signal_period: Integer period for computing the KST signal line (default: 9).
        - kst_weights: List of weights for each ROC component (default: [1, 2, 3, 4]).
        - long_only: Boolean flag; if True, restricts trading to long positions (default: True).
        - risk_params: Dictionary of risk management parameters (stop_loss_pct, take_profit_pct, slippage_pct, transaction_cost_pct).

    Methods:
        generate_signals: Fetches historical price data, computes the KST indicator and signal line,
                          generates raw trading signals via crossovers, applies risk management,
                          and returns a DataFrame with full performance metrics.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the KSTStrategy with database settings and hyperparameters.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific hyperparameters.
        """
        default_params = {
            'roc_periods': [10, 15, 20, 30],
            'sma_periods': [10, 10, 10, 15],
            'signal_period': 9,
            'kst_weights': [1, 2, 3, 4],
            'long_only': True,
            'risk_params': {
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'slippage_pct': 0.001,
                'transaction_cost_pct': 0.001
            }
        }
        params = params or {}
        for key, value in default_params.items():
            params.setdefault(key, value)
            
        super().__init__(db_config, params)
        self._validate_parameters()
        self.risk_manager = RiskManager(**self.params['risk_params'])
        self.long_only = self.params['long_only']

    def _validate_parameters(self):
        """
        Validate that all required parameters are provided and are positive integers where applicable.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        required = ['roc_periods', 'sma_periods', 'signal_period', 'kst_weights', 'risk_params']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
                
        if len(self.params['roc_periods']) != 4 or len(self.params['sma_periods']) != 4:
            raise ValueError("roc_periods and sma_periods must contain exactly 4 values")
            
        if any(p <= 0 for p in self.params['roc_periods'] + self.params['sma_periods'] + [self.params['signal_period']]):
            raise ValueError("All periods must be positive integers")

    def generate_signals(self, 
                         ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate KST trading signals with integrated risk management for one or multiple tickers.

        This method retrieves historical price data, computes the KST indicator as a weighted sum of
        multiple smoothed Rate-of-Change (ROC) components, calculates the signal line, and generates
        trading signals based on crossovers between the KST and its signal line. Risk management is then
        applied via the RiskManager to adjust for slippage, transaction costs, stop-loss, and take-profit.

        For multiple tickers, the computations are vectorized and applied per ticker using groupby,
        ensuring efficient calculation and benchmarking across tickers.

        Args:
            ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
            start_date (str, optional): Backtest start date (YYYY-MM-DD).
            end_date (str, optional): Backtest end date (YYYY-MM-DD).
            initial_position (int): Starting position (0: no position, 1: long, -1: short).
            latest_only (bool): If True, returns only the most recent trading signal(s) for forecasting.

        Returns:
            pd.DataFrame: DataFrame containing:
                - Price data: 'close', 'high', 'low'
                - Computed indicators: 'kst', 'signal_line', 'signal_strength'
                - Trading signals: raw ('signal') and risk-managed ('position')
                - Trade metrics: realized return ('return'), cumulative return ('cumulative_return')
                - Risk management action type ('exit_type')

        Raises:
            Exception: Propagates exceptions raised during data retrieval or processing.
        """
        try:
            # Determine required lookback period for valid rolling calculations.
            max_roc = max(self.params['roc_periods'])
            max_sma = max(self.params['sma_periods'])
            lookback = max_roc + max_sma + self.params['signal_period']

            # Retrieve historical price data.
            prices = self.get_historical_prices(
                ticker,
                lookback=lookback if latest_only else None,
                from_date=start_date,
                to_date=end_date
            )
            if not self._validate_data(prices, min_records=lookback):
                return pd.DataFrame()

            # Process single ticker and multiple tickers separately.
            if isinstance(ticker, (list, tuple)):
                # Multi-ticker: vectorized group-based operations.
                def compute_signals_for_group(df):
                    """
                    Compute KST and risk-managed signals for a single ticker DataFrame group.
                    """
                    close = df['close']
                    # Compute weighted ROC components.
                    kst_components = []
                    for roc_len, sma_len, weight in zip(self.params['roc_periods'],
                                                        self.params['sma_periods'],
                                                        self.params['kst_weights']):
                        roc = close.pct_change(roc_len) * 100
                        smoothed = roc.rolling(sma_len, min_periods=sma_len).mean()
                        kst_components.append(smoothed * weight)
                    # Sum components to obtain the KST indicator.
                    kst = pd.concat(kst_components, axis=1).sum(axis=1)
                    # Compute the signal line as a rolling SMA of the KST.
                    signal_line = kst.rolling(self.params['signal_period'], min_periods=self.params['signal_period']).mean()

                    # Construct the signals DataFrame.
                    sig = df[['close', 'high', 'low']].copy()
                    sig['kst'] = kst
                    sig['signal_line'] = signal_line
                    # Generate raw trading signals based on crossovers.
                    sig['position'] = 0
                    cross_above = (kst > signal_line) & (kst.shift(1) <= signal_line.shift(1))
                    cross_below = (kst < signal_line) & (kst.shift(1) >= signal_line.shift(1))
                    sig.loc[cross_above, 'position'] = 1
                    if self.long_only:
                        sig.loc[cross_below, 'position'] = 0
                    else:
                        sig.loc[cross_below, 'position'] = -1
                    # Create a continuous signal via forward filling.
                    sig['signal'] = sig['position'].replace(0, np.nan).ffill().fillna(0)
                    sig['signal_strength'] = kst - signal_line
                    # Apply risk management adjustments.
                    return self.risk_manager.apply(sig, initial_position)

                # Apply signal computation per ticker group.
                signals = prices.groupby(level='ticker', group_keys=False).apply(compute_signals_for_group)
                if latest_only:
                    signals = signals.groupby(level='ticker', group_keys=False).last()
            else:
                # Single ticker: compute vectorized KST components.
                close = prices['close']
                kst_components = []
                for roc_len, sma_len, weight in zip(self.params['roc_periods'],
                                                    self.params['sma_periods'],
                                                    self.params['kst_weights']):
                    roc = close.pct_change(roc_len) * 100
                    smoothed = roc.rolling(sma_len, min_periods=sma_len).mean()
                    kst_components.append(smoothed * weight)
                kst = pd.concat(kst_components, axis=1).sum(axis=1)
                signal_line = kst.rolling(self.params['signal_period'], min_periods=self.params['signal_period']).mean()
                signals = pd.DataFrame({
                    'close': prices['close'],
                    'high': prices['high'],
                    'low': prices['low'],
                    'kst': kst,
                    'signal_line': signal_line,
                }, index=prices.index)
                signals['position'] = 0
                cross_above = (kst > signal_line) & (kst.shift(1) <= signal_line.shift(1))
                cross_below = (kst < signal_line) & (kst.shift(1) >= signal_line.shift(1))
                signals.loc[cross_above, 'position'] = 1
                if self.long_only:
                    signals.loc[cross_below, 'position'] = 0
                else:
                    signals.loc[cross_below, 'position'] = -1
                signals['signal'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
                signals['signal_strength'] = kst - signal_line
                signals = self.risk_manager.apply(signals, initial_position)
                if latest_only:
                    signals = signals.iloc[[-1]]

            return signals

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            raise