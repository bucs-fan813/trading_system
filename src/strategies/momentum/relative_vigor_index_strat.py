# trading_system/src/strategies/relative_vigor_index_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class RVIStrategy(BaseStrategy):
    """
    Enhanced Relative Vigor Index (RVI) Strategy with Integrated Risk Management.

    Mathematical Formulation:
        Numerator(t)   = (C(t)-O(t)) + 2*(C(t-1)-O(t-1)) + 2*(C(t-2)-O(t-2)) + (C(t-3)-O(t-3))
        Denom(t)       = (H(t)-L(t)) + 2*(H(t-1)-L(t-1)) + 2*(H(t-2)-L(t-2)) + (H(t-3)-L(t-3))
        RVI_raw(t)     = Numerator(t) / Denom(t)
        
        RVI(t)         = MA(RVI_raw(t), lookback)
        
        Signal_line(t) = (RVI(t) + 2*RVI(t-1) + 2*RVI(t-2) + RVI(t-3)) / 6

    Trading Signals:
        - Buy Signal (+1)  : Generated when the RVI crosses above the signal line.
        - Sell Signal (-1) : Generated when the RVI crosses below the signal line.
          In long-only mode, sell signals are replaced with 0 (exit signals).

    Signal Strength:
        Computed as the z-score normalization (over a rolling window) of the absolute difference 
        between the RVI and the signal line.

    Risk Management:
        Integrates stop-loss and take-profit rules via the RiskManager (which also considers slippage 
        and transaction costs). This component adjusts entries, detects exit events, computes realized 
        returns, and maintains cumulative returns.

    Backtest / Forecasting:
        Supports backtesting with specified start and end dates or forecasting using a minimal lookback 
        (returning only the latest signal). In the case of multiple tickers, the processing is vectorized 
        across tickers.

    Attributes:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict): Hyperparameters for the strategy, including:
            - "lookback": Smoothing period for RVI (default: 10)
            - "signal_strength_window": Window for z-score normalization of signal strength (default: 20)
            - "stop_loss_pct": Stop loss percentage (default: 0.05)
            - "take_profit_pct": Take profit percentage (default: 0.10)
            - "trailing_stop_pct": Trailing stop percentage (default: 0.0)
            - "slippage_pct": Estimated slippage percentage (default: 0.001)
            - "transaction_cost_pct": Estimated transaction cost percentage (default: 0.001)
            - "long_only": If True, only long positions are allowed (default: True)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the RVIStrategy with a database configuration and set default hyperparameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration object for establishing the DB connection.
            params (dict, optional): Strategy hyperparameters. If not provided, defaults will be used.
        """
        super().__init__(db_config, params)
        self.params.setdefault('lookback', 10)
        self.params.setdefault('signal_strength_window', 20)
        self.params.setdefault('stop_loss_pct', 0.05)
        self.params.setdefault('take_profit_pct', 0.10)
        self.params.setdefault('trailing_stop_pct', 0.0)
        self.params.setdefault('slippage_pct', 0.001)
        self.params.setdefault('transaction_cost_pct', 0.001)
        self.params.setdefault('long_only', True)

    def generate_signals(self, 
                         ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate RVI-based trading signals with risk management adjustments.

        Retrieves historical price data (including open, high, low, and close), computes the RVI and its signal line,
        and generates trading signals based on crossovers. Signal strength is assigned as a normalized measure, and
        the RiskManager applies stop loss, take profit, slippage, and transaction cost adjustments to yield trade 
        returns and cumulative performance metrics.

        When 'ticker' is a list, processing is performed per ticker using vectorized groupby operations.

        Args:
            ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
            start_date (str, optional): Backtesting start date (YYYY-MM-DD format).
            end_date (str, optional): Backtesting end date (YYYY-MM-DD format).
            initial_position (int): Initial trading position (0 for no position, 1 for long, -1 for short).
            latest_only (bool): If True, only returns the latest available signal (per ticker in multi-ticker scenarios).

        Returns:
            pd.DataFrame: DataFrame with the following columns:
                - 'close', 'high', 'low': Price references.
                - 'rvi': Smoothed Relative Vigor Index.
                - 'signal_line': Computed signal line.
                - 'signal': Trading signal (1 for buy, -1 for sell, 0 otherwise).
                - 'signal_strength': Normalized strength of the signal.
                - (Additional columns from RiskManager: realized return, cumulative return, and exit type.)
            In the multi-ticker case, the index is a MultiIndex with (ticker, date).
        """
        lookback = self.params['lookback']
        required_days = lookback + 6  # To account for the necessary rolling and shifted periods

        if isinstance(ticker, list):
            # Retrieve historical prices for multiple tickers.
            prices_df = self.get_historical_prices(
                ticker, from_date=start_date, to_date=end_date, data_source='yfinance'
            )
            # prices_df is multi-indexed (ticker, date); group by ticker for individual processing.
            grouped = prices_df.reset_index(level=0).groupby("ticker", group_keys=False)

            def process_group(group: pd.DataFrame) -> pd.DataFrame:
                if not self._validate_data(group, min_records=required_days):
                    raise DataRetrievalError(f"Insufficient data for {group.name}")
                rvi, signal_line = self._calculate_rvi_components(group, lookback)
                sig_df = self._generate_signals_df(group, rvi, signal_line)
                risk_manager = RiskManager(
                    stop_loss_pct=self.params['stop_loss_pct'],
                    take_profit_pct=self.params['take_profit_pct'],
                    trailing_stop_pct=self.params['trailing_stop_pct'],
                    slippage_pct=self.params['slippage_pct'],
                    transaction_cost_pct=self.params['transaction_cost_pct']
                )
                return risk_manager.apply(sig_df, initial_position)

            try:
                results = grouped.apply(process_group)
            except Exception as e:
                self.logger.error(f"Error processing tickers {ticker}: {str(e)}", exc_info=True)
                return pd.DataFrame()

            if latest_only:
                # Take the latest row per ticker
                results = results.groupby(level=0).tail(1)
            return results

        else:
            # Single ticker data processing.
            if latest_only:
                prices_df = self.get_historical_prices(ticker, lookback=required_days)
            else:
                prices_df = self.get_historical_prices(
                    ticker, from_date=start_date, to_date=end_date, data_source='yfinance'
                )
            if not self._validate_data(prices_df, min_records=required_days):
                raise DataRetrievalError(f"Insufficient data for {ticker}")
            rvi, signal_line = self._calculate_rvi_components(prices_df, lookback)
            signals_df = self._generate_signals_df(prices_df, rvi, signal_line)
            risk_manager = RiskManager(
                stop_loss_pct=self.params['stop_loss_pct'],
                take_profit_pct=self.params['take_profit_pct'],
                slippage_pct=self.params['slippage_pct'],
                transaction_cost_pct=self.params['transaction_cost_pct']
            )
            result_df = risk_manager.apply(signals_df, initial_position)
            return result_df.tail(1) if latest_only else result_df

    def _calculate_rvi_components(self, prices_df: pd.DataFrame, lookback: int) -> tuple:
        """
        Calculate the Relative Vigor Index (RVI) and its signal line.

        The raw RVI is computed as a weighted sum of (close - open) differences divided by a weighted sum of 
        (high - low) differences over four consecutive periods. This raw value is then smoothed by an SMA over 
        the lookback period. The signal line is generated as a weighted average of the smoothed RVI.

        Args:
            prices_df (pd.DataFrame): Historical price data with 'open', 'high', 'low', and 'close' columns.
            lookback (int): Period for computing the SMA of the raw RVI.

        Returns:
            tuple: A pair (rvi, signal_line) where each is a pd.Series.
        """
        o = prices_df['open']
        h = prices_df['high']
        l = prices_df['low']
        c = prices_df['close']

        # Compute weighted differences.
        numerator = (c - o) + 2 * (c.shift(1) - o.shift(1)) + 2 * (c.shift(2) - o.shift(2)) + (c.shift(3) - o.shift(3))
        denominator = (h - l) + 2 * (h.shift(1) - l.shift(1)) + 2 * (h.shift(2) - l.shift(2)) + (h.shift(3) - l.shift(3))
        denominator = denominator.replace(0, np.nan).ffill()  # Prevent division by zero
        
        rvi_raw = numerator / denominator
        rvi = rvi_raw.rolling(lookback).mean()  # Smooth using simple moving average

        # Calculate the signal line as a weighted average of the smoothed RVI.
        signal_line = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6

        return rvi, signal_line

    def _generate_signals_df(self, prices_df: pd.DataFrame, 
                               rvi: pd.Series, signal_line: pd.Series) -> pd.DataFrame:
        """
        Generate a DataFrame of trading signals and their normalized strength based on RVI crossovers.

        Trading signals are determined by comparing the current RVI vs. its signal line relative to the previous period:
            - A buy signal (+1) is issued when the previous RVI is below the previous signal line and the current RVI 
              crosses above the current signal line.
            - A sell signal (-1) is issued when the previous RVI is above the previous signal line and the current RVI 
              crosses below the current signal line.
            - In long-only mode (per 'long_only' parameter), sell signals are replaced with 0.
        
        Signal strength is computed as the z-score (rolling window normalization) of the absolute difference between 
        RVI and its signal line, weighted by the magnitude of the signal.

        Args:
            prices_df (pd.DataFrame): Price data with 'close', 'high', and 'low' columns.
            rvi (pd.Series): Smoothed Relative Vigor Index.
            signal_line (pd.Series): Derived signal line from the smoothed RVI.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - 'close', 'high', 'low'
                - 'rvi': Smoothed RVI.
                - 'signal_line': Signal line.
                - 'signal': Trading signal (+1, -1, or 0).
                - 'signal_strength': Normalized measure of the signal's strength.
        """
        df = pd.DataFrame({
            'close': prices_df['close'],
            'high': prices_df['high'],
            'low': prices_df['low'],
            'rvi': rvi,
            'signal_line': signal_line
        }).dropna()

        # Use previous values for crossover detection.
        shifted_rvi = df['rvi'].shift(1)
        shifted_signal = df['signal_line'].shift(1)

        # Generate trading signals: buy when RVI crosses above and sell when it crosses below.
        df['signal'] = np.select(
            [
                (shifted_rvi < shifted_signal) & (df['rvi'] > df['signal_line']),
                (shifted_rvi > shifted_signal) & (df['rvi'] < df['signal_line'])
            ],
            [1, -1],
            default=0
        )
        # Enforce long-only mode if specified.
        if self.params['long_only']:
            df.loc[df['signal'] == -1, 'signal'] = 0

        # Compute signal strength as the z-score of the absolute difference between RVI and the signal line.
        window = self.params['signal_strength_window']
        strength = np.abs(df['rvi'] - df['signal_line']) * df['signal'].abs()
        rolling_mean = strength.rolling(window).mean()
        rolling_std = strength.rolling(window).std()
        df['signal_strength'] = (strength - rolling_mean) / rolling_std
        df['signal_strength'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['signal_strength'].fillna(0, inplace=True)

        return df[['close', 'high', 'low', 'rvi', 'signal_line', 'signal', 'signal_strength']]

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the DataFrame has a sufficient number of records and that price data is complete.

        Verifies that the number of rows is at least the minimum required and that there are no missing values
        in the 'open', 'high', 'low', and 'close' columns.

        Args:
            df (pd.DataFrame): DataFrame of historical prices.
            min_records (int): Minimum number of observations required.

        Returns:
            bool: True if the data is valid; False otherwise.
        """
        if len(df) < min_records:
            self.logger.warning(f"Data insufficient: {len(df)} records, required {min_records}.")
            return False
        if df[['open', 'high', 'low', 'close']].isnull().values.any():
            self.logger.warning("Missing required price values detected.")
            return False
        return True