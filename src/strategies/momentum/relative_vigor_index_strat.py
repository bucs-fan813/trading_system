# trading_system/src/strategies/relative_vigor_index_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class RVIStrategy(BaseStrategy):
    """
    Enhanced Relative Vigor Index (RVI) Strategy with Risk Management

    This strategy employs the Relative Vigor Index (RVI) to generate trading signals.
    Mathematically, the RVI is computed using a weighted average of the differences between
    the close and open prices over four consecutive periods as follows:

      Numerator(t)   = (C(t)-O(t)) + 2*(C(t-1)-O(t-1)) + 2*(C(t-2)-O(t-2)) + (C(t-3)-O(t-3))
      Denom(t)       = (H(t)-L(t)) + 2*(H(t-1)-L(t-1)) + 2*(H(t-2)-L(t-2)) + (H(t-3)-L(t-3))
      RVI_raw(t)     = Numerator(t) / Denom(t)

    The raw RVI is then smoothed by taking a simple moving average over the period defined by
    the "lookback" parameter:
    
      RVI(t) = MA(RVI_raw(t), lookback)

    The signal line is derived as a weighted average of the smoothed RVI:

      Signal_line(t) = (RVI(t) + 2*RVI(t-1) + 2*RVI(t-2) + RVI(t-3)) / 6

    Signal Generation:
      - A buy signal (+1) is generated if the RVI crosses above the signal line.
      - A sell signal (-1) is generated if the RVI crosses below the signal line.
      - Signal strength is computed as the z-score normalization (over a rolling window) of the
        absolute difference between RVI and the signal line.

    Risk Management:
      The RiskManager is then used to integrate stop-loss and take-profit levels, as well as account
      for slippage and transaction cost. For long positions:
        - Stop loss triggers if the low price falls to or below Entry * (1 - stop_loss_pct).
        - Take profit triggers if the high price reaches or exceeds Entry * (1 + take_profit_pct).
      For short positions the rules are reversed. The realized return is computed accordingly and
      the cumulative return is updated using the sequence of trade multipliers.

    Hyperparameters (with defaults):
      - lookback (int): Smoothing period for RVI calculation.
      - signal_strength_window (int): Window length to compute normalized signal strength.
      - stop_loss_pct (float): Percentage threshold for stop loss.
      - take_profit_pct (float): Percentage threshold for take profit.
      - slippage_pct (float): Estimated slippage percentage.
      - transaction_cost_pct (float): Estimated transaction cost percentage.
      - long_only (bool): If True, only long positions are allowed.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the RVIStrategy with risk parameters and hyperparameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings for establishing DB connection.
            params (dict, optional): Strategy hyperparameters. Defaults are set if not provided.
        """
        super().__init__(db_config, params)
        self.params.setdefault('lookback', 10)
        self.params.setdefault('signal_strength_window', 20)
        self.params.setdefault('stop_loss_pct', 0.05)
        self.params.setdefault('take_profit_pct', 0.10)
        self.params.setdefault('slippage_pct', 0.001)
        self.params.setdefault('transaction_cost_pct', 0.001)
        self.params.setdefault('long_only', True)

    def generate_signals(self, 
                         ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate RVI-based trading signals with integrated risk management for a given ticker.

        Retrieves the historical price data, calculates the RVI and its signal line,
        constructs trading signals and signal strength, and then applies risk management rules to
        compute trade returns and cumulative performance metrics.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for backtesting in 'YYYY-MM-DD' format.
            initial_position (int, optional): Initial trading position (0 for flat, 1 for long, -1 for short).
            latest_only (bool, optional): If True, returns only the latest available signal.

        Returns:
            pd.DataFrame: A DataFrame containing trading signals, managed positions, trade returns,
                          and cumulative performance. When latest_only is True, only the latest signal row is returned.
        """
        try:
            lookback = self.params['lookback']
            # Calculate the number of days needed to account for the rolling and shift operations.
            required_days = lookback + 6  

            # Retrieve historical data using explicit date range or a lookback window for the latest signal.
            if latest_only:
                prices_df = self.get_historical_prices(ticker, lookback=required_days)
            else:
                prices_df = self.get_historical_prices(
                    ticker,
                    from_date=start_date,
                    to_date=end_date,
                    data_source='yfinance'
                )

            if not self._validate_data(prices_df, min_records=required_days):
                raise DataRetrievalError(f"Insufficient data for {ticker}")

            # Calculate RVI and signal line components.
            rvi, signal_line = self._calculate_rvi_components(prices_df, lookback)
            
            # Create a DataFrame with computed RVI values and generated trading signals.
            signals_df = self._generate_signals_df(prices_df, rvi, signal_line)
            
            # Apply risk management (stop loss, take profit, slippage, transaction costs) to generate trade metrics.
            risk_manager = RiskManager(
                stop_loss_pct=self.params['stop_loss_pct'],
                take_profit_pct=self.params['take_profit_pct'],
                slippage_pct=self.params['slippage_pct'],
                transaction_cost_pct=self.params['transaction_cost_pct']
            )
            result_df = risk_manager.apply(signals_df, initial_position)

            return result_df.tail(1) if latest_only else result_df

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _calculate_rvi_components(self, prices_df: pd.DataFrame, lookback: int) -> tuple:
        """
        Calculate the Relative Vigor Index (RVI) and its signal line using vectorized operations.

        The raw RVI is computed as the weighted sum of (close - open) over 4 periods divided by the weighted sum
        of (high - low) over the same periods. This raw RVI is then smoothed using a simple moving average (SMA)
        over the specified lookback period. The signal line is derived by applying a weighted average to the
        smoothed RVI values.

        Args:
            prices_df (pd.DataFrame): Historical price data containing the 'open', 'high', 'low', and 'close' columns.
            lookback (int): Period for smoothing the raw RVI.

        Returns:
            tuple: A tuple containing two pd.Series:
                - rvi: The smoothed Relative Vigor Index.
                - signal_line: The signal line computed from the smoothed RVI.
        """
        o = prices_df['open']
        h = prices_df['high']
        l = prices_df['low']
        c = prices_df['close']

        # Compute weighted sums for numerator and denominator for RVI_raw.
        numerator = (c - o) + 2 * (c.shift(1) - o.shift(1)) + 2 * (c.shift(2) - o.shift(2)) + (c.shift(3) - o.shift(3))
        denominator = (h - l) + 2 * (h.shift(1) - l.shift(1)) + 2 * (h.shift(2) - l.shift(2)) + (h.shift(3) - l.shift(3))

        # Avoid division by zero by replacing zeros with NaN and forward filling valid values.
        denominator = denominator.replace(0, np.nan).ffill()

        rvi_raw = numerator / denominator
        # Apply a simple moving average for smoothing.
        rvi = rvi_raw.rolling(lookback).mean()
        
        # Compute the signal line as a weighted average of the smoothed RVI values.
        signal_line = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6

        return rvi, signal_line

    def _generate_signals_df(self, prices_df: pd.DataFrame, 
                               rvi: pd.Series, signal_line: pd.Series) -> pd.DataFrame:
        """
        Construct a DataFrame with trading signals and signal strength based on the RVI crossover.

        Signals are generated using vectorized detection of crossovers between the RVI and its signal line.
        A buy signal (+1) occurs when the RVI crosses above the signal line; a sell signal (-1) occurs
        when the RVI crosses below the signal line. Additionally, signal strength is computed as the normalized
        (z-score) absolute difference between the RVI and the signal line.

        Args:
            prices_df (pd.DataFrame): Historical price data containing 'close', 'high', and 'low' columns.
            rvi (pd.Series): The smoothed Relative Vigor Index.
            signal_line (pd.Series): The signal line computed from the smoothed RVI.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - 'close': Closing price.
                - 'high': High price.
                - 'low': Low price.
                - 'rvi': The computed Relative Vigor Index.
                - 'signal_line': The signal line for the RVI.
                - 'signal': Trading signal (+1 for buy, -1 for sell, 0 otherwise).
                - 'signal_strength': Normalized measure indicating the strength of the signal.
        """
        df = pd.DataFrame({
            'close': prices_df['close'],
            'high': prices_df['high'],
            'low': prices_df['low'],
            'rvi': rvi,
            'signal_line': signal_line
        }).dropna()

        # Compute shifted values for the previous period's RVI and signal line for crossover detection.
        shifted_rvi = df['rvi'].shift(1)
        shifted_signal = df['signal_line'].shift(1)
        
        # Generate buy signal (+1) if RVI crosses above the signal line, sell signal (-1) if it crosses below.
        df['signal'] = np.select(
            [
                (shifted_rvi < shifted_signal) & (df['rvi'] > df['signal_line']),
                (shifted_rvi > shifted_signal) & (df['rvi'] < df['signal_line'])
            ],
            [1, -1],
            default=0
        )

        if self.long_only:
            # Replace sell signals (-1) with 0 (exit) if long_only is True.
            df.loc[df['signal'] == -1, 'signal'] = 0

        # Calculate signal strength as the z-score of the absolute difference between RVI and the signal line.
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
        Validate that the provided DataFrame has enough records and contains all required price columns.

        Args:
            df (pd.DataFrame): Input price data.
            min_records (int): Minimum required number of records for analysis.

        Returns:
            bool: True if the data is sufficient and includes all necessary columns; False otherwise.
        """
        if len(df) < min_records:
            self.logger.warning(f"Data insufficient: {len(df)} < {min_records}")
            return False
        if df[['open', 'high', 'low', 'close']].isnull().values.any():
            self.logger.warning("Missing price values detected")
            return False
        return True