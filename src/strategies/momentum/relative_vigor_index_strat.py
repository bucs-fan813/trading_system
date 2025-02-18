# trading_system/src/strategies/rvi_strat.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class RVIStrategy(BaseStrategy):
    """
    Relative Vigor Index (RVI) strategy implementation

    Hyperparameters:
        lookback (int): Smoothing period for RVI calculation (default: 10)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params.setdefault('lookback', 10)
        # Do not override the logger from the base class
        # self.logger = logging.getLogger(__name__)

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate RVI-based trading signals

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame containing:
            - close: Closing prices
            - rvi: RVI values
            - signal_line: Signal line values
            - signal: Trading signals (1=Buy, -1=Sell, 0=Neutral)
            - signal_strength: Normalized strength of the signal
        """
        try:
            # Calculate required data points:
            # The corrected lags and subsequent rolling windows (for rvi smoothing and the 20-day signal_strength normalization)
            # require more data points.
            lookback = self.params['lookback']
            required_days = lookback + 24  # adjusted for calculation lags and rolling windows

            # Retrieve historical data
            prices_df = self.get_historical_prices(ticker, lookback=required_days)
            
            if not self._validate_data(prices_df, min_records=required_days):
                raise DataRetrievalError(f"Insufficient data for {ticker}")

            # Calculate RVI components
            rvi, signal_line = self._calculate_rvi_components(prices_df, lookback)

            # Generate signals dataframe
            signals_df = self._generate_signals_df(prices_df, rvi, signal_line)
            
            return signals_df

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_rvi_components(self, prices_df: pd.DataFrame, lookback: int) -> tuple:
        """Core RVI calculation logic with corrected lags"""
        open_ = prices_df['open']
        high = prices_df['high']
        low = prices_df['low']
        close = prices_df['close']

        # Numerator calculation with corrected lags: current, t-1, t-2, t-3
        num_terms = [
            close - open_,
            2 * (close.shift(1) - open_.shift(1)),
            2 * (close.shift(2) - open_.shift(2)),
            close.shift(3) - open_.shift(3)
        ]
        numerator = pd.concat(num_terms, axis=1).sum(axis=1)

        # Denominator calculation with corrected lags: current, t-1, t-2, t-3
        den_terms = [
            high - low,
            2 * (high.shift(1) - low.shift(1)),
            2 * (high.shift(2) - low.shift(2)),
            high.shift(3) - low.shift(3)
        ]
        denominator = pd.concat(den_terms, axis=1).sum(axis=1)

        # Handle division by zero cases
        denominator = denominator.replace(0, np.nan).ffill()

        # RVI raw calculation using 4-day sums (this is equivalent to a 4-day SMA)
        rvi_raw = numerator.rolling(4).sum() / denominator.rolling(4).sum()
        # Further smooth the RVI over the specified lookback period
        rvi = rvi_raw.rolling(lookback).mean()

        # Signal line calculation using weighted moving average with weights 1,2,2,1
        signal_terms = [
            rvi,
            2 * rvi.shift(1),
            2 * rvi.shift(2),
            rvi.shift(3)
        ]
        signal_line = pd.concat(signal_terms, axis=1).sum(axis=1) / 6

        return rvi, signal_line

    def _generate_signals_df(self, prices_df: pd.DataFrame, rvi: pd.Series, 
                               signal_line: pd.Series) -> pd.DataFrame:
        """Generate trading signals dataframe"""
        df = pd.DataFrame({
            'close': prices_df['close'],
            'rvi': rvi,
            'signal_line': signal_line
        }).dropna()

        # Generate crossover signals: detect when RVI crosses above or below the signal line
        df['cross_above'] = (df['rvi'].shift(1) < df['signal_line'].shift(1)) & (df['rvi'] > df['signal_line'])
        df['cross_below'] = (df['rvi'].shift(1) > df['signal_line'].shift(1)) & (df['rvi'] < df['signal_line'])

        # Create signal column: 1 for bullish cross, -1 for bearish cross
        df['signal'] = 0
        df.loc[df['cross_above'], 'signal'] = 1
        df.loc[df['cross_below'], 'signal'] = -1

        # Calculate signal strength based on the difference between RVI and signal line
        df['strength_raw'] = np.where(
            df['signal'] == 1, 
            (df['rvi'] - df['signal_line']),
            np.where(
                df['signal'] == -1,
                (df['signal_line'] - df['rvi']),
                0
            )
        )

        # Normalize signal strength using rolling Z-score (20-day window)
        roll_mean = df['strength_raw'].rolling(20).mean()
        roll_std = df['strength_raw'].rolling(20).std().replace(0, 1e-9)
        df['signal_strength'] = (df['strength_raw'] - roll_mean) / roll_std

        return df[['close', 'rvi', 'signal_line', 'signal', 'signal_strength']]

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """Enhanced data validation"""
        if len(df) < min_records:
            self.logger.warning(f"Only {len(df)} records found, minimum {min_records} required")
            return False
        if df.isnull().values.any():
            self.logger.warning("Missing values detected in price data")
            return False
        return True