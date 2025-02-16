# trading_system/src/strategies/rsi_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig

class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) trading strategy.
    
    Generates signals when RSI crosses user-defined thresholds with a configurable lookback period.
    Includes signal strength metrics for position sizing optimization.

    Parameters:
        rsi_period (int): Lookback period for RSI calculation (default: 14)
        overbought (float): Upper threshold for sell signals (default: 70)
        oversold (float): Lower threshold for buy signals (default: 30)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generates a DataFrame with signals, signal strengths, and RSI values.

        Args:
            ticker: Asset ticker symbol (e.g., 'AAPL')

        Returns:
            pd.DataFrame: Columns ['signal', 'signal_strength', 'rsi'], indexed by date.
        """
        # Parameter extraction with validation
        rsi_period = self.params.get('rsi_period', 14)
        overbought = self.params.get('overbought', 70.0)
        oversold = self.params.get('oversold', 30.0)

        if rsi_period < 1:
            raise ValueError("RSI period must be ≥1")
        if overbought <= oversold:
            raise ValueError("overbought must exceed oversold")
        if not (0 < oversold < overbought < 100):
            raise ValueError("Thresholds must be between 0-100")

        # Retrieve historical price data with an added buffer period for the initial RSI calculation
        df = self.get_historical_prices(
            ticker, 
            lookback=rsi_period + 50,
            data_source='yfinance'
        )
        if not self._validate_data(df, min_records=rsi_period + 1):
            self.logger.error(f"Insufficient data for {ticker}")
            return pd.DataFrame()

        # Ensure the data is in chronological (oldest first) order.
        df = df.sort_index(ascending=True)

        # RSI Calculation (using Wilder's smoothing method)
        delta = df['close'].diff().dropna()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Calculate exponential weighted moving averages to approximate Wilder's smoothing.
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()

        # Avoid division by zero by replacing zero losses with a small value.
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Signal generation:
        # - Buy when RSI crosses upward over the oversold threshold.
        # - Sell when RSI crosses downward below the overbought threshold.
        buy_condition = (rsi.shift(1) <= oversold) & (rsi > oversold)
        sell_condition = (rsi.shift(1) >= overbought) & (rsi < overbought)

        signals = pd.Series(0, index=rsi.index)
        signals[buy_condition] = 1
        signals[sell_condition] = -1

        # Signal strength calculation:
        # Corrected the formulas so that the measure is positive.
        with np.errstate(invalid='ignore'):
            buy_strength = (rsi - oversold) / oversold
            sell_strength = (overbought - rsi) / (100 - overbought)
            signal_strength = np.where(
                signals == 1,
                buy_strength,
                np.where(signals == -1, sell_strength, 0)
            )

        # Construct the final DataFrame with the latest dates first.
        signals_df = pd.DataFrame({
            'signal': signals,
            'signal_strength': signal_strength,
            'rsi': rsi
        }).dropna().sort_index(ascending=False)

        return signals_df