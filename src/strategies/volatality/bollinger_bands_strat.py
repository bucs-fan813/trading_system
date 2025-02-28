# trading_system/src/strategies/bollinger_bands_strat.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy implementation.

    Hyperparameters:
        window (int): SMA and standard deviation window (default: 20)
        std_dev (float): Number of standard deviations for bands (default: 2.0)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.window = self.params.get('window', 20)
        self.std_dev = self.params.get('std_dev', 2.0)
        # Inherited logger from BaseStrategy will be used.

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate Bollinger Bands trading signals.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with signals and relevant indicators.
        """
        # Retrieve historical price data.
        # Use a lookback period that is at least twice the SMA window or one trading year (252 days)
        lookback = max(self.window * 2, 252)
        prices = self.get_historical_prices(ticker, lookback=lookback)
        
        if not self._validate_data(prices, min_records=self.window + 1):
            self.logger.error(f"Insufficient data for {ticker}")
            return pd.DataFrame()

        close = prices['close']

        # Calculate the rolling simple moving average (SMA) and standard deviation.
        sma = close.rolling(window=self.window, min_periods=self.window).mean()
        std = close.rolling(window=self.window, min_periods=self.window).std()
        upper_bb = sma + (std * self.std_dev)
        lower_bb = sma - (std * self.std_dev)

        # Calculate percent_b indicator in a vectorized manner.
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_b = (close - lower_bb) / (upper_bb - lower_bb)
        percent_b = percent_b.replace([np.inf, -np.inf], np.nan)

        # --- Vectorized Signal Generation ---
        # Use shift(1) to compare the previous day’s values to avoid look-ahead bias.
        buy_condition = (close.shift(1) > lower_bb.shift(1)) & (close < lower_bb)
        sell_condition = (close.shift(1) < upper_bb.shift(1)) & (close > upper_bb)

        # Create a raw signal series: +1 for a buy cross, and -1 for a sell cross.
        raw_signal = pd.Series(0, index=close.index, dtype=int)
        raw_signal[buy_condition] = 1
        raw_signal[sell_condition] = -1

        # Filter out duplicate signals:
        # For each index with a raw signal, compare it to the last nonzero signal.
        # The last nonzero (active) signal is computed via ffill on raw_signal where nonzero values are kept.
        prev_signal = raw_signal.where(raw_signal != 0).ffill().shift(1).fillna(0)
        signals = raw_signal.where((raw_signal != 0) & (raw_signal != prev_signal), 0)
        # If no crossover event happens (or it is a duplicate), the signal remains 0.
        # This replicates the state management from the original loop.

        # --- Assemble the output ---
        signals_df = pd.DataFrame({
            'close': close,
            'signal': signals,
            'signal_strength': percent_b,
            'upper_bb': upper_bb,
            'lower_bb': lower_bb,
            'sma': sma,
        }, index=close.index)

        # Drop rows with NaN values, typically the initial rows without complete rolling data.
        signals_df = signals_df.dropna().copy()
        if not self._validate_data(signals_df, min_records=1):
            return pd.DataFrame()

        return signals_df