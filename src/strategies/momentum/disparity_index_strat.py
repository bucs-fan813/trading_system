# trading_system/src/strategies/disparity_index_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

class DisparityIndexStrategy(BaseStrategy):
    """
    Disparity Index Trading Strategy

    Generates signals based on the percentage deviation of price from a moving average.
    Buy when the index crosses above zero after four consecutive negative days,
    sell when it crosses below after four consecutive positive days.

    Parameters:
        di_lookback (int): Lookback period for the moving average (default: 14)
        consecutive_period (int): Number of consecutive days to check (default: 4)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        params = params or {}
        super().__init__(db_config, params)
        self.di_lookback = params.get('di_lookback', 14)
        self.consecutive_period = params.get('consecutive_period', 4)
        # Optional: Reassigning the logger (already set in BaseStrategy)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate trading signals based on Disparity Index logic.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with columns: ['close', 'di', 'signal', 'signal_strength', 'position']
        """
        # Calculate required data lookback: enough data for moving average and consecutive day conditions
        required_lookback = self.di_lookback + self.consecutive_period

        # Retrieve historical data
        prices_df = self.get_historical_prices(ticker, lookback=required_lookback)
        
        # Validate data
        if not self._validate_data(prices_df, min_records=required_lookback):
            self.logger.error(f"Insufficient data for {ticker}")
            raise DataRetrievalError(f"Insufficient historical data for {ticker}")

        # Calculate Disparity Index
        close_prices = prices_df['close']
        ma = close_prices.rolling(self.di_lookback).mean()
        di = ((close_prices - ma) / ma) * 100

        # Generate signal conditions using the previous day's DI values
        shifted_di = di.shift(1)  # Previous day's DI is used for the rolling check

        buy_condition = (
            shifted_di.rolling(window=self.consecutive_period, min_periods=self.consecutive_period)
                      .apply(lambda x: all(x < 0), raw=True)
                      .fillna(False)
                      .astype(bool)
        ) & (di > 0)

        sell_condition = (
            shifted_di.rolling(window=self.consecutive_period, min_periods=self.consecutive_period)
                      .apply(lambda x: all(x > 0), raw=True)
                      .fillna(False)
                      .astype(bool)
        ) & (di < 0)

        # Generate signals: 1 for buy, -1 for sell, 0 for no action
        signals = pd.Series(
            np.select([buy_condition, sell_condition], [1, -1], default=0),
            index=di.index, name='signal'
        )
        
        # Determine signal strength based on the DI value (adjust sign for sell signals)
        signal_strength = pd.Series(
            np.where(buy_condition, di, np.where(sell_condition, -di, 0.0)),
            index=di.index, name='signal_strength'
        )

        # Calculate position: maintain long (1) after a buy signal,
        # switch to flat (0) after a sell signal, and hold the previous state otherwise.
        signals_mapped = signals.replace({1: 1, -1: 0, 0: np.nan})
        position = signals_mapped.ffill().fillna(0).astype(int)

        # Compile final strategy DataFrame
        strategy_df = pd.concat([
            close_prices.rename('close'),
            di.rename('di'),
            signals,
            signal_strength,
            position.rename('position')
        ], axis=1).dropna(subset=['di'])

        return strategy_df