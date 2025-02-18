# trading_system/src/strategies/sma_crossover_strategy.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy

    Attributes:
        short_window (int): Short SMA window (default: 20)
        long_window (int): Long SMA window (default: 50)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        
        # Initialize strategy parameters with validation
        self.short_window = self.params.get('short_window', 20)
        self.long_window = self.params.get('long_window', 50)
        
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        
        self.logger.debug(f"Initialized SMA Strategy with short={self.short_window}, long={self.long_window}")

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate SMA crossover signals

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with columns: ['short_sma', 'long_sma', 'signal', 'position', 'strength']
        """
        try:
            # Use lookback from parameters if provided, otherwise default to 252 days.
            lookback = self.params.get('lookback', 252)
            # Ensure the lookback is not less than the long_window period.
            if lookback < self.long_window:
                lookback = self.long_window
            
            # Retrieve data with sufficient lookback for the longest SMA calculation
            df = self.get_historical_prices(ticker, lookback=lookback)
            
            # Validate data availability
            if not self._validate_data(df, min_records=self.long_window):
                self.logger.error(f"Insufficient data for {ticker}")
                return pd.DataFrame()
            
            # Calculate SMAs on the closing prices
            df['short_sma'] = df['close'].rolling(window=self.short_window).mean()
            df['long_sma'] = df['close'].rolling(window=self.long_window).mean()
            
            # Drop rows where SMAs are not yet available
            df.dropna(subset=['short_sma', 'long_sma'], inplace=True)
            
            if not self._validate_data(df, min_records=1):
                self.logger.error(f"No valid SMA data for {ticker}")
                return pd.DataFrame()
            
            # Generate crossover signals
            cross_above = (df['short_sma'] > df['long_sma']) & \
                          (df['short_sma'].shift(1) <= df['long_sma'].shift(1))
            cross_below = (df['short_sma'] < df['long_sma']) & \
                          (df['short_sma'].shift(1) >= df['long_sma'].shift(1))
            
            # Initialize signals: 0 = no signal, 1 = buy, -1 = sell
            df['signal'] = 0
            df.loc[cross_above, 'signal'] = 1  # Buy signal
            df.loc[cross_below, 'signal'] = -1 # Sell signal
            
            # Calculate positions (1 = holding, 0 = not holding)
            # Forward fill the last non-zero signal and replace sell signal (-1) with 0.
            mask = df['signal'] != 0
            df['position'] = df['signal'].where(mask).ffill().fillna(0).replace({1: 1, -1: 0})
            
            # Calculate signal strength as the normalized difference between SMAs
            df['strength'] = (df['short_sma'] - df['long_sma']) / df['close']
            
            # Return a copy of the relevant columns
            return df[['short_sma', 'long_sma', 'signal', 'position', 'strength']].copy()
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            raise