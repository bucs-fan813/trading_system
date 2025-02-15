# trading_system/src/strategies/awesome_oscillator_strat.py

import pandas as pd
import numpy as np  # Consider removing this if not required
from typing import Dict, Optional
import logging

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

# Import the configure_logging function if needed for global logging configuration
from src.config.logging_config import configure_logging

class AwesomeOscillatorStrategy(BaseStrategy):
    """
    Awesome Oscillator (AO) Crossover Strategy for production use.
    
    This strategy calculates the Awesome Oscillator (AO) as the difference between 
    a short-term and a long-term simple moving average (SMA) of the median price,
    where the median price is computed as the average of the high and low price.
    A buy signal is generated when the AO crosses above zero from a negative value,
    indicating upward momentum, and a sell signal is generated when the AO crosses
    below zero from a positive value, indicating downward momentum. Signal strength 
    is captured as the absolute value of the AO at the point of the crossover.
    
    The strategy retrieves historical price data stored in an SQLite database 
    (specifically from the 'daily_prices' table), computes the AO, and generates 
    trading signals accordingly.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize Awesome Oscillator Strategy.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (Optional[Dict]): Strategy-specific parameters.
                Expected keys: 'short_period' and 'long_period'.
        """
        default_params = {'short_period': 5, 'long_period': 34}
        # Use provided parameters or fall back to defaults
        params = params if params is not None else default_params
        super().__init__(db_config, params)
        self.short_period = int(params.get('short_period', default_params['short_period']))
        self.long_period = int(params.get('long_period', default_params['long_period']))
        # Use the strategy's class name for logging consistency
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.short_period >= self.long_period:
            raise ValueError("Short SMA period must be less than long SMA period")
            
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate Awesome Oscillator crossover trading signals with signal strength.
        
        Args:
            ticker (str): Stock ticker symbol.
            
        Returns:
            pd.DataFrame: DataFrame containing 'close', 'ao', 'signal', and 'signal_strength'
                          indexed by date.
        """
        if not isinstance(ticker, str) or not ticker:
            self.logger.error(f"Invalid ticker input: {ticker}. Must be a non-empty string.")
            raise ValueError(f"Ticker must be a non-empty string, got: {ticker}")

        # Determine the minimum number of data points needed based on the SMA periods
        required_lookback = max(self.long_period, self.short_period) + 2
        # Use a safety buffer by fetching twice as many records
        lookback_buffer = required_lookback * 2
        try:
            historical_data = self.get_historical_prices(ticker, lookback=lookback_buffer)
        except DataRetrievalError as e:
            self.logger.error(f"Data retrieval failed for {ticker}: {e}")
            raise
        
        if not self._validate_data(historical_data, min_records=required_lookback):
            self.logger.warning(f"Insufficient data for AO calculation for {ticker}. Required {required_lookback} records.")
            return pd.DataFrame()

        df = historical_data.copy()
        signals_df = self._calculate_signals(df)
        return signals_df

    def _calculate_signals(self, hist_data: pd.DataFrame) -> pd.DataFrame:
        """
        Core signal generation logic extracted to this method.
        
        Args:
            hist_data (pd.DataFrame): Historical price data.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals, AO values, close prices, and signal strength.
        """
        # Calculate the median price from high and low prices
        hist_data['median_price'] = (hist_data['high'] + hist_data['low']) / 2
        
        # Compute short-term and long-term SMAs of the median price
        short_sma = hist_data['median_price'].rolling(window=self.short_period).mean()
        long_sma = hist_data['median_price'].rolling(window=self.long_period).mean()
        
        # Awesome Oscillator (AO) is the difference between these two SMAs
        hist_data['ao'] = short_sma - long_sma
        
        # Drop rows where 'ao' is NaN (typically the first few rows before SMA windows are full)
        hist_data.dropna(subset=['ao'], inplace=True)
        
        ao = hist_data['ao']
        # Initialize a series to store the signals (1 for buy, -1 for sell, 0 for none)
        ao_signal = pd.Series(0, index=hist_data.index)
        # Initialize a series for the signal strength (absolute AO value at the moment of the crossover)
        signal_strength = pd.Series(0.0, index=hist_data.index)

        # Generate a buy signal when AO crosses above zero (from negative or zero)
        buy_condition = (ao > 0) & (ao.shift(1) <= 0)
        ao_signal[buy_condition] = 1
        signal_strength[buy_condition] = ao[buy_condition].abs()

        # Generate a sell signal when AO crosses below zero (from positive or zero)
        sell_condition = (ao < 0) & (ao.shift(1) >= 0)
        ao_signal[sell_condition] = -1
        signal_strength[sell_condition] = ao[sell_condition].abs()

        # Build the final DataFrame of signals with close prices, AO values, signals, and strength
        signals_df = pd.DataFrame(index=hist_data.index)
        signals_df['close'] = hist_data['close']
        signals_df['ao'] = ao
        signals_df['signal'] = ao_signal
        signals_df['signal_strength'] = signal_strength

        self.logger.debug(f"Generated AO signals:\n{signals_df.tail()}")
        # Return the DataFrame excluding the initial rows that might contain incomplete SMA calculations
        return signals_df.iloc[max(self.long_period, self.short_period) + 1:, :]