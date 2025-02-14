# trading_system/src/strategies/awesome_oscillator_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from src.strategies.base_strat import BaseStrategy, DataRetrievalError # Import custom exception
from src.database.config import DatabaseConfig

# Import the configure_logging function from your config file
from src.config.logging_config import configure_logging

class AwesomeOscillatorStrategy(BaseStrategy):
    """
    Awesome Oscillator (AO) Crossover Strategy for production use.

    ... (rest of the class docstring remains the same)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize Awesome Oscillator Strategy.

        ... (rest of the __init__ docstring remains the same)
        """
        default_params = {'short_period': 5, 'long_period': 34}
        params = params if params is not None else default_params
        super().__init__(db_config, params)
        self.short_period = int(params.get('short_period', default_params['short_period']))
        self.long_period = int(params.get('long_period', default_params['long_period']))
        self.logger = logging.getLogger(__name__) # Get logger for this module

        if self.short_period >= self.long_period: # Added SMA period validation
            raise ValueError("Short SMA period must be less than long SMA period")


    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate Awesome Oscillator crossover trading signals with signal strength.

        ... (rest of the generate_signals docstring remains the same)
        """
        if not isinstance(ticker, str) or not ticker:
            self.logger.error(f"Invalid ticker input: {ticker}. Must be a non-empty string.")
            raise ValueError(f"Ticker must be a non-empty string, got: {ticker}")

        lookback = max(self.long_period, self.short_period) + 2 # Corrected lookback based on longest period
        try:
            historical_data = self.get_historical_prices(ticker, lookback=lookback * 2) # Added safety buffer to lookback
        except DataRetrievalError as e:
            self.logger.error(f"Data retrieval failed for {ticker}: {e}")
            raise # Re-raise the exception to be handled upstream


        if not self._validate_data(historical_data, min_records=lookback):
            self.logger.warning(f"Insufficient data for AO calculation for {ticker}. Required {lookback} records.")
            return pd.DataFrame()


        df = historical_data.copy()
        signals_df = self._calculate_signals(df) # Call separate _calculate_signals method
        return signals_df


    def _calculate_signals(self, hist_data: pd.DataFrame) -> pd.DataFrame: # Created separate method
        """Core signal generation logic extracted to this method."""

        # Calculate Awesome Oscillator (Traditional Median Price Calculation)
        hist_data['median_price'] = (hist_data['high'] + hist_data['low']) / 2
        short_sma = hist_data['median_price'].rolling(window=self.short_period).mean()
        long_sma = hist_data['median_price'].rolling(window=self.long_period).mean()
        hist_data['ao'] = short_sma - long_sma
        hist_data.dropna(subset=['ao'], inplace=True) # Drop rows where AO is NaN after calculation

        ao = hist_data['ao']
        ao_signal = pd.Series(0, index=hist_data.index)
        signal_strength = pd.Series(0.0, index=hist_data.index) # Initialize signal strength

        # Buy signal: AO crosses above zero from negative
        buy_condition = (ao > 0) & (ao.shift(1) <= 0) # Changed to <= 0 to catch zero cross
        buy_dates = hist_data.index[buy_condition]
        ao_signal[buy_condition] = 1
        signal_strength[buy_condition] = ao[buy_condition].abs() # Strength at buy signal

        # Sell signal: AO crosses below zero from positive
        sell_condition = (ao < 0) & (ao.shift(1) >= 0) # Changed to >= 0 to catch zero cross
        sell_dates = hist_data.index[sell_condition]
        ao_signal[sell_condition] = -1
        signal_strength[sell_condition] = ao[sell_condition].abs() # Strength at sell signal

        signals_df = pd.DataFrame(index=hist_data.index) # Create signals DataFrame here
        signals_df['close'] = hist_data['close']
        signals_df['ao'] = ao # Add AO values
        signals_df['signal'] = ao_signal
        signals_df['signal_strength'] = signal_strength


        self.logger.debug(f"Generated AO signals:\n{signals_df.tail()}") # Debug log
        return signals_df.iloc[max(self.long_period, self.short_period)+1:, :] # Corrected return slice


    # Example of a basic unit test structure (conceptual - using pytest would be ideal)
    # def test_generate_signals(self):
    #     """
    #     Example unit test for generate_signals method.
    #     In a real scenario, use a testing framework like pytest and mock database interactions.
    #     """
    #     db_config = DatabaseConfig(db_name='test_db.db', db_type='sqlite') # Test DB
    #     strategy = AwesomeOscillatorStrategy(db_config)
    #     ticker = 'TEST_TICKER'
    #     signals_df = strategy.generate_signals(ticker)
    #     assert isinstance(signals_df, pd.DataFrame), "generate_signals should return a DataFrame"
    #     # Add more assertions to check for signal logic, data validation, etc.
    #     if not signals_df.empty:
    #         assert 'signal' in signals_df.columns
    #         assert 'signal_strength' in signals_df.columns

""" 
from src.database.config import DatabaseConfig
from src.strategies.ao_crossover_strat import AwesomeOscillatorStrategy

# Get default database config (or load from elsewhere if needed)
db_config = DatabaseConfig.default()

# Instantiate the strategy, passing the db_config
ao_strategy = AwesomeOscillatorStrategy(db_config, params={'short_period': 5, 'long_period': 34})

# Now you can use ao_strategy to generate signals, and it will use the database engine
signals_df = ao_strategy.generate_signals(ticker_symbol='AAPL')
"""