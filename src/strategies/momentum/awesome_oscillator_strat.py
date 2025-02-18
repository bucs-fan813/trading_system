# trading_system/src/strategies/momentum/awesome_oscillator_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from sqlalchemy import text

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

class AwesomeOscillatorStrategy(BaseStrategy):
    """
    Awesome Oscillator (AO) Crossover Strategy with backtesting support.
    
    The strategy computes the Awesome Oscillator (AO) by taking the difference
    between a short-term and a long-term SMA of the median price (average of high and low).
    
    It supports backtesting over a specified date range, calculating a 'position'
    that is updated when a buy signal (AO crosses above 0) or a sell signal (AO 
    crosses below 0) is triggered. Additionally, a flag allows you to extract just the
    last signal (useful for an end-of-day decision).
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Awesome Oscillator Strategy.
        
        Parameters:
            db_config (DatabaseConfig): Database configuration.
            params (Optional[Dict]): Strategy parameters.
                Expected keys: 'short_period' and 'long_period'.
        """
        default_params = {'short_period': 5, 'long_period': 34}
        params = params if params is not None else default_params
        super().__init__(db_config, params)
        
        self.short_period = int(params.get('short_period', default_params['short_period']))
        self.long_period = int(params.get('long_period', default_params['long_period']))
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.short_period >= self.long_period:
            raise ValueError("Short SMA period must be less than long SMA period")
    
    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Generate AO signals and simulate positions over a time period.
        
        If start_date and end_date are provided, price data is pulled for that period.
        Otherwise, a safety-buffer lookback period is used.
        
        Parameters:
            ticker (str): Stock ticker symbol.
            start_date (Optional[str]): Start date in 'YYYY-MM-DD' format for backtesting.
            end_date (Optional[str]): End date in 'YYYY-MM-DD' format for backtesting.
            initial_position (int): The starting position (e.g., 0 for no position).
            latest_only (bool): If True, return only the last date’s signal.
        
        Returns:
            pd.DataFrame: DataFrame with columns:
                'close', 'ao', 'signal', 'signal_strength', and 'position',
                indexed by date.
        """
        if not isinstance(ticker, str) or not ticker:
            self.logger.error("Ticker must be a non-empty string.")
            raise ValueError("Ticker must be a non-empty string.")
        
        # Retrieve historical data either over a provided date range or using a lookback window.
        if start_date and end_date:
            historical_data = self.get_prices_by_date(ticker, start_date, end_date)
        else:
            # Use a safety buffer lookback (twice the period) if no dates are provided.
            lookback_buffer = 2 * max(self.short_period, self.long_period)
            historical_data = self.get_historical_prices(ticker, lookback=lookback_buffer)
            # When using lookback data, sort the dates in ascending order.
            historical_data = historical_data.sort_index()
        
        required_records = max(self.short_period, self.long_period)
        if not self._validate_data(historical_data, min_records=required_records):
            self.logger.warning(
                f"Insufficient data for AO calculation for {ticker}. Required at least {required_records} records."
            )
            return pd.DataFrame()
        
        signals_df = self._calculate_signals(historical_data)
        signals_df = self._simulate_positions(signals_df, initial_position)
        
        if latest_only:
            # Return only the last available signal.
            return signals_df.tail(1)
        return signals_df

    def get_prices_by_date(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        data_source: str = "yfinance"
    ) -> pd.DataFrame:
        """
        Retrieve historical price data within a given date range.
        
        Parameters:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            data_source (str): Data source identifier.
            
        Returns:
            pd.DataFrame: Price data sorted in ascending date order.
        """
        query = text("""
            SELECT date, open, high, low, close, volume 
            FROM daily_prices
            WHERE ticker = :ticker
              AND data_source = :data_source
              AND date BETWEEN :start_date AND :end_date
            ORDER BY date ASC
        """)
        params = {
            "ticker": ticker,
            "data_source": data_source,
            "start_date": start_date,
            "end_date": end_date,
        }
        return self._execute_query(query, params, index_col="date")
    
    def _calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Awesome Oscillator and generate crossover signals.
        
        Parameters:
            data (pd.DataFrame): Historical price data.
            
        Returns:
            pd.DataFrame: DataFrame with 'close', 'ao', 'signal', and 'signal_strength'.
        """
        # Ensure the data is sorted by ascending date.
        data = data.sort_index()
        
        # Calculate the median price.
        data["median_price"] = (data["high"] + data["low"]) / 2
        
        # Compute the short-term and long-term SMAs.
        data["short_sma"] = data["median_price"].rolling(window=self.short_period, min_periods=self.short_period).mean()
        data["long_sma"] = data["median_price"].rolling(window=self.long_period, min_periods=self.long_period).mean()
        
        # The Awesome Oscillator is the difference between the two SMAs.
        data["ao"] = data["short_sma"] - data["long_sma"]
        data.dropna(subset=["ao"], inplace=True)  # Remove periods with incomplete SMA calculation.
        
        # Initialize signal and signal_strength columns.
        data["signal"] = 0
        data["signal_strength"] = 0.0
        
        # Buy signal: AO crosses upward past 0.
        buy_mask = (data["ao"] > 0) & (data["ao"].shift(1) <= 0)
        data.loc[buy_mask, "signal"] = 1
        data.loc[buy_mask, "signal_strength"] = data.loc[buy_mask, "ao"].abs()
        
        # Sell signal: AO crosses downward past 0.
        sell_mask = (data["ao"] < 0) & (data["ao"].shift(1) >= 0)
        data.loc[sell_mask, "signal"] = -1
        data.loc[sell_mask, "signal_strength"] = data.loc[sell_mask, "ao"].abs()
        
        # Select only the necessary columns.
        signals_df = data[["close", "ao", "signal", "signal_strength"]].copy()
        self.logger.debug(f"Calculated AO signals (last 5 rows):\n{signals_df.tail()}")
        return signals_df
    
    def _simulate_positions(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Update the signals DataFrame to simulate positions based on signals.
        
        The logic is simple: when a non-zero signal occurs, the position is updated;
        otherwise, the previous position is carried forward.
        
        Parameters:
            signals (pd.DataFrame): DataFrame produced from _calculate_signals.
            initial_position (int): The starting position for the backtest.
        
        Returns:
            pd.DataFrame: Signals DataFrame with an additional 'position' column.
        """
        # Replace zeros with NaN, forward-fill and then fill the first entry with initial_position.
        signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(initial_position)
        signals["position"] = signals["position"].astype(int)
        return signals