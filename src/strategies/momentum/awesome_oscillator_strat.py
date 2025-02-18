# trading_system/src/strategies/momentum/awesome_oscillator_strat.py

import pandas as pd
import numpy as np
import logging
from sqlalchemy import text
from typing import Dict, Optional

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig


class AwesomeOscillatorStrategy(BaseStrategy):
    """
    Optimized implementation of the Awesome Oscillator (AO) crossover strategy.

    Key features:
      - Improved execution speed through vectorized operations.
      - Flexible backtesting with user-specified start and end dates.
      - Simulated positions tracking with a customizable starting position.
      - Latest signal extraction for end-of-day decisions using the 'latest_only' flag.
      - Normalized signal strength calculation using rolling volatility.
      - Ensures 'close' prices are available for actionable trading decisions.

    Strategy parameters expected in `params` dictionary:
      - short_period (int): The period for the short-term SMA (default: 5)
      - long_period (int): The period for the long-term SMA (default: 34)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Awesome Oscillator Strategy.

        Parameters:
            db_config (DatabaseConfig): Database configuration settings.
            params (Optional[Dict]): Strategy parameters. Expected keys:
                'short_period' (default: 5) and 'long_period' (default: 34).
        """
        default_params = {'short_period': 5, 'long_period': 34}
        params = params or default_params
        super().__init__(db_config, params)

        self.short_period = int(params.get('short_period', default_params['short_period']))
        self.long_period = int(params.get('long_period', default_params['long_period']))

        if self.short_period >= self.long_period:
            raise ValueError("Short SMA period must be less than long SMA period.")

        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Generate trading signals and simulate positions based on the Awesome Oscillator.

        Parameters:
            ticker (str): Stock ticker symbol.
            start_date (Optional[str]): Backtesting start date in 'YYYY-MM-DD' format.
            end_date (Optional[str]): Backtesting end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting trading position (typically 0).
            latest_only (bool): If True, returns only the last date's signal (for end-of-day decisions).

        Returns:
            pd.DataFrame: DataFrame (indexed by date) containing the following columns:
                - 'close': Close price, where the signal is applicable.
                - 'ao': Calculated Awesome Oscillator value.
                - 'signal': 1 for a buy signal, -1 for a sell signal, 0 otherwise.
                - 'signal_strength': Normalized magnitude of the signal.
                - 'position': Simulated position over time.
        """
        if not isinstance(ticker, str) or not ticker:
            self.logger.error("Ticker must be a non-empty string.")
            raise ValueError("Ticker must be a non-empty string.")

        # Retrieve historical data:
        # 1. If a date range is provided, use it.
        # 2. Otherwise, use a safety lookback period (twice the maximum period) as buffer.
        if start_date and end_date:
            data = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
        else:
            lookback_buffer = 2 * max(self.short_period, self.long_period)
            data = self.get_historical_prices(ticker, lookback=lookback_buffer)
            data = data.sort_index()  # ensure ascending order if using lookback

        required_records = max(self.short_period, self.long_period)
        if not self._validate_data(data, min_records=required_records):
            self.logger.warning(
                f"Insufficient data for {ticker}: required at least {required_records} records."
            )
            return pd.DataFrame()

        # Calculate the signals and simulate positions.
        signals = self._calculate_signals(data)
        signals = self._simulate_positions(signals, initial_position)

        if latest_only:
            # Return only the final row for a quick end-of-day decision.
            return signals.tail(1)
        return signals

    def _calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Awesome Oscillator and generate trading signals.

        The Awesome Oscillator (AO) is computed as the difference between a short-term and a
        long-term SMA of the median price [(high + low) / 2]. Here we also compute a normalized
        signal strength (i.e. the AO magnitude scaled by its rolling volatility) that may be used
        for downstream optimization.

        Parameters:
            data (pd.DataFrame): Historical price data with 'high', 'low', and 'close' columns.

        Returns:
            pd.DataFrame: DataFrame containing 'close', 'ao', 'signal', and 'signal_strength'.
        """
        # Ensure that data is sorted by date.
        data = data.sort_index()

        # Calculate the median price.
        data["median_price"] = (data["high"] + data["low"]) / 2

        # Compute short-term and long-term SMAs of the median price.
        data["short_sma"] = data["median_price"].rolling(window=self.short_period, min_periods=self.short_period).mean()
        data["long_sma"] = data["median_price"].rolling(window=self.long_period, min_periods=self.long_period).mean()

        # Calculate the Awesome Oscillator.
        data["ao"] = data["short_sma"] - data["long_sma"]
        data = data.dropna(subset=["ao"])  # Remove rows without a full SMA calculation.

        # Calculate rolling standard deviation of AO to normalize signal strength.
        data["ao_std"] = data["ao"].rolling(window=self.short_period, min_periods=self.short_period).std()
        data["normalized_strength"] = data["ao"].abs() / (data["ao_std"] + 1e-6)  # Avoid division by zero

        # Initialize signal columns.
        data["signal"] = 0
        data["signal_strength"] = 0.0

        # Generate buy signals: when AO crosses upward past zero.
        buy_mask = (data["ao"] > 0) & (data["ao"].shift(1) <= 0)
        data.loc[buy_mask, "signal"] = 1
        data.loc[buy_mask, "signal_strength"] = data.loc[buy_mask, "normalized_strength"]

        # Generate sell signals: when AO crosses downward past zero.
        sell_mask = (data["ao"] < 0) & (data["ao"].shift(1) >= 0)
        data.loc[sell_mask, "signal"] = -1
        data.loc[sell_mask, "signal_strength"] = data.loc[sell_mask, "normalized_strength"]

        # Retain only the necessary output columns.
        signals_df = data[["close", "ao", "signal", "signal_strength"]].copy()
        self.logger.debug("AO signals (last 5 rows):\n%s", signals_df.tail())
        return signals_df

    def _simulate_positions(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Simulate changes in position based on the generated signals.

        This function updates the simulated position:
          - A non-zero signal updates the position.
          - If no new signal is generated, the previous position is carried forward.

        Parameters:
            signals (pd.DataFrame): DataFrame containing signals from _calculate_signals.
            initial_position (int): The starting position (e.g., 0 for no open position).

        Returns:
            pd.DataFrame: The input DataFrame augmented with a 'position' column.
        """
        signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(initial_position)
        signals["position"] = signals["position"].astype(int)
        return signals