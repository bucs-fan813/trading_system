# trading_system/src/strategies/cci_oscillator_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig


class CCIStrategy(BaseStrategy):
    """
    Commodity Channel Index (CCI) Strategy
    
    The standard CCI is calculated as:
    
        CCI = (Typical Price - SMA) / (0.015 * Mean Absolute Deviation)
        
    Typical Price is defined as (High + Low + Close) / 3.

    This strategy supports both backtesting over a defined time period and a quick
    forecast for the latest date’s signal using a restricted dataset for speed.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        # Default parameters for the strategy.
        default_params = {
            'cci_period': 20,
            'cci_upper_band': 150,
            'cci_lower_band': -150,
            # Default lookback period for backtesting if no date range is provided.
            'lookback_period': 252,
            'data_source': 'yfinance'
        }
        updated_params = default_params.copy()
        if params:
            updated_params.update(params)

        # Parameter validation.
        if updated_params['cci_upper_band'] <= updated_params['cci_lower_band']:
            raise ValueError("cci_upper_band must be greater than cci_lower_band")
        if updated_params['cci_period'] < 1:
            raise ValueError("cci_period must be at least 1")

        # Initialize the BaseStrategy.
        super().__init__(db_config, params=updated_params)
        self.cci_period = int(self.params['cci_period'])
        self.cci_upper_band = float(self.params['cci_upper_band'])
        self.cci_lower_band = float(self.params['cci_lower_band'])
        self.lookback_period = int(self.params['lookback_period'])
        self.data_source = str(self.params['data_source'])

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate CCI signals for the given ticker.
        
        Depending on the parameters, this method can be used for:
          - Backtesting: Provide a start_date and/or end_date to run over a specified period.
          - Forecasting: Set latest_only=True to calculate and return only the latest signal.
        
        The resulting DataFrame includes:
          - close: The close price (used as execution price).
          - cci: The calculated CCI value.
          - signal: Trading signal (+1 for buy, -1 for sell, 0 for no action).
          - signal_strength: Distance between the CCI value and threshold.
          - position: The trading position based on the generated signals.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Initial position for backtesting (0 for no position, 1 for long).
            latest_only (bool): If True, retrieves minimal required data and returns only the latest signal.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['close', 'cci', 'signal', 'signal_strength', 'position'].
        """
        try:
            # Determine the amount of historical data needed.
            if latest_only:
                # For a stable CCI calculation, we require at least 2 * cci_period records.
                required_records = max(2 * self.cci_period, 50)
                historical_data = self.get_historical_prices(
                    ticker,
                    lookback=required_records,
                    data_source=self.data_source
                )
            elif start_date or end_date:
                # Use the provided date range for backtesting.
                historical_data = self.get_historical_prices(
                    ticker,
                    from_date=start_date,
                    to_date=end_date,
                    data_source=self.data_source
                )
            else:
                # If no date range is provided, use the default lookback period.
                historical_data = self.get_historical_prices(
                    ticker,
                    lookback=self.lookback_period,
                    data_source=self.data_source
                )

            # Validate sufficient data for calculations.
            if not self._validate_data(historical_data, min_records=2 * self.cci_period):
                self.logger.warning("Not enough historical data to compute CCI")
                return pd.DataFrame()

            # Calculate Typical Price.
            typical_price = (historical_data['high'] +
                             historical_data['low'] +
                             historical_data['close']) / 3

            # Compute the Simple Moving Average (SMA) of the Typical Price.
            sma_tp = typical_price.rolling(window=self.cci_period, min_periods=self.cci_period).mean()

            # Compute the Mean Absolute Deviation using the built-in rolling.mad().
            mean_deviation = typical_price.rolling(window=self.cci_period, min_periods=self.cci_period).mad()
            mean_deviation = mean_deviation.replace(0, np.nan)  # Avoid division-by-zero.

            # Calculate the CCI.
            cci_indicator = (typical_price - sma_tp) / (0.015 * mean_deviation)
            cci_indicator.name = 'cci'

            # Assemble the signals DataFrame.
            signals = pd.DataFrame(index=historical_data.index)
            signals['close'] = historical_data['close']
            signals['cci'] = cci_indicator

            # Create signal column: buy when crossing above lower band; sell when crossing below upper band.
            cci_prev = signals['cci'].shift(1)  # Previous day's CCI.
            buy_signal = (cci_prev < self.cci_lower_band) & (signals['cci'] > self.cci_lower_band)
            sell_signal = (cci_prev > self.cci_upper_band) & (signals['cci'] < self.cci_upper_band)

            signals['signal'] = 0
            signals.loc[buy_signal, 'signal'] = 1
            signals.loc[sell_signal, 'signal'] = -1

            # Calculate signal strength as the distance from the threshold.
            signals['signal_strength'] = 0.0
            signals.loc[buy_signal, 'signal_strength'] = signals['cci'] - self.cci_lower_band
            signals.loc[sell_signal, 'signal_strength'] = self.cci_upper_band - signals['cci']

            # Vectorized position calculation:
            # Map buy signals to 1 and sell signals to 0.
            # Then forward-fill these values to maintain the position until a new signal occurs.
            signals['position'] = signals['signal'].map({1: 1, -1: 0}).ffill().fillna(initial_position)

            # If only the latest signal is required, return the last row.
            if latest_only:
                signals = signals.iloc[[-1]].copy()

            # Return the relevant columns.
            return signals[['close', 'cci', 'signal', 'signal_strength', 'position']]

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            return pd.DataFrame()
