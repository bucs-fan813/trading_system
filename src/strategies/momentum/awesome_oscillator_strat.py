# trading_system/src/strategies/momentum/awesome_oscillator_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

class AwesomeOscillatorStrategy(BaseStrategy):
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Awesome Oscillator Strategy.

        Parameters:
            db_config (DatabaseConfig): Database configuration settings.
            params (Optional[Dict]): Strategy parameters.
                Expected keys:
                  - 'short_period' (default: 5)
                  - 'long_period' (default: 34)
				  
				  
				  
				  
				  
		The Awesome Oscillator (AO) is computed as:
			AO = (short_sma - long_sma)
		where the SMAs are computed over the median price:
			median_price = (high + low) / 2

		Signal generation:
		  - A buy signal (signal = 1) is generated when AO crosses upward through zero.
		  - A sell signal (signal = -1) is generated when AO crosses downward through zero.

		Profit calculation:
		  - Daily returns are computed from the close price.
		  - Strategy return is computed as daily_return * previous day’s position.
		  - Cumulative return aggregates the daily strategy returns.
		  
		Strategy Parameters (via the params dictionary):
		  - short_period (int, default: 5): Period for the short-term SMA.
		  - long_period (int, default: 34): Period for the long-term SMA.
  
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
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals, simulate positions, and compute profit metrics for a given ticker.

        This method supports two modes:
          - Backtesting: When start_date and end_date are provided, the full historical data is used 
            for performance evaluation (profit, cumulative returns, etc.).
          - Forecasting: When latest_only is True, a minimal data slice is used and only the latest
            signal is returned for end-of-day decision making.

        The returned DataFrame includes:
          - 'close'               : Execution price reference.
          - 'ao'                  : Awesome Oscillator value.
          - 'signal'              : Trading signal (1 for buy, -1 for sell, 0 for none).
          - 'signal_strength'     : Normalized magnitude of the signal.
          - 'position'            : Simulated position over time.
          - 'daily_return'        : Daily returns from 'close' prices.
          - 'strategy_return'     : Strategy returns (using the previous day's position).
          - 'cumulative_return'   : Cumulative performance of the strategy.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting position (e.g., 0 for no position, 1 for long).
            latest_only (bool): If True, returns only the final row (for EOD decision making).

        Returns:
            pd.DataFrame: DataFrame with signals and performance metrics.
        """
        try:
            # Retrieve historical data.
            if start_date and end_date:
                data = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
            else:
                # For forecast mode (or if no dates provided), use a minimal lookback buffer.
                lookback_buffer = 2 * max(self.short_period, self.long_period)
                data = self.get_historical_prices(ticker, lookback=lookback_buffer)
                data = data.sort_index()  # ensure ascending date order

            # Validate data sufficiency.
            required_records = max(self.short_period, self.long_period)
            if not self._validate_data(data, min_records=required_records):
                self.logger.warning(
                    f"Insufficient data for {ticker}: required at least {required_records} records."
                )
                return pd.DataFrame()

            # Calculate the AO signals.
            signals = self._calculate_signals(data)
            # Simulate trading positions based on signals.
            signals = self._simulate_positions(signals, initial_position)
            # Compute profit and performance metrics.
            signals = self._calculate_profits(signals, initial_position)

            if latest_only:
                # Return only the latest signal if in forecast mode.
                signals = signals.iloc[[-1]].copy()
            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Awesome Oscillator (AO) and generate trading signals using vectorized operations.

        AO is calculated as:
            ao = (short_sma - long_sma)
        where:
            short_sma = SMA of median_price with window=self.short_period,
            long_sma  = SMA of median_price with window=self.long_period,
            median_price = (high + low) / 2.

        Trading signals:
          - Buy signal (1): When AO crosses upward through zero.
          - Sell signal (-1): When AO crosses downward through zero.
        Signal strength is normalized by the rolling standard deviation of AO.

        Returns:
            pd.DataFrame: DataFrame with columns ['close', 'ao', 'signal', 'signal_strength'].
        """
        data = data.sort_index()

        # Compute the median price.
        data['median_price'] = (data['high'] + data['low']) / 2

        # Compute short-term and long-term Simple Moving Averages.
        data['short_sma'] = data['median_price'].rolling(window=self.short_period, min_periods=self.short_period).mean()
        data['long_sma'] = data['median_price'].rolling(window=self.long_period, min_periods=self.long_period).mean()

        # Calculate the Awesome Oscillator.
        data['ao'] = data['short_sma'] - data['long_sma']
        data = data.dropna(subset=['ao'])

        # Normalize signal strength using the rolling standard deviation.
        data['ao_std'] = data['ao'].rolling(window=self.short_period, min_periods=self.short_period).std()
        data['normalized_strength'] = data['ao'].abs() / (data['ao_std'] + 1e-6)  # avoid division by zero

        # Vectorized signal generation.
        buy_mask = (data['ao'] > 0) & (data['ao'].shift(1) <= 0)
        sell_mask = (data['ao'] < 0) & (data['ao'].shift(1) >= 0)

        data['signal'] = 0
        data.loc[buy_mask, 'signal'] = 1
        data.loc[sell_mask, 'signal'] = -1

        data['signal_strength'] = 0.0
        data.loc[buy_mask, 'signal_strength'] = data.loc[buy_mask, 'normalized_strength']
        data.loc[sell_mask, 'signal_strength'] = data.loc[sell_mask, 'normalized_strength']

        # Return the essential columns.
        signals = data[['close', 'ao', 'signal', 'signal_strength']].copy()
        return signals

    def _simulate_positions(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Simulate positions from the trading signals in a fully vectorized fashion.

        The method replaces non-zero signals with definitive positions (1 for buy, 0 for sell),
        then forward-fills these positions, starting with the given initial position.

        Returns:
            pd.DataFrame: DataFrame with an additional 'position' column.
        """
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(initial_position)
        signals['position'] = signals['position'].astype(int)
        return signals

    def _calculate_profits(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Compute performance metrics based on trading signals and positions.

        The calculations are as follows:
          - daily_return: Percentage change in the 'close' price.
          - strategy_return: daily_return multiplied by the lagged (previous day's) position.
          - cumulative_return: Product of (1 + strategy_return) across time, minus 1.

        Returns:
            pd.DataFrame: DataFrame augmented with 'daily_return', 'strategy_return', and 'cumulative_return'.
        """
        # Calculate daily returns (close-to-close percentage change).
        signals['daily_return'] = signals['close'].pct_change().fillna(0)
        # Use the previous day's position for that day's return.
        signals['lagged_position'] = signals['position'].shift(1).fillna(initial_position)
        signals['strategy_return'] = signals['daily_return'] * signals['lagged_position']
        signals['cumulative_return'] = (1 + signals['strategy_return']).cumprod() - 1
        signals.drop(columns=['lagged_position'], inplace=True)
        return signals