# trading_system/src/strategies/supertrend_strat.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class SupertrendStrategy(BaseStrategy):
    """
    Supertrend strategy implementation.

    Hyperparameters:
        lookback (int): Period for ATR calculation. Default: 10
        multiplier (float): Multiplier for ATR to determine bands. Default: 3.0
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        # Set default parameters and update if provided
        default_params = {'lookback': 10, 'multiplier': 3.0}
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate Supertrend-based trading signals.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with columns: 'close', 'supertrend', 'signal', 'signal_strength'
        """
        # Retrieve historical data from the database
        try:
            data = self.get_historical_prices(ticker)
            # Ensure we have sufficient data
            if not self._validate_data(data, min_records=self.params['lookback'] * 2):
                self.logger.error(f"Insufficient data for {ticker}")
                return pd.DataFrame()
        except DataRetrievalError as e:
            self.logger.error(f"Data retrieval failed: {e}")
            return pd.DataFrame()

        lookback = self.params['lookback']
        multiplier = self.params['multiplier']
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range and ATR using exponential moving average
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=lookback, adjust=False).mean()

        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        # Initialize final bands as copies of the basic bands
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()

        # Iteratively adjust the final bands
        for i in range(1, len(data)):
            # Adjust final upper band
            if (basic_upper.iat[i] < final_upper.iat[i-1]) or (close.iat[i-1] > final_upper.iat[i-1]):
                final_upper.iat[i] = basic_upper.iat[i]
            else:
                final_upper.iat[i] = final_upper.iat[i-1]

            # Adjust final lower band
            if (basic_lower.iat[i] > final_lower.iat[i-1]) or (close.iat[i-1] < final_lower.iat[i-1]):
                final_lower.iat[i] = basic_lower.iat[i]
            else:
                final_lower.iat[i] = final_lower.iat[i-1]

        # Initialize Supertrend indicator and trend direction series
        supertrend = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)  # 1 = uptrend, -1 = downtrend

        # Initialize the first value:
        # Convention: if the first close is less than or equal to the final upper band,
        # the market is in a downtrend (indicator is set to final_upper),
        # otherwise it is in an uptrend (indicator is set to final_lower).
        if close.iat[0] <= final_upper.iat[0]:
            supertrend.iat[0] = final_upper.iat[0]
            trend.iat[0] = -1
        else:
            supertrend.iat[0] = final_lower.iat[0]
            trend.iat[0] = 1

        # Calculate the Supertrend for subsequent periods
        for i in range(1, len(data)):
            # If the previous Supertrend was the upper band (downtrend)
            if supertrend.iat[i-1] == final_upper.iat[i-1]:
                if close.iat[i] <= final_upper.iat[i]:
                    supertrend.iat[i] = final_upper.iat[i]
                    trend.iat[i] = -1
                else:
                    supertrend.iat[i] = final_lower.iat[i]
                    trend.iat[i] = 1
            # If the previous Supertrend was the lower band (uptrend)
            else:
                if close.iat[i] >= final_lower.iat[i]:
                    supertrend.iat[i] = final_lower.iat[i]
                    trend.iat[i] = 1
                else:
                    supertrend.iat[i] = final_upper.iat[i]
                    trend.iat[i] = -1

        # Generate trading signals based on crossovers between price and Supertrend
        signal = pd.Series(0, index=supertrend.index)
        for i in range(1, len(data)):
            # Buy signal: Price crosses from below to above the Supertrend line
            if (close.iat[i-1] < supertrend.iat[i-1]) and (close.iat[i] > supertrend.iat[i]):
                signal.iat[i] = 1
            # Sell signal: Price crosses from above to below the Supertrend line
            elif (close.iat[i-1] > supertrend.iat[i-1]) and (close.iat[i] < supertrend.iat[i]):
                signal.iat[i] = -1
            else:
                signal.iat[i] = 0

        # Compute signal strength as the absolute percentage difference between close and Supertrend,
        # multiplied by the signal (to preserve the buy/sell indication)
        signal_strength = ((close - supertrend).abs() / close) * signal

        # Create the output DataFrame
        signals_df = pd.DataFrame({
            'close': close,
            'supertrend': supertrend,
            'signal': signal,
            'signal_strength': signal_strength
        }).dropna()

        return signals_df