# trading_system/src/strategies/williams_r.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class WilliamsRStrategy(BaseStrategy):
    """
    Williams %R Trading Strategy.

    This strategy utilizes the Williams %R indicator to generate trading signals.

    A "buy" signal is generated when Williams %R crosses below the oversold threshold,
    indicating that the asset may be oversold.

    A "sell" signal is generated when Williams %R crosses above the overbought threshold,
    suggesting that the asset may be overbought.

    Hyperparameters (via params):
        - data_lookback (int): Number of days for historical data (default: 252)
        - wr_period (int): Lookback period for calculating Williams %R (default: 14)
        - oversold_threshold (float): Threshold for oversold condition (default: -80)
        - overbought_threshold (float): Threshold for overbought condition (default: -20)
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the WilliamsRStrategy.

        Args:
            db_config: Database configuration settings.
            params: Strategy-specific parameters.
        """
        super().__init__(db_config, params)
        self.logger.info("WilliamsRStrategy initialized with parameters: %s", self.params)

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve historical data, calculate the Williams %R indicator, generate buy/sell signals,
        and compute signal strengths.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - close: Closing price.
                - wr: Williams %R indicator.
                - signal: Trading signal (1 for buy, -1 for sell, 0 for none).
                - position: Position after the signal (1 for long, 0 for flat).
                - signal_strength: Magnitude of the crossing relative to the threshold.
        """
        self.logger.info("Generating Williams %R signals for ticker: %s", ticker)
        
        # Retrieve hyperparameters with default values.
        data_lookback = self.params.get("data_lookback", 252)
        wr_period = self.params.get("wr_period", 14)
        oversold_threshold = self.params.get("oversold_threshold", -80)
        overbought_threshold = self.params.get("overbought_threshold", -20)

        # Validate threshold configuration.
        if oversold_threshold >= overbought_threshold:
            raise ValueError("oversold_threshold must be less than overbought_threshold")

        # Retrieve historical daily price data from the database.
        df = self.get_historical_prices(ticker, lookback=data_lookback)
        if not self._validate_data(df, min_records=wr_period):
            error_msg = f"Insufficient historical data for {ticker} (lookback={data_lookback})"
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg)
        
        # Ensure data is sorted in chronological order.
        df = df.sort_index()

        # Step 1: Compute the Williams %R indicator.
        high_roll = df['high'].rolling(window=wr_period).max()
        low_roll = df['low'].rolling(window=wr_period).min()
        df['wr'] = -100 * ((high_roll - df['close']) / (high_roll - low_roll))
        
        # Drop initial rows where the WR cannot be computed.
        df = df.dropna(subset=['wr']).copy()

        # Step 2: Generate signals and positions.
        num_rows = len(df)
        signals = np.zeros(num_rows, dtype=int)
        positions = np.zeros(num_rows, dtype=int)
        signal_strengths = np.zeros(num_rows, dtype=float)

        # Retrieve numpy arrays for faster computation.
        wr_values = df['wr'].values
        close_values = df['close'].values

        # Initialize with a flat starting position.
        current_position = 0
        signals[0] = 0
        positions[0] = current_position
        signal_strengths[0] = 0.0

        for i in range(1, num_rows):
            # Carry forward the previous position by default.
            positions[i] = current_position

            # Buy signal: WR crosses below or touches the oversold threshold.
            if (wr_values[i-1] > oversold_threshold) and (wr_values[i] <= oversold_threshold):
                if current_position != 1:
                    signals[i] = 1  # Buy signal.
                    current_position = 1  # Enter long position.
                    positions[i] = current_position
                    signal_strengths[i] = abs(oversold_threshold - wr_values[i])
                    self.logger.debug("Buy signal on %s: WR=%.2f, close=%.2f", df.index[i], wr_values[i], close_values[i])
            # Sell signal: WR crosses above or touches the overbought threshold AND only if in a long position.
            elif (wr_values[i-1] < overbought_threshold) and (wr_values[i] >= overbought_threshold):
                if current_position == 1:  # Ensure existing long position.
                    signals[i] = -1  # Sell signal.
                    current_position = 0  # Exit to flat.
                    positions[i] = current_position
                    signal_strengths[i] = abs(wr_values[i] - overbought_threshold)
                    self.logger.debug("Sell signal on %s: WR=%.2f, close=%.2f", df.index[i], wr_values[i], close_values[i])
            else:
                signals[i] = 0
                signal_strengths[i] = 0.0

        # Append the generated signals to the DataFrame.
        df['signal'] = signals
        df['position'] = positions
        df['signal_strength'] = signal_strengths

        self.logger.info("Completed signal generation for %s", ticker)
        return df[['close', 'wr', 'signal', 'position', 'signal_strength']]