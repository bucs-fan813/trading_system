# trading_system/src/strategies/stochastic_strat.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig

class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator strategy implementation.

    Hyperparameters:
        k_period (int): Lookback period for %K calculation (default: 14)
        d_period (int): Smoothing period for %D calculation (default: 3)
        overbought (int): Overbought threshold (default: 80)
        oversold (int): Oversold threshold (default: 20)
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hyperparameters with defaults
        self.k_period = self.params.get('k_period', 14)
        self.d_period = self.params.get('d_period', 3)
        self.overbought = self.params.get('overbought', 80)
        self.oversold = self.params.get('oversold', 20)

        # Validate hyperparameters
        if self.k_period <= 0 or self.d_period <= 0:
            raise ValueError("k_period and d_period must be positive integers")
        if self.overbought <= self.oversold:
            raise ValueError("overbought must be greater than oversold")

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate Stochastic Oscillator signals.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with columns: ['close', '%k', '%d', 'signal', 'strength']
        """
        try:
            # Determine minimum required data points: (k_period + d_period - 1) ensures valid rolling calculations.
            min_data_points = self.k_period + self.d_period - 1
            prices = self.get_historical_prices(ticker, lookback=min_data_points)
            
            if not self._validate_data(prices, min_records=min_data_points):
                self.logger.error(f"Insufficient data for {ticker}")
                return pd.DataFrame()

            # Calculate rolling low and high needed for %K computation.
            low_min = prices['low'].rolling(window=self.k_period, min_periods=self.k_period).min()
            high_max = prices['high'].rolling(window=self.k_period, min_periods=self.k_period).max()
            
            # Avoid division by zero by replacing zeros in the denominator.
            denominator = (high_max - low_min).replace(0, 1e-9)
            k = 100 * (prices['close'] - low_min) / denominator
            d = k.rolling(window=self.d_period, min_periods=self.d_period).mean()

            df = pd.DataFrame({
                'close': prices['close'],
                '%k': k,
                '%d': d
            }).dropna()

            if df.empty:
                self.logger.error("No valid indicator data after applying rolling calculations")
                return pd.DataFrame()

            # Generate signals and signal strengths.
            signals = []
            strengths = []
            current_signal = 0  # Tracks current position state.

            for i in range(len(df)):
                if i == 0:
                    # First bar has no previous data; no crossover can be computed.
                    signals.append(0)
                    strengths.append(0.0)
                    continue

                current_k = df['%k'].iloc[i]
                current_d = df['%d'].iloc[i]
                prev_k = df['%k'].iloc[i - 1]
                prev_d = df['%d'].iloc[i - 1]

                buy_cond = (current_k > current_d) and (prev_k <= prev_d) and \
                           (current_k < self.oversold) and (current_d < self.oversold)
                sell_cond = (current_k < current_d) and (prev_k >= prev_d) and \
                            (current_k > self.overbought) and (current_d > self.overbought)

                new_signal = 0
                strength = 0.0

                if buy_cond and current_signal != 1:
                    new_signal = 1
                    min_level = min(current_k, current_d)
                    strength = (self.oversold - min_level) / self.oversold
                    current_signal = 1
                elif sell_cond and current_signal != -1:
                    new_signal = -1
                    max_level = max(current_k, current_d)
                    strength = (max_level - self.overbought) / (100 - self.overbought)
                    current_signal = -1

                # Ensure signal strength is between 0 and 1.
                strength = np.clip(strength, 0.0, 1.0)
                signals.append(new_signal)
                strengths.append(strength)

            df['signal'] = signals
            df['strength'] = strengths

            return df[['close', '%k', '%d', 'signal', 'strength']]

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            raise