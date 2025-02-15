# trading_system/src/strategies/cci_oscillator_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

class CCIStrategy(BaseStrategy):
    """
    Commodity Channel Index (CCI) trading strategy with corrected signal logic and robustness improvements.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        default_params = {
            'cci_period': 20,
            'cci_upper_band': 150,
            'cci_lower_band': -150,
            'lookback_period': 252,
            'data_source': 'yfinance'
        }
        updated_params = default_params.copy()
        if params:
            updated_params.update(params)

        # Add parameter validation
        if updated_params['cci_upper_band'] <= updated_params['cci_lower_band']:
            raise ValueError("Upper band must be greater than lower band")
        if updated_params['cci_period'] < 1:
            raise ValueError("CCI period must be >= 1")

        super().__init__(db_config, params=updated_params)
        self.cci_period = int(self.params['cci_period'])
        self.cci_upper_band = int(self.params['cci_upper_band'])
        self.cci_lower_band = int(self.params['cci_lower_band'])
        self.lookback_period = int(self.params['lookback_period'])
        self.data_source = str(self.params['data_source'])

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        try:
            historical_data = self.get_historical_prices(
                ticker, lookback=self.lookback_period, data_source=self.data_source
            )

            # Require 2*period for stable calculations
            if not self._validate_data(historical_data, min_records=2*self.cci_period):
                return pd.DataFrame()

            # CCI Calculation
            typical_price = (historical_data['high'] + historical_data['low'] + historical_data['close']) / 3
            sma_tp = typical_price.rolling(window=self.cci_period, min_periods=self.cci_period).mean()

            mean_deviation = typical_price.rolling(window=self.cci_period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            mean_deviation = mean_deviation.replace(0, np.nan)  # Prevent division by zero

            cci_indicator = (typical_price - sma_tp) / (0.015 * mean_deviation)
            cci_indicator = cci_indicator.rename('cci')

            signals = pd.DataFrame(index=historical_data.index)
            signals['close'] = historical_data['close']
            signals['cci'] = cci_indicator

            # Corrected signal conditions
            cci_shifted = signals['cci'].shift(1)
            buy_condition = (cci_shifted < self.cci_lower_band) & (signals['cci'] > self.cci_lower_band)
            sell_condition = (cci_shifted > self.cci_upper_band) & (signals['cci'] < self.cci_upper_band)
            neutral_condition = ~buy_condition & ~sell_condition

            # Signal generation
            signals['signal'] = 0
            signals.loc[buy_condition, 'signal'] = 1
            signals.loc[sell_condition, 'signal'] = -1

            # Signal strength calculation
            signals['signal_strength'] = 0.0
            signals.loc[buy_condition, 'signal_strength'] = signals['cci'] - self.cci_lower_band
            signals.loc[sell_condition, 'signal_strength'] = self.cci_upper_band - signals['cci']

            # --- REMOVE INCORRECT "Cleanup consecutive signals" SECTION COMPLETELY ---

            return signals[['close', 'cci', 'signal', 'signal_strength']].copy()

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            return pd.DataFrame()