# trading_system/src/strategies/supertrend_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class SupertrendStrategy(BaseStrategy):
    """
    Enhanced Supertrend strategy with vectorized operations and risk management.

    Hyperparameters:
        lookback (int): Period for ATR calculation. Default: 10
        multiplier (float): Multiplier for ATR. Default: 3.0
        long_only (bool): If True, only long positions are allowed. Default: True
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        default_params = {'lookback': 10, 'multiplier': 3.0}
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_manager = RiskManager()  # Initialize with default params
        self.long_only = bool(self.params.get('long_only', True))

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate Supertrend signals with vectorized operations and risk management.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            initial_position: Initial position for backtesting
            latest_only: If True, returns only the latest signal

        Returns:
            DataFrame with signals, prices, and performance metrics
        """
        # Data retrieval with optimized query
        data = self._get_price_data(ticker, start_date, end_date, latest_only)
        if data.empty:
            return pd.DataFrame()

        # Vectorized indicator calculation
        data = self._calculate_supertrend(data)
        signals = self._generate_trading_signals(data)
        
        # Apply risk management
        signals = self._apply_risk_management(signals, initial_position)
        
        return signals.tail(1) if latest_only else signals

    def _get_price_data(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        latest_only: bool
    ) -> pd.DataFrame:
        """Optimized data retrieval with minimal required history."""
        lookback = self.params['lookback']
        min_records = max(lookback * 3, 252)  # Ensure sufficient history
        
        try:
            return self.get_historical_prices(
                ticker=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_records if latest_only else None
            )
        except DataRetrievalError as e:
            self.logger.error(f"Data retrieval failed: {e}")
            return pd.DataFrame()

    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized calculation of Supertrend indicators."""
        lookback = self.params['lookback']
        multiplier = self.params['multiplier']
        
        # Calculate True Range and ATR
        hl = data['high'] - data['low']
        hc = (data['high'] - data['close'].shift(1)).abs()
        lc = (data['low'] - data['close'].shift(1)).abs()
        data['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        data['atr'] = data['tr'].ewm(span=lookback, adjust=False).mean()

        # Calculate bands
        hl2 = (data['high'] + data['low']) / 2
        data['basic_upper'] = hl2 + multiplier * data['atr']
        data['basic_lower'] = hl2 - multiplier * data['atr']

        # Vectorized final bands calculation
        data['final_upper'] = self._calculate_final_band(data, 'basic_upper')
        data['final_lower'] = self._calculate_final_band(data, 'basic_lower')

        # Vectorized supertrend calculation
        data['supertrend'], data['trend'] = self._calculate_supertrend_trend(data)
        return data

    def _calculate_final_band(self, data: pd.DataFrame, band_col: str) -> pd.Series:
        """Vectorized calculation of final bands."""
        condition = (data[band_col] < data[f'final_{"lower" if "lower" in band_col else "upper"}'].shift(1)) | \
                    (data['close'].shift(1) > data['final_upper'].shift(1)) if "upper" in band_col else \
                    (data['close'].shift(1) < data['final_lower'].shift(1))
        
        return np.where(condition, data[band_col], data[f'final_{"lower" if "lower" in band_col else "upper"}'].shift(1)).cummin() if "upper" in band_col else \
               np.where(condition, data[band_col], data[f'final_{"lower" if "lower" in band_col else "upper"}'].shift(1)).cummax()

    def _calculate_supertrend_trend(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Vectorized calculation of Supertrend and trend direction."""
        # Initialize arrays
        supertrend = np.zeros(len(data))
        trend = np.zeros(len(data))
        
        # Initial condition
        supertrend[0] = data['final_upper'].iloc[0] if data['close'].iloc[0] <= data['final_upper'].iloc[0] else data['final_lower'].iloc[0]
        trend[0] = -1 if data['close'].iloc[0] <= data['final_upper'].iloc[0] else 1

        # Vectorized calculation
        for i in range(1, len(data)):
            prev_trend = trend[i-1]
            current_close = data['close'].iloc[i]
            
            if prev_trend == -1:
                if current_close <= data['final_upper'].iloc[i]:
                    supertrend[i] = data['final_upper'].iloc[i]
                    trend[i] = -1
                else:
                    supertrend[i] = data['final_lower'].iloc[i]
                    trend[i] = 1
            else:
                if current_close >= data['final_lower'].iloc[i]:
                    supertrend[i] = data['final_lower'].iloc[i]
                    trend[i] = 1
                else:
                    supertrend[i] = data['final_upper'].iloc[i]
                    trend[i] = -1

        return pd.Series(supertrend, index=data.index), pd.Series(trend, index=data.index)

    def _generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vectorized trading signals."""
        # Vectorized crossover detection
        cross_above = (data['close'].shift(1) < data['supertrend'].shift(1)) & (data['close'] > data['supertrend'])
        cross_below = (data['close'].shift(1) > data['supertrend'].shift(1)) & (data['close'] < data['supertrend'])
        
        data['signal'] = np.select(
            [cross_above, cross_below],
            [1, -1],
            default=0
        )
        
        # Calculate signal strength
        data['signal_strength'] = ((data['close'] - data['supertrend']).abs() / data['close']) * data['signal']
        
        if self.long_only:
            data['signal'] = data['signal'].clip(lower=0)  # Clip negative signals for long-only strategy

        return data[['open', 'high', 'low', 'close', 'supertrend', 'signal', 'signal_strength']]

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """Apply risk management rules to signals."""
        if not signals.empty:
            return self.risk_manager.apply(
                signals,
                initial_position=initial_position
            )
        return pd.DataFrame()

    def _validate_parameters(self):
        """Validate strategy parameters."""
        if self.params['lookback'] <= 1:
            raise ValueError("Lookback period must be greater than 1")
        if self.params['multiplier'] <= 0:
            raise ValueError("Multiplier must be positive")