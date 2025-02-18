# trading_system/src/strategies/tsi_strategy.py

#TODO: TEST THIS CODE

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class TSIStrategy(BaseStrategy):
    """
    True Strength Index (TSI) strategy implementation
    
    Hyperparameters:
        long_period: First smoothing period (default: 25)
        short_period: Second smoothing period (default: 13)
        signal_period: Signal line period (default: 12)
        min_data_points: Minimum required data points (default: 100)
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        
        # Set hyperparameters with validation
        self.long_period = int(self.params.get('long_period', 25))
        self.short_period = int(self.params.get('short_period', 13))
        self.signal_period = int(self.params.get('signal_period', 12))
        self.min_data_points = int(self.params.get('min_data_points', 100))
        
        if any([p <= 0 for p in [self.long_period, self.short_period, self.signal_period]]):
            raise ValueError("All periods must be positive integers")

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate TSI trading signals
        
        Returns DataFrame with columns:
            - date, tsi, signal_line, signal, close, strength
        """
        self.logger.info(f"Generating TSI signals for {ticker}")
        
        try:
            # Retrieve price data with sufficient lookback
            prices = self.get_historical_prices(
                ticker, 
                lookback=self.min_data_points
            )
            
            if not self._validate_data(prices, self.min_data_points):
                return pd.DataFrame()
                
            close_prices = prices['close'].sort_index()
            
            # Calculate TSI components
            tsi, signal_line = self._calculate_tsi(close_prices)
            
            # Generate signals and calculate strength
            signals = self._generate_tsi_signals(tsi, signal_line)
            signals['strength'] = tsi - signal_line
            
            # Add the close price, so you know the price level when the signal was valid
            signals['close'] = close_prices
            
            return signals.reset_index().dropna()
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            raise

    def _calculate_tsi(self, close: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Core TSI calculation logic"""
        diff = close.diff()
        abs_diff = diff.abs()
        
        # First smoothing
        diff_smoothed = diff.ewm(span=self.long_period, adjust=False).mean()
        abs_smoothed = abs_diff.ewm(span=self.long_period, adjust=False).mean()
        
        # Second smoothing
        diff_double = diff_smoothed.ewm(span=self.short_period, adjust=False).mean()
        abs_double = abs_smoothed.ewm(span=self.short_period, adjust=False).mean()
        
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        tsi = (diff_double / (abs_double + epsilon)) * 100
        signal_line = tsi.ewm(span=self.signal_period, adjust=False).mean()
        
        return tsi, signal_line

    def _generate_tsi_signals(self, tsi: pd.Series, signal_line: pd.Series) -> pd.DataFrame:
        """Vectorized signal generation"""
        signals = pd.DataFrame(index=tsi.index)
        signals['tsi'] = tsi
        signals['signal_line'] = signal_line
        
        # Detect crossovers:
        above = tsi > signal_line
        below = tsi < signal_line
        
        # Generate signals using vectorized operations:
        signals['signal'] = 0
        signals['signal'] = np.where(above & below.shift(1), 1, signals['signal'])
        signals['signal'] = np.where(below & above.shift(1), -1, signals['signal'])
        
        return signals[['tsi', 'signal_line', 'signal']]

    def __repr__(self):
        return (f"TSIStrategy(long={self.long_period}, short={self.short_period}, "
                f"signal={self.signal_period})")