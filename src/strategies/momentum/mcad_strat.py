# trading_system/src/strategies/macd_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) trading strategy.
    
    Hyperparameters:
        slow (int): Slow EMA period (default: 26)
        fast (int): Fast EMA period (default: 12)
        smooth (int): Signal line EMA period (default: 9)
        lookback (int): Historical data window (default: 252)
        long_only (bool): Flag to restrict trading to long positions only (default: True)


    Long-Only Approach Confirmation:
    In your _calculate_positions method, any bearish signal (i.e. -1) is 
    replaced with 0 so that you exit a long position rather than taking a 
    short position. This is acceptable if you intend to trade only long 
    positions, but it’s something to be aware of if you ever plan to support 
    shorting.
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.logger = logging.getLogger(__name__)
        
        # Set default parameters
        self.params.setdefault('slow', 26)
        self.params.setdefault('fast', 12)
        self.params.setdefault('smooth', 9)
        self.params.setdefault('lookback', 252)
        self.params.setdefault('long_only', True)
        
        # Validate parameters
        if self.params['slow'] <= self.params['fast']:
            raise ValueError("Slow period must be greater than fast period")
        if any(not isinstance(p, int) for p in self.params.values()):
            raise ValueError("All MACD parameters must be integers")

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """Generate MACD signals and positions for the given ticker."""
        try:
            # Retrieve data
            data = self.get_historical_prices(
                ticker, 
                lookback=self.params['lookback']
            )
            
            # Validate data adequacy
            min_records = self.params['slow'] + self.params['smooth']
            if not self._validate_data(data, min_records):
                self.logger.error(f"Insufficient data for {ticker}")
                return pd.DataFrame()

            # Calculate MACD components
            close = data['close']
            macd_line, signal_line, histogram = self._calculate_macd(close)
            
            # Generate signals
            signals = self._generate_crossover_signals(macd_line, signal_line)
            
            # Calculate positions
            signals = self._calculate_positions(signals)
            
            # Build output DataFrame
            return pd.DataFrame({
                'close': close,
                'macd': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'signal': signals,
                'signal_strength': histogram  # Histogram as strength metric
            }).dropna()
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            raise

    def _calculate_macd(self, close: pd.Series) -> tuple:
        """Calculate MACD line, signal line, and histogram."""
        fast_ema = close.ewm(span=self.params['fast'], adjust=False).mean()
        slow_ema = close.ewm(span=self.params['slow'], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.params['smooth'], adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _generate_crossover_signals(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """Generate crossover signals using vectorized operations."""
        signals = pd.Series(0, index=macd.index)
        
        # Boolean series for crossovers
        above = macd > signal
        below = macd < signal
        
        # Shift the series and fill NaN with False to avoid type errors
        prev_above = above.shift(1).fillna(False)
        prev_below = below.shift(1).fillna(False)
        
        # Bullish crossover: MACD crosses above signal when it was not above in the previous period
        signals[(above & ~prev_above)] = 1
        # Bearish crossover: MACD crosses below signal when it was not below in the previous period
        signals[(below & ~prev_below)] = -1
        
        return signals

    def _calculate_positions(self, signals: pd.Series) -> pd.Series:
        """Calculate positions using signal persistence."""
        positions = signals.replace(0, method='ffill').fillna(0)
        if self.long_only:
            positions[positions == -1] = 0  # Convert sells to exit signals
        else:
            pass  # No changes needed for long/short positions
        return positions.astype(int)