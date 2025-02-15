# trading_system/src/strategies/kst_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

class KSTStrategy(BaseStrategy):
    """
    Know Sure Thing (KST) momentum strategy implementation.
    
    Hyperparameters:
        roc_periods: List of ROC lookback periods [roc1, roc2, roc3, roc4]
        sma_periods: List of SMA periods for smoothing ROC values [sma1, sma2, sma3, sma4]
        signal_period: Period for signal line SMA
        kst_weights: Weighting factors for each ROC component [w1, w2, w3, w4]
        
    Recommended parameters for daily pricing data (if not provided, defaults will be used):
        roc_periods  = [10, 15, 20, 30]
        sma_periods  = [10, 10, 10, 15]
        signal_period = 9
        kst_weights  = [1, 2, 3, 4]
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        # Default parameters for daily pricing data.
        default_params = {
            'roc_periods': [10, 15, 20, 30],
            'sma_periods': [10, 10, 10, 15],
            'signal_period': 9,
            'kst_weights': [1, 2, 3, 4]
        }
        # Use provided params or fall back to defaults.
        if params is None:
            params = default_params
        else:
            # Merge provided params with defaults for any missing keys.
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value

        super().__init__(db_config, params)
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate strategy parameters"""
        required_params = ['roc_periods', 'sma_periods', 'signal_period', 'kst_weights']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
                
        if len(self.params['roc_periods']) != 4 or len(self.params['sma_periods']) != 4:
            raise ValueError("roc_periods and sma_periods must contain exactly 4 values")
            
        if any(p <= 0 for p in self.params['roc_periods'] + self.params['sma_periods'] + [self.params['signal_period']]):
            raise ValueError("All periods must be positive integers")
    
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate KST trading signals.
        
        Returns a DataFrame with columns:
            date, close, kst, signal_line, signal, signal_strength
        """
        try:
            # Determine a lookback period sufficient for calculating all rolling averages.
            lookback = max(self.params['roc_periods']) + max(self.params['sma_periods']) + self.params['signal_period']
            prices = self.get_historical_prices(ticker, lookback=lookback)
            
            if not self._validate_data(prices, min_records=lookback):
                return pd.DataFrame()
                
            # Ensure prices are sorted in chronological order.
            close = prices['close'].sort_index(ascending=True)
            
            # Calculate KST components.
            roc_periods = self.params['roc_periods']
            sma_periods = self.params['sma_periods']
            weights = self.params['kst_weights']
            
            kst_components = []
            for roc_len, sma_len, weight in zip(roc_periods, sma_periods, weights):
                roc = self._calculate_roc(close, roc_len)
                smoothed = roc.rolling(sma_len, min_periods=1).mean()
                kst_components.append(smoothed * weight)
                
            kst = sum(kst_components)
            signal_line = kst.rolling(self.params['signal_period'], min_periods=1).mean()
            
            # Generate crossover signals.
            signals = self._generate_crossover_signals(kst, signal_line, close)
            signals['signal_strength'] = kst - signal_line
            
            return signals[['close', 'kst', 'signal_line', 'signal', 'signal_strength']].reset_index()
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            raise
            
    def _calculate_roc(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change (ROC) as a percentage change"""
        return series.pct_change(periods=period) * 100
    
    def _generate_crossover_signals(self, kst: pd.Series, signal_line: pd.Series, close: pd.Series) -> pd.DataFrame:
        """
        Generate crossover signals using vectorized operations.
        
        Returns a DataFrame with columns:
            date, close, kst, signal_line, signal
        """
        signals = pd.DataFrame(index=kst.index)
        signals['kst'] = kst
        signals['signal_line'] = signal_line
        
        # Include the close prices.
        signals['close'] = close
        
        # Initialize signal column to neutral (0).
        signals['signal'] = 0
        
        # Identify crossovers.
        above = kst > signal_line
        below = kst < signal_line
        
        # Bullish crossover: KST crosses from below to above.
        golden_cross = above & (~above.shift(1, fill_value=False))
        signals.loc[golden_cross, 'signal'] = 1
        
        # Bearish crossover: KST crosses from above to below.
        death_cross = below & (~below.shift(1, fill_value=False))
        signals.loc[death_cross, 'signal'] = -1
        
        # Forward fill close prices in case of missing data.
        signals['close'] = signals['close'].ffill()
        
        return signals