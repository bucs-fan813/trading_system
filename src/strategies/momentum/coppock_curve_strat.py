# trading_system/src/strategies/momentum/coppock_curve_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class CoppockCurveStrategy(BaseStrategy):
    """
    Coppock Curve Strategy with Monthly-to-Daily Conversion
    
    Key Fixes:
    - Properly converts monthly parameters to trading days (21 days/month)
    - Validates data availability for full calculation window
    - Production-grade error handling and logging
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self._validate_params()
        self._init_strength_params()

    def _get_validated_prices(self, ticker: str) -> pd.DataFrame:
        """Enhanced data validation with gap checking"""
        min_periods = max(self.roc1_days, self.roc2_days) + self.wma_lookback + 21
        prices = self.get_historical_prices(ticker, lookback=min_periods)
        
        if not self._validate_data(prices, min_periods):
            return pd.DataFrame()
        
        # Check for data continuity
        date_diff = prices.index.to_series().diff().dt.days
        if (date_diff > 5).any():
            self.logger.warning(f"Gaps >5 days detected in {ticker} data")
        
        return prices[['close']].ffill().dropna()

    def _calculate_coppock_curve(self, close: pd.Series) -> pd.Series:
        """
        Robust Coppock calculation with enhanced NA handling and logical correctness.

        This is a revised version of the Coppock Curve calculation method, focusing on:
            - Robustness against NaN values and empty series.
            - Logical correctness in WMA calculation.
            - Maintaining the original function structure as a method.

        Args:
            close: pandas Series representing the closing prices.

        Returns:
            pandas Series representing the Coppock Curve. Returns an empty Series if input is empty or
            if there is insufficient data to calculate the WMA.
        """
        if not isinstance(close, pd.Series):
            raise TypeError("Input 'close' must be a pandas Series.")
        if close.empty:
            return pd.Series(dtype=float)

        # Forward fill up to 5 days to handle minor gaps
        close_filled = close.ffill(limit=5)

        # Calculate Rate of Change (ROC) for two periods
        roc1 = close_filled.pct_change(self.roc1_days).mul(100)
        roc2 = close_filled.pct_change(self.roc2_days).mul(100)

        # Combine ROCs and remove any rows with NaN
        combined_roc = (roc1 + roc2).dropna()

        if len(combined_roc) < self.wma_lookback:
            return pd.Series(dtype=float, index=close.index) # Return empty Series with original index

        # Vectorized WMA calculation with min_periods for robustness
        wma_weights = np.arange(1, self.wma_lookback + 1)
        wma = combined_roc.rolling(window=self.wma_lookback, min_periods=int(self.wma_lookback * 0.8)).apply(
            lambda x: np.dot(x, wma_weights) / wma_weights.sum(),
            raw=True
        )
        return wma

    def _init_strength_params(self):
        """Initialize normalized strength parameters"""
        self.strength_window = int(self.params.get('strength_window', 504))  # 2 years in trading days
        self.normalize_strength = self.params.get('normalize_strength', True)

    def _validate_params(self):
        """Validate and convert strategy parameters"""
        # Default to classic 14-month and 11-month ROC periods
        self.roc1_months = max(1, int(self.params.get('roc1_months', 14)))
        self.roc2_months = max(1, int(self.params.get('roc2_months', 11)))
        
        # Convert months to trading days (21 days/month)
        self.roc1_days = self.roc1_months * 21
        self.roc2_days = self.roc2_months * 21
        
        # Validate WMA window
        self.wma_lookback = max(5, int(self.params.get('wma_lookback', 10)))  # Minimum 5 periods
        
        if self.roc1_days <= self.wma_lookback:
            self.logger.warning("ROC1 period should exceed WMA window for meaningful signals")

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """Generate signals with full validation chain"""
        try:
            prices_df = self._get_validated_prices(ticker)
            if prices_df.empty:
                return pd.DataFrame()

            close = prices_df['close']
            cc = self._calculate_coppock_curve(close)
            
            if cc.empty:
                self.logger.warning(f"Empty Coppock Curve for {ticker}")
                return pd.DataFrame()

            signals_df = self._create_signals_with_strength(close, cc)
            return self._postprocess_signals(signals_df)
            
        except Exception as e:
            self.logger.error(f"Signal generation failed for {ticker}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _create_signals_with_strength(self, close: pd.Series, cc: pd.Series) -> pd.DataFrame:
        """Core signal generation logic"""
        df = pd.DataFrame({'close': close, 'cc': cc}).dropna()
        if df.empty:
            return df

        # Boolean masks for signal conditions
        with np.errstate(invalid='ignore'):
            was_negative = self._get_rolling_condition(df['cc'], 'negative')
            was_positive = self._get_rolling_condition(df['cc'], 'positive')

        # Calculate signal strength
        df['strength_raw'] = self._calculate_strength(df['cc'], was_negative, was_positive)
        
        # Generate signals
        df['signal'] = np.select(
            [
                was_negative & (df['cc'] > 0),  # Buy signal condition
                was_positive & (df['cc'] < 0)   # Sell signal condition
            ],
            [1, -1],
            default=0
        )
        
        df['position'] = (
                df['signal']
                .replace(0, np.nan)
                .ffill()
                .fillna(0)
                .astype(int)
        )

        return df[['close', 'cc', 'signal', 'position', 'strength_raw']]

    def _get_rolling_condition(self, series: pd.Series, condition_type: str) -> pd.Series:
        """Create boolean masks for signal triggers"""
        window = 4  # Classic Coppock confirmation period
        if condition_type == 'negative':
            mask = (series < 0).rolling(window).apply(lambda x: np.all(x), raw=True)
        elif condition_type == 'positive':
            mask = (series > 0).rolling(window).apply(lambda x: np.all(x), raw=True)
        else:
            raise ValueError(f"Invalid condition type: {condition_type}")
        
        return (mask.fillna(False).astype(bool).shift().fillna(False))

    def _calculate_strength(self, cc: pd.Series, neg_mask: pd.Series, pos_mask: pd.Series) -> pd.Series:
        """Calculate momentum strength relative to recent history"""
        strength = pd.Series(0.0, index=cc.index)
        
        # Negative regime strength
        neg_ref = cc.rolling(21).mean().shift().where(neg_mask)
        strength[neg_mask] = (cc - neg_ref) / neg_ref.abs()
        
        # Positive regime strength
        pos_ref = cc.rolling(21).mean().shift().where(pos_mask)
        strength[pos_mask] = (cc - pos_ref) / pos_ref.abs()
        
        return strength.replace([np.inf, -np.inf], 0.0)

    def _postprocess_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final signal processing and normalization"""
        if df.empty:
            return df
            
        # Normalize strength scores
        if self.normalize_strength:
            df['strength'] = self._normalize_to_range(df['strength_raw'], self.strength_window)
        else:
            df['strength'] = df['strength_raw']
            
        # Clean final output
        return df[['close', 'cc', 'signal', 'position', 'strength']]

    def _normalize_to_range(self, series: pd.Series, window: int) -> pd.Series:
        """Normalize values to [-1, 1] range over lookback window"""
        rolling_min = series.rolling(window).min()
        rolling_max = series.rolling(window).max()
        return 2 * ((series - rolling_min) / (rolling_max - rolling_min).replace(0, 1)) - 1

    def __repr__(self):
        return (f"CoppockCurveStrategy(roc1={self.roc1_months} months, "
                f"roc2={self.roc2_months} months, wma={self.wma_lookback})")