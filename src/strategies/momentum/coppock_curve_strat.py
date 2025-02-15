# trading_system/src/strategies/momentum/coppock_curve_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError

class CoppockCurveStrategy(BaseStrategy):
    """
    Coppock Curve Strategy with enhanced data validation, signal alignment,
    and production-grade parameter checking.

    This strategy:
      - Converts ROC parameters from months to trading days (assumes 21 trading days/month).
      - Retrieves historical prices from the DB, ensuring chronological order and proper index conversion.
      - Calculates two ROC values, sums them, and then smooths the series with a rolling weighted moving average (WMA).
      - Generates buy/sell signals when the Coppock Curve crosses zero following confirmed periods of negativity or positivity.
      - Computes momentum strength that can be normalized before output.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self._validate_params()
        self._init_strength_params()

    def _validate_params(self):
        """Validate and convert strategy parameters to trading days."""
        self.roc1_months = max(1, int(self.params.get('roc1_months', 14)))
        self.roc2_months = max(1, int(self.params.get('roc2_months', 11)))
        self.roc1_days = self.roc1_months * 21
        self.roc2_days = self.roc2_months * 21

        # Ensure the WMA lookback window is at least 5 periods.
        self.wma_lookback = max(5, int(self.params.get('wma_lookback', 10)))
        if min(self.roc1_days, self.roc2_days) <= self.wma_lookback:
            self.logger.warning("ROC periods should exceed the WMA window for meaningful signals.")

    def _init_strength_params(self):
        """Initialize parameters for momentum strength calculation."""
        self.strength_window = int(self.params.get('strength_window', 504))  # Approximately 2 years of trading days
        self.normalize_strength = self.params.get('normalize_strength', True)

    def _get_validated_prices(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve and validate price data ensuring sufficient lookback.
        
        Returns:
            DataFrame with 'close' prices. Also attempts to convert the index
            to a datetime type if not already done.
        """
        min_periods = max(self.roc1_days, self.roc2_days) + self.wma_lookback + 21
        prices = self.get_historical_prices(ticker, lookback=min_periods)

        if not self._validate_data(prices, min_periods):
            self.logger.warning(f"Not enough data for ticker {ticker}. Required: {min_periods}, got: {len(prices)}")
            return pd.DataFrame()

        # Optional: Ensure the index is datetime (avoids subtraction errors later)
        if not np.issubdtype(prices.index.dtype, np.datetime64):
            try:
                prices.index = pd.to_datetime(prices.index)
            except Exception as e:
                self.logger.error(f"Failed to convert index to datetime for {ticker}: {e}")
                return pd.DataFrame()

        # The base class sorts the data, so here we simply forward-fill minor gaps.
        return prices[['close']].ffill().dropna()

    def _calculate_coppock_curve(self, close: pd.Series) -> pd.Series:
        """
        Compute the Coppock Curve using two rate-of-change (ROC) measures and a WMA smoothing.

        Steps:
          1. Compute ROC for two different periods (converted to days).
          2. Sum these ROC series.
          3. Smooth the combined ROC using a rolling weighted moving average (WMA)
             with dynamic weights based on the available window length.
        
        Returns:
            A Series representing the Coppock Curve.
        """
        roc1 = close.pct_change(self.roc1_days).mul(100)
        roc2 = close.pct_change(self.roc2_days).mul(100)
        combined_roc = (roc1 + roc2).dropna()

        if len(combined_roc) < self.wma_lookback:
            return pd.Series(dtype=float, index=close.index)

        def rolling_wma(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()

        wma = combined_roc.rolling(
            window=self.wma_lookback,
            min_periods=int(self.wma_lookback * 0.8)
        ).apply(rolling_wma, raw=True)
        return wma

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate trading signals based on the Coppock Curve indicator.

        Process:
          - Retrieve and validate historical close price data.
          - Calculate the Coppock Curve.
          - Generate raw signals by detecting a change in regime (from 
            prolonged negative to positive and vice versa) using a shifted
            rolling window check.
          - Compute and optionally normalize momentum strength.
        
        Returns:
            DataFrame with columns: ['close', 'cc', 'signal', 'position', 'strength'].
        """
        try:
            prices_df = self._get_validated_prices(ticker)
            if prices_df.empty:
                return pd.DataFrame()

            cc = self._calculate_coppock_curve(prices_df['close'])
            if cc.empty:
                self.logger.warning(f"Coppock Curve computation returned empty for {ticker}.")
                return pd.DataFrame()

            signals_df = self._create_signals_with_strength(prices_df['close'], cc)
            return self._postprocess_signals(signals_df)
        except Exception as e:
            self.logger.error(f"Signal generation failed for {ticker}: {e}", exc_info=True)
            return pd.DataFrame()

    def _create_signals_with_strength(self, close: pd.Series, cc: pd.Series) -> pd.DataFrame:
        """
        Generate the raw buy/sell signals along with momentum strength.

        Key steps:
          - Create a DataFrame combining 'close' and 'cc'.
          - Generate boolean masks by checking if the Coppock Curve was
            consistently negative or positive over a 4-day rolling window,
            using a shift(1) to reference prior periods.
          - Use np.select to produce a buy signal (1) if the previous period
            was negative and today crosses above zero, or a sell signal (-1)
            if the previous period was positive and today crosses below zero.
          - Forward-fill positions based on the last recorded non-zero signal.
          - Calculate a raw strength measure relative to a rolling 21-day mean.
        """
        df = pd.DataFrame({'close': close, 'cc': cc}).dropna()
        if df.empty:
            return df

        # Create rolling condition masks with proper shift to check the previous period.
        was_negative = (
            df['cc']
            .shift(1)
            .rolling(4, min_periods=4)
            .apply(lambda x: np.all(x < 0), raw=True)
            .fillna(0)
            .astype(bool)
        )
        was_positive = (
            df['cc']
            .shift(1)
            .rolling(4, min_periods=4)
            .apply(lambda x: np.all(x > 0), raw=True)
            .fillna(0)
            .astype(bool)
        )

        df['signal'] = np.select(
            [was_negative & (df['cc'] > 0), was_positive & (df['cc'] < 0)],
            [1, -1],
            default=0
        )
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)

        # Calculate momentum strength relative to a 21-day rolling mean reference.
        df['strength_raw'] = self._calculate_strength(df['cc'], was_negative, was_positive)
        return df

    def _calculate_strength(self, cc: pd.Series, neg_mask: pd.Series, pos_mask: pd.Series) -> pd.Series:
        """
        Calculate momentum strength relative to a shifted 21-day rolling mean.

        For both negative and positive regimes the reference is given by the same rolling average.
        The strength is computed as the relative deviation of the Coppock Curve from this reference.
        """
        strength = pd.Series(0.0, index=cc.index)
        ref = cc.rolling(21, min_periods=1).mean().shift()
        strength[neg_mask] = (cc - ref) / ref.abs()
        strength[pos_mask] = (cc - ref) / ref.abs()
        return strength.replace([np.inf, -np.inf], 0.0)

    def _postprocess_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize signal data by normalizing momentum strength if required and selecting columns.
        """
        if self.normalize_strength:
            df['strength'] = self._normalize_to_range(df['strength_raw'], self.strength_window)
        else:
            df['strength'] = df['strength_raw']
        return df[['close', 'cc', 'signal', 'position', 'strength']]

    def _normalize_to_range(self, series: pd.Series, window: int) -> pd.Series:
        """
        Normalize a series to the range [-1, 1] using a rolling window.

        This normalization computes the rolling minimum and maximum,
        then rescales the values accordingly.
        """
        rolling_min = series.rolling(window, min_periods=1).min()
        rolling_max = series.rolling(window, min_periods=1).max()
        denom = (rolling_max - rolling_min).replace(0, 1)
        return 2 * ((series - rolling_min) / denom) - 1

    def __repr__(self):
        return (f"CoppockCurveStrategy(roc1={self.roc1_months}mo, "
                f"roc2={self.roc2_months}mo, wma={self.wma_lookback})")