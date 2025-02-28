# trading_system/src/strategies/bollinger_bands_strat.py

"""
Bollinger Bands Strategy

This module implements a Bollinger Bands strategy that uses a mean-reversion logic
combined with risk management. It retrieves historical price data from the database,
computes a rolling simple moving average (SMA) and standard deviation over a defined window,
and derives upper and lower Bollinger Bands.

Indicator calculations:
    - SMA_t: Moving average over the specified window.
    - STD_t: Rolling standard deviation.
    - Upper Band = SMA_t + (std_dev × STD_t)
    - Lower Band = SMA_t - (std_dev × STD_t)
    - %B indicator = (Price - Lower Band) / (Upper Band - Lower Band)

Trading signals:
    - Long entry (signal = 1): Triggered when yesterday’s close was above the lower band and today’s close falls below the lower band.
    - Short entry (signal = -1): Triggered when yesterday’s close was below the upper band and today’s close rises above the upper band.
    
Risk management (via the RiskManager) applies stop loss and take profit rules
with adjustments for slippage and transaction cost:
    For a long trade:
        Stop loss = entry_price × (1 - stop_loss_pct)
        Target    = entry_price × (1 + take_profit_pct)
    For a short trade:
        Stop loss = entry_price × (1 + stop_loss_pct)
        Target    = entry_price × (1 - take_profit_pct)
    Realized returns are computed accordingly.

All operations (rolling indicators, signal generation, risk management) are vectorized to optimize execution speed.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager
import logging

class BollingerBandsStrategy(BaseStrategy):
    """
    BollingerBandsStrategy implements a mean-reversion trading strategy based on Bollinger Bands.
    
    The strategy computes a rolling simple moving average (SMA) and standard deviation over a specified window,
    and derives the upper and lower Bollinger Bands:
    
        Upper Band = SMA + (std_dev × standard deviation)
        Lower Band = SMA - (std_dev × standard deviation)
    
    It then calculates the %B indicator, defined as:
    
        %B = (Price - Lower Band) / (Upper Band - Lower Band)
    
    Trading signals are generated as follows:
      - A long entry (signal = 1) is triggered when the previous close was above the lower band and the current close 
        falls below the lower band, suggesting an oversold condition.
      - A short entry (signal = -1) is triggered when the previous close was below the upper band and the current close 
        moves above the upper band, suggesting an overbought condition.
      
    Subsequent duplicate signals are filtered out to avoid repeated entries. Risk management is then applied to adjust
    the signals by incorporating stop loss, take profit, slippage, and transaction cost considerations.
    
    Hyperparameters:
        window (int): Lookback period for SMA and standard deviation (default: 20).
        std_dev (float): Multiplier for standard deviation to set Bollinger Bands (default: 2.0).
        long_only (bool): Flag to allow only long trades (default: True).
        stop_loss_pct (float): Stop loss percentage for risk management (default: 0.05).
        take_profit_pct (float): Take profit percentage for risk management (default: 0.10).
        slippage_pct (float): Slippage percentage for execution price adjustments (default: 0.001).
        transaction_cost_pct (float): Transaction cost percentage per trade (default: 0.001).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the BollingerBandsStrategy with database configuration and strategy parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Dictionary of strategy parameters.
        """
        super().__init__(db_config, params)
        # Strategy parameters.
        self.window = self.params.get('window', 20)
        self.std_dev = self.params.get('std_dev', 2.0)
        self.long_only = self.params.get('long_only', True)
        
        # Risk management parameters.
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.10)
        self.slippage_pct = self.params.get('slippage_pct', 0.001)
        self.transaction_cost_pct = self.params.get('transaction_cost_pct', 0.001)

    def generate_signals(self, 
                         ticker: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals and apply risk management for the given ticker.
        
        The method retrieves historical price data from the database, computes the Bollinger Bands and %B indicator, 
        and then determines entry signals when price crosses the Bollinger Bands. Risk management is applied to 
        adjust trading signals with stop loss and take profit rules, including slippage and transaction cost impacts.
        
        Mathematical summary:
          - SMA_t = Moving Average(close, window)
          - STD_t = Standard Deviation(close, window)
          - Upper Band = SMA_t + (std_dev × STD_t)
          - Lower Band = SMA_t - (std_dev × STD_t)
          - %B = (close - Lower Band) / (Upper Band - Lower Band)
          - Long signal: if close[t-1] > LowerBand[t-1] and close[t] < LowerBand[t]
          - Short signal: if close[t-1] < UpperBand[t-1] and close[t] > UpperBand[t]
          
        Risk Management:
          - For a long trade, stop loss = entry_price × (1 - stop_loss_pct) and target = entry_price × (1 + take_profit_pct).
          - For a short trade, stop loss = entry_price × (1 + stop_loss_pct) and target = entry_price × (1 - take_profit_pct).
          - Trade return is computed based on the adjusted exit price with considerations for slippage and transaction costs.
        
        Args:
            ticker (str): Stock symbol.
            start_date (str, optional): Start date (YYYY-MM-DD) for backtesting.
            end_date (str, optional): End date (YYYY-MM-DD) for backtesting.
            initial_position (int): Starting position (1 for long, -1 for short, 0 for neutral).
            latest_only (bool): If True, return only the signal for the most recent date.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, computed indicators, generated signals, and risk-managed 
            performance metrics. Columns include 'close', 'high', 'low', 'signal', 'signal_strength', 'upper_bb', 
            'lower_bb', 'sma', 'position', 'return', 'cumulative_return', and 'exit_type'.
        """
        # Retrieve historical price data for the ticker with optional date filtering.
        prices = self.get_historical_prices(
            ticker,
            from_date=start_date,
            to_date=end_date,
            lookback=self.window * 2 if not (start_date or end_date) else None
        )

        # Validate that sufficient price data is available.
        if not self._validate_data(prices, min_records=self.window + 1):
            self.logger.error("Not enough data for computing indicators")
            return pd.DataFrame()

        # Compute Bollinger Bands indicators.
        close = prices['close']
        sma = close.rolling(self.window, min_periods=self.window).mean()
        std = close.rolling(self.window, min_periods=self.window).std()
        upper_bb = sma + std * self.std_dev
        lower_bb = sma - std * self.std_dev

        # Compute %B indicator to gauge the price position relative to the bands.
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_b = (close - lower_bb) / (upper_bb - lower_bb)
        percent_b = percent_b.replace([np.inf, -np.inf], np.nan)

        # Generate raw trading signals using look-ahead prevention.
        # Long signal: price crosses below lower band; Short signal: price crosses above upper band.
        crossed_below = (close.shift(1) > lower_bb.shift(1)) & (close < lower_bb)
        crossed_above = (close.shift(1) < upper_bb.shift(1)) & (close > upper_bb)
        raw_signals = pd.Series(0, index=close.index)
        raw_signals[crossed_below] = 1
        raw_signals[crossed_above] = -1

        # Filter out consecutive duplicate signals to ensure distinct trade entries.
        prev_signals = raw_signals.replace(0, np.nan).ffill().shift(1).fillna(0)
        signals = raw_signals.where(raw_signals != prev_signals, 0)

        if self.long_only:
            signals[signals == -1] = 0  # Filter out short signals if long_only is enabled.

        # Construct a DataFrame with computed indicators and raw signals.
        signals_df = pd.DataFrame({
            'close': close,
            'high': prices['high'],
            'low': prices['low'],
            'signal': signals,
            'signal_strength': percent_b,
            'upper_bb': upper_bb,
            'lower_bb': lower_bb,
            'sma': sma
        }).dropna()

        if signals_df.empty:
            self.logger.warning("Signals DataFrame is empty after indicator computation")
            return pd.DataFrame()

        # Apply risk management to integrate stop-loss, take-profit, slippage, and transaction costs.
        risk_manager = RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            slippage_pct=self.slippage_pct,
            transaction_cost_pct=self.transaction_cost_pct
        )
        results = risk_manager.apply(signals_df, initial_position)

        # If only the latest signal is desired, return the last row.
        return results.tail(1) if latest_only else results

    def _validate_data(self, df: pd.DataFrame, min_records: int = 1) -> bool:
        """
        Validate that the DataFrame contains sufficient and valid price data for the strategy.
        
        Args:
            df (pd.DataFrame): Price data DataFrame.
            min_records (int): Minimum number of records required.
        
        Returns:
            bool: True if data is sufficient and contains no critical NaN values, False otherwise.
        """
        if df.empty or len(df) < min_records:
            self.logger.error(f"Insufficient data: found {len(df)} records, required {min_records}")
            return False
        if df['close'].isnull().sum() > 0:
            self.logger.warning("Price data contains NaN values in 'close' column")
            return False
        return True