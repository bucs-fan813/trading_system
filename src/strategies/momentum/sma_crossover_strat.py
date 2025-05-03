# trading_system/src/strategies/sma_crossover_strategy.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager


class SMAStrategy(BaseStrategy):
    """
    SMA Crossover Strategy with Integrated Risk Management

    This strategy implements a simple moving average (SMA) crossover system to generate trading signals and simulate trades.
    The core mathematical formulas are as follows:

      Let Sₜ = (1/Nₛₕₒᵣₜ) * sum(close[t-Nₛₕₒᵣₜ+1:t]) be the short-term SMA and 
          Lₜ = (1/Nₗₒₙg) * sum(close[t-Nₗₒₙg+1:t]) be the long-term SMA.

    Trading Signals:
      - A buy signal (signal = 1) is generated when:
              Sₜ > Lₜ  and  Sₜ₋₁ ≤ Lₜ₋₁,
        indicating a bullish crossover.
      - A sell signal (signal = -1) is generated when:
              Sₜ < Lₜ  and  Sₜ₋₁ ≥ Lₜ₋₁,
        indicating a bearish crossover.
      - Signal strength is computed as:
              strength = (Sₜ - Lₜ) / close,
        so that the indicator is normalized by the current price.
      - In long-only mode, sell signals are overridden (set to 0).

    Risk Management:
      Risk management is applied via a RiskManager, which adjusts the entry price (taking into account slippage 
      and transaction costs) and then sets stop-loss and take-profit thresholds:
        - For long trades: 
              P_stop = P_entry * (1 - stop_loss_pct) and P_target = P_entry * (1 + take_profit_pct).
        - For short trades, these thresholds are reversed.
      When an exit condition is met (whether by stop-loss, take-profit, or signal reversal), the realized trade return is computed:
        - For long trades: (exit_price / entry_price) - 1.
        - For short trades: (entry_price / exit_price) - 1.
      Cumulative return is computed as the cumulative product of the trade multipliers.

    Multi-Ticker and Execution:
      The strategy supports both single and multiple tickers. When a list of tickers is provided, data retrieval is 
      done in a vectorized manner that returns a MultiIndex DataFrame (ticker, date) and processing is performed 
      per ticker via Pandas groupby operations.
      The strategy supports:
        - Backtesting (using a specified start and end date) which produces a full DataFrame (allowing calculation 
          of Sharpe ratio, max drawdown, etc.).
        - Forecasting the latest signal (using a minimal lookback of long_window + 2 bars) which is efficient for EOD decisions.
    
    Returns:
      A DataFrame with at least the following columns:
         - 'signal'             : Trade signal (-1, 0, or 1).
         - 'strength'           : Normalized signal strength.
         - 'close', 'high', 'low': Price data.
         - 'short_sma', 'long_sma': Computed SMAs.
         - 'position'           : Position after applying risk management.
         - 'return'             : Realized return on the exit event.
         - 'cumulative_return'  : Cumulative return from closed trades.
         - 'exit_type'          : Reason for trade exit.
    
    Args:
      tickers (str or List[str]): A single ticker symbol or a list of ticker symbols.
      start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
      end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
      initial_position (int, optional): Starting trade position (default is 0, i.e. no position).
      latest_only (bool, optional): If True, returns only the latest signal row per ticker (for forecasting/decision-making).

    Strategy-specific parameters provided in `params` include:
      - 'short_window'         : Window for short SMA (default: 20).
      - 'long_window'          : Window for long SMA (default: 50).
      - 'stop_loss_pct'        : Stop-loss percentage (default: 0.05).
      - 'take_profit_pct'      : Take-profit percentage (default: 0.10).
      - 'trailing_stop_pct'    : Trailing stop percentage (default: 0.00).
      - 'slippage_pct'         : Slippage percentage (default: 0.001).
      - 'transaction_cost_pct' : Transaction cost percentage (default: 0.001).
      - 'long_only'            : If True, only long trades are allowed (default: True).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the SMA Crossover Strategy with risk management parameters.

        Args:
          db_config (DatabaseConfig): Database configuration settings.
          params (Dict, optional): Strategy-specific parameters.
        
        Raises:
          ValueError: If short_window is not less than long_window.
        """
        super().__init__(db_config, params)
        self.short_window = self.params.get('short_window', 20)
        self.long_window = self.params.get('long_window', 50)
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.10)
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.00)
        self.slippage_pct = self.params.get('slippage_pct', 0.001)
        self.transaction_cost_pct = self.params.get('transaction_cost_pct', 0.001)
        self.long_only = self.params.get('long_only', True)

        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")

        self.logger.debug(f"Initialized SMA Strategy with short_window={self.short_window} and long_window={self.long_window}")

    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate SMA crossover signals and simulate trade outcomes with integrated risk management.

        In backtest mode (latest_only=False), this method retrieves historical price data between the specified 
        start and end dates, calculates the moving averages, generates signals from the crossovers, and then applies 
        risk management to simulate trade entries and exits.

        In forecast mode (latest_only=True), it uses a minimal lookback (long_window + 2) to compute stable SMAs and 
        returns only the latest signal row for each ticker.

        Args:
          tickers (str or List[str]): A single ticker symbol or a list of ticker symbols.
          start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
          end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
          initial_position (int, optional): Starting position (default is 0).
          latest_only (bool, optional): If True, return only the most current signal for each ticker.

        Returns:
          pd.DataFrame: DataFrame containing signals and risk-managed trade data.
        """
        # Latest signal mode: use a minimal lookback window
        if latest_only:
            lookback = self.long_window + 2
            if isinstance(ticker, str):
                df = self.get_historical_prices(ticker, lookback=lookback)
                if not self._validate_data(df, min_records=lookback):
                    self.logger.error(f"Insufficient data for ticker {ticker}")
                    return pd.DataFrame()
                signals = self._calculate_smas_and_signals(df)
                latest_signal = signals.iloc[-1:]
                return latest_signal[['signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma']]
            else:
                df = self.get_historical_prices(ticker, lookback=lookback)
                if df.empty:
                    return pd.DataFrame()
                def latest_per_ticker(group):
                    if not self._validate_data(group, min_records=lookback):
                        self.logger.error("Insufficient data for ticker %s", group.name)
                        return pd.DataFrame()
                    signals = self._calculate_smas_and_signals(group)
                    return signals.iloc[-1:]
                latest_signals = df.groupby(level='ticker', group_keys=False).apply(latest_per_ticker)
                return latest_signals[['signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma']]
        else:
            # Backtesting mode: retrieve full historical prices within the specified date range.
            if isinstance(ticker, str):
                df = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
                if not self._validate_data(df, min_records=self.long_window + 1):
                    self.logger.error(f"Insufficient data for ticker {ticker}")
                    return pd.DataFrame()
                signals = self._calculate_smas_and_signals(df)
                risk_manager = RiskManager(
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    trailing_stop_pct=self.trailing_stop_pct,
                    slippage_pct=self.slippage_pct,
                    transaction_cost_pct=self.transaction_cost_pct
                )
                return risk_manager.apply(signals, initial_position)
            else:
                df = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
                if df.empty:
                    return pd.DataFrame()
                risk_manager = RiskManager(
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    slippage_pct=self.slippage_pct,
                    transaction_cost_pct=self.transaction_cost_pct
                )
                def process_ticker(group):
                    if not self._validate_data(group, min_records=self.long_window + 1):
                        self.logger.error("Insufficient data for ticker %s", group.name)
                        return pd.DataFrame()
                    signals = self._calculate_smas_and_signals(group)
                    return risk_manager.apply(signals, initial_position)
                result = df.groupby(level='ticker', group_keys=False).apply(process_ticker)
                return result

    def _calculate_smas_and_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short and long SMAs from the close price and generate crossover-based trading signals.

        The method computes:
          - short_sma: Rolling mean of the close price over the short_window.
          - long_sma: Rolling mean of the close price over the long_window.
          - signal: A trade signal generated when:
                      • A bullish crossover occurs (short_sma crosses above long_sma) -> 1.
                      • A bearish crossover occurs (short_sma crosses below long_sma) -> -1.
                      • Otherwise, 0.
          - strength: Normalized difference computed as (short_sma - long_sma) / close.

        In long-only mode, negative signals (sell signals) are overridden to 0.

        Args:
          df (pd.DataFrame): Historical price data with columns 'close', 'high', and 'low'.

        Returns:
          pd.DataFrame: DataFrame with columns: 'signal', 'strength', 'close', 'high', 'low', 'short_sma', and 'long_sma'.
        """
        # Compute moving averages using vectorized rolling operations.
        df['short_sma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_window).mean()

        # Compute previous period SMAs to detect crossovers.
        df['prev_short'] = df['short_sma'].shift(1)
        df['prev_long'] = df['long_sma'].shift(1)

        # Identify crossover conditions.
        cross_above = (df['short_sma'] > df['long_sma']) & (df['prev_short'] <= df['prev_long'])
        cross_below = (df['short_sma'] < df['long_sma']) & (df['prev_short'] >= df['prev_long'])

        # Generate the trade signal: 1 for bullish crossover, -1 for bearish, else 0.
        df['signal'] = np.select(
            [cross_above, cross_below],
            [1, -1],
            default=0
        )

        # Compute normalized signal strength.
        df['strength'] = (df['short_sma'] - df['long_sma']) / df['close']

        # In long-only mode override negative signals with 0 (exit signal).
        if self.long_only:
            df['signal'] = df['signal'].clip(lower=0)

        return df[['signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma']]

    def _generate_latest_signal(self, ticker: str) -> pd.DataFrame:
        """
        Generate the latest trading signal for a given ticker using a minimal lookback window.

        Uses (long_window + 2) bars to compute stable SMAs and returns the last (most recent) row,
        which contains the computed signal, its strength, and the relevant price indicators.

        Args:
          ticker (str): Stock ticker symbol.
        
        Returns:
          pd.DataFrame: A single-row DataFrame with columns: 'signal', 'strength', 'close', 'high', 'low', 'short_sma', and 'long_sma'.
        """
        lookback = self.long_window + 2
        df = self.get_historical_prices(ticker, lookback=lookback)
        if not self._validate_data(df, min_records=lookback):
            self.logger.error(f"Insufficient data for ticker {ticker}")
            return pd.DataFrame()
        signals = self._calculate_smas_and_signals(df)
        latest = signals.iloc[-1:]
        return latest[['signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma']]

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the input DataFrame contains at least the minimum number of records necessary.

        Args:
          df (pd.DataFrame): DataFrame to be validated.
          min_records (int): Minimum required number of records.
        
        Returns:
          bool: True if the DataFrame meets the minimum record requirement; otherwise, False.
        """
        if df.empty or len(df) < min_records:
            self.logger.warning(f"Data validation failed: {len(df)} records present, {min_records} required.")
            return False
        return True