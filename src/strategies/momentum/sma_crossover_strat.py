# trading_system/src/strategies/sma_crossover_strategy.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class SMAStrategy(BaseStrategy):
    """
    SMA Crossover Strategy with Integrated Risk Management

    This strategy employs a simple moving average (SMA) crossover system to generate 
    trading signals and incorporates risk management rules (stop-loss, take-profit, slippage,
    and transaction cost adjustments).

    Let S_t denote the short-term SMA and L_t denote the long-term SMA calculated as:
    
        S_t = (1/N_short) * sum(close[t-N_short+1:t])
        L_t = (1/N_long) * sum(close[t-N_long+1:t])
    
    A buy (long) signal is generated when:
        S_t > L_t  and  S_{t-1} <= L_{t-1}
    
    A sell (short) signal is generated when:
        S_t < L_t  and  S_{t-1} >= L_{t-1}

    The signal strength is computed as:
    
        strength = (S_t - L_t) / close_t

    Risk management is applied on top of the generated signals. For a long trade entered at an 
    adjusted entry price P_entry, the stop loss is:
    
        P_stop = P_entry * (1 - stop_loss_pct)
    
    and the take profit is:
    
        P_target = P_entry * (1 + take_profit_pct)
    
    (with roles reversed for short trades). Slippage and transaction costs are applied at entry 
    and exit. The realized return for each trade and cumulative return are computed, providing a 
    full DataFrame (with close, high, low prices available) for further analytics and downstream 
    optimization.
    
    The strategy supports both backtesting (with explicit start and end dates) and efficient 
    generation of the latest signal (using a minimal lookback).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the SMA Strategy.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (Dict, optional): Strategy-specific parameters. Valid parameters include:
                - short_window (int): Window for short SMA (default: 20).
                - long_window (int): Window for long SMA (default: 50).
                - stop_loss_pct (float): Stop-loss percentage (default: 0.05).
                - take_profit_pct (float): Take-profit percentage (default: 0.10).
                - slippage_pct (float): Slippage percentage (default: 0.001).
                - transaction_cost_pct (float): Transaction cost percentage (default: 0.001).
                - long_only (bool): If True, only long trades are allowed (default: True).

        Raises:
            ValueError: If short_window is not less than long_window.
        """
        super().__init__(db_config, params)
        
        # Initialize strategy parameters with defaults and validations.
        self.short_window = self.params.get('short_window', 20)
        self.long_window = self.params.get('long_window', 50)
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.10)
        self.slippage_pct = self.params.get('slippage_pct', 0.001)
        self.transaction_cost_pct = self.params.get('transaction_cost_pct', 0.001)
        self.long_only = self.params.get('long_only', True)
        
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        
        self.logger.debug(f"Initialized SMA Strategy: short={self.short_window}, long={self.long_window}")

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate SMA crossover signals with integrated risk management.

        In backtest mode (when latest_only=False), historical price data are retrieved within 
        the specified date range and the SMA’s are computed, generating signals. The RiskManager 
        is then applied to calculate entry/exit prices (adjusted for slippage and transaction costs), 
        stop-loss and take-profit thresholds, realized returns, and the cumulative return.

        In latest_only mode, a minimal lookback (long_window + 2) is used and only the most recent 
        signal is returned. All operations are performed in a vectorized manner for speed.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Backtest start date in YYYY-MM-DD.
            end_date (str, optional): Backtest end date in YYYY-MM-DD.
            initial_position (int, optional): Initial position (0 for flat).
            latest_only (bool, optional): If True, return signal only for the latest date.

        Returns:
            pd.DataFrame: DataFrame containing at least the following columns:
                - 'signal': Trade signal (-1, 0, or 1).
                - 'strength': Normalized difference (short_sma - long_sma) / close.
                - 'close', 'high', 'low': Price data.
                - 'short_sma', 'long_sma': The computed SMAs.
                - 'position': Position after risk management.
                - 'return': Realized trade return on exit events.
                - 'cumulative_return': Cumulative return over time.
                - 'exit_type': Reason for exiting the trade.
        """
        try:
            if latest_only:
                return self._generate_latest_signal(ticker)
            
            # Backtesting mode: retrieve historical prices between start and end dates.
            df = self.get_historical_prices(
                ticker,
                from_date=start_date,
                to_date=end_date
            )
            
            if not self._validate_data(df, min_records=self.long_window+1):
                self.logger.error(f"Insufficient data for {ticker}")
                return pd.DataFrame()

            # Calculate SMAs and generate raw signals.
            signals = self._calculate_smas_and_signals(df)
            
            # Apply risk management adjustments.
            risk_manager = RiskManager(
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                slippage_pct=self.slippage_pct,
                transaction_cost_pct=self.transaction_cost_pct
            )
            
            return risk_manager.apply(signals, initial_position)

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            raise

    def _calculate_smas_and_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple moving averages and generate crossover signals.

        This method computes the short and long SMAs from the 'close' price and then determines 
        the crossover events. A buy signal (1) is generated when the short SMA crosses above the 
        long SMA, and a sell signal (-1) is generated when it crosses below. The signal strength 
        is computed as the normalized difference between the two SMAs.

        Args:
            df (pd.DataFrame): DataFrame of historical price data. Must include 'close', 'high', and 'low'.

        Returns:
            pd.DataFrame: DataFrame with new columns: 'signal', 'strength', 'short_sma', and 'long_sma'.
                        Only these and the columns 'close', 'high', and 'low' are returned.
        """
        # Compute moving averages with vectorized rolling operations.
        df['short_sma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_window).mean()
        
        # Create shifted SMAs for previous period comparison.
        df['prev_short'] = df['short_sma'].shift(1)
        df['prev_long'] = df['long_sma'].shift(1)
        
        # Determine crossover conditions (golden cross and death cross).
        cross_above = (df['short_sma'] > df['long_sma']) & (df['prev_short'] <= df['prev_long'])
        cross_below = (df['short_sma'] < df['long_sma']) & (df['prev_short'] >= df['prev_long'])

        # Generate the trade signals: 1 for long, -1 for short, 0 otherwise.
        df['signal'] = np.select(
            [cross_above, cross_below],
            [1, -1],
            default=0
        )
        
        # Calculate signal strength as the normalized difference.
        df['strength'] = (df['short_sma'] - df['long_sma']) / df['close']

        if self.long_only:
            df['signal'] = df['signal'].clip(lower=0)  # Override sell signals with 0 (exit)
        
        return df[['signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma']]

    def _generate_latest_signal(self, ticker: str) -> pd.DataFrame:
        """
        Efficiently generate the signal for the latest available date using a minimal lookback.

        The method retrieves only (long_window + 2) bars so that the SMA calculations and the 
        subsequent signal generation are stable. It then returns the most recent signal row.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with a single row containing columns: 
                          'signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma'.
        """
        lookback = self.long_window + 2  # Sufficient bars to compute both SMAs accurately.
        df = self.get_historical_prices(ticker, lookback=lookback)
        
        if not self._validate_data(df, min_records=lookback):
            self.logger.error(f"Insufficient data for {ticker}")
            return pd.DataFrame()

        signals = self._calculate_smas_and_signals(df)
        latest = signals.iloc[-1:]
        
        return latest[['signal', 'strength', 'close', 'high', 'low', 'short_sma', 'long_sma']]

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the DataFrame has sufficient records for analysis.

        Args:
            df (pd.DataFrame): DataFrame to check.
            min_records (int): Minimum required records.

        Returns:
            bool: True if the data meets the minimum requirement, False otherwise.
        """
        if df.empty or len(df) < min_records:
            self.logger.warning(f"Data validation failed: {len(df)} records vs required {min_records}")
            return False
        return True