# trading_system/src/strategies/aroon_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class AroonStrategy(BaseStrategy):
    """
    Aroon Strategy for generating trading signals based on the Aroon Oscillator with integrated risk management.
    
    The strategy computes the Aroon Up and Aroon Down indicators over a specified lookback period (default = 25 days),
    where:
    
        Aroon Up = ((k + 1) / N) * 100,
        Aroon Down = ((j + 1) / N) * 100,
        
    with N being the lookback period, k the index of highest high within the rolling window, and j the index of lowest low.
    
    A long signal (buy, represented by 1) is generated when Aroon Up ≥ 70 and Aroon Down ≤ 30,
    indicating a strong upward trend, and a short signal (sell, represented by –1) is generated when Aroon Up ≤ 30 and 
    Aroon Down ≥ 70, indicating a strong downward trend. The generated raw signals are forward filled (to maintain the uptake) 
    until a reversal is detected.
    
    Risk management rules are then applied via the RiskManager class (which incorporates stop loss, take profit, 
    transaction costs, and slippage). This strategy is designed both for backtesting over a specified date range and 
    for fast forecasting using just the minimal data needed to establish the stability of the most recent signal.
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the AroonStrategy with database configuration and strategy parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters including:
                - 'lookback': The period for computing the Aroon indicator (default 25).
                - 'stop_loss_pct': Stop loss percentage (default 0.05).
                - 'take_profit_pct': Take profit percentage (default 0.10).
                - 'slippage_pct': Slippage percentage (default 0.001).
                - 'transaction_cost_pct': Transaction cost percentage (default 0.001).
        """
        super().__init__(db_config, params)
        self.lookback = self.params.get('lookback', 25)
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )

    def _calculate_aroon(self, high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """
        Compute the Aroon Up and Aroon Down indicators for given high and low price series over a rolling window.
        
        The calculation is based on:
            Aroon Up   = ((Index of highest high + 1) / Lookback) * 100,
            Aroon Down = ((Index of lowest low + 1) / Lookback) * 100.
        
        The window index starts at 0 for the oldest value and ends at (Lookback - 1) for the most recent value.
        A value of 100 means that the highest (or lowest) price occurred on the most recent bar.
        
        Args:
            high (pd.Series): Series of high prices.
            low (pd.Series): Series of low prices.
            
        Returns:
            pd.DataFrame: DataFrame containing the 'aroon_up' and 'aroon_down' columns.
        """
        high_window = high.rolling(window=self.lookback, min_periods=self.lookback)
        low_window = low.rolling(window=self.lookback, min_periods=self.lookback)

        aroon_up = high_window.apply(lambda x: (np.argmax(x) + 1) / self.lookback * 100, raw=True)
        aroon_down = low_window.apply(lambda x: (np.argmin(x) + 1) / self.lookback * 100, raw=True)
        return pd.DataFrame({'aroon_up': aroon_up, 'aroon_down': aroon_down})

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate raw trading signals based on computed Aroon indicators.
        
        A long signal (1) is generated when Aroon Up is at least 70 and Aroon Down is at most 30;
        a short signal (–1) is generated when Aroon Up is at most 30 and Aroon Down is at least 70.
        In periods with no new signal, the previous nonzero signal is forward filled.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'aroon_up' and 'aroon_down' columns.
            
        Returns:
            pd.Series: Series of raw trading signals with integer values (1 for long, –1 for short, 0 for no position).
        """
        signals = pd.Series(0, index=df.index)
        
        # Generate long and short signals based on thresholds.
        long_cond = (df['aroon_up'] >= 70) & (df['aroon_down'] <= 30)
        short_cond = (df['aroon_up'] <= 30) & (df['aroon_down'] >= 70)
        signals = np.where(long_cond, 1, signals)
        signals = np.where(short_cond, -1, signals)
        
        # Forward fill signals until a reversal is detected.
        signals = pd.Series(signals, index=df.index).replace(0, np.nan).ffill().fillna(0)
        return signals.astype(int)

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals with integrated risk management for a given ticker.
        
        This method retrieves historical pricing data (using the efficient data retrieval method inherited from
        BaseStrategy) and computes the Aroon indicators before generating raw signals. If backtesting (latest_only=False)
        is desired, risk management is then applied to compute entry/exit prices, realized returns, cumulative returns,
        and various exit types. Additional columns (daily and strategy returns) are added so that performance metrics
        (e.g. Sharpe ratio, max drawdown) can be calculated downstream.
        
        If only the latest signal is needed (latest_only=True) for real-time decision making, only a minimal data slice 
        (equal to the lookback period) is fetched and the most recent row is returned.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Backtest start date in YYYY-MM-DD format.
            end_date (str, optional): Backtest end date in YYYY-MM-DD format.
            initial_position (int): The starting position (0 for none, 1 for long, –1 for short).
            latest_only (bool): If True, return only the latest signal; otherwise, return full backtest data.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals and risk management metrics. For backtests,
                        columns include prices, Aroon indicators, signal, position, exit_type, returns,
                        cumulative_return, and strategy returns. For latest_only=True, a minimal set of columns is returned.
        """
        # Retrieve historical data using either a full date range (for backtesting) or a minimal slice (for the latest signal).
        if latest_only:
            df = self.get_historical_prices(ticker, lookback=self.lookback)
            if not self._validate_data(df, min_records=self.lookback):
                return pd.DataFrame()
        else:
            df = self.get_historical_prices(
                ticker,
                from_date=start_date,
                to_date=end_date
            )
            if not self._validate_data(df, min_records=2*self.lookback):
                return pd.DataFrame()

        # Compute Aroon indicators over the relevant window and merge them with price data.
        aroon_df = self._calculate_aroon(df['high'], df['low'])
        df = df.join(aroon_df)
        
        # Generate trading signals based on the indicator thresholds.
        df['signal'] = self._generate_raw_signals(df)
        
        # Apply risk management when running a backtest.
        if not latest_only:
            df = self.risk_manager.apply(df, initial_position=initial_position)
            df['returns'] = df['close'].pct_change().fillna(0)
            df['strategy_returns'] = df['position'].shift(1) * df['returns']
        else:
            # For real-time forecasting, return only the minimal set of columns for the last available date.
            df = df[['open', 'high', 'low', 'close', 'aroon_up', 'aroon_down', 'signal']].iloc[[-1]]
        
        return df

    def _validate_data(self, df: pd.DataFrame, min_records: int = 1) -> bool:
        """
        Validate that the historical data contains at least the minimum number of records required.
        
        Args:
            df (pd.DataFrame): Historical price data.
            min_records (int): Minimum required record count.
        
        Returns:
            bool: True if the data is sufficient; otherwise, False.
        """
        if df.empty or len(df) < min_records:
            self.logger.warning(f"Insufficient data: {len(df)} records found, {min_records} required")
            return False
        return True