# trading_system/src/strategies/disparity_index_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig

class DisparityIndexStrategy(BaseStrategy):
    """
    Disparity Index Strategy with integrated risk management.

    Mathematical Explanation:
    ---------------------------
    The strategy calculates the Disparity Index (DI) as:
       DI = ((Close - MA) / MA) * 100,
    where MA is the moving average of the close price over a period (default 14 days).
    A buy signal is generated when DI turns positive, given that DI was negative for a series 
    of consecutive days (default 4). Similarly, a sell signal is generated when DI turns negative 
    after being positive for consecutive days. 

    The signal strength is computed as:
       signal_strength = DI (for buy signals) or -DI (for sell signals),
    indicating the magnitude of the reversal.

    Risk Management:
    ----------------
    In addition to determining entry signals, the strategy applies risk management rules:
      - Entry prices are adjusted for slippage and transaction costs.
      - Stop-loss and take-profit levels are defined as percentage moves from the entry price.
      - Exit conditions are triggered by these levels or by a signal reversal.
    Realized returns are calculated as:
       For long: (exit_price / entry_price) - 1.
       For short: (entry_price / exit_price) - 1.
    Cumulative return aggregates trade multipliers over time, allowing downstream analysis
    such as Sharpe ratio and maximum drawdown calculation.
    
    This implementation leverages vectorized computations for efficient backtesting and 
    latest signal forecasting.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Disparity Index Strategy.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (Dict, optional): Parameters for the strategy. Expected keys include:
                - di_lookback (int): Lookback period for the moving average (default 14).
                - consecutive_period (int): Number of consecutive periods for signal reversal (default 4).
                - stop_loss_pct (float): Stop loss percentage (default 0.05).
                - take_profit_pct (float): Take profit percentage (default 0.10).
                - slippage_pct (float): Estimated slippage percentage (default 0.001).
                - transaction_cost_pct (float): Transaction cost percentage (default 0.001).
                - 'long_only': Flag to allow only long positions (default: True)
        """
        params = params or {}
        super().__init__(db_config, params)
        
        # Strategy-specific parameters
        self.di_lookback = params.get('di_lookback', 14)
        self.consecutive_period = params.get('consecutive_period', 4)
        
        # Risk management parameters
        self.stop_loss_pct = params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = params.get('take_profit_pct', 0.10)
        self.slippage_pct = params.get('slippage_pct', 0.001)
        self.transaction_cost_pct = params.get('transaction_cost_pct', 0.001)
        self.long_only = self.params.get('long_only', True)
        
        # Initialize the risk manager instance with the provided risk parameters.
        self.risk_manager = RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            slippage_pct=self.slippage_pct,
            transaction_cost_pct=self.transaction_cost_pct
        )

    def generate_signals(self, 
                         ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals with integrated risk management for backtesting or forecasting.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting in YYYY-MM-DD format.
            end_date (str, optional): End date for backtesting in YYYY-MM-DD format.
            initial_position (int, optional): Starting position (0 for flat, 1 for long, -1 for short). Defaults to 0.
            latest_only (bool, optional): If True, returns only the latest computed signal for forecasting. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing:
                - Price data (open, high, low, close)
                - Calculated indicators (di, signal, signal_strength)
                - Risk-managed fields (position, return, cumulative_return, exit_type)
                - These outputs enable further downstream analysis (e.g., Sharpe ratio, max drawdown).
        """
        # Calculate the required number of historical records to ensure sufficient lookback.
        required_lookback = self.di_lookback + self.consecutive_period + 1
        
        # Retrieve historical price data using dates or lookback count based on the mode.
        prices_df = self._get_price_data(ticker, start_date, end_date, required_lookback, latest_only)
        
        # Calculate indicators: disparity index, signals, and signal strength.
        di, signals, signal_strength = self._calculate_indicators(prices_df['close'])
        
        # Create a combined DataFrame with prices and computed indicators.
        signals_df = self._create_signals_df(prices_df, di, signals, signal_strength)
        
        # Apply risk management rules to compute positions and returns.
        return self._apply_risk_management(signals_df, initial_position, latest_only)

    def _get_price_data(self, ticker: str, 
                        start_date: Optional[str], 
                        end_date: Optional[str],
                        required_lookback: int,
                        latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data from the database.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Backtest start date in YYYY-MM-DD format. (Ignored if latest_only=True)
            end_date (str): Backtest end date in YYYY-MM-DD format. (Ignored if latest_only=True)
            required_lookback (int): Minimum number of records required.
            latest_only (bool): If True, retrieves only the most recent records according to the required lookback.

        Returns:
            pd.DataFrame: DataFrame containing price data (open, high, low, close) with a datetime index.

        Raises:
            DataRetrievalError: If the available data is insufficient for analysis.
        """
        if latest_only:
            prices_df = self.get_historical_prices(ticker, lookback=required_lookback)
        else:
            prices_df = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)

        if not self._validate_data(prices_df, min_records=required_lookback):
            raise DataRetrievalError(f"Insufficient data for {ticker}")
            
        return prices_df

    def _calculate_indicators(self, close_prices: pd.Series) -> tuple:
        """
        Compute the Disparity Index and generate trading signals with signal strength.

        The calculations include:
            - Moving average (MA) computed over a window (di_lookback).
            - Disparity Index: DI = ((Close - MA) / MA) * 100.
            - Determination of buy/sell signals:
                * Buy signal (1): when DI turns positive after being negative for 'consecutive_period' days.
                * Sell signal (-1): when DI turns negative after being positive for 'consecutive_period' days.
            - Signal strength is defined as DI for buys and -DI for sells.

        Args:
            close_prices (pd.Series): Series of closing prices.

        Returns:
            tuple: (di, signals, signal_strength) where each is a pd.Series.
        """
        # Compute the moving average.
        ma = close_prices.rolling(self.di_lookback, min_periods=self.di_lookback).mean()
        # Calculate the Disparity Index.
        di = ((close_prices - ma) / ma) * 100
        
        # Use the previous period's DI for signal triggering.
        shifted_di = di.shift(1)
        # Create masks for consecutive negative or positive DI values.
        negative_mask = (shifted_di < 0).rolling(self.consecutive_period).sum() == self.consecutive_period
        positive_mask = (shifted_di > 0).rolling(self.consecutive_period).sum() == self.consecutive_period
        
        # Generate signals: 1 for a buy, -1 for a sell, 0 otherwise.
        signals = np.where(negative_mask & (di > 0), 1, 
                          np.where(positive_mask & (di < 0), -1, 0))
        
        # Establish signal strength: For a buy, strength equals DI; for a sell, the strength is -DI.
        signal_strength = np.where(signals == 1, di, 
                                  np.where(signals == -1, -di, 0.0))
        
        if self.long_only:
            # Replace sell signals (-1) with 0 (exit)
            signals[signals == -1] = 0
        
        return di, pd.Series(signals, index=di.index), pd.Series(signal_strength, index=di.index)

    def _create_signals_df(self, prices_df: pd.DataFrame,
                             di: pd.Series, 
                             signals: pd.Series,
                             signal_strength: pd.Series) -> pd.DataFrame:
        """
        Create a consolidated DataFrame merging price information with computed indicators.

        Args:
            prices_df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close'].
            di (pd.Series): Series containing the Disparity Index values.
            signals (pd.Series): Series containing the generated signals.
            signal_strength (pd.Series): Series with the magnitude of the signals.

        Returns:
            pd.DataFrame: Consolidated DataFrame with the combined data and non-null DI rows.
        """
        combined_df = pd.concat([
            prices_df[['open', 'high', 'low', 'close']],
            di.rename('di'),
            pd.Series(signals, name='signal'),
            pd.Series(signal_strength, name='signal_strength')
        ], axis=1)
        return combined_df.dropna(subset=['di'])

    def _apply_risk_management(self, 
                               signals_df: pd.DataFrame,
                               initial_position: int,
                               latest_only: bool) -> pd.DataFrame:
        """
        Apply risk management to the raw trading signals to yield final positions, returns, and exit types.

        This method adjusts trade entries and exits based on risk parameters (stop loss, take profit,
        slippage, and transaction costs) by invoking the RiskManager. The output is then merged with the
        strategy indicators.

        Args:
            signals_df (pd.DataFrame): DataFrame with price and indicator data.
            initial_position (int): Starting position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the latest row (for EOD decision-making).

        Returns:
            pd.DataFrame: DataFrame with columns:
                        ['close', 'high', 'low', 'signal', 'position', 'return',
                         'cumulative_return', 'exit_type', 'di', 'signal_strength', 'open'].
        """
        # Apply risk management to compute trade management metrics.
        managed_df = self.risk_manager.apply(
            signals_df,
            initial_position=initial_position
        )
        
        # For forecasting, return only the latest record; otherwise, return the full backtest DataFrame.
        return managed_df.tail(1) if latest_only else managed_df