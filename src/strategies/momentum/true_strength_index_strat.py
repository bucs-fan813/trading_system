# trading_system/src/strategies/tsi_strategy.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class TSIStrategy(BaseStrategy):
    """
    Enhanced True Strength Index (TSI) strategy with integrated risk management 
    for backtesting and forecasting.
    
    This strategy computes the TSI as a momentum oscillator defined by:
      1. delta = close.diff()
      2. double_smoothed = EMA(EMA(delta, span=long_period), span=short_period)
      3. abs_double_smoothed = EMA(EMA(abs(delta), span=long_period), span=short_period)
      4. TSI = 100 * (double_smoothed / (abs_double_smoothed + 1e-10))
      
    A signal line is computed as the EMA of the TSI over the signal_period.
    Trading signals are generated via crossovers between the TSI and its signal line:
      - A bullish (buy) signal is generated when TSI crosses above the signal line.
      - A bearish (sell) signal is generated when TSI crosses below the signal line.
      
    The difference (TSI - signal_line) is taken as the signal 'strength', which in turn is used
    as the basis for risk-managed entries and exits via the RiskManager.
    
    Risk management applies stop-loss and take-profit thresholds adjusted for slippage
    and transaction costs. The strategy thus outputs a complete DataFrame with signals,
    positions, returns, exit events and the associated price data to enable full performance 
    analysis downstream.
    
    Hyperparameters (with default values):
        long_period (int): First smoothing period (default: 25)
        short_period (int): Second smoothing period (default: 13)
        signal_period (int): Signal line period (default: 12)
        stop_loss_pct (float): Stop loss percentage (default: 0.05)
        take_profit_pct (float): Take profit percentage (default: 0.10)
        slippage_pct (float): Slippage percentage (default: 0.001)
        transaction_cost_pct (float): Transaction cost percentage (default: 0.001)
        long_only (bool): If True, only long positions are allowed (default: True)
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the TSIStrategy with database configuration and strategy parameters.
        
        Args:
            db_config: Database configuration settings used to initialize the DB engine.
            params (dict, optional): Dictionary of strategy parameters.
                Expected keys include 'long_period', 'short_period', 'signal_period',
                'stop_loss_pct', 'take_profit_pct', 'slippage_pct', 'transaction_cost_pct',
                and optionally 'min_data_points'.
        """
        super().__init__(db_config, params)
        
        # Strategy parameters
        self.long_period = int(self.params.get('long_period', 25))
        self.short_period = int(self.params.get('short_period', 13))
        self.signal_period = int(self.params.get('signal_period', 12))
        required_min = self.long_period + self.short_period + self.signal_period
        self.min_data_points = int(self.params.get('min_data_points', required_min))
        
        # Risk management parameters
        self.stop_loss_pct = float(self.params.get('stop_loss_pct', 0.05))
        self.take_profit_pct = float(self.params.get('take_profit_pct', 0.10))
        self.slippage_pct = float(self.params.get('slippage_pct', 0.001))
        self.transaction_cost_pct = float(self.params.get('transaction_cost_pct', 0.001))
        self.long_only = bool(self.params.get('long_only', True))
        
        self._validate_parameters()
        self.risk_manager = self._create_risk_manager()

    def generate_signals(self, ticker: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals for a specific ticker with integrated risk management.
        
        Retrieves historical price data over the specified date range, computes the TSI, 
        generates trading signals via TSI and signal line crossovers, and applies risk management 
        rules (stop-loss, take-profit, and signal-based exits). The resulting DataFrame includes:
          - 'date' index with columns: tsi, signal_line, strength, signal, position, 
            return, cumulative_return, exit_type, close, high, low.
        
        Args:
            ticker (str): Ticker symbol for which to generate signals.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): The initial market position (e.g., 0 for none).
            latest_only (bool): If True, only the signal for the latest available date is returned.
        
        Returns:
            pd.DataFrame: DataFrame containing the computed signals and risk management details.
        
        Raises:
            DataRetrievalError: When retrieved price data is insufficient.
        """
        self.logger.info(f"Processing {ticker} from {start_date} to {end_date}")
        try:
            prices = self._get_validated_prices(ticker, start_date, end_date)
            signals = self._calculate_signals(prices)
            processed_df = self._apply_risk_management(signals, initial_position)
            return self._finalize_output(processed_df, latest_only)
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            raise

    def _validate_parameters(self):
        """
        Validates that all required strategy and risk parameters are positive or non-negative.
        
        Raises:
            ValueError: If any of the period parameters are non-positive or risk parameters are negative.
        """
        if any(p <= 0 for p in [self.long_period, self.short_period, self.signal_period]):
            raise ValueError("All period parameters must be positive integers")
        if any(p < 0 for p in [self.stop_loss_pct, self.take_profit_pct,
                               self.slippage_pct, self.transaction_cost_pct]):
            raise ValueError("Risk parameters must be non-negative")

    def _create_risk_manager(self) -> RiskManager:
        """
        Initialize and return an instance of the RiskManager with the given risk parameters.
        
        Returns:
            RiskManager: Configured risk management instance.
        """
        return RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            slippage_pct=self.slippage_pct,
            transaction_cost_pct=self.transaction_cost_pct
        )

    def _get_validated_prices(self, ticker: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Retrieve historical price data from the database and validate that the number 
        of data points meets the minimum required for analysis.
        
        Args:
            ticker (str): The stock ticker.
            start_date (str or None): Start date in 'YYYY-MM-DD' format.
            end_date (str or None): End date in 'YYYY-MM-DD' format.
            
        Returns:
            pd.DataFrame: DataFrame of historical prices, indexed by date.
            
        Raises:
            DataRetrievalError: If the retrieved data has insufficient records.
        """
        prices = self.get_historical_prices(
            ticker,
            from_date=start_date,
            to_date=end_date,
            data_source='yfinance'
        )
        
        if not self._validate_data(prices, self.min_data_points):
            raise DataRetrievalError(f"Insufficient data for ticker {ticker}")
            
        return prices.sort_index()

    def _calculate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the TSI, signal line, and generate trading signals based on crossovers.
        
        This function computes:
          - TSI: Using a double-exponential smoothing of the price change.
          - Signal line: Exponentially smoothed TSI.
          - Strength: Defined as (TSI - signal line).
          - Trading signal: +1 when TSI crosses above the signal line, -1 when it crosses below.
          
        Also appends necessary price columns (close, high, and low) for risk management.
        
        Args:
            prices (pd.DataFrame): Price data containing at least the 'close', 'high', and 'low' columns.
        
        Returns:
            pd.DataFrame: DataFrame with new columns: 'tsi', 'signal_line', 'strength', 'signal',
                          and the price columns 'close', 'high', 'low'.
        """
        close = prices['close']
        tsi, signal_line = self._calculate_tsi(close)
        
        signals = pd.DataFrame(index=prices.index)
        signals['tsi'] = tsi
        signals['signal_line'] = signal_line
        signals['strength'] = tsi - signal_line
        signals['signal'] = self._generate_crossover_signals(tsi, signal_line)
        
        if self.long_only:
            signals[signals['signal'] == -1] = 0 # Disallow short positions if long_only is True
        
        # Append necessary price data for risk management calculations.
        signals[['close', 'high', 'low']] = prices[['close', 'high', 'low']]
        return signals

    def _calculate_tsi(self, close: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Vectorized computation of the True Strength Index (TSI) and its signal line.
        
        Given a Series of closing prices, this function calculates:
          1. The price change (delta).
          2. A double smoothed price change using sequential EMAs with windows 'long_period' and 'short_period'.
          3. A double smoothed absolute price change similarly.
          4. The TSI as 100 times the ratio of the two smoothed series.
          5. The signal line as an EMA of the TSI over 'signal_period'.
          
        Args:
            close (pd.Series): Series of closing prices.
        
        Returns:
            tuple: A tuple containing two pd.Series objects: (tsi, signal_line)
        """
        delta = close.diff()
        double_smoothed = delta.ewm(span=self.long_period, adjust=False).mean()\
                              .ewm(span=self.short_period, adjust=False).mean()
        abs_double_smoothed = delta.abs().ewm(span=self.long_period, adjust=False).mean()\
                                       .ewm(span=self.short_period, adjust=False).mean()
        
        tsi = (double_smoothed / (abs_double_smoothed + 1e-10)) * 100
        signal_line = tsi.ewm(span=self.signal_period, adjust=False).mean()
        return tsi, signal_line

    def _generate_crossover_signals(self, tsi: pd.Series, signal_line: pd.Series) -> pd.Series:
        """
        Detect crossovers between the TSI and its signal line to generate trading signals.
        
        A bullish signal (+1) is generated when the TSI crosses above the signal line (i.e.
        when the prior bar was below the signal line and the current bar is above). Conversely,
        a bearish signal (-1) is generated when the TSI crosses below the signal line.
        
        Args:
            tsi (pd.Series): The True Strength Index values.
            signal_line (pd.Series): The smoothed signal line computed from TSI.
        
        Returns:
            pd.Series: A Series of trading signals: 1 (buy), -1 (sell), or 0 (no change).
        """
        above = tsi > signal_line
        below = tsi < signal_line
        
        signals = pd.Series(0, index=tsi.index)
        # Generate a buy signal: previously below, now above.
        signals[(above & below.shift(1))] = 1
        # Generate a sell signal: previously above, now below.
        signals[(below & above.shift(1))] = -1
        
        return signals

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Apply risk management rules to the raw trading signals using the RiskManager component.
        
        This function subsets the necessary columns ('signal', 'high', 'low', and 'close')
        from the signals DataFrame and passes them along with the initial position to the RiskManager.
        The RiskManager computes adjusted entry prices, stop-loss/take-profit levels, exit events,
        trade returns, and cumulative returns in a fully vectorized manner.
        
        Args:
            signals (pd.DataFrame): DataFrame containing the raw signals and price data.
            initial_position (int): The starting market position (0 for no position).
        
        Returns:
            pd.DataFrame: A DataFrame with additional risk management fields including:
                          'position', 'return', 'cumulative_return', and 'exit_type'.
        """
        required_cols = ['signal', 'high', 'low', 'close']
        return self.risk_manager.apply(signals[required_cols], initial_position)

    def _finalize_output(self, df: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Finalize the output DataFrame by selecting and ordering the key columns.
        
        The returned DataFrame includes all required fields to perform downstream
        performance calculations (such as Sharpe ratio and maximum drawdown), and when
        requested, returns only the latest available date's signal.
        
        Args:
            df (pd.DataFrame): DataFrame with raw and risk-managed signal data.
            latest_only (bool): If True, only the last row (latest date) is returned.
        
        Returns:
            pd.DataFrame: The finalized DataFrame of trading signals and associated data.
        """
        df = df[['tsi', 'signal_line', 'strength', 'signal', 'position',
                 'return', 'cumulative_return', 'exit_type', 'close', 'high', 'low']]
        return df.iloc[[-1]] if latest_only else df

    def __repr__(self) -> str:
        """
        Return the string representation of the TSIStrategy instance.
        
        Returns:
            str: Representation including key strategy parameters.
        """
        return (f"TSIStrategy(long={self.long_period}, short={self.short_period}, "
                f"signal={self.signal_period}, sl={self.stop_loss_pct}, tp={self.take_profit_pct})")