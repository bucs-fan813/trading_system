# trading_system/src/strategies/choppiness_index_strat.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union, List

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager


class ChoppinessIndexStrategy(BaseStrategy):
    """
    Choppiness Index Strategy with MACD confirmation and integrated Risk Management.

    This strategy implements the following:
    
    1. Indicator Calculation:
       - **True Range (TR):**  
         TR = max(high - low, |high - previous close|, |low - previous close|).
       - **Choppiness Index (CI):**  
         CI = 100 * log10( sum(TR over n periods) / (max(high over n periods) - min(low over n periods) ) ) / log10(n),
         where n is defined by the parameter `ci_period` (default 14).
       - **MACD:**  
         MACD = EMA_fast(close) - EMA_slow(close),  
         Signal Line = EMA(MACD, macd_smooth), and  
         Histogram = MACD - Signal Line.
         
    2. Signal Generation:
       - A bullish signal is generated if MACD > Signal Line and CI < 38.2.
       - A bearish signal is generated if MACD < Signal Line and CI < 38.2.
         (In long-only mode, bearish signals are replaced by 0 for an exit.)
       - Signal strength is computed as:  
         strength = (|Histogram| / close) * (1 - (CI / 38.2)).
       - Signals are forward-filled so that positions persist until a reversal.

    3. Risk Management:
       An external RiskManager is used to adjust the raw signals. This includes:
         - Adjusting entry prices for slippage and transaction costs.
         - Determining stop-loss and take-profit thresholds.
         - Detecting exit events and computing realized and cumulative returns.

    The strategy supports both backtesting (with a start and end date) as well as forecasting
    (returning only the most recent signal). It also supports processing a list of tickers in a vectorized
    manner.

    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Dictionary of strategy parameters including:
            - 'ci_period': Period for Choppiness Index calculation (default: 14).
            - 'macd_fast': Period for fast EMA in MACD (default: 12).
            - 'macd_slow': Period for slow EMA in MACD (default: 26).
            - 'macd_smooth': Period for signal line EMA (default: 9).
            - 'long_only': If True, only long positions are allowed (default: True).
            - 'stop_loss_pct': Stop loss percentage (default: 0.05).
            - 'take_profit_pct': Take profit percentage (default: 0.10).
            - 'trailing_stop_pct': Trailing stop percentage (default: 0.0).
            - 'slippage_pct': Slippage percentage (default: 0.001).
            - 'transaction_cost_pct': Transaction cost percentage (default: 0.001).

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - 'close', 'high', 'low': Price data.
            - 'signal': Trading signal (+1 for long, -1 for short (if allowed), or 0 for exit/neutral).
            - 'strength': Computed strength of the generated signal.
            - 'position': Position after risk management adjustments.
            - 'return': Realized trade return on exit events.
            - 'cumulative_return': Cumulative return from closed trades.
            - 'exit_type': Indicator of the reason for exit (stop_loss, take_profit, or signal_exit).

    Note:
        In forecasting mode (latest_only=True), only the last observation per ticker is returned.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Choppiness Index Strategy and its RiskManager.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy and risk management parameters.
        """
        super().__init__(db_config, params)
        self.long_only = self.params.get('long_only', True)
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            trailing_stop_pct=self.params.get('trailing_stop_pct', 0.0),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )

    def generate_signals(self,
                         ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals with integrated risk management for backtesting or forecasting.

        The method retrieves historical price data with a lookback period sufficient for calculating
        the Choppiness Index and MACD indicators. It then computes the indicators in a vectorized fashion
        (grouped by ticker if multiple tickers are provided) and generates raw signals based on the criteria:
            - Bullish when MACD > signal_line and CI < 38.2.
            - Bearish when MACD < signal_line and CI < 38.2 (or exit in long-only mode).
        The signal strength is calculated as:
            strength = (|histogram| / close) * (1 - (CI / 38.2)).
        Risk management is applied through the RiskManager class, which adjusts prices for slippage and
        transaction costs, sets stop-loss/take-profit levels, and calculates performance metrics.

        Args:
            ticker (str or List[str]): A single stock ticker symbol or a list of ticker symbols.
            start_date (str, optional): Backtest start date in YYYY-MM-DD format.
            end_date (str, optional): Backtest end date in YYYY-MM-DD format.
            initial_position (int): Starting trading position (default is 0).
            latest_only (bool): If True, returns only the final observation (per ticker) for forecasting.

        Returns:
            pd.DataFrame: DataFrame containing price data, signals, risk-managed positions,
            realized returns, cumulative returns, and exit event types.
        """
        # Determine lookback period based on the maximum of CI period and MACD slow period
        lookback = max(self.params.get('ci_period', 14), 26)  # MACD slow default is 26

        # Retrieve historical price data with appropriate lookback, supporting both single and multiple tickers.
        if isinstance(ticker, list):
            prices = self.get_historical_prices(
                tickers=ticker,
                from_date=start_date,
                to_date=end_date
            )
        else:
            prices = self._get_prices_with_lookback(ticker, start_date, end_date, lookback, latest_only)

        if not self._validate_data(prices, min_records=lookback * 2):
            raise DataRetrievalError("Insufficient data for analysis")

        # Calculate CI and MACD indicators in a vectorized fashion.
        df = self._calculate_indicators(prices)

        # Generate raw trading signals based on CI and MACD conditions.
        signals = self._generate_vectorized_signals(df)

        # Apply risk management. For multi-ticker data, risk management is applied per ticker.
        if isinstance(ticker, list) or ('ticker' in signals.index.names):
            results = signals.groupby(level='ticker', group_keys=False).apply(
                lambda group: self.risk_manager.apply(group, initial_position)
            )
        else:
            results = self.risk_manager.apply(signals, initial_position)

        return self._format_output(results, latest_only)

    def _get_prices_with_lookback(self, ticker: str, start_date: Optional[str],
                                  end_date: Optional[str], lookback: int,
                                  latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data with an extended lookback period to ensure stability
        in indicator calculations during forecasting.

        When latest_only is True, the method fetches (lookback*2) bars; otherwise, it uses the
        specified backtest date range.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Backtest start date (YYYY-MM-DD) (ignored if latest_only is True).
            end_date (str): Backtest end date (YYYY-MM-DD) (ignored if latest_only is True).
            lookback (int): Base lookback period for indicator calculations.
            latest_only (bool): Indicator to return only sufficient data for a single-date forecast.

        Returns:
            pd.DataFrame: DataFrame with historical prices.
        """
        if latest_only:
            return self.get_historical_prices(
                tickers=ticker,
                lookback=lookback * 2,
                from_date=None,
                to_date=None
            )
        return self.get_historical_prices(
            tickers=ticker,
            from_date=start_date,
            to_date=end_date
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and append the Choppiness Index (CI) and MACD indicators to the price DataFrame.

        For multi-ticker data (with a 'ticker' level in the index), the calculations are performed
        per group. Otherwise, indicator calculations are applied directly.

        Args:
            df (pd.DataFrame): DataFrame with price data containing at least 'high', 'low', and 'close'.

        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - 'ci': Choppiness Index.
                - 'macd': MACD line.
                - 'signal_line': EMA-based signal line.
                - 'histogram': Difference between MACD and the signal line.
        """
        ci_period = int(self.params.get('ci_period', 14))
        fast = int(self.params.get('macd_fast', 12))
        slow = int(self.params.get('macd_slow', 26))
        smooth = int(self.params.get('macd_smooth', 9))

        if 'ticker' in df.index.names:
            # Process each ticker group separately.
            def calc_indicators(group):
                group = group.sort_index(level='date')
                tr = np.maximum(
                    group['high'] - group['low'],
                    np.maximum(
                        abs(group['high'] - group['close'].shift(1)),
                        abs(group['low'] - group['close'].shift(1))
                    )
                )
                sum_tr = tr.rolling(window=ci_period).sum()
                max_high = group['high'].rolling(window=ci_period).max()
                min_low = group['low'].rolling(window=ci_period).min()
                group['ci'] = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(ci_period)

                exp_fast = group['close'].ewm(span=fast, adjust=False).mean()
                exp_slow = group['close'].ewm(span=slow, adjust=False).mean()
                group['macd'] = exp_fast - exp_slow
                group['signal_line'] = group['macd'].ewm(span=smooth, adjust=False).mean()
                group['histogram'] = group['macd'] - group['signal_line']
                return group.dropna()

            df = df.groupby(level='ticker', group_keys=False).apply(calc_indicators)
        else:
            # Process single ticker series.
            df = df.sort_index()
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            sum_tr = tr.rolling(window=ci_period).sum()
            max_high = df['high'].rolling(window=ci_period).max()
            min_low = df['low'].rolling(window=ci_period).min()
            df['ci'] = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(ci_period)

            exp_fast = df['close'].ewm(span=fast, adjust=False).mean()
            exp_slow = df['close'].ewm(span=slow, adjust=False).mean()
            df['macd'] = exp_fast - exp_slow
            df['signal_line'] = df['macd'].ewm(span=smooth, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal_line']
            df = df.dropna()

        return df

    def _generate_vectorized_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using CI and MACD criteria in a vectorized fashion.

        Conditions applied:
            - Bullish (raw signal = 1): MACD > signal_line and CI < 38.2.
            - Bearish (raw signal = -1): MACD < signal_line and CI < 38.2.
              In long-only mode, bearish conditions result in an exit signal (0).
            - The signal strength is calculated as:
                strength = (|histogram| / close) * (1 - (CI / 38.2)).
            - Signals are forward-filled so that positions persist until a change is triggered.

        Args:
            df (pd.DataFrame): DataFrame containing price data and computed indicators.

        Returns:
            pd.DataFrame: DataFrame with columns:
                'close', 'high', 'low', 'signal', and 'strength'.
        """
        # Define bullish and bearish conditions
        bullish = (df['macd'] > df['signal_line']) & (df['ci'] < 38.2)
        bearish = (df['macd'] < df['signal_line']) & (df['ci'] < 38.2)

        # Compute strength only under trending conditions
        df['strength'] = 0.0
        mask = df['ci'] < 38.2
        df.loc[mask, 'strength'] = (abs(df['histogram']) / df['close']) * (1 - (df['ci'] / 38.2))

        # Establish raw signals: 1 for bullish; -1 for bearish (or 0 in long-only mode)
        df['raw_signal'] = 0
        df.loc[bullish, 'raw_signal'] = 1
        if self.long_only:
            df.loc[bearish, 'raw_signal'] = 0
        else:
            df.loc[bearish, 'raw_signal'] = -1

        # Forward-fill signals so that positions persist until a reversal.
        df['signal'] = df['raw_signal'].replace(0, np.nan).ffill().fillna(0)
        return df[['close', 'high', 'low', 'signal', 'strength']]

    def _format_output(self, df: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Format the output DataFrame to include essential columns for backtesting or forecasting.

        The output includes price data, generated signals, risk-managed positions,
        individual realized returns, cumulative returns, and the exit event type.
        In forecasting mode (latest_only=True), only the final observation per ticker is returned.

        Args:
            df (pd.DataFrame): DataFrame with risk-managed signals and performance metrics.
            latest_only (bool): If True, returns only the most recent observation for forecast.

        Returns:
            pd.DataFrame: A formatted DataFrame with columns:
                'close', 'high', 'low', 'signal', 'position', 'return',
                'cumulative_return', 'exit_type'.
        """
        required_cols = ['close', 'high', 'low', 'signal', 'position',
                         'return', 'cumulative_return', 'exit_type']
        df = df[required_cols]

        if latest_only:
            if 'ticker' in df.index.names:
                # Return the last row per ticker
                df = df.groupby(level='ticker', group_keys=False).tail(1)
            else:
                df = df.iloc[[-1]]
        return df

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the DataFrame has sufficient and clean data for indicator calculations.

        Checks to ensure:
            - The DataFrame is not empty.
            - The number of records meets the minimum threshold.
            - Essential price columns ('close', 'high', 'low') have no missing values.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            min_records (int): Minimum required number of records.

        Returns:
            bool: True if the data is adequate; False otherwise.
        """
        if df.empty:
            self.logger.error("No data retrieved")
            return False
        if len(df) < min_records:
            self.logger.error(f"Need at least {min_records} records, got {len(df)}")
            return False
        if df[['close', 'high', 'low']].isnull().any().any():
            self.logger.error("Missing price data in retrieved series")
            return False
        return True