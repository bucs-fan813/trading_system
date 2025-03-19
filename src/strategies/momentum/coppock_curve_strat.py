# trading_system/src/strategies/momentum/coppock_curve_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class CoppockCurveStrategy(BaseStrategy):
    """
    CoppockCurveStrategy implements the Coppock Curve trading strategy with integrated risk management.
    
    Mathematical Explanation:
        1. ROC Calculation:
           - ROC1 = 100 * ((Close / Close_shifted_by_roc1_days) - 1)
           - ROC2 = 100 * ((Close / Close_shifted_by_roc2_days) - 1)
           where roc1_days = roc1_months * 21 and roc2_days = roc2_months * 21.
           
        2. Combined ROC:
           - CombinedROC = ROC1 + ROC2
           
        3. Smoothing:
           - The CombinedROC is smoothed by calculating a weighted moving average (WMA) over a window 
             of self.wma_lookback days. Weights increase linearly (1, 2, ..., N) where N = (wma_lookback / 21).
           
        4. Signal Generation:
           - Zero-Crossing: A long signal (+1) is generated when the indicator, after a sustained negative regime (4 days),
             crosses from below 0 to above 0. A short signal (-1) is triggered when the indicator, after a sustained positive regime,
             turns negative. If long_only is set, short signals are clipped to 0.
           - An auxiliary momentum strength is computed as the divergence of the Coppock Curve from its short-term (21-day) moving average,
             optionally normalized to [-1, 1].
             
        5. Risk Management:
           - The raw signals are adjusted for stop-loss, take-profit, slippage, and transaction costs via RiskManager.
           - The adjusted risk-managed signal determines exit price, realized return (long: exit_price/entry_price - 1, short: entry_price/exit_price - 1),
             and cumulative return.
             
        6. Vectorization and Multi-Ticker Support:
           - The strategy supports processing a single ticker (string) or a list of tickers. In vectorized mode, all tickers are fetched
             in one query and then grouped by ticker for independent processing.
             
    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Dictionary containing strategy parameters:
            - 'roc1_months': Months for first ROC (default: 14)
            - 'roc2_months': Months for second ROC (default: 11)
            - 'wma_lookback': Lookback window for WMA smoothing (default: 10, multiplied by 21 trading days)
            - 'stop_loss_pct': Stop loss percentage (default: 0.05)
            - 'take_profit_pct': Take profit percentage (default: 0.10)
            - 'trailing_stop_pct': Trailing stop percentage (default: 0.0)
            - 'slippage_pct': Slippage as a fraction (default: 0.001)
            - 'transaction_cost_pct': Transaction cost as a fraction (default: 0.001)
            - 'strength_window': Lookback window for momentum strength normalization (default: 504)
            - 'normalize_strength': Flag to normalize momentum strength (default: True)
            - 'method': Signal generation method: 'zero_crossing' or 'directional' (default: 'zero_crossing')
            - 'long_only': If True, only long positions are allowed (default: True)
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self._validate_params()
        self._init_risk_params()
        self._init_strength_params()

    @staticmethod
    def generate_repeated_array(input_number: int) -> np.ndarray:
        """
        Generate a 1D numpy array where an integer number is repeated 21 times, 
        then the previous integer is repeated 21 times, and so on until 1 is repeated 21 times.
        
        Args:
            input_number (int): Starting positive integer.
            
        Returns:
            np.ndarray: Array of repeated sequences.
        """
        if not isinstance(input_number, int) or input_number <= 0:
            raise ValueError("Input must be a positive integer.")
        repetitions = 21
        number_sequence = np.arange(input_number, 0, -1)
        repeated_array = np.repeat(number_sequence, repetitions)
        return repeated_array

    def _validate_params(self):
        """
        Validate and initialize strategy parameters.
        Converts month-based inputs to trading days and verifies that the ROC periods exceed the WMA window.
        """
        self.roc1_months = max(1, int(self.params.get('roc1_months', 14)))
        self.roc2_months = max(1, int(self.params.get('roc2_months', 11)))
        self.roc1_days = self.roc1_months * 21  # Approximate trading days per month.
        self.roc2_days = self.roc2_months * 21
        # wma_lookback is specified in units of months (default 10) then converted to days.
        self.wma_lookback = max(5, int(self.params.get('wma_lookback', 10))) * 21
        self.method = self.params.get('method', 'zero_crossing')
        self.long_only = self.params.get('long_only', True)

        if min(self.roc1_days, self.roc2_days) <= self.wma_lookback:
            self.logger.warning("ROC periods should exceed the WMA window for meaningful signals")

    def _init_risk_params(self):
        """
        Initialize risk management parameters.
        """
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.10)
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.0)
        self.slippage_pct = self.params.get('slippage_pct', 0.001)
        self.transaction_cost_pct = self.params.get('transaction_cost_pct', 0.001)

    def _init_strength_params(self):
        """
        Initialize momentum strength parameters.
        """
        self.strength_window = int(self.params.get('strength_window', 504))
        self.normalize_strength = self.params.get('normalize_strength', True)

    def _get_validated_prices(self, ticker: str, start_date: str = None, 
                              end_date: str = None, min_periods: int = 252) -> pd.DataFrame:
        """
        Retrieve and validate historical OHLC price data for a given ticker.
        
        The method retrieves the data from the database and then forward fills and drops rows with missing values.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            min_periods (int): Minimum required records.
        
        Returns:
            pd.DataFrame: Validated DataFrame containing 'close', 'high', and 'low'.
            If insufficient, returns an empty DataFrame.
        """
        try:
            prices = self.get_historical_prices(
                tickers=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_periods if not (start_date or end_date) else None
            )
            if not self._validate_data(prices, min_periods):
                return pd.DataFrame()
            return prices[['close', 'high', 'low']].ffill().dropna()
        except Exception as e:
            self.logger.error(f"Price retrieval failed: {e}")
            return pd.DataFrame()

    def _calculate_coppock_curve(self, close: pd.Series) -> pd.Series:
        """
        Calculate the Coppock Curve indicator from the series of closing prices.
        
        The computation uses two ROC measures (over roc1_days and roc2_days), sums them,
        and applies a weighted moving average with linearly increasing weights.
        
        Args:
            close (pd.Series): Series of closing prices.
        
        Returns:
            pd.Series: The Coppock Curve indicator.
        """
        roc1 = close.pct_change(self.roc1_days) * 100
        roc2 = close.pct_change(self.roc2_days) * 100
        combined_roc = (roc1 + roc2).dropna()
        # Generate weights; convert window in days back to the original count (months) for weight sequence.
        weights = self.generate_repeated_array(int(self.wma_lookback / 21))
        wma = combined_roc.rolling(
            window=self.wma_lookback,
            min_periods=int(self.wma_lookback * 0.8)
        ).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
            raw=True
        )
        return wma

    def _generate_raw_signals(self, prices: pd.DataFrame, cc: pd.Series) -> pd.DataFrame:
        """
        Generate raw trading signals based on the Coppock Curve using the zero-crossing method.
        
        A long signal (1) is triggered when the Coppock Curve, after 4 consecutive negative values,
        crosses above 0; a short signal (-1) is generated when 4 consecutive positive values turn negative.
        Additionally, momentum strength is computed as the divergence from a 21-day reference.
        
        Args:
            prices (pd.DataFrame): Price data with 'close', 'high', and 'low'.
            cc (pd.Series): Coppock Curve indicator.
        
        Returns:
            pd.DataFrame: DataFrame with columns: 'close', 'high', 'low', 'signal', 'strength'.
        """
        df = prices.assign(cc=cc).dropna()
        if df.empty:
            return df

        shifted_cc = df['cc'].shift(1)
        df['was_negative'] = (shifted_cc < 0).rolling(4, min_periods=4).sum() == 4
        df['was_positive'] = (shifted_cc > 0).rolling(4, min_periods=4).sum() == 4

        df['signal'] = np.select(
            [df['was_negative'] & (df['cc'] > 0),
             df['was_positive'] & (df['cc'] < 0)],
            [1, -1],
            default=0
        )
        
        ref = df['cc'].rolling(21, min_periods=1).mean().shift()
        df['strength_raw'] = (df['cc'] - ref) / ref.abs().replace(0, 1)
        
        if self.normalize_strength:
            rolling_min = df['strength_raw'].rolling(self.strength_window, min_periods=1).min()
            rolling_max = df['strength_raw'].rolling(self.strength_window, min_periods=1).max()
            df['strength'] = 2 * ((df['strength_raw'] - rolling_min) / (rolling_max - rolling_min).replace(0, 1)) - 1
        else:
            df['strength'] = df['strength_raw']

        if self.long_only:
            df['signal'] = df['signal'].clip(lower=0)
            
        return df[['close', 'high', 'low', 'signal', 'strength']]

    def _generate_raw_direction_signals(self, prices: pd.DataFrame, cc: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on sustained directional changes in the Coppock Curve.
        
        This method computes daily changes, monitors sustained trends over a given period, evaluates
        the previous trend (using a lookback window), and then triggers signals on confirmed reversals.
        
        Args:
            prices (pd.DataFrame): DataFrame containing 'close', 'high', and 'low'.
            cc (pd.Series): Coppock Curve indicator.
        
        Returns:
            pd.DataFrame: DataFrame with columns: 'close', 'high', 'low', 'signal', 'strength'.
        """
        df = prices.assign(cc=cc).dropna()
        if df.empty:
            return df

        sustain_days = self.params.get('sustain_days', 4)
        trend_strength = self.params.get('trend_strength_threshold', 0.75)

        df['direction'] = np.sign(df['cc'].diff())
        df['sustained_up'] = (
            df['direction']
            .rolling(window=sustain_days, min_periods=sustain_days)
            .min() > 0
        )
        df['sustained_down'] = (
            df['direction']
            .rolling(window=sustain_days, min_periods=sustain_days)
            .max() < 0
        )

        lookback = max(sustain_days * 2, 10)
        df['prev_direction'] = df['direction'].shift(sustain_days)
        df['prev_trend'] = (
            df['prev_direction']
            .rolling(window=lookback, min_periods=lookback)
            .apply(
                lambda x: 1 if np.nanmean(x) > trend_strength else -1 if np.nanmean(x) < -trend_strength else 0,
                raw=True
            )
        )

        df['signal'] = np.select(
            [
                df['sustained_up'] & (df['prev_trend'] < 0) & (df['direction'] > 0),
                df['sustained_down'] & (df['prev_trend'] > 0) & (df['direction'] < 0)
            ],
            [1, -1],
            default=0
        )

        ref = df['cc'].rolling(21, min_periods=1).mean().shift()
        df['strength_raw'] = (df['cc'] - ref) / ref.abs().replace(0, 1)
        
        if self.normalize_strength:
            rolling_min = df['strength_raw'].rolling(self.strength_window, min_periods=1).min()
            rolling_max = df['strength_raw'].rolling(self.strength_window, min_periods=1).max()
            df['strength'] = 2 * ((df['strength_raw'] - rolling_min) / (rolling_max - rolling_min).replace(0, 1)) - 1
        else:
            df['strength'] = df['strength_raw']

        if self.long_only:
            df['signal'] = df['signal'].clip(lower=0)

        return df[['close', 'high', 'low', 'signal', 'strength']]

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Incorporate risk management adjustments (stop-loss, take-profit, slippage, transaction costs)
        on the raw trading signals.
        
        Uses the RiskManager to compute:
            - Adjusted entry prices.
            - Exit conditions with realized trade return.
            - Cumulative returns.
        
        Args:
            signals (pd.DataFrame): DataFrame with 'signal', 'close', 'high', 'low' and 'strength'.
            initial_position (int): Starting position (0: flat, 1: long, -1: short).
        
        Returns:
            pd.DataFrame: DataFrame including columns: 'close', 'high', 'low', 'signal', 'position',
                          'return', 'cumulative_return', 'exit_type', and 'strength'.
        """
        risk_manager = RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            slippage_pct=self.slippage_pct,
            transaction_cost_pct=self.transaction_cost_pct
        )
        try:
            managed = risk_manager.apply(
                signals[['signal', 'close', 'high', 'low']].copy(),
                initial_position=initial_position
            )
            return managed.join(signals[['strength']], how='left')
        except Exception as e:
            self.logger.error(f"Risk management failed: {e}")
            return pd.DataFrame()

    def _process_latest_signal(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the most recent signal from the signals DataFrame.
        
        This is useful for forecasting/end-of-day decisions.
        
        Args:
            signals (pd.DataFrame): DataFrame with trading signals.
        
        Returns:
            pd.DataFrame: DataFrame containing only the last available signal.
        """
        return signals.iloc[[-1]].dropna()

    def generate_signals(self, ticker: Union[str, List[str]], start_date: str = None,
                         end_date: str = None, initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals using the Coppock Curve indicator with integrated risk management.
        
        The method supports backtesting (with a specific start/end date) and forecasting (returning only the latest signal).
        It accepts either a single ticker (str) or a list of tickers. For multiple tickers, the function retrieves
        data in one query and then groups the processing per ticker.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol or list of symbols.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting trading position (default=0).
            latest_only (bool): If True, returns only the last signal (per ticker in multi-ticker scenarios).
        
        Returns:
            pd.DataFrame: DataFrame containing:
                - 'close', 'high', 'low': Price data.
                - 'signal': Raw trading signal (before risk management).
                - 'position': Final position (after risk management adjustments).
                - 'return': Realized return on exit events.
                - 'cumulative_return': Cumulative risk-managed return.
                - 'exit_type': Indicator for the type of exit (e.g. 'stop_loss', 'take_profit', 'signal_exit', 'none').
                - 'strength': Momentum strength indicator.
                In view (forecast) mode, only the last available row per ticker is returned.
        """
        # Determine required minimum periods from indicator lookbacks.
        min_periods = max(self.roc1_days, self.roc2_days) + self.wma_lookback + 21

        # Multi-ticker processing.
        if isinstance(ticker, list):
            # Retrieve prices for all tickers (returns a MultiIndex DataFrame: ticker and date)
            all_prices = self.get_historical_prices(
                tickers=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_periods if not (start_date or end_date) else None
            )
            if all_prices.empty:
                return pd.DataFrame()
            signals_list = []
            # Process each ticker group separately.
            for tkr, group in all_prices.groupby(level='ticker'):
                group = group.reset_index(level='ticker', drop=True)
                if len(group) < min_periods:
                    continue
                prices = group[['close', 'high', 'low']].ffill().dropna()
                if prices.empty:
                    continue
                cc = self._calculate_coppock_curve(prices['close'])
                if self.method == 'zero_crossing':
                    sig = self._generate_raw_signals(prices, cc)
                else:
                    sig = self._generate_raw_direction_signals(prices, cc)
                if sig.empty:
                    continue
                sig_rm = self._apply_risk_management(sig, initial_position)
                # Add a ticker column.
                sig_rm['ticker'] = tkr
                signals_list.append(sig_rm)
            if not signals_list:
                return pd.DataFrame()
            df_signals = pd.concat(signals_list)
            # Re-set multi-index as (ticker, date)
            if 'ticker' in df_signals.columns:
                df_signals.set_index('ticker', append=True, inplace=True)
            df_signals.sort_index(inplace=True)
            if latest_only:
                df_signals = df_signals.groupby(level='ticker').apply(lambda group: group.iloc[[-1]]).reset_index(level=0, drop=False)
            return df_signals

        # Single ticker processing.
        else:
            prices = self._get_validated_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                min_periods=min_periods
            )
            if prices.empty:
                return pd.DataFrame()
            cc = self._calculate_coppock_curve(prices['close'])
            if self.method == 'zero_crossing':
                signals = self._generate_raw_signals(prices, cc)
            else:
                signals = self._generate_raw_direction_signals(prices, cc)
            if signals.empty:
                return signals
            if latest_only:
                return self._process_latest_signal(signals)
            return self._apply_risk_management(signals, initial_position)

    def __repr__(self):
        """
        Return a string representation of the CoppockCurveStrategy instance including key parameters.
        """
        return (f"CoppockCurveStrategy(ROC={self.roc1_months}/{self.roc2_months}mo, "
                f"WMA={self.wma_lookback}, SL={self.stop_loss_pct*100:.1f}%, TP={self.take_profit_pct*100:.1f}%)")