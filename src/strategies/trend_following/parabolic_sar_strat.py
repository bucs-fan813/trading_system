# trading_system/src/strategies/trend_following/parabolic_sar.py

# TODO: Long Only

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class ParabolicSAR(BaseStrategy):
    """
    Parabolic SAR Strategy with Integrated Risk Management Component.

    This strategy implements the Parabolic Stop and Reverse (SAR) indicator to generate 
    trading signals for identifying trend reversals. The PSAR indicator is computed 
    recursively as follows:

        For an uptrend:
            PSAR[i] = PSAR[i-1] + AF[i-1]*(EP[i-1] - PSAR[i-1])
            Then PSAR is bounded by the minimum of the previous two lows.
            A trend reversal is signaled if the current low falls below the PSAR.

        For a downtrend:
            PSAR[i] = PSAR[i-1] - AF[i-1]*(PSAR[i-1] - EP[i-1])
            Then PSAR is bounded by the maximum of the previous two highs.
            A trend reversal is signaled if the current high exceeds the PSAR.

    Here:
        - AF (acceleration factor) begins at 'initial_af' and increments by 'af_step'
          on a new extreme price (EP) until reaching 'max_af'.
        - EP is the highest high (for an uptrend) or the lowest low (for a downtrend).

    An optional ATR filter may be applied to verify that the price movement is significant.
    Signals are generated on trend reversals:
        - A buy signal (signal = 1) is generated when the trend reverses from downtrend 
          to uptrend.
        - A sell signal (signal = -1) is generated when the trend reverses from uptrend 
          to downtrend.

    The signal strength is computed as:
        signal_strength = signal * (|close - sar| / (atr + 1e-6))
    where ATR is the Average True Range. This offers a normalized measure of how far away 
    the current price is from the SAR value.

    Risk management is integrated using an external RiskManager. After signals are generated, 
    stop loss and take profit thresholds (adjusted for slippage and transaction costs) are applied 
    and risk–managed trade returns are computed.

    Inputs:
        tickers (str or List[str]): Stock ticker symbol or list of ticker symbols.
        start_date (str, optional): Backtesting start date (YYYY-MM-DD).
        end_date (str, optional): Backtesting end date (YYYY-MM-DD).
        initial_position (int): Starting trading position (default=0, meaning no position).
        latest_only (bool): If True, returns only the final row (per ticker) for forecasting.
    
    Outputs:
        A DataFrame (or multi-index DataFrame if multiple tickers) containing the following columns:
            - open, high, low, close, volume: Price and volume data.
            - sar: Calculated PSAR value.
            - trend: Trend indicator (1 for uptrend, -1 for downtrend).
            - atr: Average True Range (ATR) value.
            - signal: Trading signal (1 for buy, -1 for sell, 0 for none).
            - signal_strength: Normalized signal strength.
            - position: Position after applying risk management.
            - return: Realized trade return when an exit is triggered.
            - cumulative_return: Cumulative return over time from risk–managed trades.
            - exit_type: Reason for risk management exit (stop_loss, take_profit, signal_exit).

    Strategy Parameters (provided via params dictionary; defaults are shown):
        - initial_af: Initial acceleration factor (default: 0.02).
        - max_af: Maximum acceleration factor (default: 0.2).
        - af_step: Acceleration factor step (default: 0.02).
        - use_atr_filter: Whether to apply ATR filtering (default: False).
        - atr_period: Lookback period for ATR calculation (default: 14).
        - atr_threshold: ATR threshold multiplier for filtering (default: 1.0).
        - stop_loss_pct: Stop loss percentage (default: 0.05).
        - take_profit_pct: Take profit percentage (default: 0.10).
        - slippage_pct: Slippage percentage (default: 0.001).
        - transaction_cost_pct: Transaction cost percentage (default: 0.001).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Parabolic SAR strategy with risk management parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters.
        """
        default_params = {
            'initial_af': 0.02,
            'max_af': 0.2,
            'af_step': 0.02,
            'use_atr_filter': False,
            'atr_period': 14,
            'atr_threshold': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)

    def generate_signals(self,
                         tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on the Parabolic SAR indicator integrated with risk management.
        
        The function retrieves historical price data from the database, computes the PSAR, trend, ATR, and 
        then generates signals and computes signal strength. Risk management rules (stop loss, take profit, 
        slippage, and transaction costs) are applied via the RiskManager component.
        
        Args:
            tickers (str or List[str]): Stock ticker symbol or list of ticker symbols.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting trading position (default=0, no initial position).
            latest_only (bool): If True, returns only the most recent signal for each ticker.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, indicator values, generated signals,
                          signal strength, and risk–managed trade metrics.
        """
        # If no explicit date filter is provided, use a default lookback of 252 records.
        lookback = None
        if not (start_date or end_date):
            lookback = 252

        # Retrieve historical price data from DB (data includes open, high, low, close, volume)
        price_data = self.get_historical_prices(tickers, lookback=lookback,
                                                from_date=start_date, to_date=end_date)
        if price_data.empty or not self._validate_data(price_data, min_records=20):
            self.logger.warning("Insufficient data to generate Parabolic SAR signals")
            return pd.DataFrame()

        # Process either a single ticker or multiple tickers using vectorized operations.
        if isinstance(tickers, str) or (isinstance(price_data.index, pd.DatetimeIndex)):
            # Single ticker; process directly.
            result = self._process_single_ticker(price_data, initial_position)
        else:
            # Multiple tickers: price_data is expected to have a MultiIndex (ticker, date).
            # Process each ticker group separately.
            def process_group(group):
                return self._process_single_ticker(group, initial_position)
            # Group by the ticker level (assumed to be level 0)
            result = price_data.groupby(level=0, group_keys=True).apply(process_group)
            # After groupby-apply the index becomes (ticker, original_date); ensure it is sorted.
            result.index = result.index.set_names(["ticker", "date"])
            result = result.sort_index()

        # If only the latest signal(s) is/are requested, filter accordingly.
        if latest_only:
            if isinstance(tickers, str):
                result = result.iloc[[-1]]  # last row for single ticker
            else:
                # For multi-ticker, take the last row for each ticker.
                result = result.reset_index(level=0).groupby("ticker", group_keys=False).apply(lambda df: df.iloc[[-1]])
                result.index.name = "date"
                result = result.sort_index()

        return result

    def _process_single_ticker(self, df: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Process a single ticker's price data to compute the Parabolic SAR, generate signals, 
        compute signal strength, and apply risk management rules.
        
        Args:
            df (pd.DataFrame): DataFrame with a DateTime index and at least the following columns:
                               'open', 'high', 'low', 'close', 'volume'.
            initial_position (int): The starting position for risk management.
            
        Returns:
            pd.DataFrame: DataFrame that includes price data, indicators, signals, and risk-managed 
                          outputs such as position, realized return, cumulative return, and exit type.
        """
        # Calculate the Parabolic SAR and trend indicators.
        sar_values, trend = self._calculate_psar(
            high=df['high'].values,
            low=df['low'].values,
            initial_af=self.params['initial_af'],
            max_af=self.params['max_af'],
            af_step=self.params['af_step']
        )

        # Compute ATR (if filtering is enabled) and define an ATR filter.
        if self.params['use_atr_filter']:
            atr = self._calculate_atr(df.copy(), period=self.params['atr_period'])
            # ATR filter: require the normalized price movement to exceed the filtered threshold.
            price_movement = (df['high'] - df['low']) / df['close']
            atr_filter = price_movement > ((atr / df['close']) * self.params['atr_threshold'])
        else:
            atr = pd.Series(0, index=df.index)
            atr_filter = pd.Series(True, index=df.index)

        # Create a result DataFrame with the price and computed indicator columns.
        result = df.copy()
        result['sar'] = sar_values
        result['trend'] = trend   # 1 for uptrend, -1 for downtrend
        result['atr'] = atr

        # Initialize signals to 0.
        result['signal'] = 0
        trend_change = result['trend'].diff().fillna(0)

        # Generate a buy signal when the trend reverses upward AND passes the ATR filter.
        buy_signal = (trend_change > 0) & atr_filter
        result.loc[buy_signal, 'signal'] = 1

        # Generate a sell signal when the trend reverses downward AND passes the ATR filter.
        sell_signal = (trend_change < 0) & atr_filter
        result.loc[sell_signal, 'signal'] = -1

        # Compute a normalized signal strength using the distance between close and sar,
        # normalized by ATR (with a small epsilon to avoid division by zero).
        result['signal_strength'] = result['signal'] * (np.abs(result['close'] - result['sar']) / (result['atr'] + 1e-6))

        # Remove any rows with missing values.
        result = result.dropna()

        # Apply the risk management rules through the RiskManager component.
        risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )
        result = risk_manager.apply(result, initial_position=initial_position)

        return result

    def _calculate_psar(self,
                        high: np.ndarray,
                        low: np.ndarray,
                        initial_af: float = 0.02,
                        max_af: float = 0.2,
                        af_step: float = 0.02) -> tuple:
        """
        Calculate the Parabolic SAR values and trend for a series of high and low prices.
        
        The computation follows typical PSAR rules:
            - For an uptrend: PSAR = previous PSAR + AF * (previous EP - previous PSAR)
              and is capped by the minimum of the two previous lows.
            - For a downtrend: PSAR = previous PSAR - AF * (previous PSAR - previous EP)
              and is bounded by the maximum of the two previous highs.
            - Trend reversals are detected based on price breaching the SAR level.
        
        Args:
            high (np.ndarray): Array of high prices.
            low (np.ndarray): Array of low prices.
            initial_af (float): Initial acceleration factor.
            max_af (float): Maximum acceleration factor.
            af_step (float): Increment step for the acceleration factor.
            
        Returns:
            tuple: A tuple containing:
                - sar (np.ndarray): The computed Parabolic SAR values.
                - trend (np.ndarray): Trend indicator (1 for uptrend, -1 for downtrend).
        """
        length = len(high)
        sar = np.zeros(length)
        trend = np.zeros(length)   # 1 for uptrend, -1 for downtrend
        ep = np.zeros(length)      # Extreme price
        af = np.zeros(length)      # Acceleration factor

        # Initialize the first bar: here we assume a downtrend by default.
        trend[0] = -1
        sar[0] = high[0]
        ep[0] = low[0]
        af[0] = initial_af

        for i in range(1, length):
            prev_trend = trend[i - 1]

            if prev_trend == 1:
                # For an uptrend: update SAR from the previous value.
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                # Enforce that SAR does not exceed the lows of the previous two bars.
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 1], low[i - 2])
                # Check for reversal: if current low breaches SAR, reverse trend.
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i - 1]  # Reset SAR to the previous extreme price.
                    ep[i] = low[i]      # New extreme is the current low.
                    af[i] = initial_af  # Reset acceleration factor.
                else:
                    trend[i] = 1
                    if high[i] > ep[i - 1]:
                        ep[i] = high[i]
                        af[i] = min(af[i - 1] + af_step, max_af)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:
                # For a downtrend: update SAR accordingly.
                sar[i] = sar[i - 1] - af[i - 1] * (sar[i - 1] - ep[i - 1])
                # Enforce that SAR does not fall below the highs of the previous two bars.
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 1], high[i - 2])
                # Check for reversal: if current high exceeds SAR, reverse trend.
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i - 1]  # Reset SAR to previous extreme.
                    ep[i] = high[i]     # New extreme is the current high.
                    af[i] = initial_af  # Reset acceleration factor.
                else:
                    trend[i] = -1
                    if low[i] < ep[i - 1]:
                        ep[i] = low[i]
                        af[i] = min(af[i - 1] + af_step, max_af)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
        return sar, trend

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR) using Wilder's smoothing method.
        
        The true range (TR) is computed as the maximum of:
            1) Current high minus current low,
            2) Absolute difference between current high and previous close,
            3) Absolute difference between current low and previous close.
        Then, the ATR is the exponentially weighted moving average (with alpha=1/period) of the TR.
        
        Args:
            price_data (pd.DataFrame): DataFrame with columns 'high', 'low', and 'close'.
            period (int): Lookback period for ATR calculation.
            
        Returns:
            pd.Series: ATR values computed for the given period.
        """
        high = price_data['high']
        low = price_data['low']
        prev_close = price_data['close'].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
        return atr