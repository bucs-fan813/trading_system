# trading_system/src/strategies/adx_strat.py

import numpy as np
import pandas as pd
from numba import njit
from typing import Optional, Dict, Union, List
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager


class ADXStrategy(BaseStrategy):
    """
    ADX Trend Following Strategy with Integrated Risk Management Component.

    Mathematical Explanation:
    ---------------------------
    The strategy calculates the Average Directional Index (ADX) along with the plus and minus
    Directional Indicators (DI+ and DI–) using the following formulas:

      • True Range (TR):
            TR(i) = max[ high(i) - low(i), |high(i) - close(i-1)|, |low(i) - close(i-1)| ]

      • Directional Movements:
            plusDM(i)  = high_diff = high(i) - high(i-1)   if high_diff > (low(i-1) - low(i)) and high_diff > 0, else 0
            minusDM(i) = low_diff  = low(i-1) - low(i)        if low_diff > (high(i) - high(i-1)) and low_diff > 0, else 0

      • Wilder's Smoothing is applied (using a lookback period, typically 14):
            ATR       = smooth(TR, lookback)
            DI+       = 100 * smooth(plusDM, lookback) / (ATR + epsilon)
            DI–       = 100 * smooth(minusDM, lookback) / (ATR + epsilon)
            DX        = 100 * |DI+ - DI–| / (DI+ + DI– + epsilon)
            ADX       = smooth(DX, lookback)

    Trading Signals:
    ----------------
    A new entry signal is generated only when ADX crosses above 25 from below. The signals are:

      - Long Signal: When ADX crosses above 25 and DI+ > DI–.
      - Short Signal: When ADX crosses above 25 and DI– > DI+.
        • If operating in "long only" mode, short signals are suppressed (set to 0).

    Signal Strength:
      signal_strength = (DI+ - DI–) * ADX / 100
      This scalar serves as a normalized measure of trend strength, which can be later fed into
      an optimizer.

    Risk Management:
    ----------------
    After generating raw signals, the strategy applies risk management (via the RiskManager class)
    to determine the adjusted entry price (incorporating slippage and transaction cost) as well as 
    the computation of stop-loss, take-profit, trailing stop thresholds. Exit events are marked if:
    
      • For long trades: the day's low touches the stop-loss level or the high reaches the target.
      • For short trades: vice versa.
      • A reversal in the raw signal occurs.

    Backtesting and Forecasting:
    -----------------------------
    The strategy supports backtesting on a full date range by computing daily returns, strategy returns,
    and risk-managed cumulative returns. For forecasting, setting latest_only=True returns only the final
    row (or the final row per ticker if multiple tickers are passed).

    Expected Data & Outputs:
    --------------------------
      - Required price data columns: open, high, low, close, volume.
      - Outputs include:
            • Price columns: open, high, low, close, volume.
            • Calculated indicators: plus_di, minus_di, adx.
            • Trading signals: signal and signal_strength.
            • Risk-managed outputs: position, return, cumulative_return, exit_type.
            • Backtest metrics: daily_return and strategy_return.

    Args:
        ticker (str or List[str]): A stock ticker symbol or a list of ticker symbols.
        start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
        end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
        initial_position (int): Starting position (default=0; 0 for none, 1 for long, -1 for short).
        latest_only (bool): If True, returns only the most recent signal (for forecasting).

    Strategy-specific parameters provided in `params` (with defaults):
        - 'adx_lookback'       : Lookback period for ADX smoothing (default: 14).
        - 'long_only'          : If True, only long positions are allowed (default: True).
        - 'stop_loss_pct'      : Stop loss percentage (default: 0.05).
        - 'take_profit_pct'    : Take profit percentage (default: 0.10).
        - 'trailing_stop_pct'  : Trailing stop percentage (default: 0.00).
        - 'slippage'           : Slippage percentage (default: 0.001).
        - 'transaction_cost'   : Transaction cost percentage (default: 0.001).
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the ADX strategy with the given database configuration and strategy parameters.

        Args:
            db_config: Database configuration object.
            params (Dict, optional): Dictionary containing strategy hyperparameters.
        """
        super().__init__(db_config, params)
        self.default_lookback = 14  # Default ADX smoothing period if not provided.
        self.long_only = self.params.get('long_only', True)
        self.adx_threshold = self.params.get('adx_threshold', 25)
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            trailing_stop_pct=self.params.get('trailing_stop_pct', 0.00),
            slippage_pct=self.params.get('slippage', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost', 0.001)
        )

    def generate_signals(self, ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Retrieve historical price data, compute ADX and directional indicators, generate trading signals,
        apply risk management, and compute backtest performance metrics.

        This method supports both single-ticker and multi-ticker (vectorized) processing.

        Args:
            ticker (str or List[str]): A stock ticker symbol or a list of ticker symbols.
            start_date (str, optional): Backtest start date (YYYY-MM-DD). If None, a sufficient lookback
                                        period is used.
            end_date (str, optional): Backtest end date (YYYY-MM-DD).
            initial_position (int): Starting position (default: 0).
            latest_only (bool): If True, returns only the latest signal (or last signal per ticker).

        Returns:
            pd.DataFrame: DataFrame containing price data, computed ADX indicators, trading signals,
                          risk-managed fields, and backtest performance metrics.
        """
        adx_lookback = self.params.get('adx_lookback', self.default_lookback)
        required_lookback = 2 * adx_lookback + 1 if start_date is None else None

        # Process a single ticker
        if isinstance(ticker, str):
            df = self._get_price_data(ticker, start_date, end_date)
            # Compute ADX and directional indicators.
            df['plus_di'], df['minus_di'], df['adx'] = self._calculate_adx(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                adx_lookback
            )
            # Generate raw vectorized signals.
            signals = self._generate_vectorized_signals(df, adx_lookback)
            df = pd.concat([df, signals], axis=1)
            # Apply risk management to adjust for stop loss / take profit, slippage, and transaction cost.
            df = self.risk_manager.apply(df, initial_position)
            # Compute daily percentage returns and strategy returns using the previous period's position.
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['position'].shift() * df['returns']
            return df.iloc[-1:] if latest_only else df

        # Process multiple tickers (vectorized group operations)
        else:
            # Retrieve multi-ticker price data (DataFrame with MultiIndex [ticker, date]).
            df = self.get_historical_prices(
                tickers=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=required_lookback
            )
            if df.empty:
                raise DataRetrievalError("No data returned for the provided tickers.")
            # Ensure only the required columns are selected.
            df = df[['open', 'high', 'low', 'close', 'volume']]
            signals_list = []

            # Group data by ticker (the base method sets a MultiIndex of [ticker, date]).
            for t, group in df.groupby(level=0):
                # Remove the ticker level so that date is the index.
                group = group.reset_index(level=0, drop=True)

                # Validate sufficient data for reliable indicator calculations.
                if len(group) < (2 * adx_lookback + 10):
                    self.logger.warning("Ticker %s has insufficient data for ADX calculation.", t)
                    continue

                # Compute ADX and directional indicators for each ticker.
                group['plus_di'], group['minus_di'], group['adx'] = self._calculate_adx(
                    group['high'].values,
                    group['low'].values,
                    group['close'].values,
                    adx_lookback
                )
                # Generate trading signals using vectorized operations.
                signals = self._generate_vectorized_signals(group, adx_lookback)
                group = pd.concat([group, signals], axis=1)
                # Apply the risk management adjustments.
                group = self.risk_manager.apply(group, initial_position)
                # Calculate returns.
                group['returns'] = group['close'].pct_change()
                group['strategy_returns'] = group['position'].shift() * group['returns']
                # Add a ticker column.
                group['ticker'] = t
                # If forecasting latest_only, retain only the last row.
                if latest_only:
                    group = group.iloc[[-1]]
                # Append the processed group.
                signals_list.append(group)

            if not signals_list:
                raise DataRetrievalError("No tickers with sufficient data were processed.")

            df_final = pd.concat(signals_list)
            # Reset index to include ticker as a column or set a MultiIndex if preferred.
            df_final = df_final.reset_index().set_index(['ticker', 'date'])
            return df_final

    def _get_price_data(self, ticker: str, 
                        start_date: Optional[str], 
                        end_date: Optional[str]) -> pd.DataFrame:
        """
        Retrieve historical price data for a single ticker and ensure that sufficient records exist
        for stable ADX calculations.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date (YYYY-MM-DD). If None, a default lookback is used.
            end_date (str, optional): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with columns 'open', 'high', 'low', 'close', 'volume' indexed by date.

        Raises:
            DataRetrievalError: If there is insufficient data to compute reliable indicators.
        """
        adx_lookback = self.params.get('adx_lookback', self.default_lookback)
        required_lookback = 2 * adx_lookback + 1 if start_date is None else None

        df = self.get_historical_prices(
            ticker=ticker,
            from_date=start_date,
            to_date=end_date,
            lookback=required_lookback
        )
        # Ensure only the columns required for ADX calculations are kept.
        df = df[['open', 'high', 'low', 'close', 'volume']]
        min_records = 2 * adx_lookback + 10  # Buffer for reliable indicator signals.
        if not self._validate_data(df, min_records):
            raise DataRetrievalError("Insufficient data for ADX calculation for ticker: " + ticker)
        return df

    @staticmethod
    @njit
    def _calculate_adx(high: np.ndarray, low: np.ndarray, 
                       close: np.ndarray, lookback: int) -> tuple:
        """
        Compute the directional indicators and the Average Directional Index (ADX) using Wilder's smoothing.

        Args:
            high (np.ndarray): Array of high prices.
            low (np.ndarray): Array of low prices.
            close (np.ndarray): Array of closing prices.
            lookback (int): Lookback period for smoothing.

        Returns:
            tuple: Three NumPy arrays representing DI+ (plus_di), DI– (minus_di), and ADX.
        """
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        # Calculate True Range (TR) and directional movements.
        for i in range(1, n):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]

            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff

        def smooth(series, window):
            """Apply Wilder's smoothing to a series."""
            smoothed = np.zeros(series.shape[0])
            sum_val = 0.0
            # Calculate the initial average.
            for j in range(window):
                sum_val += series[j]
            smoothed[window - 1] = sum_val / window
            # Apply the recursive smoothing.
            for j in range(window, series.shape[0]):
                smoothed[j] = (smoothed[j-1]*(window - 1) + series[j]) / window
            return smoothed

        atr = smooth(tr, lookback)
        epsilon = 1e-10  # Small constant to avoid division by zero.
        atr_safe = atr + epsilon

        plus_di_smoothed = smooth(plus_dm, lookback)
        minus_di_smoothed = smooth(minus_dm, lookback)

        plus_di = 100 * (plus_di_smoothed / atr_safe)
        minus_di = 100 * (minus_di_smoothed / atr_safe)

        diff_di = np.abs(plus_di - minus_di)
        sum_di = plus_di + minus_di
        sum_di_safe = np.where(sum_di == 0, epsilon, sum_di)
        dx = 100 * (diff_di / sum_di_safe)
        adx = smooth(dx, lookback)

        return plus_di, minus_di, adx

    def _generate_vectorized_signals(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Generate trading signals and signal strength based on ADX crossover conditions in a vectorized manner.

        A new signal is generated only when ADX crosses above the adx_threshold value:
            - Long Signal (1): When ADX moves from below adx_threshold to at/above adx_threshold and DI+ exceeds DI–.
            - Short Signal (-1): When ADX moves from below adx_threshold to at/above adx_threshold and DI– exceeds DI+.
              (Overridden to 0 in long-only mode.)
        Signal strength is defined as:
            signal_strength = (DI+ - DI–) * ADX / 100

        Args:
            df (pd.DataFrame): DataFrame containing at least 'adx', 'plus_di', and 'minus_di'.
            lookback (int): Lookback period (not directly used in signal computation).

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - 'signal': Trading signal (1 for long, -1 for short, 0 for no action).
                - 'signal_strength': Numeric strength of the signal.
        """
        signals = pd.DataFrame(index=df.index)
        adx = df['adx']
        pdi = df['plus_di']
        mdi = df['minus_di']

        # Identify when ADX crosses upward through 25.
        adx_cross = (adx.shift(1) < self.adx_threshold) & (adx >= self.adx_threshold)
        long_signal = adx_cross & (pdi > mdi)
        short_signal = adx_cross & (mdi > pdi)

        signal_strength = (pdi - mdi) * adx / 100
        signals['signal'] = 0
        signals.loc[long_signal, 'signal'] = 1
        if self.long_only:
            signals.loc[short_signal, 'signal'] = 0
        else:
            signals.loc[short_signal, 'signal'] = -1
        signals['signal_strength'] = signal_strength

        return signals