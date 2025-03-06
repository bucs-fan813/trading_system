# trading_system/src/strategies/volatility/supertrend_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union, List
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class SupertrendStrategy(BaseStrategy):
    """
    Supertrend Strategy with Integrated Risk Management.

    This strategy computes the Supertrend indicator which is used to generate trading signals.
    The indicator is computed as follows:

      1. True Range (TR) is calculated as:
            TR = max( High - Low, |High - Previous Close|, |Low - Previous Close| )

      2. The Average True Range (ATR) is computed using an exponential moving average over a specified lookback period:
            ATR = EMA(TR, span=lookback)

      3. Basic Bands are derived from:
            Basic Upper Band = (High + Low) / 2 + (multiplier × ATR)
            Basic Lower Band = (High + Low) / 2 - (multiplier × ATR)

      4. Final Bands are obtained recursively:
            Final Upper Band[0] = Basic Upper Band[0]
            For i > 0:
                if Basic Upper Band[i] < Final Upper Band[i-1] or Close[i-1] > Final Upper Band[i-1]:
                    Final Upper Band[i] = Basic Upper Band[i]
                else:
                    Final Upper Band[i] = Final Upper Band[i-1]
            Final Lower Band[0] = Basic Lower Band[0]
            For i > 0:
                if Basic Lower Band[i] > Final Lower Band[i-1] or Close[i-1] < Final Lower Band[i-1]:
                    Final Lower Band[i] = Basic Lower Band[i]
                else:
                    Final Lower Band[i] = Final Lower Band[i-1]

      5. Supertrend and Trend determination:
         Initialize:
             if Close[0] <= Final Upper Band[0]:
                 then Supertrend[0] = Final Upper Band[0] and trend = -1 (bearish)
             else:
                 Supertrend[0] = Final Lower Band[0] and trend = +1 (bullish)
         For i > 0:
             If previous trend is bearish (-1):
                 if Close[i] <= Final Upper Band[i]:
                     then trend remains -1 and Supertrend[i] = Final Upper Band[i]
                 else:
                     trend switches to +1 and Supertrend[i] = Final Lower Band[i]
             If previous trend is bullish (+1):
                 if Close[i] >= Final Lower Band[i]:
                     then trend remains +1 and Supertrend[i] = Final Lower Band[i]
                 else:
                     trend switches to -1 and Supertrend[i] = Final Upper Band[i]

      6. Trading Signals:
         - A buy signal (+1) is generated when the price crosses from below to above the Supertrend.
         - A sell signal (-1) is generated when the price crosses from above to below the Supertrend.
         - Signal strength is computed as:
                signal_strength = (|Close - Supertrend| / Close) * signal
           In “long only” mode (if enabled) negative signals are clipped to 0.

      7. Risk Management is applied using the RiskManager component. It adjusts the entry price taking into account
         slippage and transaction cost, and then applies stop-loss and take-profit rules. The realized return and cumulative
         return are computed across the whole dataset so that downstream performance metrics (e.g. Sharpe ratio, max drawdown)
         can be easily calculated.

    Strategy Inputs and Hyperparameters:
      - ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
      - start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
      - end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
      - initial_position (int): The starting position (0, 1, or -1).
      - latest_only (bool): If True, returns only the final row per ticker (useful for forecasting).
      - params (dict, optional): Strategy-specific parameters with defaults:
            • 'lookback'   : ATR lookback period (default: 10)
            • 'multiplier' : ATR multiplier for bands (default: 3.0)
            • 'long_only'  : If True, only long positions are allowed (default: True)

    Outputs:
      A DataFrame containing:
        - OHLC price data (open, high, low, close).
        - Computed Supertrend indicator and trend.
        - Raw trading signals and signal strength.
        - Risk-managed trading position, realized trade return, cumulative return, and exit event annotations.
    """
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        default_params = {'lookback': 10, 'multiplier': 3.0, 'long_only': True}
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_manager = RiskManager()  # Initialize with default risk parameters
        self.long_only = bool(self.params.get('long_only', True))

    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate Supertrend trading signals and apply risk management adjustments.

        Retrieves the historical price data in a vectorized manner (supporting multiple tickers),
        computes the Supertrend indicator and corresponding trend, generates raw signals based on
        price crossovers of the Supertrend line, and finally applies risk management (stop-loss,
        take-profit, slippage, transaction cost) via the RiskManager.

        Args:
            ticker (str or List[str]): Stock ticker symbol or a list of ticker symbols.
            start_date (str, optional): Backtesting start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtesting end date in 'YYYY-MM-DD' format.
            initial_position (int): The initial trading position.
            latest_only (bool): If True, only the most recent signal row is returned (per ticker for multi-ticker).

        Returns:
            pd.DataFrame: DataFrame with OHLC prices, computed indicators, raw signals, risk-managed positions,
                          trade returns, cumulative returns, and exit event annotations.
        """
        # Retrieve historical OHLC data with sufficient lookback for stable indicator estimation.
        data = self._get_price_data(ticker, start_date, end_date, latest_only)
        if data.empty:
            self.logger.warning("No historical data found.")
            return pd.DataFrame()

        # If multiple tickers are provided, process each ticker (grouped) separately.
        if isinstance(ticker, list):
            groups = []
            # Multi-index DataFrame: group by the first index level (ticker)
            for t, group in data.groupby(level=0):
                group = group.reset_index(level=0, drop=True).sort_index()
                group = self._calculate_supertrend(group)
                group = self._generate_trading_signals(group)
                group = self._apply_risk_management(group, initial_position)
                group['ticker'] = t
                groups.append(group)
            signals = pd.concat(groups).sort_index()
            if latest_only:
                signals = signals.groupby('ticker', group_keys=False).tail(1)
        else:
            # Process single ticker data.
            data = data.sort_index()
            data = self._calculate_supertrend(data)
            signals = self._generate_trading_signals(data)
            signals = self._apply_risk_management(signals, initial_position)
            if latest_only:
                signals = signals.tail(1)
        return signals

    def _get_price_data(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str],
        end_date: Optional[str],
        latest_only: bool
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLC price data from the database with a sufficient lookback.

        The number of records retrieved is at least max(lookback*3, 252) if latest_only is True;
        otherwise, all data within the specified date range is returned.

        Args:
            ticker (str or List[str]): Stock ticker symbol or list of tickers.
            start_date (str or None): Start date in 'YYYY-MM-DD' format.
            end_date (str or None): End date in 'YYYY-MM-DD' format.
            latest_only (bool): Flag to indicate if only minimal data is needed for the latest signal.

        Returns:
            pd.DataFrame: DataFrame containing historical OHLC data (and ticker column for multi-ticker).
        """
        lookback = self.params['lookback']
        min_records = max(lookback * 3, 252)  # Minimum records necessary for indicator stability
        try:
            return self.get_historical_prices(
                tickers=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_records if latest_only else None
            )
        except DataRetrievalError as e:
            self.logger.error("Data retrieval failed: %s", e)
            return pd.DataFrame()

    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Supertrend indicator components and determine the trend.

        This method computes the True Range (TR), Average True Range (ATR), basic bands, and then
        recursively calculates the final upper and lower bands. Based on these, the Supertrend
        and trend direction are determined.

        Args:
            data (pd.DataFrame): DataFrame with columns 'high', 'low', and 'close'.

        Returns:
            pd.DataFrame: Enriched DataFrame with columns for ATR, basic bands, final bands,
                          supertrend, and trend.
        """
        lookback = self.params['lookback']
        multiplier = self.params['multiplier']

        # Calculate True Range (TR) and Average True Range (ATR)
        hl = data['high'] - data['low']
        hc = (data['high'] - data['close'].shift(1)).abs()
        lc = (data['low'] - data['close'].shift(1)).abs()
        data['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        data['atr'] = data['tr'].ewm(span=lookback, adjust=False).mean()

        # Compute the Basic Upper and Lower Bands
        hl2 = (data['high'] + data['low']) / 2
        data['basic_upper'] = hl2 + multiplier * data['atr']
        data['basic_lower'] = hl2 - multiplier * data['atr']

        # Compute Final Upper and Lower Bands with recursive adjustments.
        data['final_upper'] = self._calculate_final_upper(data)
        data['final_lower'] = self._calculate_final_lower(data)

        # Compute the Supertrend line and Trend direction.
        data['supertrend'], data['trend'] = self._calculate_supertrend_trend(data)

        return data

    def _calculate_final_upper(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Final Upper Band for the Supertrend indicator.

        For each bar:
          - For the first period, Final Upper Band = Basic Upper Band.
          - For subsequent periods, if the current Basic Upper Band is lower than
            the previous Final Upper Band or if the previous close is greater than
            the previous Final Upper Band, then take the current Basic Upper Band;
            otherwise, carry forward the previous Final Upper Band.

        Args:
            data (pd.DataFrame): DataFrame containing 'basic_upper' and 'close'.

        Returns:
            pd.Series: Series of the recursively computed Final Upper Band.
        """
        final_upper = data['basic_upper'].copy()
        for i in range(1, len(final_upper)):
            if (data['basic_upper'].iloc[i] < final_upper.iloc[i-1]) or (data['close'].iloc[i-1] > final_upper.iloc[i-1]):
                final_upper.iloc[i] = data['basic_upper'].iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
        return final_upper

    def _calculate_final_lower(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Final Lower Band for the Supertrend indicator.

        For each bar:
          - For the first period, Final Lower Band = Basic Lower Band.
          - For subsequent periods, if the current Basic Lower Band is higher than
            the previous Final Lower Band or if the previous close is lower than
            the previous Final Lower Band, then take the current Basic Lower Band;
            otherwise, carry forward the previous Final Lower Band.

        Args:
            data (pd.DataFrame): DataFrame containing 'basic_lower' and 'close'.

        Returns:
            pd.Series: Series of the recursively computed Final Lower Band.
        """
        final_lower = data['basic_lower'].copy()
        for i in range(1, len(final_lower)):
            if (data['basic_lower'].iloc[i] > final_lower.iloc[i-1]) or (data['close'].iloc[i-1] < final_lower.iloc[i-1]):
                final_lower.iloc[i] = data['basic_lower'].iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
        return final_lower

    def _calculate_supertrend_trend(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Determine the Supertrend level and the trend direction.

        Initialization:
            - If the first close is less than or equal to Final Upper Band,
              then set Supertrend[0] = Final Upper Band[0] and trend = -1 (bearish);
              otherwise, set Supertrend[0] = Final Lower Band[0] and trend = +1 (bullish).
        For subsequent periods:
            - If the previous trend is bearish (-1):
                • Remains bearish if current close is below or equal to Final Upper Band,
                  setting Supertrend to Final Upper Band.
                • Otherwise, switches to bullish (+1) and Supertrend is set to Final Lower Band.
            - If the previous trend is bullish (+1):
                • Remains bullish if current close is above or equal to Final Lower Band,
                  setting Supertrend to Final Lower Band.
                • Otherwise, switches to bearish (-1) and Supertrend is set to Final Upper Band.

        Args:
            data (pd.DataFrame): DataFrame containing 'final_upper', 'final_lower', and 'close'.

        Returns:
            Tuple[pd.Series, pd.Series]:
                - Series for the computed Supertrend.
                - Series for the trend direction (-1 for bearish, +1 for bullish).
        """
        supertrend = np.zeros(len(data))
        trend = np.zeros(len(data))
        if data['close'].iloc[0] <= data['final_upper'].iloc[0]:
            supertrend[0] = data['final_upper'].iloc[0]
            trend[0] = -1
        else:
            supertrend[0] = data['final_lower'].iloc[0]
            trend[0] = 1

        for i in range(1, len(data)):
            if trend[i-1] == -1:
                if data['close'].iloc[i] <= data['final_upper'].iloc[i]:
                    supertrend[i] = data['final_upper'].iloc[i]
                    trend[i] = -1
                else:
                    supertrend[i] = data['final_lower'].iloc[i]
                    trend[i] = 1
            else:
                if data['close'].iloc[i] >= data['final_lower'].iloc[i]:
                    supertrend[i] = data['final_lower'].iloc[i]
                    trend[i] = 1
                else:
                    supertrend[i] = data['final_upper'].iloc[i]
                    trend[i] = -1
        return pd.Series(supertrend, index=data.index), pd.Series(trend, index=data.index)

    def _generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw trading signals based on the Supertrend indicator crossovers.

        A buy signal (+1) is generated when the price crosses above the Supertrend from below,
        and a sell signal (-1) is generated when the price crosses below from above. The signal
        strength is computed as the normalized distance between the close and Supertrend line,
        multiplied by the signal value. If 'long_only' is enabled, negative signals (short entries)
        are converted to 0.

        Args:
            data (pd.DataFrame): DataFrame containing 'open', 'high', 'low', 'close', and 'supertrend'.

        Returns:
            pd.DataFrame: DataFrame with columns: 'open', 'high', 'low', 'close', 'supertrend',
                          'signal' and 'signal_strength'.
        """
        cross_above = (data['close'].shift(1) < data['supertrend'].shift(1)) & (data['close'] > data['supertrend'])
        cross_below = (data['close'].shift(1) > data['supertrend'].shift(1)) & (data['close'] < data['supertrend'])
        data['signal'] = np.select([cross_above, cross_below], [1, -1], default=0)
        data['signal_strength'] = ((data['close'] - data['supertrend']).abs() / data['close']) * data['signal']
        if self.long_only:
            data['signal'] = data['signal'].clip(lower=0)
        return data[['open', 'high', 'low', 'close', 'supertrend', 'signal', 'signal_strength']]

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Apply risk management rules to raw signals.

        Uses the RiskManager component to adjust for slippage, transaction cost, and applies
        stop-loss and take-profit rules. Additionally, this function computes realized trade returns
        and cumulative returns.

        Args:
            signals (pd.DataFrame): DataFrame with raw trading signals and associated price data.
            initial_position (int): Starting trading position.

        Returns:
            pd.DataFrame: DataFrame with additional columns including 'position', 'return',
                          'cumulative_return', and 'exit_type'.
        """
        if not signals.empty:
            return self.risk_manager.apply(signals, initial_position=initial_position)
        return pd.DataFrame()