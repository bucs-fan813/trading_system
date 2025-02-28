import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class SupertrendStrategy(BaseStrategy):
    """
    Supertrend trading strategy with integrated risk management.
    
    Mathematical Description:
    
    1. True Range (TR) is computed as:
         TR = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
    
    2. Average True Range (ATR) is obtained via an exponential moving average of TR over a specified lookback.
    
    3. Basic Bands are then calculated:
         Basic Upper Band = (High + Low) / 2 + (multiplier × ATR)
         Basic Lower Band = (High + Low) / 2 - (multiplier × ATR)
    
    4. Final Bands are computed iteratively:
         For the Final Upper Band:
             final_upper[0] = basic_upper[0]
             For i > 0:
                 if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                     final_upper[i] = basic_upper[i]
                 else:
                     final_upper[i] = final_upper[i-1]
         For the Final Lower Band:
             final_lower[0] = basic_lower[0]
             For i > 0:
                 if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                     final_lower[i] = basic_lower[i]
                 else:
                     final_lower[i] = final_lower[i-1]
    
    5. Supertrend and Trend:
         Initialize:
             if close[0] <= final_upper[0]: 
                 supertrend[0] = final_upper[0]  and trend[0] = -1 (bearish)
             else:
                 supertrend[0] = final_lower[0]  and trend[0] = 1 (bullish)
         For i > 0:
             - If the previous trend was bearish and close[i] <= final_upper[i], then the trend remains bearish
               and the supertrend is set to final_upper[i]. Otherwise, the trend switches to bullish
               and the supertrend is set to final_lower[i].
             - In a bullish regime, if close[i] >= final_lower[i] then trend remains bullish,
               else the trend becomes bearish.
    
    6. Trading Signals:
         A buy (long) signal (1) is generated when a crossover occurs from below to above the supertrend.
         A sell (short) signal (-1) is generated when the price crosses from above to below the supertrend.
         Signal strength is defined as: 
             signal_strength = (|close - supertrend| / close) * signal.
    
    7. Risk Management:
         Once raw signals are generated, risk management is applied to account for slippage,
         transaction costs, stop-loss, and take-profit rules. Realized trade returns and the cumulative
         return series are then computed for the entire dataset.
    
    Hyperparameters:
         lookback (int): Period for ATR calculation (default: 10).
         multiplier (float): Multiplier for ATR to compute bands (default: 3.0).
         long_only (bool): If True, restricts trading to long positions only.
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
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate Supertrend trading signals and apply risk management.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting (YYYY-MM-DD).
            end_date (str, optional): End date for backtesting (YYYY-MM-DD).
            initial_position (int): Starting trading position (0 for none, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the most recent signal row for forecasting.
        
        Returns:
            pd.DataFrame: DataFrame containing OHLC prices, computed supertrend, the trading signal,
                          signal strength, risk-managed positions, trade returns, cumulative returns,
                          and exit event annotations.
        """
        # Retrieve historical price data.
        data = self._get_price_data(ticker, start_date, end_date, latest_only)
        if data.empty:
            return pd.DataFrame()

        # Calculate Supertrend indicator.
        data = self._calculate_supertrend(data)
        # Generate raw trading signals.
        signals = self._generate_trading_signals(data)
        # Apply risk management adjustments.
        signals = self._apply_risk_management(signals, initial_position)
        
        return signals.tail(1) if latest_only else signals

    def _get_price_data(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        latest_only: bool
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLC data with sufficient lookback for indicator stability.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str or None): Start date for data retrieval.
            end_date (str or None): End date for data retrieval.
            latest_only (bool): If True, minimizes data retrieval to compute the latest signal.
        
        Returns:
            pd.DataFrame: DataFrame containing historical open, high, low, close, and volume columns.
        """
        lookback = self.params['lookback']
        min_records = max(lookback * 3, 252)  # Ensure sufficient historical records
        
        try:
            return self.get_historical_prices(
                ticker=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_records if latest_only else None
            )
        except DataRetrievalError as e:
            self.logger.error(f"Data retrieval failed: {e}")
            return pd.DataFrame()

    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Supertrend indicator including TR, ATR, basic and final bands,
        and determine both the supertrend and trend direction.
        
        Args:
            data (pd.DataFrame): DataFrame with historical OHLC data.
        
        Returns:
            pd.DataFrame: The input DataFrame enriched with columns for ATR, basic bands,
                          final bands, supertrend, and trend direction.
        """
        lookback = self.params['lookback']
        multiplier = self.params['multiplier']
        
        # Compute True Range (TR) and Average True Range (ATR)
        hl = data['high'] - data['low']
        hc = (data['high'] - data['close'].shift(1)).abs()
        lc = (data['low'] - data['close'].shift(1)).abs()
        data['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        data['atr'] = data['tr'].ewm(span=lookback, adjust=False).mean()
        
        # Compute Basic Upper and Lower Bands.
        hl2 = (data['high'] + data['low']) / 2
        data['basic_upper'] = hl2 + multiplier * data['atr']
        data['basic_lower'] = hl2 - multiplier * data['atr']
        
        # Compute Final Upper and Lower Bands using recursive rules.
        data['final_upper'] = self._calculate_final_upper(data)
        data['final_lower'] = self._calculate_final_lower(data)
        
        # Compute the Supertrend line and Trend direction.
        data['supertrend'], data['trend'] = self._calculate_supertrend_trend(data)
        
        return data

    def _calculate_final_upper(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Final Upper Band for the Supertrend indicator.
        
        For each period:
            final_upper[0] = basic_upper[0]
            For i > 0:
                if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                    final_upper[i] = basic_upper[i]
                else:
                    final_upper[i] = final_upper[i-1]
        
        Args:
            data (pd.DataFrame): DataFrame containing 'basic_upper' and 'close' columns.
            
        Returns:
            pd.Series: Series containing the recursively computed final upper band.
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
        
        For each period:
            final_lower[0] = basic_lower[0]
            For i > 0:
                if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                    final_lower[i] = basic_lower[i]
                else:
                    final_lower[i] = final_lower[i-1]
        
        Args:
            data (pd.DataFrame): DataFrame containing 'basic_lower' and 'close' columns.
            
        Returns:
            pd.Series: Series containing the recursively computed final lower band.
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
        Determine the Supertrend and corresponding trend (-1 for bearish, 1 for bullish).
        
        Initialization:
            - If close[0] <= final_upper[0] then set supertrend[0]=final_upper[0] and trend[0]=-1;
              otherwise, set supertrend[0]=final_lower[0] and trend[0]=1.
        For each subsequent period:
            - If the previous trend is bearish and close <= final_upper, trend remains bearish.
              Otherwise, the trend switches to bullish (and vice versa in a bullish regime).
        
        Args:
            data (pd.DataFrame): DataFrame containing 'final_upper', 'final_lower', and 'close'.
        
        Returns:
            Tuple[pd.Series, pd.Series]:
                - A Series representing the computed supertrend.
                - A Series representing the trend direction.
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
        Generate raw trading signals based on the supertrend crossover.
        
        Signal Logic:
            - A buy signal (1) occurs when the close price crosses above the supertrend.
            - A sell signal (-1) occurs when the close price crosses below the supertrend.
            - Signal strength = (|close - supertrend| / close) * signal.
            - If long_only is True, negative signals are clipped to 0.
        
        Args:
            data (pd.DataFrame): DataFrame containing 'close' and 'supertrend' columns.
        
        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                          'open', 'high', 'low', 'close', 'supertrend', 'signal', 'signal_strength'.
        """
        cross_above = (data['close'].shift(1) < data['supertrend'].shift(1)) & (data['close'] > data['supertrend'])
        cross_below = (data['close'].shift(1) > data['supertrend'].shift(1)) & (data['close'] < data['supertrend'])
        
        data['signal'] = np.select(
            [cross_above, cross_below],
            [1, -1],
            default=0
        )
        data['signal_strength'] = ((data['close'] - data['supertrend']).abs() / data['close']) * data['signal']
        
        if self.long_only:
            data['signal'] = data['signal'].clip(lower=0)
        
        return data[['open', 'high', 'low', 'close', 'supertrend', 'signal', 'signal_strength']]

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Apply risk management to the raw signals, adjusting for slippage, transaction costs,
        stop-loss, and take-profit rules. Also computes realized and cumulative trade returns.
        
        Args:
            signals (pd.DataFrame): DataFrame containing raw trading signals and price data.
            initial_position (int): The starting position (0, 1, or -1).
        
        Returns:
            pd.DataFrame: A DataFrame with additional columns for 'position', 'return',
                          'cumulative_return', and 'exit_type'.
        """
        if not signals.empty:
            return self.risk_manager.apply(
                signals,
                initial_position=initial_position
            )
        return pd.DataFrame()

    def _validate_parameters(self):
        """
        Validate that the strategy hyperparameters are within acceptable ranges.
        
        Raises:
            ValueError: If the lookback period is less than or equal to 1 or if multiplier is non-positive.
        """
        if self.params['lookback'] <= 1:
            raise ValueError("Lookback period must be greater than 1")
        if self.params['multiplier'] <= 0:
            raise ValueError("Multiplier must be positive")