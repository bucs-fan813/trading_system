# trading_system/src/strategies/choppiness_index_strat.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager


class ChoppinessIndexStrategy(BaseStrategy):
    """
    Choppiness Index Strategy with MACD confirmation and integrated Risk Management.
    
    This strategy uses the Choppiness Index (CI) and the MACD indicator to generate
    trading signals. The CI is calculated from the True Range over a period and indicates
    whether the market is trending (low CI) or choppy (high CI). Only when CI is below 38.2,
    signaling trending conditions, are MACD crossovers considered valid. The signal strength
    is further derived from the MACD histogram and scaled by the trending condition.
    
    Risk management (stop-loss, take-profit, slippage, and transaction cost adjustments) is 
    applied to each trade, ensuring that positions are exited either when risk thresholds are 
    breached or when a reversal signal is detected.
    
    Mathematical formulas:
        - True Range (TR):
            TR[i] = max( high[i]-low[i], |high[i]-close[i-1]|, |low[i]-close[i-1]| )
        - Choppiness Index (CI):
            CI = 100 * log10( sum(TR over n periods) / (max(high) - min(low) over n periods) ) / log10(n)
        - MACD:
            MACD = EMA_fast(close) - EMA_slow(close)
            Signal Line = EMA(MACD, smooth)
            Histogram = MACD - Signal Line
        - Signal strength (when CI < 38.2):
            strength = (|MACD - Signal Line| / close) * (1 - (CI / 38.2))
    
    Attributes:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict): Strategy-specific parameters (e.g., ci_period, macd_fast, macd_slow, macd_smooth,
                       stop_loss_pct, take_profit_pct, slippage_pct, transaction_cost_pct).
        logger (logging.Logger): Logger instance.
        db_engine: SQLAlchemy engine for database connection.
        risk_manager (RiskManager): Instance of RiskManager for applying stop-loss, take-profit, etc.
        long_only (bool): If True, only long positions are allowed.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Choppiness Index strategy along with its RiskManager.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy and risk management parameters.
        """
        super().__init__(db_config, params)
        self.long_only = self.params.get('long_only', True)
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )
        
    def generate_signals(self,
                         ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals with integrated risk management for backtesting/forecasting.
        
        The method retrieves historical price data using a lookback period determined by the 
        maximum of the CI period and the MACD slow period. It then calculates the Choppiness Index (CI) 
        and MACD-based indicators. Trading signals are generated when the market is trending (CI < 38.2)
        and confirmed by MACD crossovers. Signal strength is computed as a combination of the MACD 
        histogram magnitude and the trending condition. Lastly, the risk management component applies 
        adjustments for slippage, transaction costs, stop-loss, and take-profit rules.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting (format YYYY-MM-DD).
            end_date (str, optional): End date for backtesting (format YYYY-MM-DD).
            initial_position (int): The initial trading position (0, 1 for long, -1 for short).
            latest_only (bool): If True, generate and return the signal for the latest available date.
            
        Returns:
            pd.DataFrame: DataFrame containing the following columns:
                - 'close': Closing prices.
                - 'high': High prices.
                - 'low': Low prices.
                - 'signal': The generated trading signal.
                - 'position': Position after risk management adjustments.
                - 'return': Realized trade return on exits.
                - 'cumulative_return': Cumulative return over time.
                - 'exit_type': Type of exit event (stop_loss, take_profit, or signal_exit).
        """
        # Determine lookback period based on indicators (ensure enough data for both CI and MACD)
        lookback = max(self.params.get('ci_period', 14), 26)  # MACD's slow period is 26 by default.
        prices = self._get_prices_with_lookback(ticker, start_date, end_date, lookback, latest_only)
        
        if not self._validate_data(prices, min_records=lookback * 2):
            raise DataRetrievalError("Insufficient data for analysis")
            
        # Calculate the Choppiness Index (CI) and MACD indicators
        df = self._calculate_indicators(prices)
        
        # Generate vectorized trading signals based on CI and MACD conditions
        signals = self._generate_vectorized_signals(df)
        
        # Integrate risk management: apply stop-loss, take-profit with slippage and transaction cost adjustments
        results = self.risk_manager.apply(signals, initial_position)
        
        return self._format_output(results, latest_only)

    def _get_prices_with_lookback(self, ticker: str, start_date: Optional[str], end_date: Optional[str], 
                                  lookback: int, latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical prices with a lookback period suitable for indicator calculations.
        
        If forecasting for the latest date, a sufficient number of bars (lookback*2) is fetched to 
        ensure stability in the indicator calculations. For backtesting, data is fetched using the provided 
        date range.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Backtest start date (or None for the latest data scenario).
            end_date (str): Backtest end date (or None for the latest data scenario).
            lookback (int): Base lookback period for indicator calculation.
            latest_only (bool): If True, fetch only enough data for a single-date forecast.
            
        Returns:
            pd.DataFrame: DataFrame containing historical prices.
        """
        if latest_only:
            return self.get_historical_prices(
                ticker=ticker,
                lookback=lookback * 2,  # Ensure enough data for indicator calculations
                from_date=None,
                to_date=None
            )
        return self.get_historical_prices(
            ticker=ticker,
            from_date=start_date,
            to_date=end_date
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and append the Choppiness Index (CI) and MACD indicators to the price DataFrame.
        
        The True Range (TR) is calculated per bar as:
            TR = max( high - low, |high - previous close|, |low - previous close| ).
        The CI is then computed as:
            CI = 100 * log10(sum(TR over n periods) / (max(high) over n periods - min(low) over n periods)) / log10(n)
        Additionally, the MACD is computed using exponential moving averages (EMA) with configurable parameters.
        
        Args:
            df (pd.DataFrame): DataFrame with columns 'high', 'low', and 'close'.
            
        Returns:
            pd.DataFrame: DataFrame with additional columns:
                - 'ci': Choppiness Index.
                - 'macd': MACD line.
                - 'signal_line': EMA of the MACD.
                - 'histogram': Difference between MACD and the signal line.
        """
        ci_period = self.params.get('ci_period', 14)
        # Calculate True Range (TR) from high, low, and previous close values.
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        sum_tr = tr.rolling(ci_period).sum()
        max_high = df['high'].rolling(ci_period).max()
        min_low = df['low'].rolling(ci_period).min()
        df['ci'] = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(ci_period)
        
        # Calculate MACD and associated indicators
        fast = self.params.get('macd_fast', 12)
        slow = self.params.get('macd_slow', 26)
        smooth = self.params.get('macd_smooth', 9)
        
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=smooth, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal_line']
        
        return df.dropna()

    def _generate_vectorized_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals in a fully vectorized manner using CI and MACD criteria.
        
        Bullish conditions:
            - MACD > signal_line and CI < 38.2.
        Bearish conditions:
            - MACD < signal_line and CI < 38.2.
        The signal strength is further computed as:
            strength = (|histogram|/close) * (1 - (CI / 38.2))
        Positions are maintained until a reversal signal is generated using forward-fill.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data and the computed indicators.
            
        Returns:
            pd.DataFrame: DataFrame with the following columns:
                - 'close', 'high', 'low': Price data.
                - 'signal': Trading signal (+1 for long, -1 for short, 0 for neutral).
                - 'strength': Signal strength metric.
        """
        # Define bullish and bearish conditions using MACD and CI levels
        bullish = (df['macd'] > df['signal_line']) & (df['ci'] < 38.2)
        bearish = (df['macd'] < df['signal_line']) & (df['ci'] < 38.2)
        
        # Compute signal strength only when in trending (CI < 38.2) conditions
        df['strength'] = 0.0
        mask = df['ci'] < 38.2
        df.loc[mask, 'strength'] = ((abs(df['histogram']) / df['close']) * (1 - (df['ci'] / 38.2)))
        
        # Generate raw signals: 1 for bullish, -1 for bearish
        df['raw_signal'] = 0
        df.loc[bullish, 'raw_signal'] = 1
        
        if self.long_only:
            df.loc[bearish, 'raw_signal'] = 0
        else:
            df.loc[bearish, 'raw_signal'] = -1
        
        # Forward-fill the signal so that positions persist until a reversal
        df['signal'] = df['raw_signal'].replace(0, np.nan).ffill().fillna(0)
        
        return df[['close', 'high', 'low', 'signal', 'strength']]

    def _format_output(self, df: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Format the output DataFrame to include essential columns for backtesting or forecasting.
        
        The output includes price data, generated signals, risk-managed positions, individual trade
        returns, cumulative returns, and the type of exit event. When forecasting in real time (latest_only),
        only the most recent observation is returned.
        
        Args:
            df (pd.DataFrame): DataFrame with risk-managed signals and performance metrics.
            latest_only (bool): Indicator whether to return only the latest available signal.
            
        Returns:
            pd.DataFrame: Reformatted DataFrame with columns:
                'close', 'high', 'low', 'signal', 'position', 'return', 
                'cumulative_return', 'exit_type'
        """
        required_cols = ['close', 'high', 'low', 'signal', 'position',
                         'return', 'cumulative_return', 'exit_type']
        df = df[required_cols]
        
        if latest_only:
            return df.iloc[[-1]]  # Return the last row for current day forecast
        return df

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the DataFrame has sufficient and clean data for indicator calculations.
        
        Checks include ensuring the DataFrame is not empty, has at least the minimum required records,
        and that there are no missing values in essential price columns (close, high, low).
        
        Args:
            df (pd.DataFrame): DataFrame to validate.
            min_records (int): Minimum number of required records.
            
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