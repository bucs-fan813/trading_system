# trading_system/src/strategies/adx_rsi_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict
from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig  # Ensure DatabaseConfig is imported for type hints

class ADXRSIStrategy(BaseStrategy):
    """
    ADX with RSI Strategy implementing vectorized operations and risk management.
    
    Mathematical Details:
    -----------------------
    - True Range (TR) is computed as:
        TR = max( high - low,
                  abs(high - prev_close),
                  abs(low - prev_close) )
    - Average True Range (ATR) is computed as the exponential moving average (EMA) of TR.
    - Directional Movements:
          PlusDM = high.diff() if high.diff() > low.diff() and high.diff() > 0, else 0.
          MinusDM = -low.diff() if -low.diff() > high.diff() and -low.diff() > 0, else 0.
    - Directional Indicators:
          PlusDI = EMA(PlusDM, period=adx_lookback) / ATR * 100
          MinusDI = EMA(MinusDM, period=adx_lookback) / ATR * 100
    - The Directional Index (DX) is given by:
          DX = abs(PlusDI - MinusDI) / (PlusDI + MinusDI + 1e-12) * 100
    - ADX is then computed as the EMA of DX over the lookback period.
    - Relative Strength Index (RSI) is computed using:
          RSI = 100 - (100 / (1 + RS))
      where RS is the ratio of EMA(gains) to EMA(losses). Gains and losses are computed
      from the change in closing prices.
      
    Signal Generation:
    -------------------
    - Buy Signal (1): When ADX > adx_threshold, PlusDI < MinusDI, and RSI < rsi_buy_level.
    - Sell Signal (-1): When ADX > adx_threshold, PlusDI > MinusDI, and RSI > rsi_sell_level.
    - No Signal (0): Otherwise.
    
    These binary signals are subsequently risk managed using the RiskManager class which applies
    stop-loss, take-profit, slippage, and transaction cost adjustments, and computes trade returns
    and cumulative performance.
    
    Attributes:
        adx_lookback (int): Lookback period for ADX and directional indicators.
        rsi_lookback (int): Lookback period for RSI calculation.
        adx_threshold (float): Minimum ADX value required to consider a signal.
        rsi_buy_level (float): RSI threshold below which a buy signal is generated.
        rsi_sell_level (float): RSI threshold above which a sell signal is generated.
        risk_manager (RiskManager): Instance to apply risk management rules.
        long_only (bool): If True, only long positions are allowed.
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the ADXRSIStrategy with database configuration and strategy parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Dictionary with strategy parameters. Possible keys include:
                - adx_lookback (int): Lookback for ADX and directional indicators.
                - rsi_lookback (int): Lookback for RSI.
                - adx_threshold (float): ADX strength threshold.
                - rsi_buy_level (float): RSI level for buy signals.
                - rsi_sell_level (float): RSI level for sell signals.
                - stop_loss_pct (float): Stop loss percentage.
                - take_profit_pct (float): Take profit percentage.
                - slippage_pct (float): Slippage percentage.
                - transaction_cost_pct (float): Transaction cost percentage.
        """
        super().__init__(db_config, params)
        self.params = params or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Strategy parameters
        self.adx_lookback = self.params.get('adx_lookback', 14)
        self.rsi_lookback = self.params.get('rsi_lookback', 14)
        self.adx_threshold = self.params.get('adx_threshold', 35)
        self.rsi_buy_level = self.params.get('rsi_buy_level', 50)
        self.rsi_sell_level = self.params.get('rsi_sell_level', 50)
        self.long_only = self.params.get('long_only', True)
        
        # Risk management initialization
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
        Generate trading signals with integrated risk management as well as performance metrics.
        
        This method retrieves historical price data (by either a date range or a minimal lookback),
        computes the ADX, RSI, and directional indicators, and then generates a binary trade signal.
        Risk management (stop-loss, take-profit, slippage, transaction cost) is applied to the signals
        so that trade returns and cumulative performance are computed. The final DataFrame output contains
        all necessary information for further analysis (e.g., Sharpe ratio, drawdown).
        
        Args:
            ticker (str): Asset ticker symbol.
            start_date (str, optional): Backtest start date (YYYY-MM-DD). Required for full backtesting.
            end_date (str, optional): Backtest end date (YYYY-MM-DD).
            initial_position (int): Starting position (0 for no position, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the most recent signal for end-of-day decision making.
        
        Returns:
            pd.DataFrame: DataFrame containing columns for prices, technical indicators, trading signals,
                          risk-managed positions, returns, cumulative returns, and exit types.
                          When latest_only is True, only the final row is returned.
        """
        # Data retrieval: use a limited lookback for real-time (latest) signal generation
        df = self._get_price_data(ticker, start_date, end_date, latest_only)
        if df.empty or not self._validate_data(df, min_records=2 * self.adx_lookback):
            self.logger.error("Insufficient data retrieved for analysis.")
            return pd.DataFrame()

        # Calculate technical indicators (ADX, RSI, plus_di, minus_di)
        df = self._calculate_indicators(df)
        
        # Generate binary trading signals (buy = 1, sell = -1, no action = 0)
        df = self._generate_vectorized_signals(df)
        
        # Apply risk management to compute trade performance metrics
        df = self._apply_risk_management(df, initial_position)
        
        # Append the ticker information and return either the full series or only the final signal as needed.
        df['ticker'] = ticker
        return df.tail(1) if latest_only else df

    def _get_price_data(self, ticker: str, start_date: Optional[str], end_date: Optional[str], latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data for the given ticker symbol.
        
        When latest_only is True, a smaller lookback (based on the maximum indicator lookback plus a buffer)
        is used to speed up computation for end-of-day decision making. Otherwise, a full date range is used.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            latest_only (bool): If True, retrieves only a limited number of recent records.
        
        Returns:
            pd.DataFrame: DataFrame containing columns ['date', 'open', 'high', 'low', 'close', 'volume'].
        """
        if latest_only:
            lookback = 2 * max(self.adx_lookback, self.rsi_lookback) + 50
            return self.get_historical_prices(ticker, lookback=lookback)
        return self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the technical indicators required for signal generation, including
        ADX (with its directional indicators) and RSI.
        
        The calculations are fully vectorized for speed:
          - True Range (TR) is derived from current high, low, and previous close.
          - ATR (Average True Range) is the EMA of TR.
          - PlusDM and MinusDM are computed from the differences in high and low prices.
          - PlusDI and MinusDI are computed as EMAs of directional movements divided by ATR.
          - DX is computed from the relative difference of PlusDI and MinusDI.
          - ADX is the EMA of DX.
          - RSI is computed from price changes, gains, and losses via an EMA.
        
        Args:
            df (pd.DataFrame): DataFrame that must contain at least 'high', 'low', and 'close' columns.
        
        Returns:
            pd.DataFrame: The input DataFrame with additional columns:
                - 'adx': The Average Directional Index.
                - 'rsi': The Relative Strength Index.
                - 'plus_di': The Plus Directional Indicator.
                - 'minus_di': The Minus Directional Indicator.
              Rows with insufficient data for indicator calculation are dropped.
        """
        # Extract price series for clarity.
        high, low, close = df['high'], df['low'], df['close']
        
        # True Range (TR) computation.
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Average True Range (ATR) computed using an exponential moving average.
        atr = tr.ewm(alpha=1 / self.adx_lookback, adjust=False).mean()
        
        # Compute directional movements.
        plus_dm = high.diff()
        minus_dm = -low.diff()
        pos_mask = (plus_dm > minus_dm) & (plus_dm > 0)
        neg_mask = (minus_dm > plus_dm) & (minus_dm > 0)
        
        # Compute directional indicators as an EMA of the directional movements divided by ATR.
        plus_di = (plus_dm.where(pos_mask, 0).ewm(alpha=1 / self.adx_lookback, adjust=False).mean() / atr) * 100
        minus_di = (minus_dm.where(neg_mask, 0).ewm(alpha=1 / self.adx_lookback, adjust=False).mean() / atr) * 100
        
        # Compute DX and then ADX.
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100  # Add a tiny constant to avoid division by zero.
        adx = dx.ewm(alpha=1 / self.adx_lookback, adjust=False).mean()
        
        # Compute RSI from price differences.
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_lookback, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_lookback, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        
        # Append indicators to DataFrame.
        df['adx'] = adx
        df['rsi'] = rsi
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Drop rows that have NaN values from these computations.
        return df.dropna()

    def _generate_vectorized_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary trading signals based on the computed technical indicators.
        
        Signal generation logic:
          - A buy signal (1) is generated when:
                ADX > adx_threshold,
                plus_di < minus_di, and
                RSI < rsi_buy_level.
          - A sell signal (-1) is generated when:
                ADX > adx_threshold,
                plus_di > minus_di, and
                RSI > rsi_sell_level.
          - Otherwise, no signal (0) is generated.
        
        Args:
            df (pd.DataFrame): DataFrame with the columns 'adx', 'rsi', 'plus_di', and 'minus_di'.
        
        Returns:
            pd.DataFrame: The same DataFrame with an additional column 'signal' containing values 1, -1, or 0.
        """
        buy_cond = (df['adx'] > self.adx_threshold) & \
                   (df['plus_di'] < df['minus_di']) & \
                   (df['rsi'] < self.rsi_buy_level)
        
        sell_cond = (df['adx'] > self.adx_threshold) & \
                    (df['plus_di'] > df['minus_di']) & \
                    (df['rsi'] > self.rsi_sell_level)
        
        df['signal'] = np.select([buy_cond, sell_cond], [1, -1], default=0)

        if self.long_only:
            # Override sell signals (-1) with 0 (exit) if long_only is True.
            df.loc[sell_cond, 'signal'] = 0

        return df

    def _apply_risk_management(self, df: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Apply risk management to the generated trading signals and compute trade performance metrics.
        
        The method verifies that required price columns ('signal', 'high', 'low', 'close')
        are available. It then delegates to the RiskManager instance, which applies the stop-loss and
        take-profit rules (with slippage and transaction cost adjustments) and computes realized as well as
        cumulative returns on a vectorized basis.
        
        Args:
            df (pd.DataFrame): DataFrame that includes at least the columns 'signal', 'high', 'low', 'close'.
            initial_position (int): The starting position (0, 1, or -1) for the backtest.
        
        Returns:
            pd.DataFrame: DataFrame containing the risk-managed output with columns such as 'position',
                          'return', 'cumulative_return', and 'exit_type', suitable for backtesting and
                          further performance analysis.
        """
        required_cols = ['signal', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            self.logger.error(f"Missing columns for risk management: {missing}")
            return pd.DataFrame()
            
        return self.risk_manager.apply(df[required_cols], initial_position=initial_position)