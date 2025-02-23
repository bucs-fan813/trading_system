# trading_system/src/strategies/rsi_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig  # Ensure DatabaseConfig is imported


class RSIStrategy(BaseStrategy):
    """
    RSI Strategy Implementation

    This strategy calculates the Relative Strength Index (RSI) using Wilder's smoothing method, mathematically defined as:

        RSI = 100 - (100 / (1 + RS))
        where RS = (Average Gain) / (Average Loss)

    The average gain and loss are computed via an exponential moving average with smoothing factor alpha = 1/period,
    ensuring proper Wilder's smoothing. A buy signal is generated when the RSI crosses above the oversold threshold 
    (e.g., 30), and a sell signal is generated when it crosses below the overbought threshold (e.g., 70). The 
    signal strength is normalized based on the distance from the threshold:
        - For buy signals: (RSI - oversold) / (overbought - oversold)
        - For sell signals: (overbought - RSI) / (overbought - oversold)

    Risk management is integrated via the RiskManager class, applying stop-loss and take-profit rules as a percentage 
    of the entry price, adjusted for slippage and transaction costs. The exit conditions include:
        - Stop loss: For long positions, triggered if the day's low falls below entry_price * (1 - stop_loss_pct);
                     For short positions, triggered if the day's high exceeds entry_price * (1 + stop_loss_pct).
        - Take profit: For long positions, triggered if the day's high exceeds entry_price * (1 + take_profit_pct);
                       For short positions, triggered if the day's low falls below entry_price * (1 - take_profit_pct).
        - Signal exit: Triggered when a reversal in the trading signal is detected.

    The strategy supports backtesting over a specified date range or generating the latest signal for end-of-day trading.
    The output DataFrame comprises all required data to compute subsequent performance metrics (e.g., Sharpe ratio,
    maximum drawdown) downstream.

    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Dictionary of strategy parameters. Expected keys include:
            - 'rsi_period': Lookback period for RSI (default 14).
            - 'overbought': Overbought threshold (default 70.0).
            - 'oversold': Oversold threshold (default 30.0).
            - 'stop_loss_pct': Stop loss percentage (e.g., 0.05).
            - 'take_profit_pct': Take profit percentage (e.g., 0.10).
            - 'slippage_pct': Slippage percentage (e.g., 0.001).
            - 'transaction_cost_pct': Transaction cost percentage (e.g., 0.001).
            - 'start_date': Backtest start date (YYYY-MM-DD). [Optional]
            - 'end_date': Backtest end date (YYYY-MM-DD). [Optional]
            - 'initial_position': Starting position (0 for flat, 1 for long, -1 for short). [Optional, default 0]
            - 'latest_only': If True, return only the latest signal for live trading. [Optional, default False]
            - 'long_only': If True, restrict trading to long positions only. [Optional, default True]

    Methods:
        generate_signals: Generates RSI signals with integrated risk management for a given ticker.

    Returns:
        pd.DataFrame: DataFrame containing 'close', 'high', 'low', 'signal', 'signal_strength', 'rsi', 'return', 
                      'position', 'cumulative_return', and 'exit_type', indexed by date.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the RSI Strategy with database configuration and parameters.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters.
        """
        super().__init__(db_config, params or {})
        self.logger = logging.getLogger(self.__class__.__name__)

    # === CHANGE 1: Update method signature to match the base class ===
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate RSI-based trading signals with integrated risk management.

        This method retrieves historical price data, computes the RSI, generates trading signals 
        and normalized signal strength based on RSI threshold crossovers, and applies risk management 
        rules including stop-loss, take-profit, slippage, and transaction costs. For backtesting, the 
        full data frame is returned, allowing calculation of performance metrics. For live trading, 
        only the latest signal is generated.

        Args:
            ticker (str): The asset ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with columns 'close', 'high', 'low', 'signal', 'signal_strength', 'rsi', 
                          'return', 'position', 'cumulative_return', and 'exit_type', indexed by date.
        """
        start_date = self.params.get('start_date')
        end_date = self.params.get('end_date')
        initial_position = self.params.get('initial_position', 0)
        latest_only = self.params.get('latest_only', False)
        long_only = self.params.get('long_only', True)

        # Validate strategy-specific parameters
        params = self._validate_parameters()
        rsi_period = params['rsi_period']
        overbought = params['overbought']
        oversold = params['oversold']

        # Retrieve historical price data with an appropriate lookback period
        df = self._get_price_data(ticker, rsi_period, start_date, end_date, latest_only)
        if df.empty:
            self.logger.warning("No historical price data retrieved.")
            return pd.DataFrame()

        # Compute RSI using vectorized operations and Wilder's smoothing
        rsi = self._calculate_rsi(df['close'], rsi_period)
        
        # Generate trading signals and corresponding normalized signal strength
        signals, signal_strength = self._generate_rsi_signals(rsi, oversold, overbought)
        
        # Combine price data with generated signals and RSI
        signals_df = df.join(pd.DataFrame({
            'signal': signals,
            'signal_strength': signal_strength,
            'rsi': rsi
        }))

        # Extract risk management related parameters from the strategy parameters
        risk_params = {k: self.params[k] for k in [
            'stop_loss_pct', 'take_profit_pct', 
            'slippage_pct', 'transaction_cost_pct'
        ] if k in self.params}

        # Apply integrated risk management rules
        return self._apply_risk_management(
            signals_df, 
            risk_params,
            initial_position,
            latest_only,
            start_date,
            end_date
        )

    def _validate_parameters(self) -> dict:
        """
        Validate and return strategy parameters.

        Returns:
            dict: Dictionary containing validated strategy parameters.
        """
        params = {
            'rsi_period': self.params.get('rsi_period', 14),
            'overbought': self.params.get('overbought', 70.0),
            'oversold': self.params.get('oversold', 30.0)
        }

        if params['rsi_period'] < 1:
            raise ValueError("RSI period must be ≥1")
        if not (0 < params['oversold'] < params['overbought'] < 100):
            raise ValueError("Thresholds must satisfy 0 < oversold < overbought < 100")
        
        return params

    def _get_price_data(self, ticker: str, rsi_period: int, 
                        start_date: Optional[str], end_date: Optional[str],
                        latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data with an adjusted lookback period.

        The method ensures sufficient data is retrieved to compute the RSI over the given period,
        including additional buffer days if a specific start date is requested for backtesting.

        Args:
            ticker (str): The asset ticker symbol.
            rsi_period (int): Lookback period for RSI calculation.
            start_date (str, optional): Start date for backtesting.
            end_date (str, optional): End date for backtesting.
            latest_only (bool): If True, retrieves minimal data required for the latest signal.

        Returns:
            pd.DataFrame: DataFrame containing historical price data indexed by date.
        """
        if latest_only:
            return self.get_historical_prices(ticker, lookback=rsi_period + 2)

        if start_date:
            start_dt = pd.to_datetime(start_date)
            adjusted_start = (start_dt - pd.DateOffset(days=rsi_period*2)).strftime('%Y-%m-%d')
            df = self.get_historical_prices(
                ticker, 
                data_source='yfinance',
                from_date=adjusted_start,
                to_date=end_date
            )
            return df[df.index >= start_dt] if not df.empty else df

        return self.get_historical_prices(
            ticker, 
            lookback=rsi_period + 252*2,  # 2 years buffer for extensive backtests
            data_source='yfinance'
        )

    def _calculate_rsi(self, close_prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.

        The RSI is computed as:
            RSI = 100 - (100 / (1 + RS))
        where RS is the ratio of the exponentially weighted moving average of gains to losses.

        Args:
            close_prices (pd.Series): Series of closing prices.
            period (int): Lookback period for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = close_prices.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = (-delta).clip(lower=0).fillna(0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _generate_rsi_signals(self, rsi: pd.Series, 
                              oversold: float, overbought: float) -> tuple:
        """
        Generate buy and sell signals based on RSI crossovers with threshold boundaries.

        A buy signal (1) is triggered when the RSI crosses above the oversold threshold, and a sell signal (-1)
        is triggered when the RSI crosses below the overbought threshold. The method also computes the normalized 
        signal strength based on the distance from the respective threshold.

        Args:
            rsi (pd.Series): Series containing RSI values.
            oversold (float): Oversold threshold.
            overbought (float): Overbought threshold.

        Returns:
            tuple: A tuple containing:
                - pd.Series: Generated signals (1 for buy, -1 for sell, 0 otherwise).
                - pd.Series: Normalized signal strength between 0 and 1.
        """
        buy_signal = (rsi.shift(1) <= oversold) & (rsi > oversold)
        sell_signal = (rsi.shift(1) >= overbought) & (rsi < overbought)

        range_width = overbought - oversold
        buy_strength = ((rsi - oversold) / range_width).clip(lower=0, upper=1)
        sell_strength = ((overbought - rsi) / range_width).clip(lower=0, upper=1)

        signals = np.select(
            [buy_signal, sell_signal],
            [1, -1],
            default=0
        )

        if self.long_only:
            signals[sell_signal] = 0  # Convert sell signals to neutral for long-only strategy

        signal_strength = np.select(
            [buy_signal, sell_signal],
            [buy_strength, sell_strength],
            default=0.0
        )

        return pd.Series(signals, index=rsi.index), pd.Series(signal_strength, index=rsi.index)

    def _apply_risk_management(self, df: pd.DataFrame, risk_params: dict,
                               initial_position: int, latest_only: bool,
                               start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Apply risk management to the generated signals and format the final output.

        This method utilizes the RiskManager class to integrate stop-loss, take-profit, slippage, and 
        transaction cost adjustments into the trading signals. For backtesting, the output DataFrame is 
        filtered to the specified date range and sorted chronologically. For live signal generation, only 
        the latest available signal is returned.

        Args:
            df (pd.DataFrame): DataFrame containing price data and generated signals.
            risk_params (dict): Dictionary of risk management parameters.
            initial_position (int): Starting trading position.
            latest_only (bool): If True, returns only the latest signal.
            start_date (str, optional): Start date for backtesting.
            end_date (str, optional): End date for backtesting.

        Returns:
            pd.DataFrame: DataFrame enriched with risk-managed positions, returns, and cumulative performance.
        """
        if df.empty:
            return df

        risk_manager = RiskManager(**risk_params)
        result = risk_manager.apply(df, initial_position)

        if start_date and not latest_only:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) if end_date else result.index.max()
            result = result.loc[start_dt:end_dt]

        if latest_only:
            return result.iloc[[-1]]  # Return only the latest row for live trading

        return result.sort_index(ascending=True)