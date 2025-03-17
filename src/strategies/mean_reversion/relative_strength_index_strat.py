# trading_system/src/strategies/rsi_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
import logging
from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig

class RSIStrategy(BaseStrategy):
    """
    RSI Strategy with Integrated Risk Management.

    Mathematical Explanation:
    -------------------------
    The Relative Strength Index (RSI) is computed using Wilder's smoothing method:
    
      RSI = 100 - (100 / (1 + RS))
    
    where RS is defined as:
    
      RS = EMA(gain, period, alpha = 1/period) / EMA(loss, period, alpha = 1/period)
    
    Gains are determined by the positive differences (and zeros for negatives) in consecutive closing
    prices, and losses are similarly processed (as the absolute value of negative differences). Hence,
    if 'oversold' is set to 30 and 'overbought' to 70, a buy signal (signal = 1) is generated when RSI
    crosses above 30 and a sell signal (signal = -1) is generated when RSI crosses below 70. Signal 
    strength is normalized by the range (overbought – oversold) and, in long-only mode, sell signals are 
    neutralized (set to 0).

    Risk Management Integration:
    ---------------------------
    The strategy then passes the combined price and signal DataFrame through a RiskManager that:
      - Adjusts the entry price for slippage and transaction costs,
      - Establishes stop-loss and take-profit thresholds,
      - Identifies exit events (whether due to stop loss, take profit, or signal reversal),
      - Computes realized trade returns and the cumulative return.
    
    The final DataFrame, indexed by date (or by (ticker, date) for multiple tickers), contains:
      - 'open', 'high', 'low', 'close': Price references.
      - 'signal': The trading signal (1 for buy, -1 for sell, 0 for neutral/exit).
      - 'signal_strength': Normalized signal strength.
      - 'rsi': The computed RSI indicator.
      - 'return': The realized trade return following risk management adjustments.
      - 'position': The updated trading position.
      - 'cumulative_return': The cumulative risk–managed return.
      - 'exit_type': The risk management exit reason (e.g., stop_loss, take_profit, or signal_exit).

    This strategy supports backtesting (with a provided date range) as well as live forecasting through
    its 'latest_only' mode. It fully supports vectorized operations for efficiency, even when processing
    a list of tickers.
    
    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Strategy parameters including:
            - 'rsi_period': Lookback period for RSI calculation (default 14).
            - 'overbought': Overbought threshold (default 70.0).
            - 'oversold': Oversold threshold (default 30.0).
            - 'stop_loss_pct': Stop loss percentage (e.g., 0.05).
            - 'take_profit_pct': Take profit percentage (e.g., 0.10).
            - 'trailing_stop_pct': Trailing stop percentage (e.g., 0.0).
            - 'slippage_pct': Slippage percentage (e.g., 0.001).
            - 'transaction_cost_pct': Transaction cost percentage (e.g., 0.001).
            - 'long_only': Flag to allow only long positions (default True).
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
        self.params.setdefault('long_only', True)

    def generate_signals(self, 
                         ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate RSI-based trading signals with integrated risk management.
        
        This method retrieves historical price data, computes the RSI indicator using vectorized and group-based 
        calculations (if multiple tickers are provided), generates buy/sell signals along with normalized signal 
        strength, and applies risk management rules (stop-loss, take-profit, slippage, and transaction costs).
        The output is a comprehensive DataFrame (containing all required columns) suitable for backtesting
        and performance metric computations (e.g., Sharpe ratio, maximum drawdown). In live forecasting mode 
        ('latest_only' True), only the latest signal(s) are returned.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol or list of symbols.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format for backtesting.
            end_date (str, optional): End date in 'YYYY-MM-DD' format for backtesting.
            initial_position (int): Starting trading position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the latest available signal for live trading.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, RSI, generated signals, signal strength, and risk-managed 
                          trade performance metrics, indexed by date (or (ticker, date) for multiple tickers).
        """
        # Validate strategy-specific parameters.
        params = self._validate_parameters()
        rsi_period = params['rsi_period']
        overbought = params['overbought']
        oversold = params['oversold']

        # Retrieve historical price data with an adjusted lookback.
        df = self._get_price_data(ticker, rsi_period, start_date, end_date, latest_only)
        if df.empty:
            self.logger.warning("No historical price data retrieved.")
            return pd.DataFrame()

        # Compute RSI and generate signals in a vectorized manner.
        if isinstance(ticker, list) or (hasattr(df.index, 'nlevels') and df.index.nlevels == 2):
            # Process each ticker group separately.
            def process_group(group):
                group = group.sort_index()  # Ensure chronological order.
                rsi = self._calculate_rsi(group['close'], rsi_period)
                signals, signal_strength = self._generate_rsi_signals(rsi, oversold, overbought)
                group['rsi'] = rsi
                group['signal'] = signals
                group['signal_strength'] = signal_strength
                return group

            signals_df = df.groupby(level=0, group_keys=False).apply(process_group)
        else:
            df = df.sort_index()  # Ensure chronological order.
            rsi = self._calculate_rsi(df['close'], rsi_period)
            signals, signal_strength = self._generate_rsi_signals(rsi, oversold, overbought)
            signals_df = df.copy()
            signals_df['rsi'] = rsi
            signals_df['signal'] = signals
            signals_df['signal_strength'] = signal_strength

        # Extract risk management parameters.
        risk_params = {k: self.params[k] for k in [
            'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct',
            'slippage_pct', 'transaction_cost_pct'
        ] if k in self.params}

        # Apply risk management rules.
        result = self._apply_risk_management(signals_df, risk_params, initial_position, latest_only, start_date, end_date)
        return result

    def _validate_parameters(self) -> dict:
        """
        Validate and return strategy parameters with defaults applied.
        
        Returns:
            dict: Validated strategy parameters.
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

    def _get_price_data(self, 
                        ticker: Union[str, List[str]], 
                        rsi_period: int, 
                        start_date: Optional[str], 
                        end_date: Optional[str],
                        latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data with an adjusted lookback to ensure sufficient data 
        for RSI computation.
        
        For live signal generation (latest_only=True), a minimal lookback of (rsi_period+2) is used.
        For backtesting, if start_date is provided, buffer days (rsi_period*2) are subtracted from start_date.
        Supports both single and multiple tickers.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol or list of symbols.
            rsi_period (int): Lookback period for RSI calculation.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            latest_only (bool): If True, retrieves only the minimal data needed.
        
        Returns:
            pd.DataFrame: Historical price data indexed by date or by (ticker, date) for multiple tickers.
        """
        if latest_only:
            return self.get_historical_prices(ticker, lookback=rsi_period + 2)
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            adjusted_start = (start_dt - pd.DateOffset(days=rsi_period * 2)).strftime('%Y-%m-%d')
            df = self.get_historical_prices(ticker, data_source='yfinance', from_date=adjusted_start, to_date=end_date)
            # For multi-ticker data (with a MultiIndex), filter by the date level.
            if hasattr(df.index, 'nlevels') and df.index.nlevels == 2:
                df = df.loc[(slice(None), start_dt):]
            else:
                df = df[df.index >= start_dt]
            return df
        return self.get_historical_prices(ticker, lookback=rsi_period + 252 * 2, data_source='yfinance')

    def _calculate_rsi(self, close_prices: pd.Series, period: int) -> pd.Series:
        """
        Compute the Relative Strength Index (RSI) using Wilder's smoothing method.
        
        Args:
            close_prices (pd.Series): Series of closing prices.
            period (int): Lookback period for the RSI calculation.
        
        Returns:
            pd.Series: RSI values computed as 100 - (100 / (1 + RS)).
        """
        delta = close_prices.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = (-delta).clip(lower=0).fillna(0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _generate_rsi_signals(self, rsi: pd.Series, oversold: float, overbought: float) -> tuple:
        """
        Generate trading signals based on RSI crossovers and compute normalized signal strength.
        
        A buy signal (1) is generated when RSI crosses above the oversold threshold.
        A sell signal (-1) is generated when RSI crosses below the overbought threshold; in a long-only mode,
        such sell signals are neutralized to 0.
        
        Args:
            rsi (pd.Series): Series of computed RSI values.
            oversold (float): Oversold threshold value.
            overbought (float): Overbought threshold value.
        
        Returns:
            tuple: A tuple of two pd.Series:
                - The first contains trading signals (1 for buy, -1 for sell, 0 otherwise).
                - The second contains the normalized signal strength between 0 and 1.
        """
        buy_signal = (rsi.shift(1) <= oversold) & (rsi > oversold)
        sell_signal = (rsi.shift(1) >= overbought) & (rsi < overbought)
        range_width = overbought - oversold
        buy_strength = ((rsi - oversold) / range_width).clip(lower=0, upper=1)
        sell_strength = ((overbought - rsi) / range_width).clip(lower=0, upper=1)
        signals = np.select([buy_signal, sell_signal], [1, -1], default=0)
        # In long-only mode, neutralize sell signals.
        if self.params.get('long_only', True):
            signals = np.where(sell_signal, 0, signals)
            buy_strength = np.where(sell_signal, 0, buy_strength)
        signal_strength = np.select([buy_signal, sell_signal], [buy_strength, sell_strength], default=0.0)
        return pd.Series(signals, index=rsi.index), pd.Series(signal_strength, index=rsi.index)

    def _apply_risk_management(self, df: pd.DataFrame, risk_params: dict,
                               initial_position: int, latest_only: bool,
                               start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Apply risk management rules to adjust trading signals and positions.
        
        This function uses the RiskManager to:
          - Adjust entry prices to account for slippage and transaction costs.
          - Calculate stop-loss and take-profit thresholds.
          - Detect exit events (stop-loss, take-profit, or signal reversal).
          - Compute the realized trade return and cumulative return.
        
        The method supports multi-ticker processing (using groupby) and ensures that if backtesting is
        requested, the results are filtered to the specified date range. In live mode ('latest_only'),
        only the most recent signal(s) per ticker are returned.
        
        Args:
            df (pd.DataFrame): DataFrame with price data, raw signals, RSI, etc.
            risk_params (dict): Dictionary containing risk management parameters.
            initial_position (int): Starting trading position.
            latest_only (bool): If True, return only the latest signal(s).
            start_date (str, optional): Backtest start date.
            end_date (str, optional): Backtest end date.
        
        Returns:
            pd.DataFrame: DataFrame containing risk-managed positions, realized returns, cumulative returns, 
                          and exit types, indexed by date (or (ticker, date) for multi-ticker data).
        """
        risk_manager = RiskManager(**risk_params)
        # Apply risk management per ticker if processing multiple tickers.
        if hasattr(df.index, 'nlevels') and df.index.nlevels == 2:
            result = df.groupby(level=0, group_keys=False).apply(lambda group: risk_manager.apply(group.sort_index(), initial_position))
        else:
            result = risk_manager.apply(df.sort_index(), initial_position)

        # For backtesting, filter the results by the provided date range.
        if start_date and not latest_only:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) if end_date else None
            if hasattr(result.index, 'nlevels') and result.index.nlevels == 2:
                if end_dt is not None:
                    result = result.loc[(slice(None), start_dt):(slice(None), end_dt)]
                else:
                    result = result.loc[(slice(None), start_dt):]
            else:
                result = result[(result.index >= start_dt) & (result.index <= end_dt) if end_dt is not None else (result.index >= start_dt)]
        # In live mode, return only the last available row per ticker.
        if latest_only:
            if hasattr(result.index, 'nlevels') and result.index.nlevels == 2:
                result = result.groupby(level=0, group_keys=False).tail(1)
            else:
                result = result.iloc[[-1]]
        return result