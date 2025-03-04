# trading_system/src/strategies/tsi_strategy.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class TSIStrategy(BaseStrategy):
    """
    True Strength Index (TSI) Strategy with Integrated Risk Management.
    
    Mathematical Explanation:
    
        Given the closing price series, first compute the price change:
            Δ = close.diff()
    
        Perform two levels of exponential smoothing:
            double_smoothed = EMA(EMA(Δ, span=long_period), span=short_period)
            abs_double_smoothed = EMA(EMA(|Δ|, span=long_period), span=short_period)
    
        The True Strength Index (TSI) is defined by:
            TSI = 100 * (double_smoothed / (abs_double_smoothed + 1e-10))
    
        The signal line is computed as the exponential moving average (EMA) of TSI:
            signal_line = EMA(TSI, span=signal_period)
    
        Trading signals are generated via crossovers:
            - A bullish (buy) signal (+1) when TSI crosses above the signal line.
            - A bearish (sell) signal (-1) when TSI crosses below the signal line.
              In "long only" mode, sell signals are overridden to 0.
    
    Risk Management:
    
        The strategy then applies risk management using the RiskManager, which:
            - Adjusts the entry price to account for slippage and transaction costs.
            - Sets stop-loss and take-profit thresholds.
            - Identifies exit events (stop-loss, take-profit, or signal reversal).
            - Computes the realized trade returns and cumulative return.
    
    Outputs:
    
        A DataFrame with the following columns:
            - 'tsi': Computed True Strength Index.
            - 'signal_line': Smoothed TSI serving as the signal line.
            - 'strength': The difference (TSI - signal_line).
            - 'signal': Trading signal (1 for buy, -1 for sell, 0 for none).
            - 'position': Risk-managed position.
            - 'return': Realized return when an exit is triggered.
            - 'cumulative_return': Cumulative return from closed trades.
            - 'exit_type': Reason for exit (e.g., stop_loss, take_profit, signal_exit, none).
            - 'close', 'high', 'low': Price data used for entry and risk management.
    
        In a multi-ticker scenario, the DataFrame is indexed by both ticker and date.
    
    Parameters in `params` (with defaults):
        - long_period (int): First smoothing period (default: 25)
        - short_period (int): Second smoothing period (default: 13)
        - signal_period (int): Period for the signal line (default: 12)
        - stop_loss_pct (float): Stop loss percentage (default: 0.05)
        - take_profit_pct (float): Take profit percentage (default: 0.10)
        - slippage_pct (float): Slippage percentage (default: 0.001)
        - transaction_cost_pct (float): Transaction cost percentage (default: 0.001)
        - long_only (bool): If True, only long positions are allowed (default: True)
        - min_data_points (int): Minimum required data points (default: long_period + short_period + signal_period)
    
    Args:
        db_config (DatabaseConfig): Database configuration instance.
        params (dict, optional): Dictionary of strategy parameters.
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the TSIStrategy with database configuration and strategy parameters.
        
        Args:
            db_config: Database configuration settings used to initialize the DB engine.
            params (dict, optional): Dictionary of strategy parameters.
        """
        super().__init__(db_config, params)
        
        self.long_period = int(self.params.get('long_period', 25))
        self.short_period = int(self.params.get('short_period', 13))
        self.signal_period = int(self.params.get('signal_period', 12))
        required_min = self.long_period + self.short_period + self.signal_period
        self.min_data_points = int(self.params.get('min_data_points', required_min))
        
        self.stop_loss_pct = float(self.params.get('stop_loss_pct', 0.05))
        self.take_profit_pct = float(self.params.get('take_profit_pct', 0.10))
        self.slippage_pct = float(self.params.get('slippage_pct', 0.001))
        self.transaction_cost_pct = float(self.params.get('transaction_cost_pct', 0.001))
        self.long_only = bool(self.params.get('long_only', True))
        
        self._validate_parameters()
        self.risk_manager = self._create_risk_manager()
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(self, tickers: Union[str, List[str]], 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals with integrated risk management for given tickers.
        
        This function retrieves historical price data for one or more tickers, computes the TSI and its signal line,
        generates trading signals based on crossovers, applies risk management rules (stop-loss, take-profit, and
        signal-based exits), and outputs a complete DataFrame for backtesting and forecasting.
        
        Args:
            tickers (str or List[str]): Stock ticker symbol or list of symbols.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting market position (0 for none, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the latest available signal row (for forecasting).
        
        Returns:
            pd.DataFrame: DataFrame containing the following columns:
                'tsi', 'signal_line', 'strength', 'signal', 'position', 'return',
                'cumulative_return', 'exit_type', 'close', 'high', 'low'.
                
            In multi-ticker mode, the DataFrame is indexed by both ticker and date.
        
        Raises:
            DataRetrievalError: If the historical price data for any ticker is insufficient.
        """
        self.logger.info(f"Processing tickers {tickers} from {start_date} to {end_date}")
        # Retrieve validated historical prices (supports single or multiple tickers)
        prices = self._get_validated_prices(tickers, start_date, end_date)
        
        # Calculate raw signals with TSI parameters (vectorized per ticker)
        raw_signals = self._calculate_signals(prices)
        
        # Apply risk management rules to the signals
        risk_df = self._apply_risk_management(raw_signals, initial_position)
        
        # Join the risk management outputs with the raw signals to combine all metrics
        combined = raw_signals.join(risk_df[['position', 'return', 'cumulative_return', 'exit_type']], how='left')
        
        # Final data ordering
        final_df = self._finalize_output(combined, latest_only)
        return final_df

    def _validate_parameters(self):
        """
        Validate that strategy and risk parameters are positive (or non-negative as applicable).
        
        Raises:
            ValueError: If any period parameter is non-positive or risk parameter is negative.
        """
        if any(p <= 0 for p in [self.long_period, self.short_period, self.signal_period]):
            raise ValueError("All period parameters must be positive integers")
        if any(p < 0 for p in [self.stop_loss_pct, self.take_profit_pct, self.slippage_pct, self.transaction_cost_pct]):
            raise ValueError("Risk parameters must be non-negative")

    def _create_risk_manager(self) -> RiskManager:
        """
        Create and return an instance of the RiskManager with configured risk parameters.
        
        Returns:
            RiskManager: Configured risk management instance.
        """
        return RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            slippage_pct=self.slippage_pct,
            transaction_cost_pct=self.transaction_cost_pct
        )

    def _get_validated_prices(self, tickers: Union[str, List[str]], 
                              start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Retrieve and validate historical price data from the database based on provided date range.
        
        In multi-ticker mode, ensures each ticker has at least the minimum required number of data points.
        The resulting DataFrame is sorted by date (and ticker if applicable).
        
        Args:
            tickers (str or List[str]): Stock ticker symbol(s).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
            pd.DataFrame: Historical price data sorted by date (and ticker for multiple tickers).
        
        Raises:
            DataRetrievalError: If any ticker has insufficient data for analysis.
        """
        prices = self.get_historical_prices(
            tickers,
            from_date=start_date,
            to_date=end_date,
            data_source='yfinance'
        )
        
        if isinstance(prices.index, pd.MultiIndex):
            insufficient = []
            for ticker, group in prices.groupby(level=0):
                if len(group) < self.min_data_points:
                    insufficient.append(ticker)
            if insufficient:
                raise DataRetrievalError(f"Insufficient data for tickers: {insufficient}")
        else:
            if not self._validate_data(prices, self.min_data_points):
                raise DataRetrievalError("Insufficient data for analysis")
        
        # Ensure data is sorted by date (or ticker and date if multi-ticker)
        return prices.sort_index()

    def _calculate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the True Strength Index (TSI), signal line, and generate trading signals.
        
        In single-ticker mode, processes the entire DataFrame; in multi-ticker mode, groups data by ticker 
        and applies the computation individually, then recombines into a single DataFrame.
        
        Args:
            prices (pd.DataFrame): Historical price data containing at least 'close', 'high', and 'low'.
        
        Returns:
            pd.DataFrame: DataFrame with computed columns 'tsi', 'signal_line', 'strength', and 'signal',
                          along with price columns 'close', 'high', and 'low'. In multi-ticker mode, the index 
                          is a MultiIndex (ticker, date).
        """
        if isinstance(prices.index, pd.MultiIndex):
            # Process each ticker group individually
            grouped = prices.groupby(level=0)
            result_list = []
            for ticker, group in grouped:
                signals = self._calculate_signals_single(group.droplevel(0))
                # Create a MultiIndex with ticker and date
                signals.index = pd.MultiIndex.from_product([[ticker], signals.index])
                result_list.append(signals)
            return pd.concat(result_list).sort_index()
        else:
            return self._calculate_signals_single(prices)

    def _calculate_signals_single(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute TSI and generate trading signals for a single ticker.
        
        Args:
            prices (pd.DataFrame): Historical price data for a single ticker, indexed by date.
        
        Returns:
            pd.DataFrame: DataFrame with 'tsi', 'signal_line', 'strength', 'signal', 'close', 'high', and 'low' columns.
        """
        close = prices['close']
        tsi, signal_line = self._calculate_tsi(close)
        
        signals = pd.DataFrame(index=prices.index)
        signals['tsi'] = tsi
        signals['signal_line'] = signal_line
        signals['strength'] = tsi - signal_line
        signals['signal'] = self._generate_crossover_signals(tsi, signal_line)
        
        if self.long_only:
            signals.loc[signals['signal'] == -1, 'signal'] = 0  # Force sell signals to 0 in long-only mode
        
        signals[['close', 'high', 'low']] = prices[['close', 'high', 'low']]
        return signals

    def _calculate_tsi(self, close: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Compute the True Strength Index (TSI) and its smoothed signal line.
        
        The TSI is computed using double exponential smoothing of the price change (delta):
            - delta = close.diff()
            - double_smoothed = EMA(EMA(delta, span=long_period), span=short_period)
            - abs_double_smoothed = EMA(EMA(|delta|, span=long_period), span=short_period)
            - tsi = 100 * (double_smoothed / (abs_double_smoothed + 1e-10))
    
        The signal line is the EMA of TSI over `signal_period`.
        
        Args:
            close (pd.Series): Series of closing prices.
        
        Returns:
            tuple: (tsi, signal_line) as two pandas Series.
        """
        delta = close.diff()
        double_smoothed = delta.ewm(span=self.long_period, adjust=False).mean().ewm(span=self.short_period, adjust=False).mean()
        abs_double_smoothed = delta.abs().ewm(span=self.long_period, adjust=False).mean().ewm(span=self.short_period, adjust=False).mean()
        
        tsi = 100 * (double_smoothed / (abs_double_smoothed + 1e-10))
        signal_line = tsi.ewm(span=self.signal_period, adjust=False).mean()
        return tsi, signal_line

    def _generate_crossover_signals(self, tsi: pd.Series, signal_line: pd.Series) -> pd.Series:
        """
        Generate trading signals based on TSI and signal line crossovers.
        
        A buy signal (+1) is triggered when TSI crosses above the signal line,
        and a sell signal (-1) is triggered when TSI crosses below the signal line.
        
        Args:
            tsi (pd.Series): TSI values.
            signal_line (pd.Series): Smoothed TSI (signal line).
        
        Returns:
            pd.Series: Series of trading signals (1 for buy, -1 for sell, 0 for no change).
        """
        above = tsi > signal_line
        below = tsi < signal_line
        
        signals = pd.Series(0, index=tsi.index)
        signals[(above & (above.shift(1) == False))] = 1
        signals[(below & (below.shift(1) == False))] = -1
        return signals

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Apply risk management to trading signals using the RiskManager component.
        
        In multi-ticker mode, processes each ticker group individually.
        Risk management adjusts positions based on stop-loss, take-profit, and signal exit rules.
        
        Args:
            signals (pd.DataFrame): DataFrame with columns 'signal', 'high', 'low', 'close' (plus TSI indicators).
            initial_position (int): Starting position (0: no position, 1: long, -1: short).
        
        Returns:
            pd.DataFrame: DataFrame with risk-managed columns including 'position', 'return', 
                          'cumulative_return', and 'exit_type', with the same index as `signals`.
        """
        if isinstance(signals.index, pd.MultiIndex):
            result_list = []
            for ticker, group in signals.groupby(level=0):
                # Remove ticker level for processing
                group = group.droplevel(0)
                rm_out = self.risk_manager.apply(group[['signal', 'high', 'low', 'close']], initial_position)
                # Restore ticker level
                rm_out.index = pd.MultiIndex.from_product([[ticker], rm_out.index])
                result_list.append(rm_out)
            return pd.concat(result_list).sort_index()
        else:
            return self.risk_manager.apply(signals[['signal', 'high', 'low', 'close']], initial_position)

    def _finalize_output(self, df: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Finalize the output DataFrame by reordering columns to a standard format.
        
        The final DataFrame contains:
            'tsi', 'signal_line', 'strength', 'signal', 'position', 
            'return', 'cumulative_return', 'exit_type', 'close', 'high', 'low'
        
        Args:
            df (pd.DataFrame): DataFrame with both indicator and risk management columns.
            latest_only (bool): If True, only returns the most recent row for each ticker 
                                (or overall for a single ticker).
        
        Returns:
            pd.DataFrame: Finalized DataFrame ready for backtesting and performance analysis.
        """
        # Ensure all required columns are present
        cols = ['tsi', 'signal_line', 'strength', 'signal', 'position', 
                'return', 'cumulative_return', 'exit_type', 'close', 'high', 'low']
        df = df[cols]
        
        if latest_only:
            if isinstance(df.index, pd.MultiIndex):
                latest = df.groupby(level=0).tail(1)
            else:
                latest = df.iloc[[-1]]
            return latest
        return df

    def __repr__(self) -> str:
        """
        Return a string representation of the TSIStrategy instance including key parameters.
        
        Returns:
            str: A formatted string with parameter values.
        """
        return (f"TSIStrategy(long_period={self.long_period}, short_period={self.short_period}, "
                f"signal_period={self.signal_period}, stop_loss_pct={self.stop_loss_pct}, "
                f"take_profit_pct={self.take_profit_pct}, slippage_pct={self.slippage_pct}, "
                f"transaction_cost_pct={self.transaction_cost_pct}, long_only={self.long_only})")