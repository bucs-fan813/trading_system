# trading_system/src/strategies/macd_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class MACDStrategy(BaseStrategy):
    """
    MACD Trading Strategy with Integrated Risk Management.
    
    Mathematical formulation:
        - Fast EMA:      fast_ema = EMA(close, span=fast)
        - Slow EMA:      slow_ema = EMA(close, span=slow)
        - MACD line:     macd = fast_ema - slow_ema
        - Signal line:   signal_line = EMA(macd, span=smooth)
        - Histogram:     histogram = macd - signal_line
    
    Trading signals are generated based on crossovers:
        - A bullish (long) signal (+1) is generated when MACD crosses above the signal line.
        - A bearish (short) signal (–1) is generated when MACD crosses below the signal line.
          Under long-only mode, bearish crossovers are ignored (resulting in no signal, i.e. 0).
    
    Signal strength is measured using the absolute value of the histogram.
    
    This strategy integrates risk management via the RiskManager component. When a trading signal is generated,
    the RiskManager adjusts entry prices for slippage and transaction costs, determines stop-loss and take-profit
    levels, identifies exit events, computes realized trade returns, and tracks cumulative returns.
    
    The strategy supports both:
      - Full historical backtesting (using a specified date range).
      - Real-time forecasting (by retrieving only the recent lookback data to ensure indicator stability).
    
    In addition, it supports vectorized processing of multiple tickers.
    
    Outputs:
        A DataFrame containing:
          - Price data:           'open', 'high', 'low', 'close', 'volume'
          - MACD indicators:      'macd', 'signal_line', 'histogram'
          - Raw trading signals:  'raw_signal' and a computed 'signal_strength'
          - Risk-managed results: 'position', 'return', 'cumulative_return', 'exit_type'
    
    Args:
        ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
        start_date (str, optional): Backtesting start date in 'YYYY-MM-DD' format.
        end_date (str, optional): Backtesting end date in 'YYYY-MM-DD' format.
        initial_position (int): Initial trading position (default: 0 for flat, 1 for long, -1 for short).
        latest_only (bool): If True, returns only the most recent signal(s) for each ticker (for forecasting).
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the MACD strategy with specified hyperparameters and database configuration.
        
        Raises:
            ValueError: If the slow period is not greater than the fast period or if MACD periods are not integers.
        """
        super().__init__(db_config, params)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default hyperparameters if not provided.
        self.params.setdefault('slow', 26)
        self.params.setdefault('fast', 12)
        self.params.setdefault('smooth', 9)
        self.params.setdefault('stop_loss_pct', 0.05)
        self.params.setdefault('take_profit_pct', 0.10)
        self.params.setdefault('trailing_stop_pct', 0.0)
        self.params.setdefault('slippage_pct', 0.001)
        self.params.setdefault('transaction_cost_pct', 0.001)
        self.params.setdefault('long_only', True)
        
        # Validate parameter consistency.
        if self.params['slow'] <= self.params['fast']:
            raise ValueError("Slow period must be greater than fast period")
        if any(not isinstance(p, int) for p in [self.params['slow'], 
                                                  self.params['fast'], 
                                                  self.params['smooth']]):
            raise ValueError("MACD periods must be integers")

    def generate_signals(self, 
                         ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate MACD trading signals with risk management adjustments.
        
        Retrieves historical price data (for single or multiple tickers), computes MACD-related indicators,
        generates raw trading signals based on MACD crossovers, and applies risk management rules to
        adjust positions and compute trade returns.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
            start_date (str, optional): Backtesting start date (YYYY-MM-DD).
            end_date (str, optional): Backtesting end date (YYYY-MM-DD).
            initial_position (int): Initial trading position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the most recent signal(s) for each ticker.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, MACD indicators, trading signals,
                          signal strength, risk-managed positions, returns, and exit types.
        """
        try:
            # Retrieve historical price data with optimized lookback if only the latest signal is needed.
            data = self._get_optimized_data(ticker, start_date, end_date, latest_only)
            
            if data.empty:
                self.logger.warning("No data retrieved for ticker(s): %s", ticker)
                return pd.DataFrame()
            
            # Determine if processing single or multiple tickers based on data index.
            if isinstance(ticker, list) or (isinstance(data.index, pd.MultiIndex)):
                # Process data for each ticker in a vectorized (grouped) manner.
                def process_group(group):
                    # Validate sufficient data for indicator computation.
                    if not self._validate_data(group, self.params['slow'] + self.params['smooth']):
                        return pd.DataFrame()
                    # Compute MACD indicators.
                    macd_line, signal_line, histogram = self._calculate_macd(group['close'])
                    # Generate raw trading signals based on MACD crossovers.
                    raw_signals = self._generate_crossover_signals(macd_line, signal_line)
                    # Apply risk management rules.
                    risk_managed = self._apply_risk_management(group, raw_signals, initial_position)
                    # Build and return the final output for the group.
                    return self._build_output(group, macd_line, signal_line, histogram, raw_signals, risk_managed)
                
                results = data.groupby(level=0, group_keys=False).apply(process_group)
                if latest_only:
                    results = results.groupby(level=0).tail(1)
                return results
            else:
                # Single ticker processing.
                if not self._validate_data(data, self.params['slow'] + self.params['smooth']):
                    return pd.DataFrame()
                macd_line, signal_line, histogram = self._calculate_macd(data['close'])
                raw_signals = self._generate_crossover_signals(macd_line, signal_line)
                risk_managed = self._apply_risk_management(data, raw_signals, initial_position)
                output = self._build_output(data, macd_line, signal_line, histogram, raw_signals, risk_managed)
                return output.iloc[[-1]] if latest_only else output

        except Exception as e:
            self.logger.error(f"Error processing ticker(s) {ticker}: {str(e)}")
            raise

    def _get_optimized_data(self, ticker: Union[str, List[str]], 
                            start_date: Optional[str],
                            end_date: Optional[str],
                            latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data optimized for MACD calculation stability.
        
        If latest_only is True, uses a minimal lookback period (based on slow + smooth parameters)
        to ensure that the indicator is stable. Supports both single and multiple tickers.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol(s).
            start_date (str, optional): Start date (YYYY-MM-DD).
            end_date (str, optional): End date (YYYY-MM-DD).
            latest_only (bool): Flag indicating retrieval of minimal data required for a stable signal.
        
        Returns:
            pd.DataFrame: DataFrame containing historical price data.
        """
        if latest_only:
            # Additional rows are used to ensure MACD indicator stability.
            lookback = self.params['slow'] + self.params['smooth'] + 10
            return self.get_historical_prices(ticker, lookback=lookback)
        return self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)

    def _calculate_macd(self, close: pd.Series) -> tuple:
        """
        Compute MACD line, Signal line, and Histogram using exponential moving averages.
        
        Mathematical formulation:
            fast_ema = EMA(close, span=fast)
            slow_ema = EMA(close, span=slow)
            MACD line = fast_ema - slow_ema
            Signal line = EMA(MACD, span=smooth)
            Histogram = MACD line - Signal line
        
        Args:
            close (pd.Series): Series of closing prices.
        
        Returns:
            tuple: A tuple containing MACD line, Signal line, and Histogram.
        """
        fast_ema = close.ewm(span=self.params['fast'], adjust=False).mean()
        slow_ema = close.ewm(span=self.params['slow'], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.params['smooth'], adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _generate_crossover_signals(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """
        Generate raw trading signals based on MACD and signal line crossovers.
        
        A bullish signal (+1) is issued when MACD crosses above the signal line.
        In non-long-only mode, a bearish signal (–1) is issued when MACD crosses below the signal line.
        In long-only mode, bearish crossovers result in no trading signal (0).
        
        Args:
            macd (pd.Series): MACD line series.
            signal (pd.Series): Signal line series.
        
        Returns:
            pd.Series: Series of raw trading signals (1, 0, -1).
        """
        # Determine positions based on MACD relative to signal line.
        above = macd > signal
        below = macd < signal
        
        # Identify crossover events.
        cross_up = above & (~above.shift(1).fillna(False))
        cross_down = below & (~below.shift(1).fillna(False))
        
        signals = pd.Series(0, index=macd.index)
        signals[cross_up] = 1
        
        if not self.params['long_only']:
            signals[cross_down] = -1
        else:
            signals[cross_down] = 0

        return signals

    def _apply_risk_management(self, data: pd.DataFrame, signals: pd.Series, initial_position: int) -> pd.DataFrame:
        """
        Integrate risk management rules with trading signals using the RiskManager.
        
        Constructs a DataFrame including required price data (close, high, low) and
        applies risk management to adjust entry prices (including slippage and transaction costs),
        compute stop-loss and take-profit thresholds, identify exit events, and compute realized
        and cumulative returns.
        
        Args:
            data (pd.DataFrame): Historical price data with at least 'close', 'high', and 'low'.
            signals (pd.Series): Raw trading signals.
            initial_position (int): Initial trading position (0 for flat, 1 for long, -1 for short).
        
        Returns:
            pd.DataFrame: DataFrame with risk-managed positions, realized returns, cumulative returns,
                          and exit event types.
        """
        signals_df = pd.DataFrame({
            'signal': signals,
            'close': data['close'],
            'high': data['high'],
            'low': data['low']
        })
        
        risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            trailing_stop_pct=self.params['trailing_stop_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )
        
        return risk_manager.apply(signals_df, initial_position)

    def _build_output(self, data: pd.DataFrame, macd_line: pd.Series, 
                      signal_line: pd.Series, histogram: pd.Series, 
                      raw_signals: pd.Series, risk_managed: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the final output DataFrame combining price data, MACD indicators,
        raw signals, signal strength, and risk-managed trading outcomes.
        
        The resulting DataFrame is designed to support both detailed backtest analysis and 
        real-time forecasting by incorporating all necessary indicators and performance metrics.
        
        Args:
            data (pd.DataFrame): Price data with columns 'open', 'high', 'low', 'close', and 'volume'.
            macd_line (pd.Series): Computed MACD line.
            signal_line (pd.Series): Computed signal line.
            histogram (pd.Series): MACD histogram.
            raw_signals (pd.Series): Raw trading signals derived from crossovers.
            risk_managed (pd.DataFrame): DataFrame with risk-managed positions, returns, and exit types.
        
        Returns:
            pd.DataFrame: Combined DataFrame with price data, indicators, raw signals, and risk-managed outcomes.
        """
        indicators = pd.DataFrame({
            'macd': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'raw_signal': raw_signals,
            'signal_strength': histogram.abs()
        }, index=macd_line.index)
        
        output = pd.concat([
            data[['open', 'high', 'low', 'close', 'volume']],
            indicators,
            risk_managed[['position', 'return', 'cumulative_return', 'exit_type']]
        ], axis=1)
        
        return output