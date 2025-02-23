# trading_system/src/strategies/macd_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class MACDStrategy(BaseStrategy):
    """
    MACD Trading Strategy with Integrated Risk Management.
    
    This strategy computes the MACD indicator based on exponential moving averages:
    
        Fast EMA:      EMA_fast = EMA(P, fast_period)
        Slow EMA:      EMA_slow = EMA(P, slow_period)
        MACD line:     MACD = EMA_fast - EMA_slow
        Signal line:   Signal = EMA(MACD, smooth_period)
        Histogram:     Histogram = MACD - Signal
    
    Trading signals are generated on crossovers:
      - A bullish (long) signal (1) is generated when MACD crosses above the signal line.
      - A bearish (short) signal (-1) is generated when MACD crosses below the signal line.
    
    The strength of the signal is measured using the absolute value of the MACD histogram.
    
    Integrated risk management applies stop-loss and take-profit rules:
      - For long positions, a stop loss is triggered if the low price falls below 
        entry_price * (1 - stop_loss_pct) and a take profit is triggered if the high 
        price exceeds entry_price * (1 + take_profit_pct).
      - For short positions, the thresholds are inverted.
    
    Entry prices are adjusted for slippage and transaction costs, and realized returns 
    are computed on exit. Cumulative returns are tracked to support downstream metrics (e.g., Sharpe ratio, max drawdown).
    
    Hyperparameters (with defaults):
        slow (int): Slow EMA period (default: 26)
        fast (int): Fast EMA period (default: 12)
        smooth (int): Signal line EMA period (default: 9)
        stop_loss_pct (float): Stop loss percentage (default: 0.05)
        take_profit_pct (float): Take profit percentage (default: 0.10)
        slippage_pct (float): Estimated execution slippage (default: 0.001)
        transaction_cost_pct (float): Transaction cost (default: 0.001)
        long_only (bool): Flag to allow only long positions (default: True)
    
    The strategy supports both full historical backtesting and real-time forecasting for the latest available date.
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the MACD strategy parameters and validate them.
        
        Args:
            db_config: Database configuration object.
            params (dict, optional): Strategy hyperparameters.
            
        Raises:
            ValueError: If the slow period is not greater than the fast period or if MACD periods 
                        are not integers.
        """
        super().__init__(db_config, params)
        self.logger = logging.getLogger(__name__)
        
        # Set default hyperparameters if not provided.
        self.params.setdefault('slow', 26)
        self.params.setdefault('fast', 12)
        self.params.setdefault('smooth', 9)
        self.params.setdefault('stop_loss_pct', 0.05)
        self.params.setdefault('take_profit_pct', 0.10)
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
                         ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate MACD trading signals and apply risk management rules.
        
        The method retrieves historical price data, calculates the MACD indicator metrics,
        determines crossover signals, and integrates risk management (stop-loss, take-profit,
        slippage, and transaction costs). Returns either a full historical DataFrame for backtesting
        or only the latest signal for real-time decision making.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Backtesting start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtesting end date in 'YYYY-MM-DD' format.
            initial_position (int): Initial trading position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, only the most recent signal row is returned.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, MACD indicators (macd, signal_line, histogram),
                          raw crossover signals, a computed signal strength, risk-managed position,
                          realized returns, cumulative returns, and exit event types.
        """
        try:
            # Retrieve historical price data with optimized lookback if only latest signal is needed.
            data = self._get_optimized_data(ticker, start_date, end_date, latest_only)
            
            # Validate that the retrieved data has sufficient records for MACD calculation.
            if not self._validate_data(data, self.params['slow'] + self.params['smooth']):
                return pd.DataFrame()

            # Compute MACD-related indicators in a vectorized manner.
            macd_line, signal_line, histogram = self._calculate_macd(data['close'])
            
            # Generate trading signals based on MACD and signal line crossovers.
            signals = self._generate_crossover_signals(macd_line, signal_line)
            
            # Apply integrated risk management to adjust positions and compute returns.
            risk_managed = self._apply_risk_management(data, signals, initial_position)
            
            # Construct the final output DataFrame by merging price data, indicators, and risk-managed signals.
            output = self._build_output(data, macd_line, signal_line, histogram, signals, risk_managed)
            
            return output.iloc[[-1]] if latest_only else output

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            raise

    def _get_optimized_data(self, ticker: str, 
                            start_date: Optional[str],
                            end_date: Optional[str],
                            latest_only: bool) -> pd.DataFrame:
        """
        Retrieve and optimize historical price data retrieval.
        
        If latest_only is True, a minimal lookback period is used to compute indicators.
        Otherwise, full historical data for the specified date range is returned.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for data retrieval.
            end_date (str, optional): End date for data retrieval.
            latest_only (bool): Flag indicating if only the most recent data is needed.
        
        Returns:
            pd.DataFrame: Historical price data.
        """
        if latest_only:
            # Use a buffer of additional rows to ensure indicator stability.
            lookback = self.params['slow'] + self.params['smooth'] + 10
            return self.get_historical_prices(ticker, lookback=lookback)
        
        return self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)

    def _calculate_macd(self, close: pd.Series) -> tuple:
        """
        Calculate MACD, Signal line, and Histogram using vectorized exponential moving averages.
        
        Mathematical Formulation:
            fast_ema = EMA(close, span=fast)
            slow_ema = EMA(close, span=slow)
            MACD line = fast_ema - slow_ema
            Signal line = EMA(MACD line, span=smooth)
            Histogram = MACD line - Signal line
        
        Args:
            close (pd.Series): Series of closing prices.
        
        Returns:
            tuple: (macd_line, signal_line, histogram) as pandas Series.
        """
        fast_ema = close.ewm(span=self.params['fast'], adjust=False).mean()
        slow_ema = close.ewm(span=self.params['slow'], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.params['smooth'], adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _generate_crossover_signals(self, 
                                    macd: pd.Series, 
                                    signal: pd.Series) -> pd.Series:
        """
        Generate trading signals based on crossovers between MACD and its signal line.
        
        A bullish signal (1) is generated when MACD crosses above the signal line.
        A bearish signal (-1) is generated when MACD crosses below the signal line.
        Otherwise, a signal of 0 is returned.
        
        Args:
            macd (pd.Series): MACD line.
            signal (pd.Series): Signal line.
        
        Returns:
            pd.Series: Series of raw trading signals.
        """
        above = macd > signal
        below = macd < signal
        
        cross_up = above & (~above.shift(1).fillna(False))
        cross_down = below & (~below.shift(1).fillna(False))
        
        signals = pd.Series(0, index=macd.index)
        signals[cross_up] = 1
        
        if self.params['long_only']:
            signals[cross_down] = 0
        else:
            signals[cross_down] = -1

        return signals

    def _apply_risk_management(self, 
                               data: pd.DataFrame,
                               signals: pd.Series,
                               initial_position: int) -> pd.DataFrame:
        """
        Integrate the risk management rules with the generated signals using the RiskManager class.
        
        Constructs a DataFrame with the required price (close, high, low) and signal data,
        applies vectorized risk management (stop-loss, take-profit, slippage, and transaction costs),
        and computes trade returns and cumulative performance.
        
        Args:
            data (pd.DataFrame): Historical price data.
            signals (pd.Series): Raw trading signals derived from MACD crossovers.
            initial_position (int): Initial trading position.
        
        Returns:
            pd.DataFrame: DataFrame with risk-managed position, realized return, cumulative return,
                          and exit event type.
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
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )
        
        return risk_manager.apply(signals_df, initial_position)

    def _build_output(self, 
                      data: pd.DataFrame,
                      macd_line: pd.Series,
                      signal_line: pd.Series,
                      histogram: pd.Series,
                      signals: pd.Series,
                      risk_managed: pd.DataFrame) -> pd.DataFrame:
        """
        Build and return the final output DataFrame combining price data, MACD indicators,
        raw and risk-managed signals.
        
        This DataFrame is structured to support both detailed backtesting analysis (e.g., Sharpe ratio,
        maximum drawdown computations) and real-time forecasting.
        
        Args:
            data (pd.DataFrame): Price data containing 'open', 'high', 'low', 'close', and 'volume'.
            macd_line (pd.Series): Calculated MACD line.
            signal_line (pd.Series): Calculated signal line.
            histogram (pd.Series): MACD histogram.
            signals (pd.Series): Raw crossover trading signals.
            risk_managed (pd.DataFrame): DataFrame with risk-managed positions and returns.
        
        Returns:
            pd.DataFrame: Combined DataFrame with all required indicators and signals.
        """
        output = pd.concat([
            data[['open', 'high', 'low', 'close', 'volume']],
            pd.DataFrame({
                'macd': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'raw_signal': signals,
                'signal_strength': histogram.abs()
            }, index=macd_line.index),
            risk_managed[['position', 'return', 'cumulative_return', 'exit_type']]
        ], axis=1)
        
        # For full historical output, return the complete DataFrame.
        return output