# trading_system/src/strategies/triangle_breakout.py

# TODO Long only

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class TriangleBreakout(BaseStrategy):
    """
    Triangle Breakout Strategy with Integrated Risk Management.
    
    This strategy detects consolidation patterns (triangles) in a stock's price action
    by fitting two converging trend lines through local highs and lows within a rolling window.
    
    Mathematical Formulation:
      Consider a rolling window of 'max_lookback' bars. Within this window, identify local
      highs (peaks) and local lows (troughs). Apply linear regression to the set of highs and 
      lows separately to obtain the equations of the upper and lower trend lines:
      
          Upper Line:    y = m_upper * x + c_upper
          Lower Line:    y = m_lower * x + c_lower
          
      where x is the relative index in the window (0 to max_lookback-1). The triangle pattern 
      is defined by these two lines. Its significance is measured by the "pattern height" (the 
      difference between the two lines at x = max_lookback-1) as a fraction of the current price.
      
      A breakout is signaled when:
          - Upward Breakout: current_close > upper_line * (1 + breakout_threshold)
          - Downward Breakout: current_close < lower_line * (1 - breakout_threshold)
      
      Optionally, the breakout is confirmed only if current volume exceeds a 20-day rolling average.
    
    Operating Modes:
      - Backtesting: When start_date and end_date are provided, signals are generated over the date range.
      - Forecasting: With latest_only=True, only the latest signal for each ticker is returned.
      - Multi-ticker: Accepts a list of tickers and processes each in a vectorized fashion.
    
    Risk Management:
      After signal generation, the RiskManager module applies stop loss, take profit, slippage,
      and transaction cost adjustments. It then computes the realized return when an exit occurs 
      and tracks the cumulative return.
    
    Strategy-specific Parameters (with defaults):
      - min_points (int): Minimum number of local extremes required to form a valid triangle (5).
      - max_lookback (int): Rolling window size in bars for triangle pattern detection (60).
      - breakout_threshold (float): Percentage buffer to confirm a breakout (0.005).
      - volume_confirm (bool): Use volume confirmation on breakout (True).
      - min_pattern_size (float): Minimum triangle height relative to price (0.03).
      - stop_loss_pct (float): Stop loss percentage (0.05).
      - take_profit_pct (float): Take profit percentage (0.10).
      - slippage_pct (float): Estimated slippage percentage (0.001).
      - transaction_cost_pct (float): Estimated transaction cost percentage (0.001).
      - long_only (bool): If True, only long positions are allowed (True).
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Triangle Breakout strategy with the given parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Dictionary of strategy-specific parameters.
        """
        default_params = {
            'min_points': 5,
            'max_lookback': 60,
            'breakout_threshold': 0.005,
            'volume_confirm': True,
            'min_pattern_size': 0.03,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            'long_only': True
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
    
    def generate_signals(self,
                         tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on triangle breakout patterns and apply risk management.
        
        Retrieves historical price data from the database, scans for triangle patterns using a
        rolling window, and generates signals when the price breaks out above or below the fitted 
        trend lines. Afterward, the RiskManager is applied to adjust positions and compute trade returns.
        
        Args:
            tickers (str or List[str]): Single or multiple stock ticker symbols.
            start_date (str, optional): Start date (YYYY-MM-DD) for backtesting.
            end_date (str, optional): End date (YYYY-MM-DD) for backtesting.
            initial_position (int): Starting position (0 for no position, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the most recent signal row(s) for forecasting.
            
        Returns:
            pd.DataFrame: DataFrame with columns including:
                - Price data: 'open', 'high', 'low', 'close', 'volume'.
                - Pattern info: 'upper_line', 'lower_line', 'triangle_type', 'in_pattern'.
                - Signal: 'signal' (1 for upward breakout, -1 for downward breakout, 0 otherwise).
                - Risk management output: 'position', 'return', 'cumulative_return', 'exit_type', etc.
                The result is structured for downstream performance analysis.
        """
        # Retrieve historical price data. Use date filters if provided; otherwise, use a default lookback.
        if start_date or end_date:
            price_data = self.get_historical_prices(tickers, from_date=start_date, to_date=end_date)
        else:
            price_data = self.get_historical_prices(tickers, lookback=252)
            
        if price_data.empty:
            self.logger.warning("No historical price data retrieved.")
            return pd.DataFrame()
        
        def process_single_ticker(df: pd.DataFrame) -> pd.DataFrame:
            """
            Process price data for one ticker to generate triangle breakout signals.
            
            Args:
                df (pd.DataFrame): Price data for a single ticker with date index.
            
            Returns:
                pd.DataFrame: DataFrame with triangle breakout signals computed.
            """
            if len(df) < self.params['max_lookback'] + 10:
                self.logger.warning("Insufficient data for ticker %s to generate signals.", df.name)
                return pd.DataFrame()
            
            df = df.sort_index()
            # Initialize the result DataFrame with price columns.
            result = pd.DataFrame(index=df.index)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                result[col] = df[col]
            
            # Initialize columns for pattern lines and breakout signals.
            result['upper_line'] = np.nan
            result['lower_line'] = np.nan
            result['triangle_type'] = None
            result['in_pattern'] = False
            result['signal'] = 0
            result['avg_volume'] = df['volume'].rolling(window=20).mean()
            
            # Loop over the data starting at index = max_lookback.
            for i in range(self.params['max_lookback'], len(result)):
                window = df.iloc[i - self.params['max_lookback']:i]
                # If already in a pattern, update the regression lines.
                if result['in_pattern'].iloc[i - 1]:
                    upper_points, lower_points = self._get_triangle_points(window)
                    if (len(upper_points) >= self.params['min_points'] and
                        len(lower_points) >= self.params['min_points']):
                        upper_slope, upper_intercept = self._linear_regression(upper_points)
                        lower_slope, lower_intercept = self._linear_regression(lower_points)
                        # Use relative index: predict at x = max_lookback - 1.
                        x_predict = self.params['max_lookback'] - 1
                        result.iloc[i, result.columns.get_loc('upper_line')] = upper_slope * x_predict + upper_intercept
                        result.iloc[i, result.columns.get_loc('lower_line')] = lower_slope * x_predict + lower_intercept
                        triangle_type = self._determine_triangle_type(upper_slope, lower_slope)
                        result.iloc[i, result.columns.get_loc('triangle_type')] = triangle_type
                        
                        current_close = result['close'].iloc[i]
                        upper_line = result['upper_line'].iloc[i]
                        lower_line = result['lower_line'].iloc[i]
                        pattern_height = upper_line - lower_line
                        price_percentage = pattern_height / current_close if current_close != 0 else 0
                        
                        if price_percentage >= self.params['min_pattern_size']:
                            # Upward breakout check.
                            if current_close > upper_line * (1 + self.params['breakout_threshold']):
                                if (not self.params['volume_confirm'] or 
                                    result['volume'].iloc[i] > result['avg_volume'].iloc[i]):
                                    result.iloc[i, result.columns.get_loc('signal')] = 1
                                    result.iloc[i, result.columns.get_loc('in_pattern')] = False
                                else:
                                    result.iloc[i, result.columns.get_loc('in_pattern')] = True
                            # Downward breakout check.
                            elif current_close < lower_line * (1 - self.params['breakout_threshold']):
                                if (not self.params['volume_confirm'] or 
                                    result['volume'].iloc[i] > result['avg_volume'].iloc[i]):
                                    if self.params['long_only']:
                                        result.iloc[i, result.columns.get_loc('signal')] = 0
                                    else:
                                        result.iloc[i, result.columns.get_loc('signal')] = -1
                                    result.iloc[i, result.columns.get_loc('in_pattern')] = False
                                else:
                                    result.iloc[i, result.columns.get_loc('in_pattern')] = True
                            else:
                                result.iloc[i, result.columns.get_loc('in_pattern')] = True
                        else:
                            result.iloc[i, result.columns.get_loc('in_pattern')] = False
                    else:
                        result.iloc[i, result.columns.get_loc('in_pattern')] = False
                else:
                    # Look for a new triangle pattern.
                    upper_points, lower_points = self._get_triangle_points(window)
                    if (len(upper_points) >= self.params['min_points'] and
                        len(lower_points) >= self.params['min_points']):
                        upper_slope, upper_intercept = self._linear_regression(upper_points)
                        lower_slope, lower_intercept = self._linear_regression(lower_points)
                        triangle_type = self._determine_triangle_type(upper_slope, lower_slope)
                        if triangle_type:
                            x_predict = self.params['max_lookback'] - 1
                            result.iloc[i, result.columns.get_loc('upper_line')] = upper_slope * x_predict + upper_intercept
                            result.iloc[i, result.columns.get_loc('lower_line')] = lower_slope * x_predict + lower_intercept
                            result.iloc[i, result.columns.get_loc('triangle_type')] = triangle_type
                            
                            current_close = result['close'].iloc[i]
                            pattern_height = result['upper_line'].iloc[i] - result['lower_line'].iloc[i]
                            price_percentage = pattern_height / current_close if current_close != 0 else 0
                            if price_percentage >= self.params['min_pattern_size']:
                                result.iloc[i, result.columns.get_loc('in_pattern')] = True
                    # Otherwise, leave the default values.
            result = result.dropna(subset=['close'])
            return result
        
        # Process multi-ticker (or single ticker) data.
        if isinstance(tickers, list):
            groups = price_data.groupby(level=0)
            processed_list = []
            for ticker, group in groups:
                group.name = ticker
                res = process_single_ticker(group)
                if not res.empty:
                    res['ticker'] = ticker
                    processed_list.append(res)
            if not processed_list:
                return pd.DataFrame()
            signals_df = pd.concat(processed_list).sort_index()
        else:
            price_data.name = tickers
            signals_df = process_single_ticker(price_data)
            signals_df['ticker'] = tickers
        
        # Apply the Risk Management component to adjust signals for stop-loss, take profit, etc.
        rm = RiskManager(
            stop_loss_pct=self.params.get("stop_loss_pct", 0.05),
            take_profit_pct=self.params.get("take_profit_pct", 0.10),
            slippage_pct=self.params.get("slippage_pct", 0.001),
            transaction_cost_pct=self.params.get("transaction_cost_pct", 0.001)
        )
        signals_with_rm = rm.apply(signals_df, initial_position=initial_position)
        
        # If latest_only is True, return only the latest row for each ticker.
        if latest_only:
            if isinstance(tickers, list):
                signals_with_rm = signals_with_rm.groupby('ticker', group_keys=False).apply(lambda x: x.iloc[[-1]])
            else:
                signals_with_rm = signals_with_rm.iloc[[-1]]
        
        return signals_with_rm
    
    def _get_triangle_points(self, data: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Identify potential upper and lower points for triangle patterns.
        
        Scans the data window for local highs and lows.
        
        Args:
            data (pd.DataFrame): A subset of the price data.
            
        Returns:
            Tuple:
                - List of (index, high) tuples for local highs.
                - List of (index, low) tuples for local lows.
        """
        upper_points = []
        lower_points = []
        
        for i in range(1, len(data) - 1):
            # Detect local high.
            if data['high'].iloc[i] > data['high'].iloc[i - 1] and data['high'].iloc[i] > data['high'].iloc[i + 1]:
                upper_points.append((i, data['high'].iloc[i]))
            # Detect local low.
            if data['low'].iloc[i] < data['low'].iloc[i - 1] and data['low'].iloc[i] < data['low'].iloc[i + 1]:
                lower_points.append((i, data['low'].iloc[i]))
        return upper_points, lower_points
    
    def _linear_regression(self, points: List[Tuple[int, float]]) -> Tuple[float, float]:
        """
        Compute the least squares regression line for a set of points.
        
        Args:
            points (List[Tuple[int, float]]): List of (x, y) pairs representing the points.
            
        Returns:
            Tuple (slope, intercept) of the best-fit line.
        """
        if not points:
            return 0, 0
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        if len(x) < 2:
            return 0, y[0] if len(y) > 0 else 0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        return slope, intercept
    
    def _determine_triangle_type(self, upper_slope: float, lower_slope: float) -> Optional[str]:
        """
        Determine the triangle type based on the slopes of the trend lines.
        
        Valid triangle types:
          - "symmetrical": upper line is descending and lower line is ascending.
          - "ascending":   upper line is nearly flat while lower line is ascending.
          - "descending":  upper line is descending while lower line is nearly flat.
        If the slopes do not form a valid converging pattern (or are too steep), None is returned.
        
        Args:
            upper_slope (float): Slope of the upper trend line.
            lower_slope (float): Slope of the lower trend line.
            
        Returns:
            str or None: The triangle type ('symmetrical', 'ascending', 'descending') or None if invalid.
        """
        if upper_slope * lower_slope > 0:
            return None
        if abs(upper_slope) > 0.5 or abs(lower_slope) > 0.5:
            return None
        
        if upper_slope < -0.01 and lower_slope > 0.01:
            return 'symmetrical'
        if abs(upper_slope) < 0.01 and lower_slope > 0.01:
            return 'ascending'
        if upper_slope < -0.01 and abs(lower_slope) < 0.01:
            return 'descending'
        return None