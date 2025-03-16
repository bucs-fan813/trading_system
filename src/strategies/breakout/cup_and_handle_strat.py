# trading_system/src/strategies/cup_and_handle.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, Optional, List, Tuple, Union
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager

class CupAndHandle(BaseStrategy):
    """
    Cup and Handle Breakout Strategy with Integrated Risk Management

    This strategy identifies a "cup" formation followed by a "handle" formation in the price action.
    Mathematically, define the price series P(t) and let:
    
        • t₁ = time of left peak (P₁ = price at t₁)
        • t₂ = time of right peak (P₂ = price at t₂)
        • P_bottom = minimum price observed between t₁ and t₂
    
    Then the resistance level is:
    
        resistance = max(P₁, P₂)
    
    and the cup depth is:
    
        D = resistance - P_bottom,
    
    so that the relative cup depth is D / resistance. For a valid cup, the relative cup depth must not 
    exceed a specified threshold (e.g. 0.3). Following the cup, a handle formation is expected—a minor 
    dip that does not retrace the full cup depth (e.g. no deeper than 50% of D). A breakout (and hence a 
    buy signal) is triggered once the price exceeds the resistance by a small breakout threshold (e.g. 0.005).
    
    The signal strength is computed as:
    
        signal_strength = (close / resistance) - 1,
    
    meaning that a higher signal strength indicates a stronger breakout above resistance.
    
    Risk management is then applied via the RiskManager which accounts for stop loss, take profit, slippage,
    and transaction costs to compute realized trade returns and cumulative returns. The strategy also 
    supports backtesting over a user-specified date range as well as quick generation for only the latest signal.
    
    Strategy Parameters (defaults):
        - min_cup_duration (int): Minimum bars required for a valid cup (default: 30).
        - max_cup_duration (int): Maximum bars allowed for a cup (default: 150).
        - min_handle_duration (int): Minimum bars for a valid handle (default: 5).
        - max_handle_duration (int): Maximum bars allowed for a handle (default: 30).
        - cup_depth_threshold (float): Maximum allowable cup depth (fraction, default: 0.3).
        - handle_depth_threshold (float): Maximum allowable handle depth relative to cup depth (default: 0.5).
        - breakout_threshold (float): Percentage above resistance to trigger breakout (default: 0.005).
        - volume_confirm (bool): Require volume confirmation for breakout (default: True).
        - stop_loss_pct (float): Stop loss percentage (default: 0.05).
        - take_profit_pct (float): Take profit percentage (default: 0.10).
        - trailing_stop_pct (float): Trailing stop percentage (default: 0.0).
        - slippage_pct (float): Estimated slippage percentage (default: 0.001).
        - transaction_cost_pct (float): Estimated transaction cost percentage (default: 0.001).
        - long_only (bool): If True, only long positions are allowed (default: True).
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Cup and Handle strategy with specific parameters.

        Args:
            db_config (DatabaseConfig): Database configuration for retrieving price data.
            params (Dict, optional): Strategy-specific parameters. If not provided,
                                     default parameters are used.
        """
        default_params = {
            'min_cup_duration': 30,
            'max_cup_duration': 150,
            'min_handle_duration': 5,
            'max_handle_duration': 30,
            'cup_depth_threshold': 0.3,         # Cup depth must be less than 30% of resistance
            'handle_depth_threshold': 0.5,      # Handle depth must be less than 50% of cup depth
            'breakout_threshold': 0.005,        # Breakout occurs when price exceeds resistance by 0.5%
            'volume_confirm': True,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trailing_stop_pct': 0.0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            'long_only': True                   # This is a long only strategy by design and would work the same irrespecyive of the parameter
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
    
    def generate_signals(self, ticker: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on cup and handle breakout patterns and apply risk management.

        This method retrieves historical price data, identifies cup and handle patterns,
        computes breakout signals when price exceeds the resistance level, and integrates
        risk management via the RiskManager class. It supports processing for a single ticker
        or a list of tickers in a vectorized fashion. Additionally, when backtesting,
        an extended lookback period is used to ensure robust detection of patterns before the
        start of the backtest period.

        Args:
            ticker (str or List[str]): Stock ticker symbol or list of symbols.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Initial trading position (0 for no position, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the most recent signal per ticker.

        Returns:
            pd.DataFrame: DataFrame with columns including 'open', 'high', 'low', 'close', 'volume',
                          computed technical indicators, signal (1 for breakout), signal_strength, and
                          risk-managed return metrics (position, return, cumulative_return, exit_type).
        """
        extra_days = self.params['max_cup_duration'] + self.params['max_handle_duration'] + 20
        
        # Determine effective start date for lookback if backtesting or latest signal generation
        if start_date:
            effective_start = (pd.Timestamp(start_date) - pd.Timedelta(days=extra_days)).strftime('%Y-%m-%d')
        else:
            effective_start = None
        
        # For latest_only, use a minimal lookback; otherwise, use an extended period.
        if latest_only:
            price_data = self.get_historical_prices(ticker, lookback=extra_days, from_date=effective_start, to_date=end_date)
        else:
            price_data = self.get_historical_prices(ticker, from_date=effective_start, to_date=end_date)
        
        # Process signals for each ticker independently.
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker
        
        signals_list = []
        # When multiple tickers, price_data will have a multi-index or a ticker column.
        for t, group in price_data.groupby('ticker') if ('ticker' in price_data.columns or isinstance(price_data.index, pd.MultiIndex)) else [(tickers[0], price_data)]:
            group = group.sort_index()
            ts_result = self._generate_signals_for_series(group.copy())
            ts_result['ticker'] = t
            signals_list.append(ts_result)
        
        result = pd.concat(signals_list)
        
        # Filter backtest period if start_date and/or end_date provided.
        if start_date:
            result = result[result.index >= pd.to_datetime(start_date)]
        if end_date:
            result = result[result.index <= pd.to_datetime(end_date)]
        
        # If latest_only is True, return only the last row per ticker.
        if latest_only:
            if 'ticker' in result.columns:
                result = result.groupby('ticker', group_keys=False).apply(lambda df: df.iloc[[-1]])
            else:
                result = result.iloc[[-1]]
        
        # Apply risk management adjustments (stop-loss, take-profit, slippage, transaction costs)
        rm = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct'],
            trailing_stop_pct=self.params['trailing_stop_pct']
        )
        result = rm.apply(result, initial_position=initial_position)
        
        return result
    
    def _generate_signals_for_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cup and handle signals for a single ticker's time series.

        This helper function resets the index to ensure integer positional indexing,
        computes simple moving averages, identifies local extrema for potential pattern points,
        and iteratively searches for cup and handle formations in the data. Once a breakout is
        confirmed, a buy signal is generated.

        Args:
            df (pd.DataFrame): Historical price data for a single ticker with a datetime index.

        Returns:
            pd.DataFrame: DataFrame indexed by date with columns for price data, technical indicators,
                          pattern formation markers, signal (1 for breakout), and the computed signal_strength.
        """
        # Ensure the DataFrame is sorted by date and reset index for positional operations.
        df = df.sort_index()
        df_reset = df.reset_index()  # 'date' becomes a column.
        
        result = pd.DataFrame()
        result['date'] = df_reset['date']
        result['open'] = df_reset['open']
        result['high'] = df_reset['high']
        result['low'] = df_reset['low']
        result['close'] = df_reset['close']
        result['volume'] = df_reset['volume']
        
        # Compute technical indicators
        result['sma20'] = result['close'].rolling(window=20, min_periods=1).mean()
        result['avg_volume'] = result['volume'].rolling(window=20, min_periods=1).mean()
        
        # Initialize pattern tracking and signal columns.
        result['in_cup'] = False
        result['in_handle'] = False
        result['cup_start'] = np.nan
        result['cup_bottom'] = np.nan
        result['cup_end'] = np.nan
        result['handle_start'] = np.nan
        result['handle_bottom'] = np.nan
        result['handle_end'] = np.nan
        result['pattern_resistance'] = np.nan
        result['signal'] = 0
        
        # Calculate local extrema for potential pattern detection.
        order = 5
        arr_high = result['high'].values
        arr_low = result['low'].values
        local_max_indices = argrelextrema(arr_high, np.greater, order=order)[0]
        local_min_indices = argrelextrema(arr_low, np.less, order=order)[0]
        result.loc[local_max_indices, 'is_local_max'] = True
        result.loc[local_min_indices, 'is_local_min'] = True
        result['is_local_max'] = result['is_local_max'].fillna(False)
        result['is_local_min'] = result['is_local_min'].fillna(False)
        
        # Iteratively detect cup and handle formations.
        for i in range(order, len(result) - self.params['min_handle_duration']):
            # If no existing pattern is in progress, search for a new cup formation.
            if not result.at[i-1, 'in_cup'] and not result.at[i-1, 'in_handle']:
                window_start = max(0, i - (self.params['max_cup_duration'] + self.params['max_handle_duration']))
                cup_start, cup_bottom, cup_end = self._find_cup(result.iloc[window_start:i])
                if cup_start is not None and cup_bottom is not None and cup_end is not None:
                    abs_cup_start = window_start + cup_start
                    abs_cup_bottom = window_start + cup_bottom
                    abs_cup_end = window_start + cup_end
                    left_height = result.at[abs_cup_start, 'high']
                    right_height = result.at[abs_cup_end, 'high']
                    bottom_price = result.at[abs_cup_bottom, 'low']
                    resistance = max(left_height, right_height)
                    cup_depth = resistance - bottom_price
                    relative_depth = cup_depth / resistance if resistance != 0 else np.inf
                    if relative_depth <= self.params['cup_depth_threshold']:
                        result.at[i, 'in_cup'] = True
                        result.at[i, 'cup_start'] = abs_cup_start
                        result.at[i, 'cup_bottom'] = abs_cup_bottom
                        result.at[i, 'cup_end'] = abs_cup_end
                        result.at[i, 'pattern_resistance'] = resistance
            elif result.at[i-1, 'in_cup']:
                # Continue cup formation tracking.
                abs_cup_start = int(result.at[i-1, 'cup_start'])
                abs_cup_bottom = int(result.at[i-1, 'cup_bottom'])
                abs_cup_end = int(result.at[i-1, 'cup_end'])
                resistance = result.at[i-1, 'pattern_resistance']
                result.at[i, 'in_cup'] = True
                result.at[i, 'cup_start'] = abs_cup_start
                result.at[i, 'cup_bottom'] = abs_cup_bottom
                result.at[i, 'cup_end'] = abs_cup_end
                result.at[i, 'pattern_resistance'] = resistance
                time_since_cup = i - abs_cup_end
                if 1 <= time_since_cup <= self.params['max_handle_duration']:
                    handle_start, handle_bottom, handle_end = self._find_handle(result.iloc[abs_cup_end:i+1], resistance)
                    if handle_start is not None and handle_bottom is not None and handle_end is not None:
                        result.at[i, 'in_cup'] = False
                        result.at[i, 'in_handle'] = True
                        result.at[i, 'handle_start'] = abs_cup_end + handle_start
                        result.at[i, 'handle_bottom'] = abs_cup_end + handle_bottom
                        result.at[i, 'handle_end'] = abs_cup_end + handle_end
                elif time_since_cup > self.params['max_handle_duration']:
                    result.at[i, 'in_cup'] = False
            elif result.at[i-1, 'in_handle']:
                # Continue handle formation tracking and detect breakout.
                abs_handle_start = int(result.at[i-1, 'handle_start'])
                abs_handle_bottom = int(result.at[i-1, 'handle_bottom'])
                abs_handle_end = int(result.at[i-1, 'handle_end'])
                resistance = result.at[i-1, 'pattern_resistance']
                result.at[i, 'in_handle'] = True
                result.at[i, 'cup_start'] = result.at[i-1, 'cup_start']
                result.at[i, 'cup_bottom'] = result.at[i-1, 'cup_bottom']
                result.at[i, 'cup_end'] = result.at[i-1, 'cup_end']
                result.at[i, 'handle_start'] = abs_handle_start
                result.at[i, 'handle_bottom'] = abs_handle_bottom
                result.at[i, 'handle_end'] = abs_handle_end
                result.at[i, 'pattern_resistance'] = resistance
                if result.at[i, 'close'] > resistance * (1 + self.params['breakout_threshold']):
                    if not self.params['volume_confirm'] or result.at[i, 'volume'] > result.at[i, 'avg_volume']:
                        result.at[i, 'signal'] = 1
                    result.at[i, 'in_handle'] = False
                time_in_handle = i - abs_handle_start
                if time_in_handle > self.params['max_handle_duration']:
                    result.at[i, 'in_handle'] = False
        
        # Compute signal strength as the normalized breakout magnitude.
        result['signal_strength'] = np.where(
            (result['signal'] == 1) & result['pattern_resistance'].notna() & (result['pattern_resistance'] != 0),
            (result['close'] / result['pattern_resistance']) - 1,
            0
        )
        
        # Retain only relevant columns and set the date as the index.
        cols_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'sma20', 
                        'cup_start', 'cup_bottom', 'cup_end', 'handle_start', 'handle_bottom', 
                        'handle_end', 'pattern_resistance', 'signal', 'signal_strength']
        result = result[cols_to_keep].set_index('date')
        return result
    
    def _find_cup(self, data: pd.DataFrame) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Identify a cup formation in the provided data segment based on local extrema.

        The method searches for two local maxima (left and right peaks) with a local
        minimum (cup bottom) between them. The cup's bottom should occur roughly in the middle,
        and the relative depth of the cup (difference between resistance and bottom) should be
        within the specified threshold.

        Args:
            data (pd.DataFrame): A slice of the price DataFrame with local extrema flags.

        Returns:
            Tuple[Optional[int], Optional[int], Optional[int]]:
                - cup_start: Position index of the left peak.
                - cup_bottom: Position index of the cup bottom.
                - cup_end: Position index of the right peak.
            Returns (None, None, None) if a valid cup is not found.
        """
        if len(data) < self.params['min_cup_duration']:
            return None, None, None
        
        # Retrieve indices of local maxima and minima.
        max_indices = data.index[data['is_local_max']]
        min_indices = data.index[data['is_local_min']]
        
        if len(max_indices) < 2 or len(min_indices) < 1:
            return None, None, None
        
        # Check for cup pattern by iterating over possible left and right peaks.
        for left_idx in max_indices:
            left_pos = data.index.get_loc(left_idx)
            for right_idx in max_indices:
                right_pos = data.index.get_loc(right_idx)
                if right_pos <= left_pos:
                    continue
                cup_duration = right_pos - left_pos
                if cup_duration < self.params['min_cup_duration'] or cup_duration > self.params['max_cup_duration']:
                    continue
                cup_slice = data.iloc[left_pos:right_pos+1]
                if len(cup_slice) < 3:
                    continue
                bottom_idx = cup_slice['low'].idxmin()
                bottom_pos = data.index.get_loc(bottom_idx)
                if bottom_pos <= left_pos + cup_duration * 0.2 or bottom_pos >= left_pos + cup_duration * 0.8:
                    continue
                left_price = data.loc[left_idx, 'high']
                right_price = data.loc[right_idx, 'high']
                if abs(left_price - right_price) / max(left_price, right_price) > 0.1:
                    continue
                bottom_region = cup_slice.iloc[cup_duration // 3: (cup_duration * 2) // 3]
                bottom_range = bottom_region['low'].max() - bottom_region['low'].min()
                cup_depth = max(left_price, right_price) - data.loc[bottom_idx, 'low']
                if bottom_range > cup_depth * 0.2:
                    continue
                return left_pos, bottom_pos, right_pos
        return None, None, None
    
    def _find_handle(self, data: pd.DataFrame, resistance: float) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Identify a handle formation in the given data segment after the cup formation.

        The method looks for a handle which is characterized by a minor dip in price following the cup.
        The handle formation starts at the cup's right rim and should end with a local high near the resistance level.

        Args:
            data (pd.DataFrame): Price data from the cup end to the current bar.
            resistance (float): The resistance level determined from the cup formation.

        Returns:
            Tuple[Optional[int], Optional[int], Optional[int]]:
                - handle_start: Relative index in the data segment where the handle starts.
                - handle_bottom: Relative index where the handle's lowest price occurs.
                - handle_end: Relative index where the handle formation ends (local high).
            Returns (None, None, None) if a valid handle is not detected.
        """
        if len(data) < self.params['min_handle_duration']:
            return None, None, None
        
        max_indices = data.index[data['is_local_max']]
        min_indices = data.index[data['is_local_min']]
        
        if len(max_indices) < 1 or len(min_indices) < 1:
            return None, None, None
        
        # The handle formation typically starts at the beginning of the data segment.
        handle_start = data.index[0]
        start_pos = 0
        for min_idx in min_indices:
            min_pos = data.index.get_loc(min_idx)
            if min_pos <= start_pos:
                continue
            duration_to_min = min_pos - start_pos
            if duration_to_min < 2 or duration_to_min > self.params['max_handle_duration'] // 2:
                continue
            handle_price = data.loc[min_idx, 'low']
            handle_depth = data.loc[handle_start, 'high'] - handle_price
            for max_idx in max_indices:
                max_pos = data.index.get_loc(max_idx)
                if max_pos <= min_pos:
                    continue
                total_duration = max_pos - start_pos
                if total_duration < self.params['min_handle_duration'] or total_duration > self.params['max_handle_duration']:
                    continue
                handle_end_price = data.loc[max_idx, 'high']
                if handle_end_price < resistance * 0.95 or handle_end_price > resistance:
                    continue
                return start_pos, min_pos, max_pos
        return None, None, None