# trading_system/src/strategies/cup_and_handle.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, Optional, List, Tuple, Union, Any
import logging

from src.strategies.base_strat import BaseStrategy, DataRetrievalError # Import BaseStrategy and potential error
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager # Assuming RiskManager is in this path

class CupAndHandle(BaseStrategy):
    """
    Cup and Handle Breakout Strategy with Integrated Risk Management.

    This strategy identifies a bullish continuation pattern consisting of a "cup"
    followed by a "handle".

    Mathematical Definition:
    1. Cup Formation:
        - Identify two local price peaks (using high prices), P_left at t_left and
          P_right at t_right, within a duration [min_cup_duration, max_cup_duration].
        - Resistance R = max(P_left, P_right).
        - Find the lowest price (low) P_bottom between t_left and t_right.
        - Cup Depth D = R - P_bottom.
        - Validation:
            - Duration constraints met.
            - Peaks P_left and P_right are relatively close (e.g., within 10-15% of R).
            - Relative Depth (D / R) <= cup_depth_threshold (cup isn't too deep).
            - Bottom occurs roughly in the middle third of the cup duration.

    2. Handle Formation:
        - After t_right, look for a consolidation period within duration
          [min_handle_duration, max_handle_duration].
        - Characterized by a minor pullback. Let the low be P_handle_bottom.
        - Handle Depth D_handle = P_right - P_handle_bottom.
        - Validation:
            - Duration constraints met.
            - Relative Depth: D_handle <= D * handle_depth_threshold (handle is shallow).
            - Handle low P_handle_bottom should remain above the cup's midpoint.

    3. Breakout Signal (Long Only):
        - Triggered when close price breaks above the resistance R by a threshold.
        - Condition: close > R * (1 + breakout_threshold).
        - Optional Volume Confirmation: volume > average_volume on breakout.
        - Signal = 1 (Buy).

    4. Signal Strength:
        - Measures the intensity of the breakout.
        - Formula: signal_strength = (close / R) - 1, for breakout signals.

    Risk Management:
    - Uses the RiskManager class to apply stop-loss, take-profit, trailing stops,
      slippage, and transaction costs.
    - Calculates risk-adjusted positions and returns.

    Strategy Parameters (defaults):
        - min_cup_duration (int): Min bars for cup (default: 30).
        - max_cup_duration (int): Max bars for cup (default: 150).
        - min_handle_duration (int): Min bars for handle (default: 5).
        - max_handle_duration (int): Max bars for handle (default: 30).
        - cup_depth_threshold (float): Max relative cup depth (default: 0.3).
        - handle_depth_threshold (float): Max handle depth relative to cup depth (default: 0.5).
        - peak_similarity_threshold (float): Max relative difference between cup peaks (default: 0.15).
        - breakout_threshold (float): Percentage above resistance for breakout (default: 0.005).
        - volume_confirm (bool): Require volume confirmation (default: True).
        - extrema_order (int): Order for scipy.signal.argrelextrema (default: 5).
        - stop_loss_pct (float): Stop loss percentage (default: 0.05).
        - take_profit_pct (float): Take profit percentage (default: 0.10).
        - trailing_stop_pct (float): Trailing stop percentage (default: 0.0).
        - slippage_pct (float): Slippage estimate (default: 0.001).
        - transaction_cost_pct (float): Transaction cost estimate (default: 0.001).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Cup and Handle strategy.

        Args:
            db_config (DatabaseConfig): Database configuration object.
            params (Dict[str, Any], optional): Strategy parameters overriding defaults.
        """
        default_params = {
            'min_cup_duration': 30,
            'max_cup_duration': 150,
            'min_handle_duration': 5,
            'max_handle_duration': 30,
            'cup_depth_threshold': 0.3,         # Max relative depth D/R
            'handle_depth_threshold': 0.5,      # Max relative depth D_handle/D
            'peak_similarity_threshold': 0.15,  # Max |P_left - P_right| / R
            'breakout_threshold': 0.005,        # Breakout trigger: close > R * (1 + threshold)
            'volume_confirm': True,
            'extrema_order': 5,                 # Order for local extrema detection
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trailing_stop_pct': 0.0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
        }
        if params:
            default_params.update(params)

        super().__init__(db_config, default_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            trailing_stop_pct=self.params['trailing_stop_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )

    def generate_signals(self,
                         ticker: Union[str, List[str]], # <<< CHANGED BACK TO 'ticker' (singular)
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generates Cup and Handle trading signals for one or more tickers.

        Retrieves price data, identifies patterns, generates signals, and applies
        risk management. Supports backtesting and latest signal generation.

        Args:
            ticker (Union[str, List[str]]): Ticker symbol or list of symbols. # <<< Updated docstring
            start_date (str, optional): Backtest start date ('YYYY-MM-DD').
                                        Data prior to this is used for lookback.
            end_date (str, optional): Backtest end date ('YYYY-MM-DD').
            initial_position (int): Initial position (0, 1) per ticker before start_date.
                                   (Default: 0). Note: Strategy is long-only.
            latest_only (bool): If True, return only the latest signal row per ticker.

        Returns:
            pd.DataFrame: DataFrame indexed by date (or MultiIndex ['ticker', 'date'])
                          with columns including prices, signal, signal_strength,
                          and risk management outputs (position, return, cumulative_return, etc.).
                          Returns an empty DataFrame if no data is found or processing fails.
        """
        if initial_position not in [0, 1]:
             self.logger.warning("Initial position must be 0 or 1 for this long-only strategy. Setting to 0.")
             initial_position = 0

        required_lookback = int(self.params['max_cup_duration']) + int(self.params['max_handle_duration']) \
                            + int(self.params['extrema_order']) + 20 # Buffer

        effective_start = None
        lookback_for_latest = None

        if latest_only:
            lookback_for_latest = required_lookback
        elif start_date:
            try:
                start_dt = pd.Timestamp(start_date)
                effective_start_dt = start_dt - pd.Timedelta(days=int(required_lookback * 1.5))
                effective_start = effective_start_dt.strftime('%Y-%m-%d')
            except ValueError:
                self.logger.error(f"Invalid start_date format: {start_date}. Use 'YYYY-MM-DD'.")
                return pd.DataFrame()

        self.logger.info(f"Fetching historical prices for {ticker}...") # <<< Use 'ticker' here
        try:
            price_data = self.get_historical_prices(
                tickers=ticker, # <<< Pass 'ticker' to get_historical_prices
                from_date=effective_start,
                to_date=end_date,
                lookback=lookback_for_latest
            )
        except DataRetrievalError as e: # Catch specific error from BaseStrategy
             self.logger.error(f"Data retrieval error for {ticker}: {e}")
             return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error fetching price data for {ticker}: {e}")
            return pd.DataFrame()

        if price_data.empty:
            self.logger.warning(f"No price data retrieved for {ticker} with given parameters.")
            return pd.DataFrame()

        # --- Data Structure Handling ---
        # Determine if the input 'ticker' was a single string or a list
        # And structure the DataFrame accordingly for groupby
        input_is_list = isinstance(ticker, list)
        df_needs_ticker_col = False

        if isinstance(price_data.index, pd.MultiIndex):
             # Data already has ['ticker', 'date'] index from get_historical_prices
             is_multi_ticker = True
             # Ensure the date level is DatetimeIndex
             price_data.index = price_data.index.set_levels(pd.to_datetime(price_data.index.levels[1]), level=1)

        elif input_is_list and len(ticker) == 1:
             # Input was a list of one, but get_historical_prices might return a single index DF
             # Assign the single ticker name and create MultiIndex
             ticker_name = ticker[0]
             price_data['ticker'] = ticker_name
             price_data.index = pd.to_datetime(price_data.index)
             price_data = price_data.set_index(['ticker', price_data.index])
             is_multi_ticker = True # Treat as multi for consistency
             price_data.index.names = ['ticker', 'date'] # Ensure names are set

        elif not input_is_list: # Input was a single string ticker
             # get_historical_prices for single ticker returns single index DF
             ticker_name = ticker
             price_data['ticker'] = ticker_name
             price_data.index = pd.to_datetime(price_data.index)
             price_data = price_data.set_index(['ticker', price_data.index])
             is_multi_ticker = True # Treat as multi for consistency
             price_data.index.names = ['ticker', 'date'] # Ensure names are set
        else:
            # This case (input_is_list > 1, but no MultiIndex) should ideally not happen
            # based on get_historical_prices logic, but handle defensively.
            self.logger.error("Inconsistent data structure received from get_historical_prices.")
            return pd.DataFrame()

        self.logger.info("Generating C&H signals...")

        # Define the function to apply signal generation + risk management
        def apply_strategy_and_rm(group_df):
            # group_df comes with MultiIndex, need to drop ticker level for processing
            single_ticker_df = group_df.reset_index(level='ticker', drop=True)
            # Ensure index is sorted datetime after reset
            single_ticker_df = single_ticker_df.sort_index()

            # Generate signals for this single ticker
            signals_only_df = self._generate_signals_for_ticker(single_ticker_df)

            if signals_only_df.empty:
                return None # Return None if no signals generated for this group

            # Apply risk management
            rm_df = self.risk_manager.apply(signals_only_df, initial_position=initial_position)
            return rm_df


        # Apply the combined function using groupby
        # Using list comprehension and concat handles cases where some groups return None
        results_list = [apply_strategy_and_rm(group) for _, group in price_data.groupby(level='ticker', group_keys=False)]
        # Filter out None results and concatenate
        valid_results = [df for df in results_list if df is not None]

        if not valid_results:
             self.logger.warning("No valid signals or risk management results generated for any ticker.")
             return pd.DataFrame()

        # Concatenate results - need to reintroduce the ticker index/column
        # Get ticker names from the original price_data index
        ticker_names_in_order = price_data.index.get_level_values('ticker').unique().tolist()
        final_result = pd.concat(valid_results, keys=ticker_names_in_order, names=['ticker', 'date'])


        if final_result.empty:
             self.logger.warning("Signal generation and RM resulted in an empty DataFrame.")
             return pd.DataFrame()

        # --- Filtering and Final Selection ---
        # Filter results to the requested backtest period *after* all processing
        if start_date:
            final_result = final_result[final_result.index.get_level_values('date') >= pd.Timestamp(start_date)]
        if end_date:
            final_result = final_result[final_result.index.get_level_values('date') <= pd.Timestamp(end_date)]


        if final_result.empty:
             self.logger.warning("No signals found within the specified date range after filtering.")
             return pd.DataFrame()

        # Handle latest_only: Get the last row for each ticker
        if latest_only:
            # Use tail(1) within groupby for multi-index robustness
            final_result = final_result.groupby(level='ticker', group_keys=False).tail(1)

        self.logger.info("Cup and Handle strategy processing complete.")
        # Ensure index names are correct before returning
        final_result.index.names = ['ticker', 'date']
        return final_result

    # --- Helper methods _generate_signals_for_ticker, _find_cup, _find_handle remain the same ---
    # (No changes needed in the helper methods themselves regarding the 'ticker' parameter name)

    def _generate_signals_for_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates Cup and Handle signals for a single ticker DataFrame.

        Args:
            df (pd.DataFrame): Price data for one ticker, indexed by datetime.
                               Must contain 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            pd.DataFrame: DataFrame with signals and pattern info, indexed by datetime.
                          Includes 'open', 'high', 'low', 'close', 'volume', 'sma20',
                          'avg_volume', 'signal', 'signal_strength', and pattern details.
                          Returns empty DataFrame if input data is insufficient.
        """
        min_required_data = int(self.params['max_cup_duration']) + int(self.params['max_handle_duration'])
        if len(df) < min_required_data:
             # Log ticker name if available (might not be if called directly)
             ticker_name = df.name if hasattr(df, 'name') else 'Unknown Ticker'
             self.logger.warning(f"Insufficient data for {ticker_name} (need {min_required_data}, have {len(df)}). Skipping.")
             return pd.DataFrame()

        # --- Precompute Indicators ---
        df = df.sort_index() # Ensure chronological order
        result = df[['open', 'high', 'low', 'close', 'volume']].copy()
        result['sma20'] = result['close'].rolling(window=20, min_periods=1).mean()
        result['avg_volume'] = result['volume'].rolling(window=20, min_periods=1).mean()

        # --- Find Local Extrema ---
        order = int(self.params['extrema_order'])
        highs = result['high'].values
        lows = result['low'].values
        local_max_indices = argrelextrema(highs, np.greater_equal, order=order)[0] # Use greater_equal for plateaus
        local_min_indices = argrelextrema(lows, np.less_equal, order=order)[0]    # Use less_equal for plateaus

        # Store extrema indices relative to the start of the DataFrame
        result['is_local_max'] = False
        result['is_local_min'] = False
        if len(local_max_indices) > 0:
            # Use get_loc for safety with potential non-standard indices
            max_cols_loc = result.columns.get_loc('is_local_max')
            result.iloc[local_max_indices, max_cols_loc] = True
        if len(local_min_indices) > 0:
            min_cols_loc = result.columns.get_loc('is_local_min')
            result.iloc[local_min_indices, min_cols_loc] = True

        # --- Initialize State Columns ---
        result['signal'] = 0
        result['signal_strength'] = 0.0
        result['pattern_resistance'] = np.nan
        result['cup_depth'] = np.nan # Store cup depth for handle check

        # --- Pattern Detection Loop ---
        start_iter_idx = int(self.params['min_cup_duration']) + int(self.params['min_handle_duration'])
        last_breakout_idx = -1 # Prevent immediate re-entry after a breakout

        # Use .iloc for positional access in the loop
        for i in range(start_iter_idx, len(result)):
            if i <= last_breakout_idx + int(self.params['min_handle_duration']):
                 continue

            window_end_idx = i - int(self.params['min_handle_duration']) # Potential end of cup / start of handle
            window_start_idx = max(0, window_end_idx - int(self.params['max_cup_duration']))

            if window_end_idx <= window_start_idx + int(self.params['min_cup_duration']):
                continue

            # Pass a slice using iloc for positional indexing consistency
            cup_data_slice = result.iloc[window_start_idx : window_end_idx]
            # Reset index for internal helper consistency if they rely on 0-based index
            cup_data_slice_reset = cup_data_slice.reset_index(drop=True)


            cup_found = self._find_cup(cup_data_slice_reset) # Pass reset index slice

            if cup_found:
                cup_left_idx_rel, cup_bottom_idx_rel, cup_right_idx_rel, resistance, cup_depth = cup_found
                # Indices relative to the start of the *slice* (cup_data_slice)
                # Need absolute index in the *full* result DataFrame
                abs_cup_left_idx = window_start_idx + cup_left_idx_rel
                abs_cup_bottom_idx = window_start_idx + cup_bottom_idx_rel
                abs_cup_right_idx = window_start_idx + cup_right_idx_rel

                handle_window_start_idx = abs_cup_right_idx # Start handle check right after cup end
                handle_window_end_idx = i + 1 # Include current index i
                if handle_window_end_idx <= handle_window_start_idx + int(self.params['min_handle_duration']):
                    continue

                # Pass slice using iloc
                handle_data_slice = result.iloc[handle_window_start_idx : handle_window_end_idx]
                handle_data_slice_reset = handle_data_slice.reset_index(drop=True)

                handle_found = self._find_handle(handle_data_slice_reset, resistance, cup_depth)

                if handle_found:
                    # Check for breakout at the current index 'i'.
                    current_close = result.iloc[i]['close']
                    breakout_level = resistance * (1 + self.params['breakout_threshold'])

                    if current_close > breakout_level:
                        volume_ok = True
                        if self.params['volume_confirm']:
                             current_volume = result.iloc[i]['volume']
                             avg_volume = result.iloc[i]['avg_volume']
                             if pd.notna(avg_volume) and avg_volume > 0:
                                 volume_ok = current_volume > avg_volume
                             else:
                                 # If avg_volume is NaN or 0, cannot confirm with volume
                                 volume_ok = False # Treat as non-confirmed if required

                        if volume_ok:
                             # --- Breakout Confirmed ---
                             # Use get_loc for column positions
                             signal_col_loc = result.columns.get_loc('signal')
                             res_col_loc = result.columns.get_loc('pattern_resistance')
                             depth_col_loc = result.columns.get_loc('cup_depth')
                             strength_col_loc = result.columns.get_loc('signal_strength')

                             result.iloc[i, signal_col_loc] = 1
                             result.iloc[i, res_col_loc] = resistance
                             result.iloc[i, depth_col_loc] = cup_depth

                             if resistance > 0:
                                 strength = (current_close / resistance) - 1
                                 result.iloc[i, strength_col_loc] = strength

                             last_breakout_idx = i

        # --- Final Cleanup ---
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'signal', 'signal_strength',
                        'pattern_resistance', 'cup_depth']
        final_df = result[cols_to_keep]

        return final_df


    def _find_cup(self, data: pd.DataFrame) -> Optional[Tuple[int, int, int, float, float]]:
        """
        Identifies the best valid cup formation within the provided data slice.
        Args:
            data (pd.DataFrame): Slice with 0-based integer index. Must contain
                                 'high', 'low', 'is_local_max', 'is_local_min'.
        Returns:
            Optional[Tuple[int, int, int, float, float]]: Relative indices within the slice,
            resistance, and cup depth, or None.
        """
        n = len(data)
        if n < int(self.params['min_cup_duration']):
            return None

        # Find indices relative to the 0-based index of the slice 'data'
        max_indices_rel = data.index[data['is_local_max']].tolist()
        min_indices_rel = data.index[data['is_local_min']].tolist()


        if len(max_indices_rel) < 2 or len(min_indices_rel) < 1:
            return None

        best_cup = None
        max_resistance = -np.inf

        for i in range(len(max_indices_rel)):
            left_idx_rel = max_indices_rel[i]

            for j in range(i + 1, len(max_indices_rel)):
                right_idx_rel = max_indices_rel[j]

                duration = right_idx_rel - left_idx_rel
                if not (int(self.params['min_cup_duration']) <= duration <= int(self.params['max_cup_duration'])):
                    continue

                # Find Bottom between peaks (using relative indices)
                relevant_min_indices_rel = [m_idx for m_idx in min_indices_rel if left_idx_rel < m_idx < right_idx_rel]
                if not relevant_min_indices_rel:
                     # Check if there's any low point between peaks, even if not flagged by argrelextrema
                     if data.iloc[left_idx_rel+1 : right_idx_rel]['low'].empty:
                         continue # No data between peaks
                     bottom_idx_rel_provisional = data.iloc[left_idx_rel+1 : right_idx_rel]['low'].idxmin()
                     # Verify this provisional bottom isn't too close to edges
                     if not (left_idx_rel + duration * 0.2 < bottom_idx_rel_provisional < left_idx_rel + duration * 0.8):
                         continue
                     # It's a potential bottom, proceed with this index
                     bottom_idx_rel = bottom_idx_rel_provisional
                else:
                     # Find the minimum among the flagged local minima between peaks
                     cup_slice_mins = data.iloc[relevant_min_indices_rel]
                     bottom_idx_rel = cup_slice_mins['low'].idxmin()
                     # Re-check position constraint for the identified minimum
                     if not (left_idx_rel + duration * 0.2 < bottom_idx_rel < left_idx_rel + duration * 0.8):
                          continue


                p_left = data.iloc[left_idx_rel]['high']
                p_right = data.iloc[right_idx_rel]['high']
                resistance = max(p_left, p_right)
                p_bottom = data.iloc[bottom_idx_rel]['low']

                if resistance <= 0: continue

                if abs(p_left - p_right) / resistance > self.params['peak_similarity_threshold']:
                    continue

                cup_depth = resistance - p_bottom
                if cup_depth / resistance > self.params['cup_depth_threshold']:
                    continue

                if resistance > max_resistance:
                     max_resistance = resistance
                     best_cup = (left_idx_rel, bottom_idx_rel, right_idx_rel, resistance, cup_depth)

        return best_cup


    def _find_handle(self, data: pd.DataFrame, resistance: float, cup_depth: float) -> Optional[Tuple[int, int, int]]:
        """
        Identifies a valid handle formation within the provided data slice.
        Args:
            data (pd.DataFrame): Slice with 0-based integer index. Must contain
                                 'high', 'low', 'is_local_min'.
            resistance (float): Resistance from the cup.
            cup_depth (float): Depth from the cup.
        Returns:
            Optional[Tuple[int, int, int]]: Relative indices for handle start, bottom, end, or None.
        """
        n = len(data)
        if n < int(self.params['min_handle_duration']):
            return None

        handle_start_idx_rel = 0
        # Use high of the first point in the handle slice as reference
        handle_start_price_ref = data.iloc[handle_start_idx_rel]['high']

        # Find indices relative to the 0-based index of the slice 'data'
        min_indices_rel = data.index[data['is_local_min']].tolist()

        possible_handles = []

        # Consider *any* low point within the handle duration as a potential bottom
        # Not just those flagged by argrelextrema with the cup's order
        potential_bottoms = []
        if n > 1: # Need at least two points to have a bottom different from start
            search_end = min(n, int(self.params['max_handle_duration'])) # Look within max duration
            relevant_slice = data.iloc[1:search_end] # Exclude the start point itself
            if not relevant_slice.empty:
                bottom_idx_rel_provisional = relevant_slice['low'].idxmin()
                potential_bottoms.append(bottom_idx_rel_provisional)

        # Combine with flagged minima if they fall in range
        for idx in min_indices_rel:
            if 0 < idx < int(self.params['max_handle_duration']) and idx not in potential_bottoms:
                potential_bottoms.append(idx)

        if not potential_bottoms:
            return None # No potential bottom found

        for bottom_idx_rel in sorted(potential_bottoms): # Check earlier bottoms first
            duration_to_bottom = bottom_idx_rel - handle_start_idx_rel
            # Min duration to bottom check (e.g., needs at least 1 bar drop)
            if duration_to_bottom < 1: # Adjusted from 2 to allow immediate dip
                 continue

            p_handle_bottom = data.iloc[bottom_idx_rel]['low']
            # Use the high at the actual start of the handle data slice
            handle_depth = handle_start_price_ref - p_handle_bottom

            if cup_depth <= 0: continue # Avoid division by zero
            if handle_depth < 0: continue # Handle bottom cannot be higher than handle start high

            # Check depth relative to cup depth
            if handle_depth / cup_depth > self.params['handle_depth_threshold']:
                 continue

            # Check handle low doesn't undercut too much of the cup's rise
            # E.g., stays above 60% retracement from resistance
            if p_handle_bottom < resistance - cup_depth * 0.6:
                 continue

            # The handle "end" is considered the last point of the data slice passed in,
            # as breakout is checked externally.
            handle_end_idx_rel = n - 1

            total_duration = handle_end_idx_rel - handle_start_idx_rel
            if not (int(self.params['min_handle_duration']) <= total_duration <= int(self.params['max_handle_duration'])):
                 continue

            # Found a valid handle structure ending at the end of this slice
            possible_handles.append((handle_start_idx_rel, bottom_idx_rel, handle_end_idx_rel))
            break # Take the first valid handle bottom found


        if possible_handles:
             return possible_handles[0]
        else:
             return None