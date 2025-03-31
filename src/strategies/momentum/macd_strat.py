# trading_system/src/strategies/macd_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
# Assuming RiskManager is correctly imported from its location
# from src.strategies.risk_management import RiskManager 
# Use the provided definition or ensure correct import path
try:
    # Assuming risk_management.py is in the same directory for this example
    from risk_management import RiskManager 
except ImportError:
    # Fallback if structure is different, adjust as needed
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
          - Ticker symbol (in index if multiple tickers)
          - Price data:           'open', 'high', 'low', 'close', 'volume'
          - MACD indicators:      'macd', 'signal_line', 'histogram'
          - Raw trading signals:  'raw_signal' and a computed 'signal_strength'
          - Risk-managed results: 'position', 'return', 'cumulative_return', 'exit_type'
    
    Args:
        db_config: Database configuration.
        params (Optional[Dict]): Strategy hyperparameters.
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the MACD strategy with specified hyperparameters and database configuration.
        
        Raises:
            ValueError: If the slow period is not greater than the fast period or if MACD periods are not integers.
        """
        # Ensure params is a dictionary if None is passed
        if params is None:
            params = {}
            
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

        # Ensure MACD periods are integers
        try:
            self.params['slow'] = int(self.params['slow'])
            self.params['fast'] = int(self.params['fast'])
            self.params['smooth'] = int(self.params['smooth'])
        except (ValueError, TypeError) as e:
             raise ValueError(f"MACD periods (slow, fast, smooth) must be convertible to integers: {e}")

        
        # Validate parameter consistency.
        if self.params['slow'] <= self.params['fast']:
            raise ValueError("Slow period must be greater than fast period")
        # Redundant check as int conversion is done above, but kept for clarity
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
        adjust positions and compute trade returns. Preserves indices including ticker level for multi-ticker requests.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol or list of ticker symbols.
            start_date (str, optional): Backtesting start date (YYYY-MM-DD).
            end_date (str, optional): Backtesting end date (YYYY-MM-DD).
            initial_position (int): Initial trading position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the most recent signal(s) for each ticker.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, MACD indicators, trading signals,
                          signal strength, risk-managed positions, returns, and exit types.
                          Index will be DatetimeIndex for single ticker, MultiIndex 
                          (level 0: ticker, level 1: date) for multiple tickers.
        """
        try:
            # Retrieve historical price data with optimized lookback if only the latest signal is needed.
            data = self._get_optimized_data(ticker, start_date, end_date, latest_only)
            
            if data.empty:
                self.logger.warning("No data retrieved for ticker(s): %s", ticker)
                return pd.DataFrame()

            # Determine if processing single or multiple tickers based on data index type.
            # Use isinstance check on the index directly.
            is_multi_ticker = isinstance(data.index, pd.MultiIndex)

            if is_multi_ticker:
                # Process data for each ticker in a vectorized (grouped) manner.
                # group_keys=False prevents the group key (ticker) from being added as an extra index level
                # The original MultiIndex structure is typically preserved by apply.
                results = data.groupby(level=0, group_keys=False).apply(
                    self._process_single_ticker_group, 
                    initial_position=initial_position
                )
                
                if latest_only:
                    # Get the last row for each ticker (group)
                    results = results.groupby(level=0).tail(1)
                return results
            else:
                # Single ticker processing.
                # No need to check index type again, process directly.
                output = self._process_single_ticker_group(data, initial_position)
                return output.iloc[[-1]] if latest_only else output

        except DataRetrievalError as e:
             self.logger.error(f"Data retrieval failed for {ticker}: {e}")
             # Re-raise or return empty based on desired error handling
             return pd.DataFrame() # Return empty frame on data error
        except Exception as e:
            self.logger.exception(f"Error processing ticker(s) {ticker}: {str(e)}") # Use logger.exception for traceback
            # Re-raise the exception to allow higher-level handling
            raise 

    def _process_single_ticker_group(self, group_data: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Processes data for a single ticker (or a group corresponding to one ticker).
        Calculates indicators, generates signals, applies risk management, and builds output.
        Assumes group_data has a DatetimeIndex (or is a slice of a MultiIndex).
        """
        # Validate sufficient data for indicator computation.
        min_required_data = self.params['slow'] + self.params['smooth'] # Minimum needed for EMAs
        if not self._validate_data(group_data, min_required_data):
            ticker_name = group_data.index.get_level_values(0)[0] if isinstance(group_data.index, pd.MultiIndex) else "single ticker"
            self.logger.warning(f"Insufficient data for {ticker_name} ({len(group_data)} rows, need {min_required_data}). Skipping.")
            return pd.DataFrame() # Return empty frame for this group

        # Compute MACD indicators. Index will match group_data.index
        macd_line, signal_line, histogram = self._calculate_macd(group_data['close'])
        
        # Generate raw trading signals based on MACD crossovers. Index will match group_data.index
        raw_signals = self._generate_crossover_signals(macd_line, signal_line)
        
        # Apply risk management rules. This function now handles index preservation internally.
        # It expects the group_data and raw_signals with their original index.
        risk_managed = self._apply_risk_management(group_data, raw_signals, initial_position)
        
        # Build and return the final output for the group.
        # All inputs to _build_output should now share the same index (group_data.index).
        return self._build_output(group_data, macd_line, signal_line, histogram, raw_signals, risk_managed)


    def _get_optimized_data(self, ticker: Union[str, List[str]], 
                            start_date: Optional[str],
                            end_date: Optional[str],
                            latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data optimized for MACD calculation stability.
        
        If latest_only is True, uses a minimal lookback period (based on slow + smooth parameters + buffer)
        to ensure that the indicator is stable. Supports both single and multiple tickers.
        
        Args:
            ticker (str or List[str]): Stock ticker symbol(s).
            start_date (str, optional): Start date (YYYY-MM-DD).
            end_date (str, optional): End date (YYYY-MM-DD).
            latest_only (bool): Flag indicating retrieval of minimal data required for a stable signal.
        
        Returns:
            pd.DataFrame: DataFrame containing historical price data, indexed appropriately
                          (DatetimeIndex for single ticker, MultiIndex for list).
        
        Raises:
            DataRetrievalError: If data cannot be fetched from the source.
        """
        try:
            if latest_only:
                # Stable EMA calculation needs roughly 3x the period, plus signal smoothing period. Add buffer.
                # Lookback = (slow_period * 3) + smooth_period + buffer
                lookback = (self.params['slow'] * 3) + self.params['smooth'] + 30 # Generous buffer
                self.logger.debug(f"Latest only mode: Using lookback={lookback} bars.")
                return self.get_historical_prices(ticker, lookback=lookback)
            else:
                self.logger.debug(f"Fetching data for {ticker} from {start_date} to {end_date}.")
                return self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
        except Exception as e:
             # Catch specific data errors if BaseStrategy defines them, or broad Exception
             self.logger.error(f"Failed to retrieve price data for {ticker}: {e}")
             # Wrap in a custom exception type for clarity
             raise DataRetrievalError(f"Failed to retrieve price data for {ticker}: {e}")


    def _validate_data(self, data: pd.DataFrame, min_rows: int) -> bool:
        """Check if the DataFrame has enough rows."""
        if len(data) < min_rows:
            return False
        # Optional: Check for NaNs in 'close' if required
        if data['close'].isnull().any():
             self.logger.warning("NaN values found in 'close' price data.")
             # Decide if this should invalidate the data (return False) or just warn
             # return False 
        return True

    def _calculate_macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MACD line, Signal line, and Histogram using exponential moving averages.
        Index of output Series will match the input 'close' Series index.
        
        Args:
            close (pd.Series): Series of closing prices.
        
        Returns:
            tuple: (MACD line, Signal line, Histogram), all pd.Series with matching index.
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
        Index of output Series will match the input Series index.
        
        Args:
            macd (pd.Series): MACD line series.
            signal (pd.Series): Signal line series.
        
        Returns:
            pd.Series: Series of raw trading signals (1, 0, -1) with matching index.
        """
        # Determine positions based on MACD relative to signal line.
        above = macd > signal
        below = macd < signal
        
        # Identify crossover events. Ensure consistent index handling with fillna(False)
        cross_up = above & (~above.shift(1).fillna(False))
        cross_down = below & (~below.shift(1).fillna(False))
        
        # Initialize signals Series with the same index as macd
        signals = pd.Series(0, index=macd.index, dtype=int) 
        signals.loc[cross_up] = 1
        
        if not self.params['long_only']:
            signals.loc[cross_down] = -1
        else:
            # Explicitly set to 0 for long_only, though initialization covers this
            signals.loc[cross_down] = 0 

        # Ensure the first signal is 0 if no crossover happened immediately
        signals.iloc[0] = 0 # Or handle initial state based on strategy rules if needed

        return signals

    def _apply_risk_management(self, data: pd.DataFrame, signals: pd.Series, initial_position: int) -> pd.DataFrame:
        """
        Integrate risk management using RiskManager, preserving the original index.
        
        Constructs the necessary input DataFrame for RiskManager (with a DatetimeIndex),
        applies risk management, and then restores the original index to the results.
        
        Args:
            data (pd.DataFrame): Historical price data with 'close', 'high', 'low', 
                                 and an index (DatetimeIndex or MultiIndex slice).
            signals (pd.Series): Raw trading signals with matching index.
            initial_position (int): Initial trading position.
        
        Returns:
            pd.DataFrame: DataFrame with risk-managed results ('position', 'return', etc.)
                          preserving the original index of 'data' and 'signals'.
        """
        # Store the original index
        original_index = data.index
        
        # Prepare the input DataFrame for RiskManager. It needs 'signal', 'close', 'high', 'low'.
        # Crucially, ensure its index is a simple DatetimeIndex as required by RiskManager.
        sig_for_rm = pd.DataFrame({
            'signal': signals,
            'close': data['close'],
            'high': data['high'],
            'low': data['low']
        }, index=original_index) # Keep original index for now

        # If the original index IS a MultiIndex, create a temporary DataFrame with just the DatetimeIndex.
        # Otherwise, use the DataFrame as is (with a copy to be safe).
        if isinstance(original_index, pd.MultiIndex):
            # Assuming date is level 1 (adjust if necessary)
            datetime_index = original_index.get_level_values(1) 
            # Create a temporary DataFrame with the DatetimeIndex for the RiskManager
            temp_sig_for_rm = sig_for_rm.copy()
            temp_sig_for_rm.index = datetime_index
            if not isinstance(temp_sig_for_rm.index, pd.DatetimeIndex):
                 self.logger.warning("Failed to create DatetimeIndex for RiskManager. Risk processing may fail.")
                 # Fallback or raise error
                 # For now, proceed but log warning. The RiskManager will likely raise TypeError.
                 pass 
            input_to_rm = temp_sig_for_rm
        else:
             # Index is already DatetimeIndex, just use a copy
             input_to_rm = sig_for_rm.copy()
             # Verify it's a DatetimeIndex as expected by RiskManager
             if not isinstance(input_to_rm.index, pd.DatetimeIndex):
                  raise TypeError(f"RiskManager requires a DatetimeIndex, but received {type(input_to_rm.index)}")


        # Initialize and apply RiskManager
        try:
            risk_manager = RiskManager(
                stop_loss_pct=self.params['stop_loss_pct'],
                take_profit_pct=self.params['take_profit_pct'],
                trailing_stop_pct=self.params['trailing_stop_pct'],
                slippage_pct=self.params['slippage_pct'],
                transaction_cost_pct=self.params['transaction_cost_pct']
            )
            
            # Apply risk management using the DatetimeIndexed data
            rm_results_temp_index = risk_manager.apply(input_to_rm, initial_position)

        except TypeError as e:
             # Catch TypeErrors likely caused by incorrect index in RiskManager
             self.logger.error(f"TypeError during RiskManager application, likely due to index issues: {e}")
             # Return an empty DataFrame or re-raise, matching desired error handling
             # Create an empty DataFrame with expected columns and original index
             rm_cols = ['position', 'return', 'cumulative_return', 'exit_type']
             return pd.DataFrame(index=original_index, columns=rm_cols).fillna(0) # Fill with 0 or NaN as appropriate

        # --- CRITICAL STEP: Restore the original index ---
        # The result from RiskManager has the temporary DatetimeIndex.
        # We need to put the original index (single or MultiIndex) back on.
        rm_results_original_index = rm_results_temp_index.copy()
        rm_results_original_index.index = original_index

        # Select only the columns added by the risk manager
        risk_managed_output_cols = ['position', 'return', 'cumulative_return', 'exit_type']
        # Ensure columns exist, handling potential errors in RM.apply
        for col in risk_managed_output_cols:
             if col not in rm_results_original_index.columns:
                  rm_results_original_index[col] = 0 # Or np.nan
                  self.logger.warning(f"RiskManager output missing expected column '{col}'. Added with default value.")
                  
        return rm_results_original_index[risk_managed_output_cols]


    def _build_output(self, data: pd.DataFrame, macd_line: pd.Series, 
                      signal_line: pd.Series, histogram: pd.Series, 
                      raw_signals: pd.Series, risk_managed: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the final output DataFrame combining all components.
        Assumes all input DataFrames/Series share the same index.
        
        Args:
            data (pd.DataFrame): Price data ('open', 'high', 'low', 'close', 'volume').
            macd_line (pd.Series): Computed MACD line.
            signal_line (pd.Series): Computed signal line.
            histogram (pd.Series): MACD histogram.
            raw_signals (pd.Series): Raw trading signals.
            risk_managed (pd.DataFrame): Risk-managed results ('position', 'return', etc.).
        
        Returns:
            pd.DataFrame: Combined DataFrame. Index is preserved from inputs.
        """
        # Create DataFrame for indicators, ensuring index matches 'data'
        indicators = pd.DataFrame({
            'macd': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'raw_signal': raw_signals,
            'signal_strength': histogram.abs() # Use abs() for magnitude
        }, index=data.index) # Explicitly use data.index
        
        # Select required columns from data
        price_data = data[['open', 'high', 'low', 'close', 'volume']]

        # Concatenate along columns. Indices must align perfectly.
        # If _apply_risk_management correctly restored the index, this works.
        try:
             output = pd.concat([
                 price_data,
                 indicators,
                 risk_managed # Already contains the correctly indexed RM results
             ], axis=1)
        except Exception as e:
             self.logger.error(f"Failed to concatenate final output DataFrame, likely index mismatch: {e}")
             # Attempt to show index differences for debugging
             if not price_data.index.equals(indicators.index):
                  self.logger.error("Index mismatch between price_data and indicators.")
             if not price_data.index.equals(risk_managed.index):
                  self.logger.error("Index mismatch between price_data and risk_managed.")
                  self.logger.error(f"Price Index Head:\n{price_data.index[:5]}")
                  self.logger.error(f"Risk Managed Index Head:\n{risk_managed.index[:5]}")

             # Return an empty frame or re-raise
             return pd.DataFrame()

        return output