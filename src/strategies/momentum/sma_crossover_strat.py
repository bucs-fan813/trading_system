# trading_system/src/strategies/momentum/sma_crossover_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager # Ensure correct import path

# Configure logger for this module
logger = logging.getLogger(__name__)

class SMAStrategy(BaseStrategy):
    """
    Implements a Simple Moving Average (SMA) Crossover trading strategy with
    integrated risk management.

    Signal Generation Logic:
    - Calculates a short-term SMA and a long-term SMA based on the 'close' price.
    - Generates a buy signal (1) when the short SMA crosses above the long SMA.
    - Generates a sell signal (-1) when the short SMA crosses below the long SMA.
    - If 'long_only' is True, sell signals are converted to hold/exit signals (0).
    - Signal strength is calculated as the normalized difference between the SMAs.

    Risk Management:
    - Utilizes the RiskManager class to apply stop-loss, take-profit, trailing
      stop-loss, slippage, and transaction costs to the raw signals.
    - Calculates risk-managed positions and returns.

    Output Columns:
    - Includes original OHLCV data, calculated SMAs, raw signal, signal strength,
      daily returns, simple strategy returns (based on raw signal position),
      and the outputs from RiskManager (risk-managed position, returns, cumulative
      returns, and exit actions).

    Compatibility:
    - Designed to integrate with StrategyOptimizer and PerformanceEvaluator.
    - Handles both single and multiple ticker inputs.
    - Returns data in a format compatible with portfolio-level analysis (typically
      a DataFrame with a 'ticker' column for multi-ticker inputs).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the SMA Crossover Strategy.

        Args:
            db_config (DatabaseConfig): Database configuration object.
            params (Optional[Dict]): Strategy parameters dictionary. Expected keys:
                - 'short_window' (int): Window for the short SMA (default: 20).
                - 'long_window' (int): Window for the long SMA (default: 50).
                - 'stop_loss_pct' (float): Stop-loss percentage (default: 0.05).
                - 'take_profit_pct' (float): Take-profit percentage (default: 0.10).
                - 'trailing_stop_pct' (float): Trailing stop percentage (default: 0.0).
                - 'slippage_pct' (float): Slippage per transaction (default: 0.001).
                - 'transaction_cost_pct' (float): Cost per transaction (default: 0.001).
                - 'long_only' (bool): If True, restrict to long trades (default: True).

        Raises:
            ValueError: If short_window is not less than long_window after casting to int,
                        or if window parameters cannot be cast to int.
        """
        default_params = {
            'short_window': 20,
            'long_window': 50,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trailing_stop_pct': 0.0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            'long_only': True
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(db_config, merged_params) # Pass merged params to base

        # --- Strategy Specific Parameters ---
        # Cast window parameters to int immediately to prevent rolling() errors
        try:
            # Use .get with default from merged_params to handle missing keys robustly
            self.short_window = int(self.params.get('short_window'))
            self.long_window = int(self.params.get('long_window'))
        except (ValueError, TypeError) as e:
            logger.error(f"Could not convert window parameters to integers: {e}")
            logger.error(f"Received short_window: {self.params.get('short_window')}, long_window: {self.params.get('long_window')}")
            raise ValueError("Window parameters (short_window, long_window) must be convertible to integers.") from e

        self.long_only = self.params.get('long_only')

        # --- Validation ---
        if self.short_window >= self.long_window:
            raise ValueError(f"short_window ({self.short_window}) must be less than long_window ({self.long_window}).")

        # --- Risk Management Initialization ---
        risk_params = {
            'stop_loss_pct': self.params.get('stop_loss_pct'),
            'take_profit_pct': self.params.get('take_profit_pct'),
            'trailing_stop_pct': self.params.get('trailing_stop_pct'),
            'slippage_pct': self.params.get('slippage_pct'),
            'transaction_cost_pct': self.params.get('transaction_cost_pct')
        }
        self.risk_manager = RiskManager(**risk_params)

        logger.debug(
            f"Initialized SMA Strategy: short={self.short_window}, long={self.long_window}, "
            f"long_only={self.long_only}, Risk Params={risk_params}"
        )

    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate SMA crossover signals and apply risk management.

        Handles single/multiple tickers and backtesting/latest signal modes.

        Args:
            ticker (Union[str, List[str]]): Ticker symbol or list of symbols.
            start_date (str, optional): Start date 'YYYY-MM-DD' for backtesting.
            end_date (str, optional): End date 'YYYY-MM-DD' for backtesting.
            initial_position (int): Starting position (0, 1, -1). Default is 0.
            latest_only (bool): If True, return only the latest signal row(s)
                                without full risk management simulation. Default is False.

        Returns:
            pd.DataFrame: DataFrame containing price data, indicators, signals,
                          risk-managed results, and performance metrics.
                          For multi-ticker requests, typically includes a 'ticker' column
                          or is a MultiIndex DataFrame.
                          Returns an empty DataFrame on failure or insufficient data.
        """
        # Determine lookback needed for the longest SMA calculation + buffer
        # Requires long_window periods for SMA + 1 for previous value comparison
        lookback_required = self.long_window + 1

        try:
            # --- Mode 1: Latest Signal Only ---
            if latest_only:
                logger.debug(f"Generating latest signal only for {ticker} with lookback {lookback_required}")
                # Fetch minimal data needed (BaseStrategy handles lookback parameter here)
                data = self.get_historical_prices(ticker, lookback=lookback_required)
                if data.empty:
                     logger.warning(f"No data retrieved for latest signal generation: {ticker}")
                     return pd.DataFrame()

                # Define columns needed for latest signal output
                latest_signal_cols = ['signal', 'strength', 'close', 'short_sma', 'long_sma']

                if isinstance(ticker, list):
                    latest_signals_list = []
                    # Group by ticker level
                    for t, group_raw in data.groupby(level='ticker', group_keys=False):
                        # Reset index to ensure DatetimeIndex for processing
                        group = group_raw.reset_index(level='ticker', drop=True)
                        if not isinstance(group.index, pd.DatetimeIndex):
                            logger.error(f"Ticker {t}: Group index is not DatetimeIndex after reset. Type: {type(group.index)}. Skipping.")
                            continue

                        if not self._validate_data(group, min_records=lookback_required):
                            logger.warning(f"Insufficient data for latest signal ({t}): {len(group)} < {lookback_required}")
                            continue

                        signals_df = self._calculate_smas_and_signals(group.copy())
                        if not signals_df.empty:
                            latest_row = signals_df.iloc[-1:].copy()
                             # Ensure all desired columns are present, add NaN if missing
                            for col in latest_signal_cols:
                                if col not in latest_row:
                                    latest_row[col] = np.nan
                            latest_row['ticker'] = t
                            latest_signals_list.append(latest_row[ ['ticker'] + latest_signal_cols]) # Select and order cols

                    if not latest_signals_list:
                        logger.warning("No latest signals generated for any ticker in the list.")
                        return pd.DataFrame()

                    final_df = pd.concat(latest_signals_list)
                    # Set index back to Ticker, Date for consistency? Or return with ticker col?
                    # Let's return with ticker column and date index for simplicity here.
                    final_df = final_df.set_index(final_df.index.rename('date')) # Ensure index name is 'date'
                    return final_df

                else: # Single ticker
                    # No need to reset index for single ticker if get_historical_prices returns DatetimeIndex
                    if not isinstance(data.index, pd.DatetimeIndex):
                         logger.error(f"Single Ticker {ticker}: Expected DatetimeIndex, got {type(data.index)}. Cannot proceed.")
                         return pd.DataFrame()

                    if not self._validate_data(data, min_records=lookback_required):
                        logger.warning(f"Insufficient data for latest signal ({ticker}): {len(data)} < {lookback_required}")
                        return pd.DataFrame()

                    signals_df = self._calculate_smas_and_signals(data.copy())
                    if signals_df.empty:
                        return pd.DataFrame()
                    latest_row = signals_df.iloc[-1:]
                    # Ensure all desired columns are present
                    for col in latest_signal_cols:
                         if col not in latest_row:
                              latest_row[col] = np.nan
                    return latest_row[latest_signal_cols] # Select columns


            # --- Mode 2: Backtesting (Full Period) ---
            else:
                logger.debug(f"Generating full backtest signals for {ticker} from {start_date} to {end_date}")

                # --- Calculate buffer start date to fetch lookback data ---
                # Need lookback_required periods *before* the start_date
                # Use 1.5x multiplier for business days vs calendar days buffer
                try:
                    buffer_days = int(lookback_required * 1.5) + 5 # Add extra days for safety
                    buffer_start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
                    fetch_start_date = buffer_start_dt.strftime('%Y-%m-%d')
                    logger.debug(f"Fetching data from {fetch_start_date} to {end_date} to ensure lookback for fold start {start_date}")
                except Exception as date_err:
                    logger.error(f"Error calculating buffer start date from {start_date}: {date_err}")
                    return pd.DataFrame()

                # Fetch data including the buffer period BEFORE the fold start date
                # NOTE: We remove the 'lookback' parameter here because BaseStrategy currently
                # ignores it when from_date/to_date are specified. The fetch_start_date handles the lookback.
                data = self.get_historical_prices(ticker, from_date=fetch_start_date, to_date=end_date)

                if data.empty:
                     logger.warning(f"No data retrieved for backtest: {ticker} [{fetch_start_date}-{end_date}]")
                     return pd.DataFrame()

                # Convert original start/end dates for final filtering
                original_start_dt = pd.to_datetime(start_date)
                original_end_dt = pd.to_datetime(end_date)

                # --- Process Data (Single vs Multi-Ticker) ---
                if isinstance(ticker, list):
                    signals_list = []
                    # Group by the first level of the MultiIndex (should be 'ticker')
                    for t, group_raw in data.groupby(level='ticker', group_keys=False):

                        # --- Explicitly set the date index for the group ---
                        group = group_raw.reset_index(level='ticker', drop=True) # Drop the ticker level
                        if not isinstance(group.index, pd.DatetimeIndex):
                            logger.error(f"Ticker {t}: Group index is not DatetimeIndex after reset. Type: {type(group.index)}. Skipping.")
                            continue
                        # --- End index fix ---

                        logger.debug(f"Processing ticker: {t} with {len(group)} rows (fetched from {fetch_start_date})")

                        # Validate AFTER potentially fetching more data, check against requirement for calculation
                        if not self._validate_data(group, min_records=lookback_required):
                            # This check should ideally pass now if data fetching worked
                            logger.warning(f"Insufficient data for backtesting {t} even after fetching buffer: {len(group)} < {lookback_required}")
                            continue

                        # 1. Calculate Raw Signals (on the buffered data)
                        signals = self._calculate_smas_and_signals(group.copy())
                        if signals.empty:
                             logger.warning(f"Signal calculation failed for {t}. Skipping.")
                             continue

                        # 2. Apply Risk Management (on the buffered data)
                        risk_managed_signals = self.risk_manager.apply(signals, initial_position)

                        # 3. Add Performance Metrics & Rename Columns
                        risk_managed_signals['daily_return'] = risk_managed_signals['close'].pct_change().fillna(0)
                        risk_managed_signals['strategy_return'] = risk_managed_signals['daily_return'] * risk_managed_signals['position'].shift(1).fillna(0)
                        risk_managed_signals.rename(columns={'return': 'rm_strategy_return',
                                                             'cumulative_return': 'rm_cumulative_return',
                                                             'exit_type': 'rm_action'}, inplace=True)

                        # 4. Add Ticker Identifier
                        risk_managed_signals['ticker'] = t
                        signals_list.append(risk_managed_signals)

                    if not signals_list:
                        logger.warning("No signals generated for any ticker during backtest after processing.")
                        return pd.DataFrame()

                    # Combine results from all tickers (includes buffer period data)
                    final_signals_unfiltered = pd.concat(signals_list)
                    # Create the MultiIndex before filtering
                    final_signals_unfiltered = final_signals_unfiltered.set_index(['ticker', final_signals_unfiltered.index.rename('date')])


                else: # Single Ticker Processing
                    # Ensure index is correct
                    if not isinstance(data.index, pd.DatetimeIndex):
                        logger.error(f"Single Ticker {ticker}: Expected DatetimeIndex, got {type(data.index)}. Cannot proceed.")
                        return pd.DataFrame()

                    logger.debug(f"Processing single ticker: {ticker} with {len(data)} rows (fetched from {fetch_start_date})")
                    if not self._validate_data(data, min_records=lookback_required):
                        logger.warning(f"Insufficient data for backtesting {ticker} even after fetching buffer: {len(data)} < {lookback_required}")
                        return pd.DataFrame()

                    # 1. Calculate Raw Signals
                    signals = self._calculate_smas_and_signals(data.copy())
                    if signals.empty:
                         logger.warning(f"Signal calculation failed for {ticker}. Returning empty.")
                         return pd.DataFrame()

                    # 2. Apply Risk Management
                    risk_managed_signals = self.risk_manager.apply(signals, initial_position)

                    # 3. Add Performance Metrics & Rename Columns
                    risk_managed_signals['daily_return'] = risk_managed_signals['close'].pct_change().fillna(0)
                    risk_managed_signals['strategy_return'] = risk_managed_signals['daily_return'] * risk_managed_signals['position'].shift(1).fillna(0)
                    risk_managed_signals.rename(columns={'return': 'rm_strategy_return',
                                                         'cumulative_return': 'rm_cumulative_return',
                                                         'exit_type': 'rm_action'}, inplace=True)
                    final_signals_unfiltered = risk_managed_signals


                # --- FINAL Filtering step AFTER all calculations ---
                if final_signals_unfiltered.empty:
                    logger.warning("Final signals DataFrame is empty before date filtering.")
                    return pd.DataFrame()

                # Filter results strictly within the requested date range
                logger.debug(f"Filtering final results from {original_start_dt} to {original_end_dt}")
                if isinstance(final_signals_unfiltered.index, pd.MultiIndex):
                     date_level_values = final_signals_unfiltered.index.get_level_values('date')
                     mask = (date_level_values >= original_start_dt) & (date_level_values <= original_end_dt)
                     final_signals = final_signals_unfiltered[mask]
                elif isinstance(final_signals_unfiltered.index, pd.DatetimeIndex): # Single ticker case
                     final_signals = final_signals_unfiltered.loc[original_start_dt:original_end_dt]
                else:
                     logger.warning("Could not filter final signals by original date range due to unexpected index.")
                     final_signals = final_signals_unfiltered # Return unfiltered if index is weird

                if final_signals.empty:
                     logger.warning(f"Final signals DataFrame is empty AFTER filtering for dates {start_date} to {end_date}.")

                return final_signals

        except DataRetrievalError as dre:
             logger.error(f"Data retrieval error for {ticker}: {dre}")
             return pd.DataFrame()
        except ValueError as ve: # Catch specific errors like window size validation
             logger.error(f"Configuration or calculation error for SMA strategy: {ve}", exc_info=True)
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error generating signals for {ticker}: {e}", exc_info=True) # Log traceback
            return pd.DataFrame()


    def _calculate_smas_and_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMAs and generate raw crossover signals for a single ticker's data.

        Args:
            df (pd.DataFrame): Input DataFrame for a single ticker. Must contain
                               'open', 'high', 'low', 'close' columns and be
                               indexed by a DatetimeIndex.

        Returns:
            pd.DataFrame: DataFrame with original price data plus added columns:
                          'short_sma', 'long_sma', 'signal', 'strength'.
                          Returns an empty DataFrame if calculation fails or input is invalid.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("_calculate_smas_and_signals received non-DatetimeIndex.")
            return pd.DataFrame()

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             logger.warning(f"Input DataFrame missing required OHLC columns for SMA calculation: {missing}")
             # Cannot proceed without close, high, low for RM. Return empty.
             return pd.DataFrame()

        df_calc = df.copy() # Work on a copy

        # Ensure data is sorted by date (important for rolling calculations and shifts)
        df_calc = df_calc.sort_index()

        # Calculate SMAs using the integer window sizes
        try:
            df_calc['short_sma'] = df_calc['close'].rolling(window=self.short_window, min_periods=self.short_window).mean()
            df_calc['long_sma'] = df_calc['close'].rolling(window=self.long_window, min_periods=self.long_window).mean()
        except ValueError as e:
             logger.error(f"Error during rolling calculation (check window sizes): {e}")
             logger.error(f"Short window: {self.short_window} (type: {type(self.short_window)}), Long window: {self.long_window} (type: {type(self.long_window)})")
             return pd.DataFrame() # Calculation failed

        # Calculate previous SMAs to detect the crossover event
        df_calc['prev_short'] = df_calc['short_sma'].shift(1)
        df_calc['prev_long'] = df_calc['long_sma'].shift(1)

        # Identify crossover conditions precisely
        crossed_above = (df_calc['short_sma'] > df_calc['long_sma']) & (df_calc['prev_short'] <= df_calc['prev_long'])
        crossed_below = (df_calc['short_sma'] < df_calc['long_sma']) & (df_calc['prev_short'] >= df_calc['prev_long'])

        # Generate signals based on crossover events
        df_calc['signal'] = 0 # Default to hold
        df_calc.loc[crossed_above, 'signal'] = 1
        df_calc.loc[crossed_below, 'signal'] = -1

        # Calculate normalized signal strength
        # Use np.where to handle potential division by zero if close is 0
        df_calc['strength'] = np.where(
             np.abs(df_calc['close']) > 1e-9, # Check if close is non-zero
            (df_calc['short_sma'] - df_calc['long_sma']) / df_calc['close'],
             0.0 # Assign 0 strength if close is 0 or near zero
        )
        # Fill NaN strength resulting from initial SMA NaNs
        df_calc['strength'].fillna(0.0, inplace=True)

        # Apply long-only constraint if enabled
        if self.long_only:
            df_calc['signal'] = df_calc['signal'].clip(lower=0) # Replace -1 with 0

        # Select and return relevant columns including OHLC needed for RiskManager
        # and indicators for analysis. Drop temporary columns.
        return_cols = ['open', 'high', 'low', 'close', 'volume', # Pass through original data
                       'short_sma', 'long_sma', 'signal', 'strength']
        # Ensure all expected columns exist before returning
        output_df = df_calc[[col for col in return_cols if col in df_calc.columns]].copy() # Use copy

        return output_df


    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the input DataFrame (expected for a single ticker) is suitable.

        Checks for non-empty, minimum records, DatetimeIndex, and required columns.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            min_records (int): Minimum number of records required.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        if df is None or df.empty:
            # logger.warning("Data validation failed: DataFrame is None or empty.") # Reduced verbosity
            return False

        # --- CRITICAL: Check index type HERE ---
        if not isinstance(df.index, pd.DatetimeIndex):
             logger.warning(f"Data validation failed: Index is not a DatetimeIndex (type: {type(df.index)}).")
             return False
        # --- End index check ---

        if len(df) < min_records:
            # logger.warning(f"Data validation failed: {len(df)} records present, {min_records} required.") # Reduced verbosity
            return False

        required_cols = ['open', 'high', 'low', 'close'] # Check essential columns
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             logger.warning(f"Data validation failed: Missing required columns: {missing}")
             return False

        return True