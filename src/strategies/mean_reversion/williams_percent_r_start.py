# trading_system/src/strategies/williams_r.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class WilliamsRStrategy(BaseStrategy):
    """
    Williams %R Trading Strategy with Integrated Risk Management.

    The Williams %R indicator is computed as:
        WR = -100 * ((Highest High - Close) / (Highest High - Lowest Low))
    where Highest High and Lowest Low are computed over a lookback period (default 14 days).

    Trading signals are generated as follows:
        - A buy signal (signal = 1) is triggered when the previous WR value is above the
          oversold threshold (default -80) and the current WR value crosses below or equals -80.
        - A sell signal (signal = -1) is triggered when the previous WR value is below the
          overbought threshold (default -20) and the current WR value crosses above or equals -20.
        - In a long-only strategy, sell signals are neutralized (i.e. set to 0).
    
    Signal strength is computed as the absolute difference between the WR value and the 
    relevant threshold:
        - For a buy:    |WR - (oversold)|
        - For a sell:   |WR - (overbought)|

    After signal generation, the RiskManager is applied. This component:
        - Adjusts the entry price for slippage and transaction cost.
        - Computes stop-loss and take-profit thresholds.
        - Determines exit events (for stop loss, take profit, or a signal reversal).
        - Computes the realized return and cumulative return.
    
    The strategy supports:
        - Backtesting with a specific date range.
        - End-of-day forecasting (using minimal lookback data).
        - Vectorized processing of a single ticker or a list of tickers.
    
    Output:
        A DataFrame containing for each date (and ticker if applicable):
          - 'open', 'high', 'low', 'close': Price data.
          - 'wr': Williams %R indicator.
          - 'signal': Trading signal (1 for buy, -1 for sell, 0 for hold).
          - 'signal_strength': Strength of the signal.
          - 'position': Trading position after risk management.
          - 'return': Realized trade return on exit events.
          - 'cumulative_return': Cumulative strategy return.
          - 'exit_type': Reason for exit (stop_loss, take_profit, signal_exit).

    Args:
        db_config: Database configuration details.
        params (dict, optional): Strategy-specific parameters including:
            - 'wr_period': Lookback period for WR calculation (default: 14).
            - 'oversold_threshold': Oversold level (default: -80).
            - 'overbought_threshold': Overbought level (default: -20).
            - 'data_lookback': Number of records to retrieve if no date range is given (default: 252).
            - 'long_only': Flag for long-only trading (default: True).
            - 'stop_loss_pct': Stop loss percentage (default: 0.05).
            - 'take_profit_pct': Take profit percentage (default: 0.10).
            - 'slippage_pct': Slippage percentage (default: 0.001).
            - 'transaction_cost_pct': Transaction cost percentage (default: 0.001).

    Methods:
        generate_signals: Generates the Williams %R signals, applies risk management, and returns
                          a DataFrame for backtesting or EOD signal forecast.
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the Williams %R Strategy.

        Args:
            db_config: A database configuration object.
            params (dict, optional): Strategy-specific parameters.
        """
        super().__init__(db_config, params)
        self.logger.info("WilliamsRStrategy initialized with parameters: %s", self.params)
        self.long_only = bool(self.params.get('long_only', True))
    
    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals using the Williams %R indicator, then apply risk management rules.

        For a single ticker:
          - Retrieves historical prices from the database for the specified date range (or minimal lookback if latest_only).
          - Computes the Williams %R indicator over the given lookback period (wr_period).
          - Generates a buy signal when WR crosses below the oversold threshold and a sell signal when it 
            crosses above the overbought threshold.
          - Computes signal strength as the absolute difference between WR and the relevant threshold.
          - If long_only is enabled, sell signals are neutralized.
          - Applies risk management via RiskManager to adjust prices (slippage & transaction cost) and compute
            stop-loss/take-profit triggered exits, realized returns, and cumulative returns.

        For a list of tickers:
          - Retrieves data for all tickers and applies the above calculations groupwise in a vectorized manner.

        Args:
            ticker (str or List[str]): Single ticker symbol or list of ticker strings.
            start_date (str, optional): Backtest start in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end in 'YYYY-MM-DD' format.
            initial_position (int): Starting position (default: 0).
            latest_only (bool): If True, returns only the final row (per ticker in multi-ticker mode) for EOD forecasting.

        Returns:
            pd.DataFrame: A DataFrame containing price data, indicator values, raw signals, risk-managed positions, 
            realized returns, cumulative returns, and exit types.
        """
        wr_period = self.params.get("wr_period", 14)
        oversold = self.params.get("oversold_threshold", -80)
        overbought = self.params.get("overbought_threshold", -20)
        data_lookback = self.params.get("data_lookback", 252)

        if oversold >= overbought:
            raise ValueError("oversold_threshold must be less than overbought_threshold")

        # Instantiate RiskManager with strategy-specific risk and cost parameters.
        risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )
        
        if isinstance(ticker, list):
            # Retrieve historical data for multiple tickers.
            if latest_only:
                df_all = self.get_historical_prices(ticker, lookback=wr_period + 1)
            else:
                df_all = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date, lookback=data_lookback)
            if df_all.empty:
                error_msg = f"No data retrieved for tickers: {ticker}"
                self.logger.error(error_msg)
                raise DataRetrievalError(error_msg)
            
            # Reset index to ensure 'ticker' is available as a column.
            df_all = df_all.reset_index(level=0) if isinstance(df_all.index, pd.MultiIndex) else df_all
            
            def process_group(group: pd.DataFrame) -> pd.DataFrame:
                """
                Process each ticker group: compute WR, generate signals and signal strength.
                """
                group = group.sort_index()
                # Calculate rolling highest high and lowest low.
                high_roll = group['high'].rolling(wr_period, min_periods=wr_period).max()
                low_roll = group['low'].rolling(wr_period, min_periods=wr_period).min()
                # Compute Williams %R.
                group['wr'] = -100 * ((high_roll - group['close']) / (high_roll - low_roll).replace(0, 1e-9))
                group = group.dropna(subset=['wr'])
                # Compute previous WR (vectorized) for crossover detection.
                wr_vals = group['wr'].values
                prev_wr = np.roll(wr_vals, 1)
                prev_wr[0] = np.nan  # No previous value for the first observation.
                buy_signals = (prev_wr > oversold) & (wr_vals <= oversold)
                sell_signals = (prev_wr < overbought) & (wr_vals >= overbought)
                group['signal'] = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
                # In long-only mode, force sell signals to 0.
                if self.long_only:
                    group['signal'] = group['signal'].clip(lower=0)
                # Compute signal strength.
                group['signal_strength'] = np.abs(group['wr'] - np.where(group['signal'] == 1, oversold, overbought))
                return group
            
            # Process each ticker individually in a vectorized manner.
            df_processed = df_all.groupby('ticker', group_keys=False).apply(process_group)
            df_processed.sort_index(inplace=True)
            
            # Apply risk management groupwise.
            df_risk = df_processed.groupby('ticker', group_keys=False).apply(lambda grp: risk_manager.apply(grp, initial_position))
            
            if latest_only:
                # Return only the latest row per ticker for end-of-day forecasting.
                df_risk = df_risk.groupby('ticker', group_keys=False).tail(1)
            return df_risk
        
        else:
            # Single ticker processing.
            if latest_only:
                df = self.get_historical_prices(ticker, lookback=wr_period + 1)
            else:
                df = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date, lookback=data_lookback)
            if not self._validate_data(df, min_records=wr_period):
                error_msg = f"Insufficient data for {ticker} (min {wr_period} records needed)"
                self.logger.error(error_msg)
                raise DataRetrievalError(error_msg)
            df = df.sort_index()
            # Calculate rolling highest high and lowest low.
            high_roll = df['high'].rolling(wr_period, min_periods=wr_period).max()
            low_roll = df['low'].rolling(wr_period, min_periods=wr_period).min()
            # Compute Williams %R.
            df['wr'] = -100 * ((high_roll - df['close']) / (high_roll - low_roll).replace(0, 1e-9))
            df = df.dropna(subset=['wr']).copy()
            # Compute previous WR.
            wr_vals = df['wr'].values
            prev_wr = np.roll(wr_vals, 1)
            prev_wr[0] = np.nan
            buy_signals = (prev_wr > oversold) & (wr_vals <= oversold)
            sell_signals = (prev_wr < overbought) & (wr_vals >= overbought)
            df['signal'] = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
            if self.long_only:
                df['signal'] = df['signal'].clip(lower=0)
            df['signal_strength'] = np.abs(df['wr'] - np.where(df['signal'] == 1, oversold, overbought))
            
            # Apply risk management rules.
            result_df = risk_manager.apply(df, initial_position)
            return result_df.iloc[-1:] if latest_only else result_df