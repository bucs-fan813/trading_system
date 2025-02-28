# trading_system/src/strategies/williams_r.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class WilliamsRStrategy(BaseStrategy):
    """
    Williams %R Trading Strategy with Embedded Risk Management.
    
    Mathematical formulation:
    
      WR_t = -100 * ((Highest High_{t-N+1:t} - Close_t) / (Highest High_{t-N+1:t} - Lowest Low_{t-N+1:t}))
      
    where N is the lookback period (default 14 days).  
    A long (buy) signal is generated when the Williams %R value crosses below the oversold threshold (default -80):
    
      WR_{t-1} > -80  and  WR_t <= -80
      
    A signal to exit (sell) is generated when the value crosses above the overbought threshold (default -20):
    
      WR_{t-1} < -20  and  WR_t >= -20
      
    After signal generation, a RiskManager is applied to enforce stop-loss and take-profit rules with adjustments 
    for slippage and transaction costs. The resulting DataFrame includes the price data, computed indicator, 
    signals, positions, realized returns, and cumulative returns – which facilitates backtesting (Sharpe, drawdown, etc.)
    and also end-of-day (latest) forecast.
    
    Attributes:
        db_config: Database configuration details.
        params (dict): Strategy-specific parameters including 'wr_period', 
            'oversold_threshold', 'overbought_threshold', 'stop_loss_pct', 'take_profit_pct', 
            'slippage_pct', 'transaction_cost_pct', and 'data_lookback'.
    """
    
    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the WilliamsRStrategy with a database configuration and strategy parameters.
        
        Args:
            db_config: A database configuration object.
            params (dict, optional): Strategy-specific parameters.
        """
        super().__init__(db_config, params)
        self.logger.info("WilliamsRStrategy initialized with parameters: %s", self.params)
        self.long_only = bool(self.params.get('long_only', True))

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Generate trading signals based on the Williams %R indicator, integrating risk management rules.
        
        This method retrieves historical price data, computes the Williams %R indicator over the designated period,
        and then creates buy signals when the indicator crosses below the oversold threshold and sell signals 
        when it crosses above the overbought threshold. A state machine then builds the position history (long or flat),
        and a signal strength is derived as the absolute deviation from the respective threshold.
        
        Finally, a RiskManager applies stop-loss, take-profit, slippage, and transaction cost adjustments,
        computing the realized returns and cumulative strategy performance.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting in YYYY-MM-DD format.
            end_date (str, optional): End date for backtesting in YYYY-MM-DD format.
            initial_position (int): The initial position (0 for no position; 1 for long).
            latest_only (bool): If True, returns only the most recent row (for EOD forecasting); otherwise, returns
                                the full time series for backtesting.
        
        Returns:
            pd.DataFrame: A DataFrame with columns including 'close', 'high', 'low', 'signal', 'position', 
                          'return', 'cumulative_return', and 'exit_type'. In `latest_only` mode, a single-row 
                          DataFrame is returned.
        """
        self.logger.info("Generating Williams %%R signals for %s (latest_only=%s)", ticker, latest_only)

        # Retrieve and validate strategy parameters.
        wr_period = self.params.get("wr_period", 14)
        oversold = self.params.get("oversold_threshold", -80)
        overbought = self.params.get("overbought_threshold", -20)
        data_lookback = self.params.get("data_lookback", 252)

        if oversold >= overbought:
            raise ValueError("oversold_threshold must be less than overbought_threshold")

        # Retrieve historical market data.
        if latest_only:
            df = self.get_historical_prices(ticker, lookback=wr_period + 1)
        else:
            df = self.get_historical_prices(
                ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=data_lookback
            )

        if not self._validate_data(df, min_records=wr_period):
            error_msg = f"Insufficient data for {ticker} (min {wr_period} records needed)"
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg)

        # Calculate Williams %R indicator using vectorized rolling windows.
        high_roll = df['high'].rolling(wr_period, min_periods=wr_period).max()
        low_roll = df['low'].rolling(wr_period, min_periods=wr_period).min()
        df['wr'] = -100 * ((high_roll - df['close']) / (high_roll - low_roll).replace(0, 1e-9))
        df = df.dropna(subset=['wr']).copy()

        # Generate buy and sell signals based on threshold crossings.
        # Buy when the value crosses below the oversold threshold.
        # Sell when the value crosses above the overbought threshold.
        wr_vals = df['wr'].values
        prev_wr = np.roll(wr_vals, 1)
        prev_wr[0] = np.nan  # First element has no previous value.

        buy_signals = (prev_wr > oversold) & (wr_vals <= oversold)
        sell_signals = (prev_wr < overbought) & (wr_vals >= overbought)

        # Build the position series using a sequential (state machine) approach.
        positions = np.zeros(len(df), dtype=int)
        current_pos = initial_position
        for i in range(len(df)):
            if buy_signals[i] and current_pos != 1:
                current_pos = 1
            elif sell_signals[i] and current_pos == 1:
                current_pos = 0
            positions[i] = current_pos

        # Map discrete signals: 1 for buy, -1 for sell, and 0 for no action.
        df['signal'] = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
        df['position'] = positions
        # Compute signal strength as the absolute distance from the threshold relevant to the signal.
        df['signal_strength'] = np.abs(df['wr'] - np.where(df['signal'] == 1, oversold, overbought))

        if self.long_only:
            df['signal'] = df['signal'].clip(lower=0) # Clip negative signals for long-only strategy.

        # Apply risk management rules (stop-loss, take-profit, slippage, transaction cost)
        risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )
        result_df = risk_manager.apply(df, initial_position)

        return result_df.iloc[-1:] if latest_only else result_df