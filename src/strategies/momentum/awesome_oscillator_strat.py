# src/strategies/momentum/awesome_oscillator_strat.py

"""
Awesome Oscillator Strategy with Integrated Risk Management Component

This strategy implements the Awesome Oscillator indicator to generate trading signals,
simulate positions, and compute profit metrics. Afterwards, it delegates the risk management
responsibility (stop-loss, take-profit, slippage, and transaction cost adjustments) to the
external RiskManager component.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig

# Import the external Risk Management Component.
from src.strategies.risk_management import RiskManager

class AwesomeOscillatorStrategy(BaseStrategy):
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Awesome Oscillator Strategy.

        Parameters:
            db_config (DatabaseConfig): Database configuration settings.
            params (Optional[Dict]): Strategy parameters.
                Expected keys:
                  - 'short_period' (default: 5)
                  - 'long_period' (default: 34)
				  - Risk management related parameters:
						'stop_loss_pct' (default: 0.05)
						'take_profit_pct' (default: 0.10)
						'slippage_pct' (default: 0.001)
						'transaction_cost_pct' (default: 0.001)
		The Awesome Oscillator (AO) is computed as:
			AO = (short_sma - long_sma)
		where the SMAs are computed over the median price:
			median_price = (high + low) / 2

		Signal generation:
		  - A buy signal (signal = 1) is generated when AO crosses upward through zero.
		  - A sell signal (signal = -1) is generated when AO crosses downward through zero.									  
        """
        default_params = {'short_period': 5, 'long_period': 34}
        params = params or default_params
        super().__init__(db_config, params)
        self.short_period = int(params.get('short_period', default_params['short_period']))
        self.long_period = int(params.get('long_period', default_params['long_period']))
        if self.short_period >= self.long_period:
            raise ValueError("Short SMA period must be less than long SMA period.")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize the RiskManager. Any risk-related parameters are passed through.
        risk_params = {
            'stop_loss_pct': params.get('stop_loss_pct', 0.05),
            'take_profit_pct': params.get('take_profit_pct', 0.10),
            'slippage_pct': params.get('slippage_pct', 0.001),
            'transaction_cost_pct': params.get('transaction_cost_pct', 0.001)
        }
        self.risk_manager = RiskManager(**risk_params)

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Generate trading signals, simulate positions, compute profit metrics,
        and apply external risk management adjustments for the given ticker.
		
		This method supports two modes:
          - Backtesting: When start_date and end_date are provided, the full historical data is used 
            for performance evaluation (profit, cumulative returns, etc.).
          - Forecasting: When latest_only is True, a minimal data slice is used and only the latest
            signal is returned for end-of-day decision making.

        Returns a DataFrame with the following columns:
            - 'close'              : Trading price reference.
            - 'high', 'low'        : Prices used for risk management checks.			
            - 'ao'                 : Awesome Oscillator value.
            - 'signal'             : Trading signal (1 for buy, -1 for sell, 0 otherwise).
            - 'signal_strength'    : Normalized strength of the signal.
            - 'position'           : Simulated trading position.
            - 'daily_return'       : Daily percentage change in close price.
            - 'strategy_return'    : Return based on the previous day’s position.
            - 'cumulative_return'  : Cumulative return from strategy_return.
            - 'rm_strategy_return' : Risk-managed trade return.
            - 'rm_cumulative_return': Cumulative return after risk management.
            - 'rm_action'          : Risk-management action taken.
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting position (e.g., 0 for no position, 1 for long).
            latest_only (bool): If True, returns only the final row (for EOD decision making).

        Returns:
            pd.DataFrame: DataFrame with signals and performance metrics.											 
        """
        try:
            # Retrieve historical data.
            if start_date and end_date:
                data = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
            else:
                # In forecast mode or absent defined dates, use a minimal lookback buffer.
                lookback_buffer = 2 * max(self.short_period, self.long_period)
                data = self.get_historical_prices(ticker, lookback=lookback_buffer)
                data = data.sort_index()

            # Ensure there’s enough data.
            required_records = max(self.short_period, self.long_period)
            if not self._validate_data(data, min_records=required_records):
                self.logger.warning(f"Insufficient data for {ticker}: required at least {required_records} records.")
                return pd.DataFrame()

            # 1. Generate signals.
            signals = self._calculate_signals(data)
            # 2. Calculate profit metrics and delegate risk management to the external component.
            signals = self.risk_manager.apply(signals, initial_position)

            if latest_only:
                signals = signals.iloc[[-1]].copy()
            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Awesome Oscillator (AO) and generate trading signals using vectorized operations.

        AO is calculated as:
            ao = (short_sma - long_sma)
        where:
            short_sma = SMA of median_price with window=self.short_period,
            long_sma  = SMA of median_price with window=self.long_period,
            median_price = (high + low) / 2.

        Trading signals:
          - Buy signal (1): When AO crosses upward through zero.
          - Sell signal (-1): When AO crosses downward through zero.
        Signal strength is normalized by the rolling standard deviation of AO.

        Returns:
            pd.DataFrame: DataFrame with columns ['close', 'ao', 'signal', 'signal_strength'].																			  
        """
        data = data.sort_index()
        df = data.copy()
        df['median_price'] = (df['high'] + df['low']) / 2
        df['short_sma'] = df['median_price'].rolling(window=self.short_period, min_periods=self.short_period).mean()
        df['long_sma'] = df['median_price'].rolling(window=self.long_period, min_periods=self.long_period).mean()
        df['ao'] = df['short_sma'] - df['long_sma']
        df = df.dropna(subset=['ao'])
        df['ao_std'] = df['ao'].rolling(window=self.short_period, min_periods=self.short_period).std()
        df['normalized_strength'] = df['ao'].abs() / (df['ao_std'] + 1e-6)
        buy_mask = (df['ao'] > 0) & (df['ao'].shift(1) <= 0)
        sell_mask = (df['ao'] < 0) & (df['ao'].shift(1) >= 0)
        df['signal'] = 0
        df.loc[buy_mask, 'signal'] = 1
        df.loc[sell_mask, 'signal'] = -1
        df['signal_strength'] = 0.0
        df.loc[buy_mask, 'signal_strength'] = df.loc[buy_mask, 'normalized_strength']
        df.loc[sell_mask, 'signal_strength'] = df.loc[sell_mask, 'normalized_strength']
        signals = df[['open', 'close', 'high', 'low', 'ao', 'signal', 'signal_strength']].copy()
        return signals