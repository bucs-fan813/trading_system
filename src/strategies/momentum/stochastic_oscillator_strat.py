# trading_system/src/strategies/stochastic_oscillator_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig

class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator strategy with vectorized operations and integrated risk management.
    
    The strategy computes the %K and %D indicators as:
       %K = 100 * (close - min(low, over k_period)) / (max(high, over k_period) - min(low, over k_period))
       %D = moving average of %K over d_period

    Signal Generation:
       - A buy signal is generated when %K crosses above %D when both are below the oversold threshold.
       - A sell signal is generated when %K crosses below %D when both are above the overbought threshold.
       - Signal strength is computed as the normalized distance from the threshold.
    
    Risk management is applied via a RiskManager instance which incorporates stop-loss, take-profit, slippage,
    and transaction cost adjustments. The full backtested DataFrame includes trade returns and cumulative return
    metrics for downstream analysis. The latest available signal can be easily extracted for end-of-day decisions.
    
    Hyperparameters:
        k_period (int): Lookback period for %K calculation (default: 14)
        d_period (int): Smoothing period for %D calculation (default: 3)
        overbought (int): Overbought threshold (default: 80)
        oversold (int): Oversold threshold (default: 20)
        stop_loss_pct (float): Stop loss percentage (default: 0.05)
        take_profit_pct (float): Take profit percentage (default: 0.10)
        slippage_pct (float): Slippage percentage (default: 0.001)
        transaction_cost_pct (float): Transaction cost percentage (default: 0.001)
        long_only (bool): If True, only long positions are allowed (default: True)
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the stochastic strategy with database configuration and strategy parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Hyperparameters for the strategy.
        """
        super().__init__(db_config, params)
        self.params.setdefault('long_only', True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_and_set_params()

    def _validate_and_set_params(self):
        """
        Validate and set hyperparameters for the stochastic oscillator.
        
        Raises:
            ValueError: If k_period or d_period is non-positive or if overbought <= oversold.
        """
        self.k_period = self.params.get('k_period', 14)
        self.d_period = self.params.get('d_period', 3)
        self.overbought = self.params.get('overbought', 80)
        self.oversold = self.params.get('oversold', 20)

        if self.k_period <= 0 or self.d_period <= 0:
            raise ValueError("k_period and d_period must be positive integers")
        if self.overbought <= self.oversold:
            raise ValueError("overbought must be greater than oversold")

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate Stochastic Oscillator signals and apply integrated risk management.
        
        This function retrieves historical prices from the database, computes the %K and %D indicators,
        generates vectorized buy/sell signals based on crossover conditions in overbought/oversold regions,
        and then applies risk management (including stop-loss and take-profit) via the RiskManager.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting (inclusive, 'YYYY-MM-DD'). 
            end_date (str, optional): End date for backtesting (inclusive, 'YYYY-MM-DD').
            initial_position (int): The initial trading position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, returns only the latest available signal row.
        
        Returns:
            pd.DataFrame: DataFrame containing columns such as 'close', 'high', 'low', 'signal', 
                          'strength', 'return', 'cumulative_return', and 'exit_type' to support full
                          backtest analysis and downstream metrics.
        """
        try:
            # Ensure minimum number of data points for indicator calculations.
            min_data_points = self.k_period + self.d_period - 1
            # Retrieve historical price data.
            prices = self.get_historical_prices(
                ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_data_points if latest_only else None
            )

            if not self._validate_data(prices, min_records=min_data_points):
                return pd.DataFrame()

            # Calculate the stochastic indicators %K and %D.
            df = self._calculate_indicators(prices)

            # Generate trading signals (buy: 1, sell: -1, otherwise 0) and corresponding signal strengths.
            signals, strengths = self._generate_vectorized_signals(df, initial_position)

            # Apply the risk management rules (stop loss, take profit, slippage, transaction cost).
            return self._apply_risk_management(df, signals, strengths, initial_position, latest_only)

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _calculate_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the %K and %D stochastic oscillator indicators.
        
        The %K line is defined as:
            %K = 100 * (close - rolling_min(low)) / (rolling_max(high) - rolling_min(low))
        and %D is the moving average of %K over the d_period.
        
        Args:
            prices (pd.DataFrame): DataFrame with columns 'close', 'high', and 'low'.
        
        Returns:
            pd.DataFrame: DataFrame containing 'close', 'high', 'low', '%k', and '%d' columns with NaN
                          values removed.
        """
        low_min = prices['low'].rolling(self.k_period, min_periods=self.k_period).min()
        high_max = prices['high'].rolling(self.k_period, min_periods=self.k_period).max()

        # Prevent division by zero by replacing 0 with a very small number.
        denominator = (high_max - low_min).replace(0, 1e-9)
        k = 100 * (prices['close'] - low_min) / denominator
        d = k.rolling(self.d_period, min_periods=self.d_period).mean()

        df = pd.DataFrame({
            'close': prices['close'],
            'high': prices['high'],
            'low': prices['low'],
            '%k': k,
            '%d': d
        }).dropna()
        return df

    def _generate_vectorized_signals(
        self, df: pd.DataFrame, initial_position: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals based on %K and %D crossovers and compute signal strengths.
        
        A buy signal (1) is generated when a crossover (current %K > %D and previous %K <= %D)
        occurs in oversold territory (both %K and %D are below the 'oversold' threshold).  
        Similarly, a sell signal (-1) is generated for a crossover in overbought territory.
        Signal strength is computed as a normalized difference from the oversold or overbought levels.
        
        Args:
            df (pd.DataFrame): DataFrame containing '%k' and '%d' columns.
            initial_position (int): The initial position from which trading is started.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - signals: An integer numpy array where 1 indicates a buy, -1 a sell, and 0 no change.
                - strengths: A float numpy array with the normalized signal strength.
        """
        k = df['%k'].values
        d = df['%d'].values
        prev_k = np.roll(k, 1)
        prev_d = np.roll(d, 1)
        prev_k[0] = np.nan  # First element has no previous data

        with np.errstate(invalid='ignore'):
            buy_cond = (
                (k > d) &
                (prev_k <= prev_d) &
                (k < self.oversold) &
                (d < self.oversold)
            )
            sell_cond = (
                (k < d) &
                (prev_k >= prev_d) &
                (k > self.overbought) &
                (d > self.overbought)
            )

        # Stateful signal generation: update in a loop to ensure only a signal change triggers a trade.
        signals = np.zeros(len(df), dtype=int)
        current_signal = initial_position
        for i in range(len(df)):
            if i == 0:
                signals[i] = initial_position
            elif buy_cond[i] and current_signal != 1:
                signals[i] = 1
                current_signal = 1
            elif sell_cond[i] and current_signal != -1:
                signals[i] = -1
                current_signal = -1
            else:
                signals[i] = 0

        # Compute signal strength:
        strengths = np.zeros(len(df), dtype=float)
        buy_mask = signals == 1
        sell_mask = signals == -1
        
        if buy_mask.any():
            # For buy signals, strength is the normalized distance from the oversold threshold.
            min_level = np.minimum(k[buy_mask], d[buy_mask])
            strengths[buy_mask] = np.clip((self.oversold - min_level) / self.oversold, 0, 1)
        if sell_mask.any():
            # For sell signals, strength is the normalized distance above the overbought threshold.
            max_level = np.maximum(k[sell_mask], d[sell_mask])
            strengths[sell_mask] = np.clip((max_level - self.overbought) / (100 - self.overbought), 0, 1)

        if self.params.get('long_only', True):
            signals[signals == -1] = 0  # Override sell signals with 0 (exit) for long-only strategy

        return signals, strengths

    def _apply_risk_management(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        strengths: np.ndarray,
        initial_position: int,
        latest_only: bool
    ) -> pd.DataFrame:
        """
        Apply risk management to the generated signals and return the final backtest DataFrame.
        
        The risk management (via RiskManager) adjusts entry prices for slippage/transaction cost,
        computes stop-loss and take-profit thresholds, and determines the trade exit events.
        After applying risk management, the function returns either the full processed DataFrame (for
        full backtesting and downstream analytics) or only the last row (if 'latest_only' is True).
        
        Args:
            df (pd.DataFrame): DataFrame with trading indicator columns.
            signals (np.ndarray): Array of trading signals (1 for buy, -1 for sell, 0 for no change).
            strengths (np.ndarray): Array of signal strengths.
            initial_position (int): The initial trading position.
            latest_only (bool): If True, only the last row of the risk-managed DataFrame is returned.
        
        Returns:
            pd.DataFrame: A DataFrame including 'close', 'high', 'low', 'signal', 'strength',
                          'return', 'cumulative_return', and 'exit_type' columns.
        """
        df = df.copy()
        df['signal'] = signals
        df['strength'] = strengths

        risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )

        result = risk_manager.apply(df, initial_position=initial_position)
        return result.iloc[[-1]] if latest_only else result

    def _validate_data(self, df: pd.DataFrame, min_records: int) -> bool:
        """
        Validate that the DataFrame has the minimum required number of records.
        
        Args:
            df (pd.DataFrame): The historical price data.
            min_records (int): Minimum required records for indicator computation.
            
        Returns:
            bool: True if data is sufficient; otherwise, False.
        """
        if len(df) < min_records:
            self.logger.warning(f"Insufficient data: {len(df)} < {min_records}")
            return False
        return True