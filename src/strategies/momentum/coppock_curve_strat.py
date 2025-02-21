# trading_system/src/strategies/momentum/coppock_curve_strat.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager

class CoppockCurveStrategy(BaseStrategy):
    """
    CoppockCurveStrategy implements the Coppock Curve trading strategy.
    
    Mathematical Explanation:
    The Coppock Curve is calculated as a Weighted Moving Average (WMA) of the sum of two rate-of-change (ROC)
    measurements:
    
    1. Calculate ROC values:
       - ROC1 = 100 * ((Close / Close shifted by n1 days) - 1)
       - ROC2 = 100 * ((Close / Close shifted by n2 days) - 1)
       where n1 and n2 are derived from the number of months (converted to trading days, e.g. 14 and 11 months).
       
    2. Combine the two ROC values:
       CombinedROC = ROC1 + ROC2
       
    3. Smooth the CombinedROC using a weighted moving average (WMA):
       Coppock Curve = WMA(CombinedROC, window=self.wma_lookback) 
       where weights increase linearly (1, 2, ..., wma_lookback).
       
    Signal Generation:
    - A long signal (+1) is generated when the Coppock Curve crosses from below 0 to above 0 after a sustained negative regime.
    - A short signal (-1) is generated when the Coppock Curve crosses from above 0 to below 0 after a sustained positive regime.
    - A signal of 0 indicates no actionable change.
    
    Momentum Strength:
    The strategy also computes a momentum strength metric by comparing the Coppock Curve with its short‐term average,
    and optionally normalizes the strength to a range [-1, 1]. This additional indicator can be used downstream for
    position sizing or further optimization.
    
    Risk Management:
    The strategy integrates risk management using the RiskManager class to apply stop-loss, take-profit, slippage, and 
    transaction cost adjustments. This ensures that entry and exit signals are adjusted according to risk parameters,
    and the returns are computed accordingly for accurate backtesting.
    
    Backtesting and Signal Generation:
    This implementation supports vectorized backtesting over a specified date range, and also provides quick latest
    signal generation for end-of-day decision making.
    
    Attributes:
        roc1_months (int): Lookback period in months for the first ROC calculation.
        roc2_months (int): Lookback period in months for the second ROC calculation.
        roc1_days (int): Converted trading days for first ROC (roc1_months * 21).
        roc2_days (int): Converted trading days for second ROC (roc2_months * 21).
        wma_lookback (int): Lookback period for weighted moving average smoothing.
        stop_loss_pct (float): Stop loss percentage.
        take_profit_pct (float): Take profit percentage.
        slippage_pct (float): Slippage as a fraction of price.
        transaction_cost_pct (float): Transaction cost as a fraction of price.
        strength_window (int): Lookback window for normalizing momentum strength.
        normalize_strength (bool): Flag to normalize momentum strength.
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Coppock Curve Strategy with database configuration and parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Dictionary of strategy parameters including:
                - 'roc1_months': Months for first ROC (default: 14)
                - 'roc2_months': Months for second ROC (default: 11)
                - 'wma_lookback': Lookback window for weighting (default: 10)
                - 'stop_loss_pct': Stop loss percentage (default: 0.05)
                - 'take_profit_pct': Take profit percentage (default: 0.10)
                - 'slippage_pct': Slippage percentage (default: 0.001)
                - 'transaction_cost_pct': Transaction cost percentage (default: 0.001)
                - 'strength_window': Lookback window for momentum strength normalization (default: 504)
                - 'normalize_strength': Boolean flag to normalize strength (default: True)
        """
        super().__init__(db_config, params)
        self._validate_params()
        self._init_risk_params()
        self._init_strength_params()

    def _validate_params(self):
        """
        Validate strategy parameters and convert month-based inputs to trading days.

        Ensures that the ROC windows (in days) are larger than the WMA window.
        Logs a warning if the WMA window exceeds or equals the shorter ROC period.
        """
        self.roc1_months = max(1, int(self.params.get('roc1_months', 14)))
        self.roc2_months = max(1, int(self.params.get('roc2_months', 11)))
        self.roc1_days = self.roc1_months * 21  # Approximate trading days per month.
        self.roc2_days = self.roc2_months * 21
        self.wma_lookback = max(5, int(self.params.get('wma_lookback', 10)))

        if min(self.roc1_days, self.roc2_days) <= self.wma_lookback:
            self.logger.warning("ROC periods should exceed WMA window for meaningful signals")

    def _init_risk_params(self):
        """
        Initialize risk management parameters from strategy parameters.
        """
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.params.get('take_profit_pct', 0.10)
        self.slippage_pct = self.params.get('slippage_pct', 0.001)
        self.transaction_cost_pct = self.params.get('transaction_cost_pct', 0.001)

    def _init_strength_params(self):
        """
        Initialize momentum strength parameters from strategy parameters.
        """
        self.strength_window = int(self.params.get('strength_window', 504))
        self.normalize_strength = self.params.get('normalize_strength', True)

    def _get_validated_prices(self, ticker: str, start_date: str = None, 
                              end_date: str = None, min_periods: int = 252) -> pd.DataFrame:
        """
        Retrieve and validate historical OHLC price data for a given ticker.

        Retrieves data from the database using the base class method, applies forward fill and drops missing values.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for the data in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for the data in 'YYYY-MM-DD' format.
            min_periods (int): Minimum number of records required for validity.

        Returns:
            pd.DataFrame: Validated DataFrame with columns: ['close', 'high', 'low'].
            If insufficient data is found, returns an empty DataFrame.
        """
        try:
            prices = self.get_historical_prices(
                ticker=ticker,
                from_date=start_date,
                to_date=end_date,
                lookback=min_periods if not (start_date or end_date) else None
            )
            if not self._validate_data(prices, min_periods):
                return pd.DataFrame()

            return prices[['close', 'high', 'low']].ffill().dropna()
        except Exception as e:
            self.logger.error(f"Price retrieval failed: {e}")
            return pd.DataFrame()

    def _calculate_coppock_curve(self, close: pd.Series) -> pd.Series:
        """
        Calculate the Coppock Curve using vectorized operations.

        The Coppock Curve is calculated by computing two rates of change (ROC) over different periods,
        summing them, and then smoothing the result using a weighted moving average (WMA).

        Mathematical details:
            ROC1 = 100 * ((Close / Close_shifted_by_roc1_days) - 1)
            ROC2 = 100 * ((Close / Close_shifted_by_roc2_days) - 1)
            CombinedROC = ROC1 + ROC2
            Coppock Curve = WMA(CombinedROC, window=self.wma_lookback) where weights are linearly increasing.

        Args:
            close (pd.Series): Series of closing prices.

        Returns:
            pd.Series: Series representing the Coppock Curve.
        """
        roc1 = close.pct_change(self.roc1_days, fill_method=None) * 100
        roc2 = close.pct_change(self.roc2_days, fill_method=None) * 100
        combined_roc = (roc1 + roc2).dropna()

        # Precompute weights for the weighted moving average.
        weights = np.arange(1, self.wma_lookback + 1)
        wma = combined_roc.rolling(
            window=self.wma_lookback,
            min_periods=int(self.wma_lookback * 0.8)
        ).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
            raw=True
        )
        return wma

    def _generate_raw_signals(self, prices: pd.DataFrame, cc: pd.Series) -> pd.DataFrame:
        """
        Generate raw trading signals based on the Coppock Curve and momentum regime detection.

        Determines regime shifts by checking consecutive days of negative or positive Coppock Curve values 
        from the previous day, and generates a long signal when a negative regime turns positive, and a 
        short signal when a positive regime turns negative.

        Additionally calculates a momentum strength metric based on the divergence of the current Coppock 
        Curve value from its short-term moving average, with optional normalization.

        Args:
            prices (pd.DataFrame): DataFrame containing 'close', 'high', and 'low' prices.
            cc (pd.Series): Series representing the Coppock Curve.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - 'close': Closing price.
                - 'high': High price.
                - 'low': Low price.
                - 'signal': Trading signal (1 for long, -1 for short, 0 for no action).
                - 'strength': Calculated momentum strength indicator.
        """
        df = prices.assign(cc=cc).dropna()
        if df.empty:
            return df

        # Detect regime: check if the previous Coppock Curve (shifted_cc) was consistently negative or positive for 4 days.
        shifted_cc = df['cc'].shift(1)
        df['was_negative'] = (shifted_cc < 0).rolling(4, min_periods=4).sum() == 4
        df['was_positive'] = (shifted_cc > 0).rolling(4, min_periods=4).sum() == 4

        # Generate signals: long when curve shifts from negative to positive, short when it shifts from positive to negative.
        df['signal'] = np.select(
            [df['was_negative'] & (df['cc'] > 0),
             df['was_positive'] & (df['cc'] < 0)],
            [1, -1],
            default=0
        )
        
        # Calculate raw momentum strength based on divergence from a 21-day moving average.
        ref = df['cc'].rolling(21, min_periods=1).mean().shift()
        df['strength_raw'] = (df['cc'] - ref) / ref.abs().replace(0, 1)
        
        # Normalize momentum strength to range [-1, 1] if required.
        if self.normalize_strength:
            rolling_min = df['strength_raw'].rolling(self.strength_window, min_periods=1).min()
            rolling_max = df['strength_raw'].rolling(self.strength_window, min_periods=1).max()
            df['strength'] = 2 * ((df['strength_raw'] - rolling_min) / (rolling_max - rolling_min).replace(0, 1)) - 1
        else:
            df['strength'] = df['strength_raw']
            
        return df[['close', 'high', 'low', 'signal', 'strength']]

    def _apply_risk_management(self, signals: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Apply risk management rules to trading signals using the RiskManager.

        Adjusts for stop-loss, take-profit, slippage, and transaction costs to calculate entry and exit prices,
        realized returns, and cumulative performance. Integrates with a vectorized risk manager strategy.

        Args:
            signals (pd.DataFrame): DataFrame containing trading signals and OHLC data with columns:
                - 'signal'
                - 'close'
                - 'high'
                - 'low'
            initial_position (int): The starting trading position (0 for flat, 1 for long, -1 for short).

        Returns:
            pd.DataFrame: Updated DataFrame containing risk-managed columns:
                - 'close', 'high', 'low', 'signal', 'position', 'return', 
                  'cumulative_return', 'exit_type'
            Also includes the 'strength' column after joining.
        """
        risk_manager = RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            slippage_pct=self.slippage_pct,
            transaction_cost_pct=self.transaction_cost_pct
        )
        try:
            managed = risk_manager.apply(
                signals[['signal', 'close', 'high', 'low']].copy(),
                initial_position=initial_position
            )
            # Join the calculated momentum strength from the raw signals.
            return managed.join(signals[['strength']], how='left')
        except Exception as e:
            self.logger.error(f"Risk management failed: {e}")
            return pd.DataFrame()

    def _process_latest_signal(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve the latest available signal for end-of-day decision making.

        This method extracts the most recent row from the signals DataFrame, ensuring minimal data access and 
        stable signal calculation.

        Args:
            signals (pd.DataFrame): DataFrame containing trading signals and associated metrics.

        Returns:
            pd.DataFrame: A DataFrame containing only the most recent signal.
        """
        return signals.iloc[[-1]].dropna()

    def generate_signals(self, ticker: str, start_date: str = None,
                         end_date: str = None, initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals applying Coppock Curve logic and risk management, 
        supporting both backtesting and latest signal evaluation.

        Retrieves historical prices, calculates the Coppock Curve indicator, derives raw signals based on 
        regime shifts and momentum strength, and then applies risk management adjustments including 
        stop-loss and take-profit rules with slippage and transaction cost.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for backtesting in 'YYYY-MM-DD' format.
            initial_position (int): The starting position (0: flat, 1: long, -1: short).
            latest_only (bool, optional): If True, returns only the latest signal for immediate trading decision.

        Returns:
            pd.DataFrame: DataFrame containing:
                - 'close', 'high', 'low': OHLC price data.
                - 'signal': Raw trading signal.
                - 'position': Risk-managed position.
                - 'return': Realized return on exit events.
                - 'cumulative_return': Cumulative trade return.
                - 'exit_type': Type of exit event (e.g., 'stop_loss', 'take_profit', 'signal_exit', 'none').
                - 'strength': Momentum strength indicator.
            In backtesting mode, the full time series is returned. In latest mode, only the most recent row is returned.
        """
        min_periods = max(self.roc1_days, self.roc2_days) + self.wma_lookback + 21
        
        try:
            prices = self._get_validated_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                min_periods=min_periods
            )
            if prices.empty:
                return pd.DataFrame()

            # Compute the Coppock Curve indicator.
            cc = self._calculate_coppock_curve(prices['close'])
            # Generate raw signals based on the Coppock Curve.
            signals = self._generate_raw_signals(prices, cc)
            
            # If latest_only flag is set, return only the most recent signal.
            if latest_only:
                return self._process_latest_signal(signals)
            
            # Apply risk management rules and return the enriched signals DataFrame.
            return self._apply_risk_management(signals, initial_position)
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}", exc_info=True)
            return pd.DataFrame()

    def __repr__(self):
        """
        Return a string representation of the strategy instance with key parameters.

        Returns:
            str: Representation of the strategy configuration.
        """
        return (f"CoppockCurveStrategy(ROC={self.roc1_months}/{self.roc2_months}mo, "
                f"WMA={self.wma_lookback}, SL={self.stop_loss_pct*100:.1f}%, "
                f"TP={self.take_profit_pct*100:.1f}%)")