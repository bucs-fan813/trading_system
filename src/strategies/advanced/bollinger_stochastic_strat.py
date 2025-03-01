# trading_system/src/strategies/bollinger_stochastic_strat.py

import pandas as pd
import numpy as np
from typing import Optional, Dict
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig 

class BollingerStochasticStrategy(BaseStrategy):
    """
    Bollinger Bands & Stochastic Oscillator Strategy with integrated risk management.

    This strategy calculates Bollinger Bands and the Stochastic Oscillator to generate 
    long and short signals. The signal strength is computed as a normalized metric on a 
    0–100 scale combining both the Bollinger Bands and Stochastics information. Generated 
    signals are then passed through a risk management module that applies stop loss, 
    take profit, slippage, and transaction costs to compute trade returns.
    
    The strategy supports both full backtesting over a specified date range and live 
    prediction using a short lookback period.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Bollinger-Stochastic strategy.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): A dictionary of strategy-specific parameters.
        """
        super().__init__(db_config, params)
        self._set_default_params()
        self.risk_manager = self._create_risk_manager()

    def _set_default_params(self):
        """
        Initialize strategy parameters with default values if not provided.
        
        Defaults include:
            bb_window: Rolling window for Bollinger Bands.
            stoch_k: Lookback period for the Stochastic Oscillator.
            stoch_d: Smoothing period for %K to obtain %D.
            stop_loss: Stop loss percentage.
            take_profit: Take profit percentage.
            slippage: Slippage percentage.
            commission: Transaction cost percentage.
            min_history: Minimum number of data points required for indicator calculation.
            long_only: If True, only long positions are allowed.
        """
        defaults = {
            'bb_window': 20,
            'stoch_k': 14,
            'stoch_d': 3,
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'slippage': 0.001,
            'commission': 0.001,
            'min_history': 50,
            'long_only': True
        }
        for k, v in defaults.items():
            self.params.setdefault(k, v)

    def _create_risk_manager(self) -> RiskManager:
        """
        Instantiate the RiskManager with the strategy's risk parameters.
        
        Returns:
            RiskManager: An instance configured with stop loss, take profit, slippage, 
                         and commission requirements.
        """
        return RiskManager(
            stop_loss_pct=self.params['stop_loss'],
            take_profit_pct=self.params['take_profit'],
            slippage_pct=self.params['slippage'],
            transaction_cost_pct=self.params['commission']
        )

    def generate_signals(self, 
                         ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals for a given ticker and apply risk management.
        
        Depending on the parameters, this method can either perform a full backtest 
        over a date range or generate a single latest prediction.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for backtesting in 'YYYY-MM-DD' format.
            initial_position (int): The initial position (0: none, 1: long, -1: short).
            latest_only (bool): If True, generate a signal only for the latest date.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, indicators, raw signals, 
                          risk-managed positions, returns, cumulative returns, and 
                          exit event types.
        """
        # Retrieve historical price data.
        df = self._get_price_data(ticker, start_date, end_date, latest_only)
        if df.empty or not self._validate_data(df, self.params['min_history']):
            return pd.DataFrame()

        # Calculate Bollinger Bands and Stochastic Oscillator indicators.
        df = self._calculate_indicators(df)
        
        # Generate raw entry signals and compute signal strength.
        df = self._generate_raw_signals(df)
        
        # Apply risk management adjustments (stop loss, take profit, etc.).
        return self._apply_risk_management(df, initial_position, latest_only)

    def _get_price_data(self, 
                        ticker: str,
                        start_date: Optional[str],
                        end_date: Optional[str],
                        latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data with optimized lookback handling.
        
        For live predictions (latest_only=True), a minimal lookback is used so that the 
        signal is computed on the most recent stable data.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for data retrieval.
            end_date (str, optional): End date for data retrieval.
            latest_only (bool): If True, only a fixed lookback dataset is fetched.
        
        Returns:
            pd.DataFrame: DataFrame of historical prices with 'date' as index.
        """
        if latest_only:
            lookback = self.params['bb_window'] + self.params['stoch_k'] + 10
            df = self.get_historical_prices(ticker, lookback=lookback)
            return df.iloc[-lookback:] if not df.empty else df
        
        return self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and Stochastic Oscillator indicators using vectorized operations.
        
        Computes:
            - Middle, Upper, and Lower Bollinger Bands.
            - %K and %D of the Stochastic Oscillator.
        
        Args:
            df (pd.DataFrame): DataFrame containing at least 'close', 'high', and 'low' prices.
        
        Returns:
            pd.DataFrame: DataFrame with new indicator columns added, with incomplete rows dropped.
        """
        # Bollinger Bands calculation.
        window = self.params['bb_window']
        df['middle_bb'] = df['close'].rolling(window).mean()
        df['upper_bb'] = df['middle_bb'] + 2 * df['close'].rolling(window).std()
        df['lower_bb'] = df['middle_bb'] - 2 * df['close'].rolling(window).std()

        # Stochastic Oscillator calculation.
        k_window = self.params['stoch_k']
        d_window = self.params['stoch_d']
        df['lowest_low'] = df['low'].rolling(k_window).min()
        df['highest_high'] = df['high'].rolling(k_window).max()
        df['%k'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'] + 1e-9)
        df['%d'] = df['%k'].rolling(d_window).mean()
        
        return df.dropna()

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw entry signals based on indicator crossovers and price extremes.
        
        A long signal (1) is generated when:
            - Prior day's %K and %D > 30 and current day's %K and %D < 30.
            - Close price is less than or equal to the lower Bollinger Band.
        
        A short signal (-1) is generated when:
            - Prior day's %K and %D < 70 and current day's %K and %D > 70.
            - Close price is greater than or equal to the upper Bollinger Band.
        
        The method also computes a normalized signal strength.

        Args:
            df (pd.DataFrame): DataFrame containing price data and calculated indicators.
        
        Returns:
            pd.DataFrame: DataFrame with 'signal' and 'strength' columns added.
        """
        # Define the conditions for long and short entries.
        k, d = df['%k'], df['%d']
        prev_k, prev_d = k.shift(1), d.shift(1)
        
        long_cond = (
            (prev_k > 30) & (prev_d > 30) &
            (k < 30) & (d < 30) &
            (df['close'] <= df['lower_bb'])
        )
        
        short_cond = (
            (prev_k < 70) & (prev_d < 70) &
            (k > 70) & (d > 70) &
            (df['close'] >= df['upper_bb'])
        )

        # Create the signal column: 1 for long, -1 for short, 0 for neutral.
        df['signal'] = np.select([long_cond, short_cond], [1, -1], 0)

        if self.long_only:
            # Replace short signals with 0 (exit).
            df.loc[short_cond, 'signal'] = 0
        
        # Compute signal strength on a 0–100 scale.
        df['strength'] = 0.0
        df.loc[long_cond, 'strength'] = self._calculate_strength(df, long_cond, 'long')
        df.loc[short_cond, 'strength'] = self._calculate_strength(df, short_cond, 'short')
        
        return df

    def _calculate_strength(self, 
                            df: pd.DataFrame,
                            mask: pd.Series,
                            side: str) -> pd.Series:
        """
        Calculate normalized signal strength on a 0–100 scale for the given signal.
        
        For long signals:
            strength = [((lower_bb - close) / (upper_bb - lower_bb)) + ((30 - %k) / 30)] / 2 * 100
        For short signals:
            strength = [((close - upper_bb) / (upper_bb - lower_bb)) + ((%k - 70) / 30)] / 2 * 100
        
        Args:
            df (pd.DataFrame): DataFrame containing price and indicator data.
            mask (pd.Series): Boolean series indicating where the signal condition is met.
            side (str): Either 'long' or 'short', indicating the trade direction.
        
        Returns:
            pd.Series: A series of normalized signal strengths.
        """
        subset = df[mask]
        
        if side == 'long':
            bb_strength = (subset['lower_bb'] - subset['close']) / (subset['upper_bb'] - subset['lower_bb'] + 1e-9)
            stoch_strength = (30 - subset['%k']) / 30
        else:
            bb_strength = (subset['close'] - subset['upper_bb']) / (subset['upper_bb'] - subset['lower_bb'] + 1e-9)
            stoch_strength = (subset['%k'] - 70) / 30
            
        # Average the clipped components and scale to 0-100.
        combined = (bb_strength.clip(0, 1) + stoch_strength.clip(0, 1)) / 2 * 100
        return combined.abs()

    def _apply_risk_management(self,
                               df: pd.DataFrame,
                               initial_position: int,
                               latest_only: bool) -> pd.DataFrame:
        """
        Apply risk management rules to the raw signals and merge with indicator data.
        
        Uses the RiskManager to adjust for stop loss, take profit, slippage, 
        and transaction costs. After processing, the results include the updated 
        position, realized return, cumulative return, and exit event type.
        
        Args:
            df (pd.DataFrame): DataFrame containing raw signals along with price data.
            initial_position (int): The starting position for the backtest (0, 1, or -1).
            latest_only (bool): If True, returns only the latest prediction row.
        
        Returns:
            pd.DataFrame: A DataFrame that merges the original indicators with risk 
                          management outputs.
        """
        required_cols = ['signal', 'high', 'low', 'close']
        signals = df[required_cols].copy()
        
        try:
            managed = self.risk_manager.apply(signals, initial_position)
        except KeyError as e:
            self.logger.error(f"Missing required columns for risk management: {e}")
            return pd.DataFrame()

        # Merge the original DataFrame with risk management results.
        result = df.join(managed[['position', 'return', 'cumulative_return', 'exit_type']])
        
        # For live prediction, return only the latest row.
        return result.tail(1) if latest_only else result