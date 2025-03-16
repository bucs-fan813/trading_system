# trading_system/src/strategies/breakout/volume_breakout.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class VolumeBreakout(BaseStrategy):
    """
    Volume Breakout Strategy with Integrated Risk Management.
    
    This strategy identifies price breakouts that are confirmed by significant volume surges.
    It computes rolling resistance and support levels, and verifies the breakout using volume and
    price change thresholds. Optionally, it employs an ATR filter to account for volatility.
    
    Mathematical Formulation:
      - Resistanceₜ = max(highₜ₋ₙ₊₁, ..., highₜ)
      - Supportₜ = min(lowₜ₋ₙ₊₁, ..., lowₜ)
      - Average Volumeₜ = mean(volumeₜ₋ₘ₊₁, ..., volumeₜ)
      - Price Change = pct_change(closeₜ)
      
      **Buy Signal (signal = 1) Conditions:**  
      1. A number of consecutive bars (≥ consecutive_bars) exist where  
         closeₜ > Resistanceₜ₋₁.
      2. volumeₜ > volume_threshold × Average Volumeₜ.
      3. |Price Change| > price_threshold.
      4. (Optional) |Price Change| > atr_threshold × (ATR / closeₜ) if ATR filtering is enabled.
      
      **Sell Signal (signal = –1) Conditions:**  
      1. A number of consecutive bars (≥ consecutive_bars) exist where  
         closeₜ < Supportₜ₋₁.
      2. volumeₜ > volume_threshold × Average Volumeₜ.
      3. |Price Change| > price_threshold.
      4. (Optional) |Price Change| > atr_threshold × (ATR / closeₜ) if ATR filtering is enabled.
      
      The signal strength is defined as the normalized absolute price change:
          signal_strength = |Price Change| / (price_volatility + 1e-6),
      where price_volatility is (ATR / closeₜ) when ATR filtering is enabled, otherwise |Price Change|.
      
      Risk Management:
      After generating raw signals, this strategy integrates with the external RiskManager,
      which adjusts the entry prices for slippage and transaction costs and applies stop-loss and
      take-profit rules. For long trades:
          Stop Loss = entry_price × (1 – stop_loss_pct)
          Take Profit = entry_price × (1 + take_profit_pct)
      For short trades, these levels are inverted.
    
    Args:
        tickers (str or List[str]): Stock ticker symbol or list of tickers.
        start_date (str, optional): Start date for backtesting in 'YYYY-MM-DD' format.
        end_date (str, optional): End date for backtesting in 'YYYY-MM-DD' format.
        initial_position (int): Initial trading position (0 for none, 1 for long, -1 for short).
        latest_only (bool): If True, returns only the latest signal row per ticker.
    
    Returns:
        pd.DataFrame: A DataFrame containing price data and breakout indicators along with:
          - 'open', 'high', 'low', 'close', 'volume'
          - 'resistance', 'support', 'avg_volume', 'price_change', 'atr', 'price_volatility'
          - 'above_resistance_count', 'below_support_count'
          - 'signal' and 'signal_strength'
          - Risk-managed outputs: 'position', 'rm_strategy_return', 'rm_cumulative_return', 'rm_action'
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Volume Breakout strategy with parameters.
        
        Parameters in `params` (with defaults):
            - lookback_period (int): Period for resistance/support calculation (default: 20).
            - volume_threshold (float): Multiplier to confirm volume breakout (default: 1.5).
            - price_threshold (float): Minimum price move as a fraction (default: 0.02).
            - volume_avg_period (int): Moving average period for volume (default: 20).
            - consecutive_bars (int): Required consecutive bars to validate breakout (default: 1).
            - use_atr_filter (bool): Enable ATR-based volatility filtering (default: False).
            - atr_period (int): Lookback period for ATR calculation (default: 14).
            - atr_threshold (float): ATR multiplier threshold for signal filtering (default: 1.0).
            - stop_loss_pct (float): Stop loss percentage (default: 0.05).
            - take_profit_pct (float): Take profit percentage (default: 0.10).
            - trailing_stop_pct (float): Trailing stop percentage (default: 0.0).
            - slippage_pct (float): Slippage percentage (default: 0.001).
            - transaction_cost_pct (float): Transaction cost percentage (default: 0.001).
            - long_only (bool): If True, only long trades are allowed (default: True).
        """
        default_params = {
            'lookback_period': 20,
            'volume_threshold': 1.5,
            'price_threshold': 0.02,
            'volume_avg_period': 20,
            'consecutive_bars': 1,
            'use_atr_filter': False,
            'atr_period': 14,
            'atr_threshold': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trailing_stop_pct': 0.0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            'long_only': True
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
    
    def generate_signals(self,
                         tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals using the Volume Breakout strategy with risk management.
        
        Retrieves historical price data, computes resistance/support levels, average volume, 
        and ATR (if enabled). It then calculates consecutive breakout counts in a vectorized manner,
        generates buy/sell signals based on volume and price move thresholds, computes normalized 
        signal strength, and integrates risk management adjustments.
        
        Args:
            tickers (str or List[str]): Stock ticker symbol or list of tickers.
            start_date (str, optional): Backtesting start date (YYYY-MM-DD).
            end_date (str, optional): Backtesting end date (YYYY-MM-DD).
            initial_position (int): Starting trading position (0, 1, or -1).
            latest_only (bool): If True, returns only the most recent signal row per ticker.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, breakout indicators, raw signals,
                          normalized signal strength, and risk-managed trade metrics.
        """
        
        # Determine extra lookback buffer for indicator calculation.
        extra_buffer = 50
        lookback_length = max(
            self.params['lookback_period'], 
            self.params['volume_avg_period'],
            self.params['atr_period'] if self.params['use_atr_filter'] else 0
        ) + extra_buffer
        
        # Retrieve historical price data; supports both single and multiple tickers.
        price_data = self.get_historical_prices(tickers, lookback=lookback_length, from_date=start_date, to_date=end_date)
        if price_data.empty:
            self.logger.warning("Insufficient data to generate Volume Breakout signals.")
            return pd.DataFrame()
        
        # Process data differently for multi-ticker (MultiIndex) and single ticker (DatetimeIndex)
        if isinstance(price_data.index, pd.MultiIndex):
            price_data = price_data.sort_index()
            # Compute rolling metrics using groupby transform.
            resistance = price_data.groupby(level=0)['high'].transform(
                lambda x: x.rolling(window=self.params['lookback_period']).max()
            )
            support = price_data.groupby(level=0)['low'].transform(
                lambda x: x.rolling(window=self.params['lookback_period']).min()
            )
            avg_volume = price_data.groupby(level=0)['volume'].transform(
                lambda x: x.rolling(window=self.params['volume_avg_period']).mean()
            )
            price_change = price_data.groupby(level=0)['close'].transform(lambda x: x.pct_change())
            if self.params['use_atr_filter']:
                atr = price_data.groupby(level=0).apply(
                    lambda grp: self._calculate_atr(grp, period=self.params['atr_period'])
                )
                atr = atr.sort_index()
                price_volatility = atr / price_data['close']
            else:
                atr = pd.Series(0, index=price_data.index)
                price_volatility = pd.Series(0, index=price_data.index)
            
            result = price_data.copy()
            result['resistance'] = resistance
            result['support'] = support
            result['avg_volume'] = avg_volume
            result['price_change'] = price_change
            result['atr'] = atr
            result['price_volatility'] = price_volatility
            
            # Vectorized calculation of consecutive breakout counts per ticker.
            above_mask = result['close'] > result.groupby(level=0)['resistance'].shift(1)
            below_mask = result['close'] < result.groupby(level=0)['support'].shift(1)
            def count_consecutive(b):
                return b.groupby((~b).cumsum()).cumcount() + 1
            result['above_resistance_count'] = above_mask.groupby(result.index.get_level_values(0)).transform(count_consecutive)
            result['above_resistance_count'] = result['above_resistance_count'].where(above_mask, 0)
            result['below_support_count'] = below_mask.groupby(result.index.get_level_values(0)).transform(count_consecutive)
            result['below_support_count'] = result['below_support_count'].where(below_mask, 0)
        else:
            price_data = price_data.sort_index()
            resistance = price_data['high'].rolling(window=self.params['lookback_period']).max()
            support = price_data['low'].rolling(window=self.params['lookback_period']).min()
            avg_volume = price_data['volume'].rolling(window=self.params['volume_avg_period']).mean()
            price_change = price_data['close'].pct_change()
            if self.params['use_atr_filter']:
                atr = self._calculate_atr(price_data, period=self.params['atr_period'])
                price_volatility = atr / price_data['close']
            else:
                atr = pd.Series(0, index=price_data.index)
                price_volatility = pd.Series(0, index=price_data.index)
            
            result = price_data.copy()
            result['resistance'] = resistance
            result['support'] = support
            result['avg_volume'] = avg_volume
            result['price_change'] = price_change
            result['atr'] = atr
            result['price_volatility'] = price_volatility
            
            # Vectorized calculation of consecutive breakout counts.
            above_mask = result['close'] > resistance.shift(1)
            below_mask = result['close'] < support.shift(1)
            def count_consecutive(x):
                return x.groupby((~x).cumsum()).cumcount() + 1
            result['above_resistance_count'] = count_consecutive(above_mask)
            result['above_resistance_count'] = result['above_resistance_count'].where(above_mask, 0)
            result['below_support_count'] = count_consecutive(below_mask)
            result['below_support_count'] = result['below_support_count'].where(below_mask, 0)
        
        # Generate raw trading signals based on breakout criteria.
        result['signal'] = 0
        buy_signal = (
            (result['above_resistance_count'] >= self.params['consecutive_bars']) &
            (result['volume'] > result['avg_volume'] * self.params['volume_threshold']) &
            (result['price_change'].abs() > self.params['price_threshold'])
        )
        if self.params['use_atr_filter']:
            buy_signal &= (result['price_change'].abs() > result['price_volatility'] * self.params['atr_threshold'])
        
        sell_signal = (
            (result['below_support_count'] >= self.params['consecutive_bars']) &
            (result['volume'] > result['avg_volume'] * self.params['volume_threshold']) &
            (result['price_change'].abs() > self.params['price_threshold'])
        )
        if self.params['use_atr_filter']:
            sell_signal &= (result['price_change'].abs() > result['price_volatility'] * self.params['atr_threshold'])
        
        result.loc[buy_signal, 'signal'] = 1
        if self.params['long_only']:
            result.loc[sell_signal, 'signal'] = 0
        else:
            result.loc[sell_signal, 'signal'] = -1
        
        # Compute normalized signal strength.
        if self.params['use_atr_filter']:
            result['signal_strength'] = result['price_change'].abs() / (result['price_volatility'] + 1e-6)
        else:
            result['signal_strength'] = result['price_change'].abs()
        
        # Drop rows with essential NaN values.
        result.dropna(subset=['close', 'resistance', 'support', 'avg_volume'], inplace=True)
        
        # If only the latest signal is desired, select the last row for each ticker.
        if latest_only:
            if isinstance(result.index, pd.MultiIndex):
                result = result.groupby(level=0).tail(1)
            else:
                result = result.tail(1)
        
        # Integrate risk management using the RiskManager class.
        risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            trailing_stop_pct=self.params['trailing_stop_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )
        result = risk_manager.apply(result, initial_position=initial_position)
        # Rename risk management output columns to align with the Awesome Oscillator structure.
        result = result.rename(columns={
            'return': 'rm_strategy_return', 
            'cumulative_return': 'rm_cumulative_return', 
            'exit_type': 'rm_action'
        })
        
        return result
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for volatility measurement.
        
        ATR is calculated using Wilder's smoothing method:
            True Range (TR)ₜ = max( highₜ – lowₜ, |highₜ – closeₜ₋₁|, |lowₜ – closeₜ₋₁| )
            ATRₜ = EMA(TR, alpha = 1/period)
        
        Args:
            price_data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
            period (int): Lookback period for ATR calculation (default: 14).
        
        Returns:
            pd.Series: Series of ATR values.
        """
        high = price_data['high']
        low = price_data['low']
        prev_close = price_data['close'].shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
        return atr