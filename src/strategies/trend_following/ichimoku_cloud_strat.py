# trading_system/src/strategies/trend_following/ichimoku_cloud.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class IchimokuCloud(BaseStrategy):
    """
    Ichimoku Cloud Strategy with Integrated Risk Management Component.

    This strategy implements the Ichimoku Cloud (Ichimoku Kinko Hyo)
    indicator system. The indicator components are defined mathematically as:
    
        Tenkan-sen    = (Highest High + Lowest Low) / 2 over `tenkan_period`
                      (default 9)
        Kijun-sen     = (Highest High + Lowest Low) / 2 over `kijun_period`
                      (default 26)
        Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted forward by `displacement`
                      (default 26)
        Senkou Span B = (Highest High + Lowest Low) / 2 over `senkou_b_period`
                      (default 52), shifted forward by `displacement`
        Chikou Span   = Close price shifted backward by `displacement`
    
    Trading signals are generated as follows:
    
      - A long (buy) signal (signal = 1) is triggered when any of the following
        conditions occur (if enabled):
          • Tenkan-sen crosses above Kijun-sen.
          • Price crosses above Kijun-sen.
          • Price crosses above the upper bound of the cloud (max(Senkou Span A, Senkou Span B)).
    
      - A short (sell) signal (signal = -1) is triggered when:
          • Tenkan-sen crosses below Kijun-sen.
          • Price crosses below Kijun-sen.
          • Price crosses below the lower bound of the cloud (min(Senkou Span A, Senkou Span B)).
    
    Risk Management is applied via the RiskManager class. It adjusts the entry price
    (accounting for slippage and transaction costs), computes stop-loss and take-profit
    thresholds, identifies exit events (also on signal reversal), and computes realized and
    cumulative returns.
    
    The strategy supports both backtesting (through start_date and end_date selections)
    and forecasting (using the latest available data). It also efficiently processes a list
    of tickers in a vectorized fashion.
    
    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Strategy-specific parameters with defaults:
            - 'tenkan_period': int, period for Tenkan-sen (default: 9)
            - 'kijun_period': int, period for Kijun-sen (default: 26)
            - 'senkou_b_period': int, period for Senkou Span B (default: 52)
            - 'displacement': int, shift period for Senkou Span A/B and Chikou Span (default: 26)
            - 'use_cloud_breakouts': bool, whether to use cloud breakout signals (default: True)
            - 'use_tk_cross': bool, whether to use TK cross signals (default: True)
            - 'use_price_cross': bool, whether to use price-Kijun cross signals (default: False)
            - 'stop_loss_pct': float, stop loss percentage (default: 0.05)
            - 'take_profit_pct': float, take profit percentage (default: 0.10)
            - 'slippage_pct': float, estimated slippage as a fraction (default: 0.001)
            - 'transaction_cost_pct': float, transaction cost as a fraction (default: 0.001)
            - 'long_only': bool, whether to allow only long positions (default: True)
    
    Outputs:
        A pandas DataFrame containing, at a minimum:
            - 'open', 'high', 'low', 'close', 'volume': Price data
            - 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span': Ichimoku components
            - 'cloud_bullish': Boolean indicator if Senkou Span A > Senkou Span B
            - 'signal': Trading signal (1, -1, or 0)
            - Additional columns from RiskManager including 'position', 'return',
              'cumulative_return', and 'exit_type'.
    
        This output is designed to support downstream metrics computation such as the Sharpe ratio,
        maximum drawdown, and to provide a stable final daily signal for portfolio decisions.
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        default_params = {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26,
            'use_cloud_breakouts': True,
            'use_tk_cross': True,
            'use_price_cross': False,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            'long_only': True
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
    
    def generate_signals(self, tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate risk-managed trading signals using the Ichimoku Cloud indicator.

        This method retrieves required historical data (including extra periods for rolling
        calculations), computes Ichimoku components in a vectorized fashion (supporting multiple
        tickers via group operations), generates buy/sell signals based on TK crosses, price-Kijun
        crosses and cloud breakout rules, and then applies risk management (stop-loss, take-profit,
        slippage, transaction costs).

        Args:
            tickers (str or List[str]): A single ticker symbol or a list of ticker symbols.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format. When provided,
                                        additional historical data is fetched for indicator calculation.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): The starting market position (0 for flat, 1 for long, -1 for short).
            latest_only (bool): If True, only the most recent signal row is returned (per ticker in multi-ticker mode).

        Returns:
            pd.DataFrame: DataFrame that includes price data, Ichimoku components, raw trading signals,
            and risk management outputs (including positions and returns) ready for backtest analysis
            and downstream optimization.
        """
        # Convert to list if a single ticker is provided
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Define necessary indicator periods from parameters
        displacement = self.params['displacement']
        tenkan_period = self.params['tenkan_period']
        kijun_period = self.params['kijun_period']
        senkou_b_period = self.params['senkou_b_period']
        extra_periods = displacement + max(tenkan_period, kijun_period, senkou_b_period)
        
        # If a start_date is provided, adjust it backwards to supply extra data for rolling calculations
        if start_date:
            start_dt = pd.to_datetime(start_date)
            adjusted_start_date = (start_dt - pd.Timedelta(days=extra_periods)).strftime('%Y-%m-%d')
        else:
            adjusted_start_date = None
        
        # Retrieve historical price data; when start_date is not provided, use a lookback period (e.g., 252 trading days)
        lookback = None if start_date else (252 + extra_periods)
        price_data = self.get_historical_prices(
            tickers,
            lookback=lookback,
            from_date=adjusted_start_date,
            to_date=end_date
        )
        if price_data.empty:
            self.logger.warning("No price data available for tickers: %s", tickers)
            return pd.DataFrame()
        
        # Determine whether the data is for multiple tickers (MultiIndex) or single ticker
        multi_ticker = isinstance(price_data.index, pd.MultiIndex)
        
        if multi_ticker:
            # Group data by ticker for vectorized rolling operations.
            groups = price_data.groupby(level='ticker')
            # Calculate Tenkan-sen: (highest high + lowest low) / 2 over tenkan_period.
            tenkan_high = groups['high'].rolling(window=tenkan_period, min_periods=tenkan_period).max().reset_index(level=0, drop=True)
            tenkan_low = groups['low'].rolling(window=tenkan_period, min_periods=tenkan_period).min().reset_index(level=0, drop=True)
            price_data['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

            # Calculate Kijun-sen: (highest high + lowest low) / 2 over kijun_period.
            kijun_high = groups['high'].rolling(window=kijun_period, min_periods=kijun_period).max().reset_index(level=0, drop=True)
            kijun_low = groups['low'].rolling(window=kijun_period, min_periods=kijun_period).min().reset_index(level=0, drop=True)
            price_data['kijun_sen'] = (kijun_high + kijun_low) / 2

            # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, shifted forward by displacement.
            senkou_a = (price_data['tenkan_sen'] + price_data['kijun_sen']) / 2
            price_data['senkou_span_a'] = senkou_a.groupby(price_data.index.get_level_values('ticker')).shift(displacement)

            # Senkou Span B: (highest high + lowest low) / 2 over senkou_b_period, shifted forward by displacement.
            senkou_b_high = groups['high'].rolling(window=senkou_b_period, min_periods=senkou_b_period).max().reset_index(level=0, drop=True)
            senkou_b_low = groups['low'].rolling(window=senkou_b_period, min_periods=senkou_b_period).min().reset_index(level=0, drop=True)
            senkou_b = (senkou_b_high + senkou_b_low) / 2
            price_data['senkou_span_b'] = senkou_b.groupby(price_data.index.get_level_values('ticker')).shift(displacement)

            # Chikou Span: Current close shifted backward by displacement.
            price_data['chikou_span'] = groups['close'].shift(-displacement).reset_index(level=0, drop=True)

            # Compute previous values of Tenkan-sen, Kijun-sen and close for cross detection.
            price_data['prev_tenkan'] = groups['tenkan_sen'].shift(1).reset_index(level=0, drop=True)
            price_data['prev_kijun'] = groups['kijun_sen'].shift(1).reset_index(level=0, drop=True)
            price_data['prev_close'] = groups['close'].shift(1).reset_index(level=0, drop=True)
            
            # For cloud breakout signals compute previous cloud boundaries.
            cloud_top = price_data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            cloud_bottom = price_data[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            price_data['prev_cloud_top'] = cloud_top.groupby(price_data.index.get_level_values('ticker')).shift(1)
            price_data['prev_cloud_bottom'] = cloud_bottom.groupby(price_data.index.get_level_values('ticker')).shift(1)
        else:
            # For a single ticker compute indicator components without grouping.
            price_data.index = pd.to_datetime(price_data.index)
            tenkan_high = price_data['high'].rolling(window=tenkan_period, min_periods=tenkan_period).max()
            tenkan_low = price_data['low'].rolling(window=tenkan_period, min_periods=tenkan_period).min()
            price_data['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

            kijun_high = price_data['high'].rolling(window=kijun_period, min_periods=kijun_period).max()
            kijun_low = price_data['low'].rolling(window=kijun_period, min_periods=kijun_period).min()
            price_data['kijun_sen'] = (kijun_high + kijun_low) / 2

            senkou_a = (price_data['tenkan_sen'] + price_data['kijun_sen']) / 2
            price_data['senkou_span_a'] = senkou_a.shift(displacement)

            senkou_b_high = price_data['high'].rolling(window=senkou_b_period, min_periods=senkou_b_period).max()
            senkou_b_low = price_data['low'].rolling(window=senkou_b_period, min_periods=senkou_b_period).min()
            senkou_b = (senkou_b_high + senkou_b_low) / 2
            price_data['senkou_span_b'] = senkou_b.shift(displacement)

            price_data['chikou_span'] = price_data['close'].shift(-displacement)

            price_data['prev_tenkan'] = price_data['tenkan_sen'].shift(1)
            price_data['prev_kijun'] = price_data['kijun_sen'].shift(1)
            price_data['prev_close'] = price_data['close'].shift(1)
            cloud_top = price_data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            cloud_bottom = price_data[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            price_data['prev_cloud_top'] = cloud_top.shift(1)
            price_data['prev_cloud_bottom'] = cloud_bottom.shift(1)
        
        # Compute cloud bullishness.
        price_data['cloud_bullish'] = price_data['senkou_span_a'] > price_data['senkou_span_b']
        
        # Initialize the raw signal column.
        price_data['signal'] = 0
        buy_signals = pd.Series(False, index=price_data.index)
        sell_signals = pd.Series(False, index=price_data.index)
        
        # Generate TK Cross signals.
        if self.params.get('use_tk_cross', True):
            tk_cross_up = (price_data['tenkan_sen'] > price_data['kijun_sen']) & (price_data['prev_tenkan'] <= price_data['prev_kijun'])
            tk_cross_down = (price_data['tenkan_sen'] < price_data['kijun_sen']) & (price_data['prev_tenkan'] >= price_data['prev_kijun'])
            buy_signals |= tk_cross_up
            sell_signals |= tk_cross_down
        
        # Generate Price-Kijun Cross signals.
        if self.params.get('use_price_cross', False):
            price_cross_up = (price_data['close'] > price_data['kijun_sen']) & (price_data['prev_close'] <= price_data['prev_kijun'])
            price_cross_down = (price_data['close'] < price_data['kijun_sen']) & (price_data['prev_close'] >= price_data['prev_kijun'])
            buy_signals |= price_cross_up
            sell_signals |= price_cross_down
        
        # Generate Cloud Breakout signals.
        if self.params.get('use_cloud_breakouts', True):
            cloud_top = price_data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            cloud_bottom = price_data[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            cloud_breakout_up = (price_data['close'] > cloud_top) & (price_data['prev_close'] <= price_data['prev_cloud_top'])
            cloud_breakout_down = (price_data['close'] < cloud_bottom) & (price_data['prev_close'] >= price_data['prev_cloud_bottom'])
            buy_signals |= cloud_breakout_up
            sell_signals |= cloud_breakout_down
        
        # Set signals: +1 for buys and -1 for sells.
        price_data.loc[buy_signals, 'signal'] = 1
        if self.params.get('long_only', True):
            price_data.loc[sell_signals, 'signal'] = 0
        else:
            price_data.loc[sell_signals, 'signal'] = -1
        
        # Drop rows with NA values that may result from rolling calculations.
        price_data = price_data.dropna()
        
        # Apply risk management adjustments (stop loss, take profit, slippage, transaction cost).
        rm = RiskManager(
            stop_loss_pct=self.params.get("stop_loss_pct", 0.05),
            take_profit_pct=self.params.get("take_profit_pct", 0.10),
            slippage_pct=self.params.get("slippage_pct", 0.001),
            transaction_cost_pct=self.params.get("transaction_cost_pct", 0.001)
        )
        result = rm.apply(price_data, initial_position=initial_position)
        
        # If latest_only flag is set, return only the most recent signal row for each ticker.
        if latest_only:
            if multi_ticker:
                result = result.groupby(level='ticker').tail(1)
            else:
                result = result.tail(1)
        
        return result