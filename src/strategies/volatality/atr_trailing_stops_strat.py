# trading_system/src/strategies/volatality/atr_trailing_stops.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig

class ATRTrailingStops(BaseStrategy):
    """
    ATR Trailing Stops Strategy with Integrated Risk Management.

    This strategy uses the Average True Range (ATR) as a volatility measure to set
    dynamic trailing stop levels for trend-following trades. The main logic is as follows:

    1. ATR Calculation:
       - True Range (TR) is computed as:
             TR = max(high - low, |high - previous_close|, |low - previous_close|)
       - ATR is then calculated using an exponential moving average with smoothing factor
         alpha = 1/atr_period, replicating Wilder's smoothing method.

    2. Trend Determination:
       - A Simple Moving Average (SMA) of the close price over a specified trend_period acts as a trend filter.
       - When close > SMA, the trend is considered upward (favoring long positions).
       - Otherwise, a downward trend is assumed (favoring short positions).

    3. Trailing Stop Mechanism:
       - For long trades:
         - Entry is initiated when the trend becomes up.
         - The trailing stop is set as:
               trailing_stop_long = current_close - (atr × atr_multiplier)
         - On subsequent bars the stop is updated as the maximum of
               previous_stop and (current_close - (atr × atr_multiplier)).
         - An exit is triggered if the low price breaches the previous trailing stop.
       - For short trades:
         - Entry is initiated when the trend turns down.
         - The trailing stop is set as:
               trailing_stop_short = current_close + (atr × atr_multiplier)
         - On subsequent bars the stop is updated as the minimum of
               previous_stop and (current_close + (atr × atr_multiplier)).
         - An exit is triggered if the high price breaches the previous trailing stop.

    4. Signal Generation:
       - A buy signal (signal = 1) is generated when a new long is initiated.
       - A sell signal (signal = -1) is generated when a new short is initiated.
       - An exit signal (signal = 0) is generated when the trailing stop is breached.
       - A signal strength metric is computed based on the normalized difference between
         the close price and the corresponding trailing stop level.

    5. Integration with Risk Management:
       - After signal generation, the signals are passed to the RiskManager which applies
         stop-loss, take-profit, slippage, and transaction cost adjustments.
       - The final output DataFrame contains detailed performance metrics (such as position,
         realized returns, cumulative returns, and risk management actions) to allow further
         downstream calculations (e.g., Sharpe ratio, maximum drawdown).

    Args:
        tickers (str or List[str]): Stock ticker symbol or list of tickers.
        start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
        end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
        initial_position (int): Starting trading position (default is 0: no position,
                                  1 for long, -1 for short).
        latest_only (bool): If True, return only the latest signal per ticker for forecasting.

    Strategy-specific parameters via `params` (with defaults):
        - atr_period (int): Lookback period for ATR calculation (default: 14).
        - atr_multiplier (float): Multiplier for ATR to set the stop distance (default: 3.0).
        - trend_period (int): Period for computing SMA for trend detection (default: 20).
        - stop_loss_pct (float): Stop-loss percentage for risk management (default: 0.05).
        - take_profit_pct (float): Take-profit percentage for risk management (default: 0.10).
        - trailing_stop_pct (float): Trailing stop percentage for risk management (default: 0.0).
        - slippage_pct (float): Slippage percentage for risk management (default: 0.001).
        - transaction_cost_pct (float): Transaction cost percentage for risk management (default: 0.001).
        - long_only (bool): If True, restricts trading to long positions only (default: True).

    Returns:
        pd.DataFrame: DataFrame with the following key columns:
            - open, high, low, close, volume: Price data.
            - atr, sma: Calculated ATR and SMA.
            - trailing_stop_long, trailing_stop_short: Dynamic trailing stop levels.
            - signal: Trading signal (1 for buy, -1 for sell, 0 for exit).
            - signal_strength: Normalized indicator of signal confidence.
            - position: Updated position after applying risk management.
            - return: Realized trade return upon exit events.
            - cumulative_return: Cumulative return from closed trades.
            - rm_action: Indicator of the risk management exit action.
            - Additional risk management metrics provided by the RiskManager.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the ATR Trailing Stops strategy with specified parameters.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters.
        """
        default_params = {
            'atr_period': 14,
            'atr_multiplier': 3.0,
            'trend_period': 20,
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

    def generate_signals(self, tickers: Union[str, List[str]], start_date: Optional[str] = None,
                         end_date: Optional[str] = None, initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals using ATR-based trailing stops with integrated risk adjustments.

        Retrieves historical price data (filtered by date if provided), calculates the ATR and the trend
        (via an SMA), computes dynamic trailing stop levels, and generates entry or exit signals for long
        and short positions. The signal strength is calculated as the normalized distance between the close 
        and the relevant trailing stop value. Finally, the signals are processed by the RiskManager to
        incorporate stop-loss, take-profit, slippage, and transaction cost adjustments.

        Args:
            tickers (str or List[str]): Stock ticker symbol or list of tickers.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting trading position (default: 0).
            latest_only (bool): If True, returns only the latest signal per ticker.

        Returns:
            pd.DataFrame: DataFrame containing trading signals and risk-managed performance metrics.
        """
        # Retrieve historical price data based on provided dates or using a default 252-day lookback
        if start_date or end_date:
            price_data = self.get_historical_prices(tickers, from_date=start_date, to_date=end_date)
        else:
            price_data = self.get_historical_prices(tickers, lookback=252)

        # Process data for a single ticker or multiple tickers
        if isinstance(tickers, str):
            if not self._validate_data(price_data, min_records=int(self.params['atr_period']) + 10):
                self.logger.warning(f"Insufficient data for {tickers} to generate ATR trailing stops signals")
                return pd.DataFrame()
            groups = [(tickers, price_data)]
        else:
            if price_data.empty:
                self.logger.warning("Price data is empty for provided tickers")
                return pd.DataFrame()
            groups = list(price_data.groupby(level=0))

        signal_dfs = []

        # Process each ticker group individually
        for ticker, df in groups:
            df = df.copy().reset_index(level=0, drop=True).sort_index()
            # Calculate ATR and compute SMA for trend filtering
            atr = self._calculate_atr(df, period=int(self.params['atr_period']))
            sma = df['close'].rolling(window=int(self.params['trend_period']), min_periods=int(self.params['trend_period'])).mean()
            trend_up = df['close'] > sma

            # Initialize working DataFrame with price and indicator data
            df_signals = pd.DataFrame(index=df.index)
            df_signals['open'] = df['open']
            df_signals['high'] = df['high']
            df_signals['low'] = df['low']
            df_signals['close'] = df['close']
            df_signals['volume'] = df['volume']
            df_signals['atr'] = atr
            df_signals['sma'] = sma
            df_signals['trend_up'] = trend_up

            # Preallocate columns for trailing stops and position flags
            df_signals['trailing_stop_long'] = np.nan
            df_signals['trailing_stop_short'] = np.nan
            df_signals['in_long'] = False
            df_signals['in_short'] = False

            # Initialize state variables for positions and trailing stop levels
            in_long = False
            in_short = False
            trailing_stop_long = np.nan
            trailing_stop_short = np.nan

            # Iteratively compute trailing stops and update position status
            for i, current_date in enumerate(df_signals.index):
                current_close = df_signals.at[current_date, 'close']
                current_atr = df_signals.at[current_date, 'atr']
                current_high = df_signals.at[current_date, 'high']
                current_low = df_signals.at[current_date, 'low']
                current_trend = df_signals.at[current_date, 'trend_up']

                # For the first observation, initialize the trade based on the trend direction.
                if i == 0:
                    if current_trend:
                        in_long = True
                        trailing_stop_long = current_close - (current_atr * self.params['atr_multiplier'])
                        df_signals.at[current_date, 'trailing_stop_long'] = trailing_stop_long
                        df_signals.at[current_date, 'in_long'] = True
                    else:
                        in_short = True
                        trailing_stop_short = current_close + (current_atr * self.params['atr_multiplier'])
                        df_signals.at[current_date, 'trailing_stop_short'] = trailing_stop_short
                        df_signals.at[current_date, 'in_short'] = True
                    continue

                # If not in any position, start a new trade based on the trend.
                if not in_long and not in_short:
                    if current_trend:
                        in_long = True
                        trailing_stop_long = current_close - (current_atr * self.params['atr_multiplier'])
                        df_signals.at[current_date, 'trailing_stop_long'] = trailing_stop_long
                        df_signals.at[current_date, 'in_long'] = True
                    else:
                        in_short = True
                        trailing_stop_short = current_close + (current_atr * self.params['atr_multiplier'])
                        df_signals.at[current_date, 'trailing_stop_short'] = trailing_stop_short
                        df_signals.at[current_date, 'in_short'] = True
                # If in a long position, update the trailing stop and check for an exit.
                elif in_long:
                    candidate_stop = current_close - (current_atr * self.params['atr_multiplier'])
                    prev_stop = df_signals.iloc[i-1]['trailing_stop_long']
                    trailing_stop_long = max(trailing_stop_long, candidate_stop)
                    df_signals.at[current_date, 'trailing_stop_long'] = trailing_stop_long
                    if current_low <= prev_stop:
                        in_long = False
                        df_signals.at[current_date, 'in_long'] = False
                    else:
                        df_signals.at[current_date, 'in_long'] = True
                # If in a short position, update the trailing stop and check for an exit.
                elif in_short:
                    candidate_stop = current_close + (current_atr * self.params['atr_multiplier'])
                    prev_stop = df_signals.iloc[i-1]['trailing_stop_short']
                    trailing_stop_short = min(trailing_stop_short, candidate_stop)
                    df_signals.at[current_date, 'trailing_stop_short'] = trailing_stop_short
                    if current_high >= prev_stop:
                        in_short = False
                        df_signals.at[current_date, 'in_short'] = False
                    else:
                        df_signals.at[current_date, 'in_short'] = True

            # Generate signals from changes in the position flags.
            df_signals['signal'] = 0

            long_entry = df_signals['in_long'] & ~df_signals['in_long'].shift(1).fillna(False)
            short_entry = df_signals['in_short'] & ~df_signals['in_short'].shift(1).fillna(False)
            df_signals.loc[long_entry, 'signal'] = 1
            df_signals.loc[short_entry, 'signal'] = -1
            long_exit = ~df_signals['in_long'] & df_signals['in_long'].shift(1).fillna(False)
            short_exit = ~df_signals['in_short'] & df_signals['in_short'].shift(1).fillna(False)
            df_signals.loc[long_exit | short_exit, 'signal'] = 0

            # Compute signal strength as the normalized distance between the close price and the trailing stop.
            df_signals['signal_strength'] = np.where(
                df_signals['signal'] == 1,
                (df_signals['close'] - df_signals['trailing_stop_long']) /
                (df_signals['atr'] * self.params['atr_multiplier']),
                np.where(
                    df_signals['signal'] == -1,
                    (df_signals['trailing_stop_short'] - df_signals['close']) /
                    (df_signals['atr'] * self.params['atr_multiplier']),
                    0
                )
            )

            if self.params['long_only']:
                df_signals['signal'] = df_signals['signal'].clip(lower=0)

            # Drop temporary flags
            df_signals.drop(columns=['in_long', 'in_short', 'trend_up'], inplace=True)
            # Add a ticker identifier for multi-ticker outputs.
            df_signals['ticker'] = ticker
            signal_dfs.append(df_signals)

        # Concatenate all ticker signal DataFrames and sort the result.
        final_signals = pd.concat(signal_dfs)
        final_signals = final_signals.reset_index().set_index(['ticker', 'date']).sort_index()

        # If only the latest signal is required, filter by the last row per ticker.
        if latest_only:
            if isinstance(tickers, str):
                final_signals = final_signals.iloc[[-1]]
            else:
                final_signals = final_signals.groupby('ticker', group_keys=False).apply(lambda grp: grp.iloc[[-1]])

        # Incorporate risk management (stop-loss, take-profit, slippage, transaction cost) via RiskManager.
        from src.strategies.risk_management import RiskManager
        rm = RiskManager(stop_loss_pct=self.params['stop_loss_pct'],
                         take_profit_pct=self.params['take_profit_pct'],
                         slippage_pct=self.params['slippage_pct'],
                         transaction_cost_pct=self.params['transaction_cost_pct'])
        final_signals = rm.apply(final_signals, initial_position=initial_position)

        return final_signals

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR) from historical OHLC data.

        The True Range (TR) is defined as:
            TR = max(high - low, |high - previous_close|, |low - previous_close|)
        ATR is then computed as an exponential moving average of TR with a smoothing factor
        of alpha = 1/period.

        Args:
            price_data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
            period (int): Lookback period for the ATR calculation (default: 14).

        Returns:
            pd.Series: The calculated ATR values.
        """
        high = price_data['high']
        low = price_data['low']
        prev_close = price_data['close'].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        return atr