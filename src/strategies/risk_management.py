"""
Risk Management Component

This module provides the RiskManager class that applies stop-loss and take-profit
rules, along with slippage and transaction cost adjustments. In addition, this
version integrates profit and cumulative return calculations. The implementation
is fully vectorized for efficiency and correctly handles both long and short positions.

Usage:
    Initialize the RiskManager with desired risk parameters. Then, call its apply()
    method with daily price data and a signals DataFrame. The returned DataFrame includes
    the updated position, realized return (when a trade exit is triggered), cumulative return,
    and the risk management event type (e.g., stop_loss, take_profit, trailing_stop, or signal_exit).
"""

import pandas as pd
import numpy as np

class RiskManager:
    """
    A class to apply risk management to trading signals with built-in adjustments
    for slippage, transaction costs, stop-loss, take-profit, and trailing stop logic.
    It calculates the trade returns and cumulative returns based on risk-managed exits.

    Attributes:
        stop_loss_pct (float): Fractional drop from the entry price to trigger a stop loss.
        take_profit_pct (float): Fractional gain from the entry price to trigger a take profit.
        trailing_stop_pct (float): Fractional distance from the peak (for long) or trough (for short)
                                   that the price is allowed to reverse before the trade is exited.
        slippage_pct (float): Estimated slippage as a fraction of the price.
        transaction_cost_pct (float): Transaction cost as a fraction of the price.
    """
    
    def __init__(self,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 trailing_stop_pct: float = 0.0,
                 slippage_pct: float = 0.001,
                 transaction_cost_pct: float = 0.001):
        """
        Initialize the RiskManager with risk and cost parameters.
        
        Args:
            stop_loss_pct (float): Percentage drop from the entry price to trigger stop loss.
            take_profit_pct (float): Percentage increase from the entry price to trigger take profit.
            trailing_stop_pct (float): Percentage for the trailing stop (0 to disable).
            slippage_pct (float): Estimated slippage percentage during execution.
            transaction_cost_pct (float): Estimated transaction cost percentage per trade.
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.slippage_pct = slippage_pct
        self.transaction_cost_pct = transaction_cost_pct

    def apply(self, signals: pd.DataFrame, initial_position: int = 0) -> pd.DataFrame:
        """
        Apply vectorized risk management rules and compute trade returns.

        This function processes signals DataFrame (with at least 'signal', 'high', 'low' and 'close' columns).
        It determines the entry price (adjusted for slippage and transaction costs) on a change in position,
        computes stop-loss and take-profit thresholds for long and short trades, identifies exit events,
        calculates the realized return when an exit is triggered, and derives the cumulative product of
        the trade multipliers. Finally, it updates the position to 0 on exit events.

        Args:
            signals (pd.DataFrame): Trading signals with at least 'signal', 'high', 'low' and 'close' columns.
            initial_position (int): The initial position (0 for none, 1 for long, -1 for short).

        Returns:
            pd.DataFrame: DataFrame containing:
                - 'close': The closing price.
                - 'high', 'low': High and low prices (used for risk checks).
                - 'signal': Original trading signal.
                - 'position': Updated position after risk management.
                - 'return': Realized trade return on exit events.
                - 'cumulative_return': Cumulative return over time from closed trades.
                - 'exit_type': Indicator of the reason for exiting the trade.
        """
        # Merge signals with relevant price data.
        df = signals.copy()

        # Determine the underlying trading position based on signals.
        # Non-zero signals trigger an entry and the position is forward-filled.
        df['raw_position'] = df['signal'].replace(0, np.nan).ffill().fillna(initial_position)

        # Identify when a new trade is entered, i.e., when the position changes.
        entry_mask = df['raw_position'].diff().abs() >= 1

        # Calculate entry prices adjusted for slippage and transaction costs.
        # On a new trade, the entry price is the current 'close' adjusted by the cost premium.
        df['entry_price'] = np.nan
        df.loc[entry_mask, 'entry_price'] = df['close'] * (
            1 + (df['raw_position'] * (self.slippage_pct + self.transaction_cost_pct))
        )
        # If the first row has an open position and is not marked as an entry,
        # explicitly set the entry price.
        if len(df) > 0 and pd.isna(df.iloc[0]['entry_price']) and df.iloc[0]['raw_position'] != 0:
            df.iat[0, df.columns.get_loc('entry_price')] = df.iloc[0]['close'] * (
                1 + (df.iloc[0]['raw_position'] * (self.slippage_pct + self.transaction_cost_pct))
            )
        # Forward fill the entry price until a new entry occurs.
        df['entry_price'] = df['entry_price'].ffill()

        # Calculate stop-loss and take-profit thresholds for each trade.
        df['stop_level'] = np.nan
        df['target_level'] = np.nan

        # For long positions, a stop loss is below the entry and the target is above.
        long_mask = df['raw_position'] == 1
        df.loc[long_mask, 'stop_level'] = df['entry_price'] * (1 - self.stop_loss_pct)
        df.loc[long_mask, 'target_level'] = df['entry_price'] * (1 + self.take_profit_pct)

        # For short positions, the stop loss is above the entry and the target is below.
        short_mask = df['raw_position'] == -1
        df.loc[short_mask, 'stop_level'] = df['entry_price'] * (1 + self.stop_loss_pct)
        df.loc[short_mask, 'target_level'] = df['entry_price'] * (1 - self.take_profit_pct)

        # Determine exit conditions:
        #  - Stop loss: For long positions if the day's low falls to or below stop_level;
        #               for short positions if the day's high rises to or above stop_level.
        #  - Take profit: For long positions if the day's high reaches or exceeds target_level;
        #                 for short positions if the day's low reaches or falls below target_level.
        #  - Signal exit: When a reversal in the signal is detected.
        stop_exit = (long_mask & (df['low'] <= df['stop_level'])) | (short_mask & (df['high'] >= df['stop_level']))
        target_exit = (long_mask & (df['high'] >= df['target_level'])) | (short_mask & (df['low'] <= df['target_level']))
        signal_exit = df['signal'].abs().diff().abs() >= 1

        # --- Trailing Stop Logic Added Here ---
        # If trailing_stop_pct is enabled (> 0), then compute the dynamic trailing stop level.
        # For long positions, the trailing stop level is set to the cumulative maximum high
        # (since trade entry) multiplied by (1 - trailing_stop_pct). For short positions, it is
        # the cumulative minimum low multiplied by (1 + trailing_stop_pct).
        if self.trailing_stop_pct > 0:
            # Mark the bar when a new trade is initiated.
            df['entry_flag'] = False
            df.loc[entry_mask, 'entry_flag'] = True
            if len(df) > 0 and df.iloc[0]['raw_position'] != 0:
                df.iat[0, df.columns.get_loc('entry_flag')] = True
            # Create a trade identifier that increases at every entry.
            df['trade_id'] = np.where(df['raw_position'] != 0, df['entry_flag'].cumsum(), np.nan)

            # Compute cumulative high (for long trades) and cumulative low (for short trades) within each trade.
            df['cummax_high'] = df.groupby('trade_id')['high'].cummax()
            df['trailing_stop_long'] = df['cummax_high'] * (1 - self.trailing_stop_pct)

            df['cummin_low'] = df.groupby('trade_id')['low'].cummin()
            df['trailing_stop_short'] = df['cummin_low'] * (1 + self.trailing_stop_pct)

            # Define trailing stop exit conditions:
            # For long positions: exit if the day's low falls to or below the trailing stop level.
            # For short positions: exit if the day's high rises to or above the trailing stop level.
            trailing_stop_exit = (long_mask & (df['low'] <= df['trailing_stop_long'])) | (short_mask & (df['high'] >= df['trailing_stop_short']))
        else:
            trailing_stop_exit = pd.Series(False, index=df.index)
        # ------------------------------------------------

        # Create a combined mask for any exit event.
        exit_mask = stop_exit | target_exit | trailing_stop_exit | signal_exit

        # Label the exit event type.
        if self.trailing_stop_pct > 0:
            df['exit_type'] = np.select(
                [trailing_stop_exit, stop_exit, target_exit, signal_exit],
                ['trailing_stop', 'stop_loss', 'take_profit', 'signal_exit'],
                default='none'
            )
        else:
            df['exit_type'] = np.select(
                [stop_exit, target_exit, signal_exit],
                ['stop_loss', 'take_profit', 'signal_exit'],
                default='none'
            )

        # Compute the exit price based on which exit condition was met.
        if self.trailing_stop_pct > 0:
            df['exit_price'] = np.select(
                [
                    trailing_stop_exit & long_mask,
                    trailing_stop_exit & short_mask,
                    stop_exit & long_mask,
                    stop_exit & short_mask,
                    target_exit & long_mask,
                    target_exit & short_mask,
                    signal_exit
                ],
                [
                    df['trailing_stop_long'] * (1 - self.slippage_pct - self.transaction_cost_pct),
                    df['trailing_stop_short'] * (1 + self.slippage_pct + self.transaction_cost_pct),
                    df['stop_level'] * (1 - self.slippage_pct - self.transaction_cost_pct),
                    df['stop_level'] * (1 + self.slippage_pct + self.transaction_cost_pct),
                    df['target_level'] * (1 - self.slippage_pct - self.transaction_cost_pct),
                    df['target_level'] * (1 + self.slippage_pct + self.transaction_cost_pct),
                    df['close'] * (1 - (df['raw_position'] * (self.slippage_pct + self.transaction_cost_pct)))
                ],
                default=np.nan
            )
        else:
            df['exit_price'] = np.select(
                [
                    stop_exit & long_mask,
                    target_exit & long_mask,
                    stop_exit & short_mask,
                    target_exit & short_mask,
                    signal_exit
                ],
                [
                    df['stop_level'] * (1 - self.slippage_pct - self.transaction_cost_pct),
                    df['target_level'] * (1 - self.slippage_pct - self.transaction_cost_pct),
                    df['stop_level'] * (1 + self.slippage_pct + self.transaction_cost_pct),
                    df['target_level'] * (1 + self.slippage_pct + self.transaction_cost_pct),
                    df['close'] * (1 - (df['raw_position'] * (self.slippage_pct + self.transaction_cost_pct)))
                ],
                default=np.nan
            )

        # Calculate the realized return for trades where an exit event occurred.
        # For long trades: return = (exit_price / entry_price) - 1.
        # For short trades: return = (entry_price / exit_price) - 1.
        trade_return_long = (df['exit_price'] / df['entry_price']) - 1
        trade_return_short = (df['entry_price'] / df['exit_price']) - 1
        df['return'] = np.where(
            exit_mask & long_mask, trade_return_long,
            np.where(exit_mask & short_mask, trade_return_short, 0)
        )

        # Once an exit event occurs, the position is reset to 0.
        df['position'] = df['raw_position']
        df.loc[exit_mask, 'position'] = 0

        # For cumulative return, we use a trade multiplier:
        #   For long trades: multiplier = exit_price / entry_price.
        #   For short trades: multiplier = entry_price / exit_price.
        # On bars with no exit, use 1 (neutral effect).
        df['trade_multiplier'] = np.where(
            exit_mask,
            np.where(long_mask, df['exit_price'] / df['entry_price'],
                     np.where(short_mask, df['entry_price'] / df['exit_price'], 1)),
            1
        )
        df['cumulative_return'] = (df['trade_multiplier']).cumprod() - 1

        # Drop temporary and helper columns.
        cols_to_drop = ['raw_position', 'entry_price', 'stop_level',
                        'target_level', 'trade_multiplier', 'exit_price']
        if self.trailing_stop_pct > 0:
            cols_to_drop.extend(['entry_flag', 'trade_id', 'cummax_high', 'cummin_low', 'trailing_stop_long', 'trailing_stop_short'])
        result = df.drop(columns=cols_to_drop)

        return result