"""
Risk Management Component

This module provides the RiskManager class to apply stop-loss and take-profit rules,
and adjust for slippage and transaction costs.

Usage:
    Initialize the RiskManager with desired parameters. Then, call its
    apply() method with daily price data and a signals DataFrame. The method
    returns a modified signals DataFrame with risk-managed returns and actions.

Methods:
    apply: Processes daily price data and trading signals and returns risk-managed
           columns including:
             - rm_strategy_return
             - rm_cumulative_return
             - rm_action
"""

import pandas as pd

class RiskManager:
    def __init__(self, 
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 slippage_pct: float = 0.001,
                 transaction_cost_pct: float = 0.001):
        """
        Initialize the RiskManager.

        Parameters:
            stop_loss_pct (float): Percentage drop from entry price triggering stop loss.
            take_profit_pct (float): Percentage increase from entry price triggering take profit.
            slippage_pct (float): Percentage impact of slippage.
            transaction_cost_pct (float): Percentage representing trading cost.
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.slippage_pct = slippage_pct
        self.transaction_cost_pct = transaction_cost_pct

    def apply(self, data: pd.DataFrame, signals: pd.DataFrame, initial_position: int = 0) -> pd.DataFrame:
        """
        Apply risk management rules along with slippage and transaction cost adjustments.

        This method simulates trade exits when a stop-loss, take-profit or an explicit sell signal occurs.
        It expects 'data' to include at least the columns 'high', 'low', and 'close'. The 'signals' DataFrame
        is expected to contain 'signal', 'position', and 'close' price (for entry/exit reference).

        Args:
            data (pd.DataFrame): Daily price data.
            signals (pd.DataFrame): Trading signals DataFrame.
            initial_position (int): The initial position (e.g., 0 for no open trade).

        Returns:
            pd.DataFrame: A copy of the signals DataFrame augmented with:
                - 'rm_strategy_return': The risk-managed return for the day.
                - 'rm_cumulative_return': Cumulative return based on the risk-managed returns.
                - 'rm_action': String flag indicating the risk management event.
        """
        rm_signals = signals.copy()
        # Initialize new columns.
        rm_signals['rm_strategy_return'] = 0.0
        rm_signals['rm_action'] = 'none'
        in_trade = False
        entry_price = None

        # Loop over each day (daily data is not huge – this loop is acceptable).
        for idx in rm_signals.index:
            row_data = data.loc[idx]
            row_signal = rm_signals.loc[idx, 'signal']
            if not in_trade:
                if row_signal == 1:
                    in_trade = True
                    # Adjust entry price for slippage and transaction cost.
                    entry_price = row_data['close'] * (1 + self.slippage_pct + self.transaction_cost_pct)
                    rm_signals.at[idx, 'rm_action'] = 'buy'
                else:
                    rm_signals.at[idx, 'rm_action'] = 'none'
            else:
                # Calculate risk thresholds.
                stop_level = entry_price * (1 - self.stop_loss_pct)
                tp_level = entry_price * (1 + self.take_profit_pct)
                if row_data['low'] <= stop_level:
                    exit_price = stop_level * (1 - self.slippage_pct - self.transaction_cost_pct)
                    trade_return = (exit_price / entry_price) - 1
                    rm_signals.at[idx, 'rm_strategy_return'] = trade_return
                    rm_signals.at[idx, 'rm_action'] = 'stop_loss'
                    in_trade = False
                    entry_price = None
                elif row_data['high'] >= tp_level:
                    exit_price = tp_level * (1 - self.slippage_pct - self.transaction_cost_pct)
                    trade_return = (exit_price / entry_price) - 1
                    rm_signals.at[idx, 'rm_strategy_return'] = trade_return
                    rm_signals.at[idx, 'rm_action'] = 'take_profit'
                    in_trade = False
                    entry_price = None
                elif row_signal == -1:
                    exit_price = row_data['close'] * (1 - self.slippage_pct - self.transaction_cost_pct)
                    trade_return = (exit_price / entry_price) - 1
                    rm_signals.at[idx, 'rm_strategy_return'] = trade_return
                    rm_signals.at[idx, 'rm_action'] = 'sell_signal'
                    in_trade = False
                    entry_price = None
                else:
                    rm_signals.at[idx, 'rm_action'] = 'hold'
                    rm_signals.at[idx, 'rm_strategy_return'] = 0.0

        # Calculate cumulative return based on the risk-managed returns.
        rm_signals['rm_cumulative_return'] = (1 + rm_signals['rm_strategy_return']).cumprod() - 1
        return rm_signals