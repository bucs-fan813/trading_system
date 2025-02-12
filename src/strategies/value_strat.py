# trading_system/src/strategies/value_strat.py

import pandas as pd
from src.strategies.base_strat import BaseStrategy 


class ValueStrategy(BaseStrategy):
    """Fundamental value investing strategy"""

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        # Get financial data
        balance_sheet = self.get_financials(ticker, 'balance_sheet')
        income_stmt = self.get_financials(ticker, 'income_statement')

        if not self._validate_data(balance_sheet) or not self._validate_data(income_stmt):
            return pd.DataFrame()

        # Calculate fundamental metrics
        latest_bs = balance_sheet.iloc[0]
        latest_is = income_stmt.iloc[0]

        metrics = {
            'pe_ratio': latest_is['currentprice'] / latest_is['trailingeps'],
            'pb_ratio': latest_bs['bookvalue'] / latest_bs['marketcap'],
            'debt_to_equity': latest_bs['totaldebt'] / latest_bs['total_equity_gross_minority_interest']
        }

        # Generate signals based on value metrics
        signals = pd.DataFrame(index=[pd.Timestamp.today()])
        signals['value_score'] = 0

        if metrics['pe_ratio'] < 15:
            signals['value_score'] += 1
        if metrics['pb_ratio'] < 1:
            signals['value_score'] += 1
        if metrics['debt_to_equity'] < 0.5:
            signals['value_score'] += 1

        return signals