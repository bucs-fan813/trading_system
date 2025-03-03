# trading_system/src/strategies/acquirers_multiple_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from datetime import datetime

from src.strategies.base_strat import BaseStrategy, DataRetrievalError


class AcquirersMultipleStrategy(BaseStrategy):
    """
    Implementation of Tobias Carlisle's Acquirer's Multiple strategy with vectorized operations.
    
    The strategy evaluates companies based on historical Enterprise Value/Operating Earnings (EV/EBIT)
    ratios calculated at financial statement dates. Utilizes financial statements and historical prices
    for accurate temporal alignment of fundamental data and market prices.
    
    Path: trading_system/src/strategies/acquirers_multiple_strategy.py
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params = params or {
            'min_market_cap': 50e6,  # $50M minimum
            'max_multiple': 8.0,     # EV/EBIT threshold
            'min_ebit': 5e6          # Minimum operating earnings
        }

    def generate_signals(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate historical signals for a ticker based on financial statement dates.
        
        Args:
            ticker: Single ticker symbol to analyze
            start_date: Optional start date filter
            end_date: Optional end date filter
            initial_position: Not used in this strategy
            latest_only: If True, returns only the most recent signal
            
        Returns:
            DataFrame with columns: [ticker, date, operating_earnings, enterprise_value, 
                                    market_cap, acquirers_multiple, signal]
        """
        try:
            # Retrieve financial data with full history
            income_stmt = self.get_financials(ticker, 'income_statement', lookback=None)
            balance_sheet = self.get_financials(ticker, 'balance_sheet', lookback=None)
            
            # Validate required data exists
            if income_stmt.empty or balance_sheet.empty:
                self.logger.warning(f"Incomplete financials for {ticker}")
                return pd.DataFrame()

            # Merge financial statements on date
            merged = self._merge_financials(income_stmt, balance_sheet)
            if merged.empty:
                return pd.DataFrame()

            # Get prices for financial statement dates
            price_data = self._get_realtime_prices(ticker, merged.index.get_level_values('date'))
            
            # Calculate fundamental metrics
            results = self._calculate_metrics(merged, price_data)
            if results.empty:
                return pd.DataFrame()

            # Apply date filters
            if start_date:
                results = results[results.index >= start_date]
            if end_date:
                results = results[results.index <= end_date]

            # Generate signals based on parameters
            results['signal'] = self._generate_signal_column(results)

            if latest_only:
                return results.iloc[[-1]].reset_index()
                
            return results.reset_index()

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _merge_financials(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.DataFrame:
        """Merge income statement and balance sheet data on date index."""
        required_columns = {
            'income': ['ebit', 'operating_income'],
            'balance': ['total_debt', 'cash_and_cash_equivalents', 'common_stock_shares_outstanding']
        }

        # Find first valid EBIT source
        income_source = next((col for col in required_columns['income'] 
                            if col in income_stmt.columns), None)
        if not income_source:
            self.logger.warning("Missing EBIT/operating income data")
            return pd.DataFrame()

        # Validate balance sheet columns
        missing_balance = [col for col in required_columns['balance'] 
                         if col not in balance_sheet.columns]
        if missing_balance:
            self.logger.warning(f"Missing balance sheet columns: {missing_balance}")
            return pd.DataFrame()

        return pd.merge(
            income_stmt[[income_source]],
            balance_sheet[required_columns['balance']],
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=('_income', '_balance')
        )

    def _get_realtime_prices(self, ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Retrieve closing prices for specific financial statement dates."""
        if dates.empty:
            return pd.Series()

        # Get price data for all available dates
        prices = self.get_historical_prices(
            ticker,
            from_date=dates.min().strftime('%Y-%m-%d'),
            to_date=dates.max().strftime('%Y-%m-%d')
        )

        # Align prices with financial dates using forward fill
        aligned_prices = prices.reindex(dates.union(prices.index)).sort_index().ffill()
        return aligned_prices.loc[dates, 'close']

    def _calculate_metrics(self, merged_data: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Calculate enterprise value and acquirer's multiple."""
        if prices.empty or merged_data.empty:
            return pd.DataFrame()

        # Calculate market capitalization
        shares = merged_data['common_stock_shares_outstanding']
        merged_data['market_cap'] = shares * prices.values

        # Calculate enterprise value
        merged_data['enterprise_value'] = (
            merged_data['market_cap'] +
            merged_data['total_debt'] -
            merged_data['cash_and_cash_equivalents']
        )

        # Calculate acquirer's multiple (EV/EBIT)
        ebit_col = merged_data.columns[merged_data.columns.str.contains('ebit|operating_income')][0]
        merged_data['acquirers_multiple'] = (
            merged_data['enterprise_value'] / 
            merged_data[ebit_col].replace(0, np.nan)
        )

        # Filter valid calculations
        valid_data = merged_data[
            (merged_data['enterprise_value'] > 0) &
            (merged_data[ebit_col] >= self.params['min_ebit']) &
            (merged_data['market_cap'] >= self.params['min_market_cap']) &
            (merged_data['acquirers_multiple'].notnull())
        ]

        return valid_data[['enterprise_value', 'market_cap', 'acquirers_multiple', ebit_col]]

    def _generate_signal_column(self, results: pd.DataFrame) -> pd.Series:
        """Generate binary signals based on multiple threshold."""
        return np.where(
            results['acquirers_multiple'] <= self.params['max_multiple'], 1, 0
        ).astype(int)