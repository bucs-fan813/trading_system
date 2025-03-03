# trading_system/src/strategies/magic_formula_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, List

from src.strategies.base_strat import BaseStrategy, DataRetrievalError


class MagicFormulaStrategy(BaseStrategy):
    """
    Implementation of Joel Greenblatt's Magic Formula strategy with historical analysis.

    Enhances the original strategy by:
    1. Calculating metrics for all historical financial statement dates
    2. Using vectorized operations for efficient computation
    3. Strictly using financial statement data with fallback to company info only when necessary

    Parameters:
    - min_market_cap: Minimum market capitalization filter (default: $50M)
    - min_ebit: Minimum EBIT filter (default: $0)
    - min_enterprise_value: Minimum enterprise value filter (default: $0)
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params = {
            'min_market_cap': 50e6,
            'min_ebit': 0,
            'min_enterprise_value': 0
        }
        if params:
            self.params.update(params)

    def generate_signals(self, ticker: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate Magic Formula signals for all available historical periods.

        Args:
            ticker: Single ticker symbol to analyze
            start_date: Not used, maintained for base class compatibility
            end_date: Not used, maintained for base class compatibility
            initial_position: Not used, maintained for base class compatibility
            latest_only: If True, returns only the most recent signal

        Returns:
            DataFrame with columns: date, ticker, ebit, enterprise_value, 
            earnings_yield, return_on_capital, signal
        """
        try:
            # Retrieve and process financial statements
            financial_data = self._get_processed_financials(ticker)
            if financial_data.empty:
                return pd.DataFrame()

            # Merge with price data
            merged_data = self._merge_price_data(financial_data, ticker)
            if merged_data.empty:
                return pd.DataFrame()

            # Calculate core metrics
            calculated_data = self._calculate_metrics(merged_data)
            if calculated_data.empty:
                return pd.DataFrame()

            # Apply strategy filters and generate signals
            filtered_data = self._apply_filters(calculated_data)
            signals = self._generate_signals(filtered_data)
            
            if latest_only and not signals.empty:
                return signals.iloc[[-1]].reset_index()
            
            return signals.reset_index()

        except DataRetrievalError as e:
            self.logger.error(f"Data retrieval error for {ticker}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            return pd.DataFrame()

    def _get_processed_financials(self, ticker: str) -> pd.DataFrame:
        """Retrieve and harmonize financial statements."""
        # Income statement processing
        income_stmt = self.get_financials(ticker, 'income_statement', lookback=None)
        if income_stmt.empty:
            return pd.DataFrame()
        income_stmt = self._standardize_columns(income_stmt, 'income', ['ebit', 'operating_income'])
        
        # Balance sheet processing
        balance_sheet = self.get_financials(ticker, 'balance_sheet', lookback=None)
        if balance_sheet.empty:
            return pd.DataFrame()
        balance_sheet = self._standardize_columns(balance_sheet, 'balance', [
            'current_assets', 'current_liabilities', 'net_ppe',
            'short_term_debt', 'long_term_debt', 'cash', 
            'shares_outstanding'
        ])
        
        # Merge financial statements
        return income_stmt.merge(balance_sheet, left_index=True, right_index=True, how='inner')

    def _standardize_columns(self, df: pd.DataFrame, stmt_type: str, fields: List[str]) -> pd.DataFrame:
        """Handle alternative column names in financial statements."""
        column_map = {
            'income': {
                'ebit': ['ebit', 'operating_income']
            },
            'balance': {
                'current_assets': ['current_assets', 'total_current_assets'],
                'current_liabilities': ['current_liabilities', 'total_current_liabilities'],
                'net_ppe': ['net_ppe', 'property_plant_equipment_net'],
                'short_term_debt': ['short_term_debt', 'current_debt'],
                'long_term_debt': ['long_term_debt', 'noncurrent_debt'],
                'cash': ['cash', 'cash_and_cash_equivalents'],
                'shares_outstanding': ['shares_outstanding', 'common_stock_shares_outstanding']
            }
        }
        
        standardized = pd.DataFrame(index=df.index)
        for field in fields:
            possible_names = column_map[stmt_type][field]
            available = [name for name in possible_names if name in df.columns]
            if not available:
                self.logger.warning(f"Missing {field} in {stmt_type} for ticker")
                return pd.DataFrame()
            standardized[field] = df[available[0]]
        
        return standardized

    def _merge_price_data(self, financial_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Merge financial data with historical prices."""
        dates = financial_data.index
        prices = self.get_historical_prices(
            ticker,
            from_date=dates.min().strftime('%Y-%m-%d'),
            to_date=dates.max().strftime('%Y-%m-%d')
        )
        return financial_data.merge(
            prices[['close']], 
            left_index=True, 
            right_index=True, 
            how='left'
        ).dropna(subset=['close'])

    def _calculate_metrics(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Magic Formula metrics using vectorized operations."""
        merged_data['market_cap'] = merged_data['shares_outstanding'] * merged_data['close']
        merged_data['enterprise_value'] = (
            merged_data['market_cap'] +
            merged_data['short_term_debt'] +
            merged_data['long_term_debt'] -
            merged_data['cash']
        )
        merged_data['earnings_yield'] = merged_data['ebit'] / merged_data['enterprise_value']
        merged_data['return_on_capital'] = merged_data['ebit'] / (
            merged_data['current_assets'] - 
            merged_data['current_liabilities'] + 
            merged_data['net_ppe']
        )
        return merged_data.replace([np.inf, -np.inf], np.nan).dropna()

    def _apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply strategy filters using vectorized operations."""
        mask = (
            (data['market_cap'] >= self.params['min_market_cap']) &
            (data['ebit'] >= self.params['min_ebit']) &
            (data['enterprise_value'] >= self.params['min_enterprise_value']) &
            ((data['current_assets'] - data['current_liabilities'] + data['net_ppe']) > 0)
        )
        return data[mask]

    def _generate_signals(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on calculated metrics."""
        filtered_data['signal'] = np.where(
            (filtered_data['earnings_yield'] > 0) & 
            (filtered_data['return_on_capital'] > 0),
            1, 0
        )
        return filtered_data[['ebit', 'enterprise_value', 'earnings_yield', 
                              'return_on_capital', 'signal']]