# trading_system/src/strategies/graham_defensive_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime

from src.strategies.base_strat import BaseStrategy, DataRetrievalError


class GrahamDefensiveStrategy(BaseStrategy):
    """
    Implements Benjamin Graham's Defensive Investor strategy with:
    - Vectorized operations for high performance
    - Multi-ticker processing
    - Historical financial statement date alignment
    - Financial statement-driven calculations
    
    Strategy Criteria:
    1. Market Cap ≥ $2B
    2. Current Ratio ≥ 2
    3. 5+ Years Positive Earnings
    4. Consistent Dividends
    5. 10%+ Earnings Growth
    6. P/E ≤ 15
    7. P/B ≤ 1.5
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params = params or {
            'min_market_cap': 2e9,
            'min_current_ratio': 2.0,
            'min_years_positive_earnings': 5,
            'min_earnings_growth': 0.1,
            'max_pe_ratio': 15,
            'max_pb_ratio': 1.5,
            'min_criteria_met': 5
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(self, tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate historical signals for multiple tickers using vectorized operations
        
        Args:
            tickers: Single or list of tickers
            latest_only: Return only latest signal if True
            
        Returns:
            DataFrame with columns: 
            [ticker, date, market_cap, current_ratio, pe_ratio, pb_ratio, 
             dividend_record, criteria_met, signal] + criteria flags
        """
        tickers = [tickers] if isinstance(tickers, str) else tickers
        results = []

        for ticker in tickers:
            try:
                df = self._process_ticker(ticker)
                if not df.empty:
                    results.append(df)
            except Exception as e:
                self.logger.error(f"Error processing {ticker}: {str(e)}")

        final_df = pd.concat(results, axis=0) if results else pd.DataFrame()
        
        if latest_only and not final_df.empty:
            final_df = final_df.groupby('ticker', group_keys=False).last().reset_index()
            
        return final_df

    def _process_ticker(self, ticker: str) -> pd.DataFrame:
        """Vectorized processing pipeline for a single ticker"""
        # Get raw financial data
        financials = self._get_all_financials(ticker)
        if financials.empty: return pd.DataFrame()
        
        # Get prices for financial statement dates
        prices = self.get_historical_prices(
            ticker, 
            from_date=financials.index.min().strftime('%Y-%m-%d'),
            to_date=financials.index.max().strftime('%Y-%m-%d')
        )
        
        # Merge and calculate metrics
        merged = self._merge_financials_with_prices(financials, prices)
        if merged.empty: return pd.DataFrame()
        
        # Calculate fundamental metrics
        metrics = self._calculate_metrics(merged)
        
        # Apply Graham criteria
        criteria = self._apply_graham_criteria(metrics)
        
        return criteria

    def _get_all_financials(self, ticker: str) -> pd.DataFrame:
        """Retrieve and align financial statements using vectorized operations"""
        dfs = []
        
        # Get all statements
        for stmt_type in ['balance_sheet', 'income_statement', 'cash_flow']:
            stmt = self.get_financials(ticker, stmt_type, lookback=None)
            if not stmt.empty:
                stmt = stmt.add_suffix(f'_{stmt_type}')
                dfs.append(stmt)
        
        if not dfs: return pd.DataFrame()
        
        # Combine statements on date index
        combined = pd.concat(dfs, axis=1)
        return combined.sort_index()

    def _merge_financials_with_prices(self, financials: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Vectorized merge of financial data with price data"""
        if prices.empty: return pd.DataFrame()
        
        # Resample prices to financial dates using forward fill
        prices = prices['close'].resample('D').ffill()
        
        # Merge using financial dates
        merged = financials.merge(
            prices, 
            left_index=True, 
            right_index=True,
            how='left'
        )
        
        # Forward fill missing prices
        merged['close'] = merged['close'].ffill()
        return merged.dropna(subset=['close'])

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized calculation of financial metrics"""
        metrics = df.copy()
        
        # Market Cap (from balance sheet)
        metrics['market_cap'] = (
            metrics['common_stock_shares_outstanding_balance_sheet'] * 
            metrics['close']
        )
        
        # Current Ratio
        metrics['current_ratio'] = (
            metrics['current_assets_balance_sheet'] / 
            metrics['current_liabilities_balance_sheet']
        )
        
        # P/E Ratio
        metrics['pe_ratio'] = (
            metrics['market_cap'] / 
            metrics['net_income_income_statement']
        ).replace([np.inf, -np.inf], np.nan)
        
        # P/B Ratio
        metrics['pb_ratio'] = (
            metrics['market_cap'] / 
            metrics['total_equity_balance_sheet']
        ).replace([np.inf, -np.inf], np.nan)
        
        # Dividend Record (from cash flow)
        metrics['dividend_record'] = (
            metrics['dividends_paid_cash_flow'] < 0
        )
        
        return metrics

    def _apply_graham_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized application of Graham criteria"""
        criteria = df.copy()
        
        # 1. Adequate Size
        criteria['adequate_size'] = (
            criteria['market_cap'] >= self.params['min_market_cap']
        )
        
        # 2. Strong Financial Condition
        criteria['strong_financial'] = (
            criteria['current_ratio'] >= self.params['min_current_ratio']
        )
        
        # 3. Earnings Stability (vectorized rolling calculation)
        criteria['earnings_stability'] = (
            criteria['net_income_income_statement']
            .rolling(window=self.params['min_years_positive_earnings'], min_periods=1)
            .apply(lambda x: (x > 0).all())
        )
        
        # 4. Dividend Record (already calculated)
        
        # 5. Earnings Growth (vectorized)
        criteria['earnings_growth'] = (
            criteria['net_income_income_statement'].pct_change(
                periods=self.params['min_years_positive_earnings']
            ) >= self.params['min_earnings_growth']
        )
        
        # 6. Moderate P/E
        criteria['moderate_pe'] = (
            criteria['pe_ratio'].between(0, self.params['max_pe_ratio'])
        )
        
        # 7. Moderate P/B
        criteria['moderate_pb'] = (
            criteria['pb_ratio'].between(0, self.params['max_pb_ratio'])
        )
        
        # Count met criteria
        criteria_cols = [
            'adequate_size', 'strong_financial', 'earnings_stability',
            'dividend_record', 'earnings_growth', 'moderate_pe', 'moderate_pb'
        ]
        criteria['criteria_met'] = criteria[criteria_cols].sum(axis=1)
        criteria['signal'] = (
            criteria['criteria_met'] >= self.params['min_criteria_met']
        ).astype(int)
        
        # Add ticker and format
        criteria['ticker'] = criteria.index.get_level_values('ticker')[0]
        criteria = criteria.reset_index().rename(columns={'date': 'date'})
        
        return criteria[
            ['ticker', 'date', 'market_cap', 'current_ratio', 'pe_ratio',
             'pb_ratio', 'dividend_record', 'criteria_met', 'signal'] + criteria_cols
        ]