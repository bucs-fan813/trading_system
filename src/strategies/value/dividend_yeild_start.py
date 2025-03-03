# trading_system/src/strategies/dividend_yield_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from datetime import datetime

from src.strategies.base_strat import BaseStrategy, DataRetrievalError


class DividendYieldStrategy(BaseStrategy):
    """
    Vectorized Dividend Yield strategy implementation using financial statements.
    
    Key Features:
    - Processes multiple dates from financial statements
    - Uses vectorized operations for efficient calculations
    - Relies on balance sheet, income statement, and cash flow data
    - Generates historical signals for financial statement dates
    
    Parameters:
    - min_market_cap: Minimum market capitalization filter
    - min_dividend_yield: Minimum acceptable dividend yield
    - max_payout_ratio: Maximum acceptable payout ratio
    - min_years_dividend: Minimum consecutive years with dividends
    - min_criteria_met: Minimum criteria to trigger buy signal
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params = params or {
            'min_market_cap': 500e6,
            'min_dividend_yield': 0.03,
            'max_payout_ratio': 0.75,
            'min_years_dividend': 3,
            'min_criteria_met': 3
        }

    def generate_signals(self, ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate signals for a ticker based on financial statement dates.
        
        Args:
            ticker: Single ticker symbol
            start_date: Not used, maintained for base class compatibility
            end_date: Not used, maintained for base class compatibility
            initial_position: Starting position (unused)
            latest_only: If True, return only latest signal
            
        Returns:
            pd.DataFrame: Signals with metrics for each financial statement date
        """
        try:
            # Retrieve financial data
            financial_data = self._get_merged_financials(ticker)
            if financial_data.empty:
                return pd.DataFrame()

            # Calculate fundamental metrics
            metrics = self._calculate_metrics(financial_data)
            
            # Calculate criteria
            signals = self._calculate_criteria(metrics)
            
            # Format output
            results = self._format_output(ticker, metrics, signals)
            
            return results.iloc[[-1]] if latest_only else results
            
        except DataRetrievalError as e:
            self.logger.error(f"Data error for {ticker}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Processing error for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _get_merged_financials(self, ticker: str) -> pd.DataFrame:
        """Merge financial statements and price data for analysis dates."""
        # Retrieve financial statements
        bs = self.get_financials(ticker, 'balance_sheet', lookback=None)
        is_ = self.get_financials(ticker, 'income_statement', lookback=None)
        cf = self.get_financials(ticker, 'cash_flow', lookback=None)
        prices = self.get_historical_prices(ticker, lookback=0)

        # Get common columns from financials
        financial_dfs = []
        for df, cols in [(bs, ['Common Stock Shares Outstanding']),
                         (is_, ['Net Income']),
                         (cf, ['Cash Dividends Paid'])]:
            if not df.empty and cols[0] in df.columns:
                financial_dfs.append(df[cols].reset_index())

        # Merge financial data
        merged = pd.concat([d.set_index('date') for d in financial_dfs], axis=1)
        merged = merged.ffill().dropna().reset_index()
        
        if merged.empty:
            return pd.DataFrame()

        # Merge with prices using nearest date
        merged['date'] = pd.to_datetime(merged['date'])
        prices = prices.reset_index()
        prices['date'] = pd.to_datetime(prices['date'])
        merged = pd.merge_asof(
            merged.sort_values('date'),
            prices[['date', 'close']].sort_values('date'),
            on='date',
            direction='nearest'
        )
        
        return merged.dropna()

    def _calculate_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial metrics using vectorized operations."""
        df = data.copy()
        df['shares_outstanding'] = df['Common Stock Shares Outstanding']
        df['dividend_paid'] = df['Cash Dividends Paid'].abs()
        df['net_income'] = df['Net Income']
        
        # Calculate per-share metrics
        df['dividend_per_share'] = df['dividend_paid'] / df['shares_outstanding']
        df['eps'] = df['net_income'] / df['shares_outstanding']
        
        # Market cap calculation
        df['market_cap'] = df['shares_outstanding'] * df['close']
        
        # Yield and payout ratio
        df['dividend_yield'] = df['dividend_per_share'] / df['close']
        df['payout_ratio'] = np.where(
            df['eps'] > 0,
            df['dividend_per_share'] / df['eps'],
            np.nan
        )
        
        # Dividend history tracking
        df['year'] = df['date'].dt.year
        df['has_dividend'] = (df['dividend_per_share'] > 0).astype(int)
        df['div_year_count'] = df.groupby('year')['has_dividend'].transform('max')
        df['cumulative_div_years'] = df['div_year_count'].cumsum()
        
        return df.dropna()

    def _calculate_criteria(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Calculate buy criteria using vectorized operations."""
        df = metrics.copy()
        params = self.params
        
        # Market cap criteria
        df['market_cap_met'] = df['market_cap'] >= params['min_market_cap']
        
        # Dividend yield criteria
        df['yield_met'] = df['dividend_yield'] >= params['min_dividend_yield']
        
        # Payout ratio criteria
        df['payout_met'] = df['payout_ratio'] <= params['max_payout_ratio']
        
        # Dividend history criteria
        df['history_met'] = df['cumulative_div_years'] >= params['min_years_dividend']
        
        # Dividend growth criteria
        df['div_growth'] = df['dividend_per_share'].pct_change(periods=4).fillna(0)
        df['growth_met'] = (df['div_growth'] > 0).astype(int)
        
        # Total criteria met
        criteria_cols = ['market_cap_met', 'yield_met', 'payout_met', 'history_met', 'growth_met']
        df['criteria_met'] = df[criteria_cols].sum(axis=1)
        
        return df

    def _format_output(self, ticker: str, metrics: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Format output with required columns and structure."""
        output = pd.DataFrame({
            'ticker': ticker,
            'date': signals['date'],
            'market_cap': metrics['market_cap'],
            'dividend_yield': metrics['dividend_yield'],
            'payout_ratio': metrics['payout_ratio'],
            'years_with_dividends': signals['cumulative_div_years'],
            'criteria_met': signals['criteria_met'],
            'signal': (signals['criteria_met'] >= self.params['min_criteria_met']).astype(int)
        })
        
        # Add individual criteria flags
        criteria_cols = [c for c in signals.columns if c.endswith('_met')]
        output = pd.concat([output, signals[criteria_cols]], axis=1)
        
        return output.sort_values('date').reset_index(drop=True)