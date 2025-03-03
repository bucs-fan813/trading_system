# trading_system/src/strategies/deep_value_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
from datetime import datetime

from src.strategies.base_strat import BaseStrategy, DataRetrievalError


class DeepValueStrategy(BaseStrategy):
    """
    Optimized Deep Value investing strategy using vectorized operations and financial statement dates.
    
    Features:
    - Processes multiple tickers simultaneously
    - Uses financial statement dates for ratio calculations
    - Vectorized calculations for optimal performance
    - Relies primarily on financial statement data
    
    Strategy Logic:
    1. Collects financial statement dates from all three statements
    2. Retrieves closing prices for these specific dates
    3. Calculates value metrics using financials aligned with price dates
    4. Scores metrics across 4 dimensions with configurable weights
    5. Generates signals based on composite value score
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.ratio_columns = ['pe_ratio', 'pb_ratio', 'dividend_yield', 'debt_to_equity']
        self.default_params = {
            'min_market_cap': 1e8,  # $100M default
            'pe_thresholds': [10, 15, 20],
            'pb_thresholds': [1.0, 1.5, 2.0],
            'div_thresholds': [0.04, 0.03, 0.02],
            'de_thresholds': [0.3, 0.6, 1.0],
            'score_weights': [1.0, 1.0, 0.8, 0.7],  # PE, PB, Div, DE
            'min_statements': 3  # Minimum financial statements required
        }
        self.params = {**self.default_params, **(params or {})}

    def generate_signals(self, tickers: Union[str, List[str]], 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        initial_position: int = 0,
                        latest_only: bool = False) -> pd.DataFrame:
        """
        Generate signals for multiple tickers using financial statement dates.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Not used (maintained for ABC compatibility)
            end_date: Not used (maintained for ABC compatibility)
            initial_position: Starting position (unused in this strategy)
            latest_only: If True, returns only latest signal per ticker
            
        Returns:
            DataFrame with value metrics and signals indexed by (ticker, date)
        """
        ticker_list = [tickers] if isinstance(tickers, str) else tickers
        try:
            # Step 1: Get financial statement dates and fundamental data
            combined_fin = self._get_all_financials_with_dates(ticker_list)
            if combined_fin.empty:
                return pd.DataFrame()

            # Step 2: Get prices at financial statement dates
            price_dates = combined_fin.index.get_level_values('date').unique().tolist()
            prices = self.get_historical_prices(
                tickers=ticker_list,
                from_date=min(price_dates).strftime('%Y-%m-%d'),
                to_date=max(price_dates).strftime('%Y-%m-%d')
            )
            
            # Step 3: Merge financials with prices and calculate ratios
            merged = self._merge_financials_prices(combined_fin, prices)
            if merged.empty:
                return pd.DataFrame()

            # Step 4: Calculate and score metrics
            scored = self._calculate_vectorized_scores(merged)
            
            # Step 5: Filter and format results
            return self._format_output(scored, latest_only)

        except Exception as e:
            self.logger.error(f"Error in DeepValue strategy: {str(e)}")
            return pd.DataFrame()

    def _get_all_financials_with_dates(self, tickers: List[str]) -> pd.DataFrame:
        """Combine financial statements and collect reporting dates."""
        dfs = []
        for stmt_type in ['balance_sheet', 'income_statement', 'cash_flow']:
            try:
                df = self.get_financials(tickers, stmt_type)
                if not df.empty:
                    dfs.append(df.reset_index())
            except DataRetrievalError:
                continue

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, axis=0)
        date_counts = combined.groupby(['ticker', 'date']).size()
        valid_dates = date_counts[date_counts >= self.params['min_statements']].index
        
        return combined.set_index(['ticker', 'date']).loc[valid_dates]

    def _merge_financials_prices(self, financials: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Merge financial statement data with closing prices."""
        if not isinstance(prices.index, pd.MultiIndex):
            prices = prices.reset_index().set_index(['ticker', 'date'])

        # Extract closing prices at financial report dates
        closes = prices['close'].unstack('ticker').ffill().bfill()
        aligned_prices = closes.reindex(index=financials.index.get_level_values('date').unique(),
                                        method='nearest').stack()
        
        # Merge and filter valid entries
        merged = financials.join(aligned_prices.rename('price'), how='left')
        merged = merged[merged['price'].notna() & (merged['price'] > 0)]
        
        # Calculate market cap if available
        if 'shares_outstanding' in merged:
            merged['market_cap'] = merged['price'] * merged['shares_outstanding']
        else:
            merged['market_cap'] = np.nan

        return merged.dropna(subset=['market_cap'])

    def _calculate_vectorized_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate value metrics and scores using vectorized operations."""
        # Calculate fundamental ratios
        data['pe_ratio'] = data['price'] / data.get('eps', np.nan)
        data['pb_ratio'] = data['price'] / data.get('book_value', np.nan)
        data['dividend_yield'] = data.get('dividend_per_share', 0) / data['price']
        data['debt_to_equity'] = data.get('total_debt', np.nan) / data.get('total_equity', np.nan)
        
        # Score each metric
        metrics = ['pe_ratio', 'pb_ratio', 'dividend_yield', 'debt_to_equity']
        thresholds = [
            self.params['pe_thresholds'],
            self.params['pb_thresholds'],
            self.params['div_thresholds'],
            self.params['de_thresholds']
        ]
        directions = [False, False, True, False]
        
        for metric, thresh, higher_is_better in zip(metrics, thresholds, directions):
            data[f'{metric}_score'] = self._vectorized_score(
                data[metric], 
                thresholds=thresh,
                higher_is_better=higher_is_better
            )
        
        # Calculate weighted composite score
        weights = np.array(self.params['score_weights'])
        scores = data[[f'{m}_score' for m in metrics]]
        data['value_score'] = scores.dot(weights) / (3 * weights.sum())
        
        # Apply market cap filter
        data = data[data['market_cap'] >= self.params['min_market_cap']]
        data['signal'] = (data['value_score'] >= 0.6).astype(int)
        
        return data

    def _vectorized_score(self, series: pd.Series, thresholds: List[float], higher_is_better: bool) -> pd.Series:
        """Vectorized implementation of metric scoring."""
        conditions = [
            series < thresholds[0] if not higher_is_better else series > thresholds[0],
            series < thresholds[1] if not higher_is_better else series > thresholds[1],
            series < thresholds[2] if not higher_is_better else series > thresholds[2]
        ]
        
        scores = np.select(
            condlist=conditions,
            choicelist=[3, 2, 1],
            default=0 if not higher_is_better else 3
        )
        return pd.Series(scores, index=series.index)

    def _format_output(self, data: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """Format final output with required columns."""
        cols = [
            'market_cap', 'price', 'pe_ratio', 'pb_ratio',
            'dividend_yield', 'debt_to_equity', 'value_score', 'signal'
        ]
        result = data[cols].reset_index()
        result['date'] = pd.to_datetime(result['date'])
        
        if latest_only:
            return result.sort_values('date').groupby('ticker').last().reset_index()
        return result

    def __del__(self):
        """Ensure proper cleanup."""
        super().__del__()
