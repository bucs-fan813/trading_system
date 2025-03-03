# trading_system/src/strategies/piotroski_fscore_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy, DataRetrievalError


class PiotroskiFScoreStrategy(BaseStrategy):
    """
    Implements Piotroski F-Score strategy using vectorized operations for multiple dates.
    
    Key Enhancements:
    1. Vectorized calculations across all historical financial dates
    2. Dynamic market cap calculation using shares outstanding and price data
    3. Full integration with financial statement dates for ratio calculations
    4. Batch processing of multiple dates per ticker
    
    Output Format:
    | ticker | date       | market_cap | f_score | signal | price |
    |--------|------------|------------|---------|--------|-------|
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params = params or {
            'min_market_cap': 100e6,
            'strong_score': 7,
            'moderate_score': 5
        }
        self._column_map = {
            'income': ['net_income', 'gross_profit', 'total_revenue'],
            'balance': ['total_assets', 'long_term_debt', 'current_assets',
                        'current_liabilities', 'common_stock'],
            'cashflow': ['operating_cash_flow']
        }

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generate historical F-Score signals for all available financial dates.
        
        Args:
            ticker: Single ticker symbol to process
            
        Returns:
            DataFrame with F-Score and price data indexed by financial statement dates
        """
        try:
            # Retrieve and validate financial data
            financial_data = self._get_combined_financials(ticker)
            if financial_data.empty:
                return pd.DataFrame()

            # Merge with price data and calculate market cap
            enriched_data = self._enrich_with_prices(ticker, financial_data)
            if enriched_data.empty:
                return pd.DataFrame()

            # Calculate F-Score components
            scored_data = self._calculate_fscore_components(enriched_data)
            
            # Filter by market cap and generate signals
            filtered_data = scored_data[scored_data['market_cap'] >= self.params['min_market_cap']]
            if filtered_data.empty:
                self.logger.info(f"{ticker} below market cap threshold")
                return pd.DataFrame()

            return self._format_output(filtered_data, ticker)

        except DataRetrievalError as e:
            self.logger.error(f"Data error for {ticker}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Processing error for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _get_combined_financials(self, ticker: str) -> pd.DataFrame:
        """Integrate financial statements with consistent date index"""
        statements = {
            'income': self.get_financials(ticker, 'income_statement', lookback=None),
            'balance': self.get_financials(ticker, 'balance_sheet', lookback=None),
            'cashflow': self.get_financials(ticker, 'cash_flow', lookback=None)
        }

        # Validate data availability
        if any(df.empty for df in statements.values()):
            self.logger.warning(f"Incomplete financial data for {ticker}")
            return pd.DataFrame()

        # Align statements on date index with forward fill
        combined = pd.concat(
            [df.add_suffix(f'_{stmt}') for stmt, df in statements.items()],
            axis=1
        ).sort_index().ffill()

        return combined[self._get_required_columns()]

    def _get_required_columns(self) -> list:
        """Generate list of required columns with suffixes"""
        return [f"{col}_{stmt}" 
                for stmt, cols in self._column_map.items() 
                for col in cols]

    def _enrich_with_prices(self, ticker: str, financial_data: pd.DataFrame) -> pd.DataFrame:
        """Merge financial data with historical prices"""
        dates = financial_data.index
        prices = self.get_historical_prices(
            ticker, 
            from_date=dates.min(), 
            to_date=dates.max()
        )['close'].rename('price')

        if prices.empty:
            self.logger.warning(f"No price data for {ticker}")
            return pd.DataFrame()

        # Calculate market cap using shares from balance sheet
        enriched = financial_data.join(prices, how='left')
        shares_field = next((f for f in ['common_stock_balance', 'share_issued_balance']
                             if f in enriched.columns), None)
        if not shares_field:
            return pd.DataFrame()

        enriched['market_cap'] = enriched[shares_field] * enriched['price']
        return enriched.dropna(subset=['market_cap', 'price'])

    def _calculate_fscore_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized calculation of all F-Score components"""
        # Profitability criteria
        data['roa'] = data['net_income_income'] / data['total_assets_balance']
        data['positive_roa'] = (data['roa'] > 0).astype(int)
        data['positive_ocf'] = (data['operating_cash_flow_cashflow'] > 0).astype(int)
        data['increasing_roa'] = (data['roa'] > data['roa'].shift(1)).astype(int)
        data['ocf_exceeds_ni'] = (data['operating_cash_flow_cashflow'] > 
                                 data['net_income_income']).astype(int)

        # Leverage & liquidity
        data['debt_ratio'] = data['long_term_debt_balance'] / data['total_assets_balance']
        data['decreasing_leverage'] = (data['debt_ratio'] < data['debt_ratio'].shift(1)).astype(int)
        data['current_ratio'] = (data['current_assets_balance'] / 
                                data['current_liabilities_balance'])
        data['improving_liquidity'] = (data['current_ratio'] > 
                                      data['current_ratio'].shift(1)).astype(int)

        # Share issuance
        data['no_new_shares'] = (data['common_stock_balance'] <= 
                                data['common_stock_balance'].shift(1)).astype(int)

        # Operational efficiency
        data['gross_margin'] = (data['gross_profit_income'] / 
                               data['total_revenue_income'])
        data['improving_margin'] = (data['gross_margin'] > 
                                   data['gross_margin'].shift(1)).astype(int)
        data['asset_turnover'] = (data['total_revenue_income'] / 
                                 data['total_assets_balance'])
        data['improving_turnover'] = (data['asset_turnover'] > 
                                     data['asset_turnover'].shift(1)).astype(int)

        # Calculate final score
        criteria = ['positive_roa', 'positive_ocf', 'increasing_roa', 'ocf_exceeds_ni',
                    'decreasing_leverage', 'improving_liquidity', 'no_new_shares',
                    'improving_margin', 'improving_turnover']
        data['f_score'] = data[criteria].sum(axis=1)
        return data

    def _format_output(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Create final output format with signals"""
        signals = [
            (data['f_score'] >= self.params['strong_score']),
            (data['f_score'] >= self.params['moderate_score']) & 
            (data['f_score'] < self.params['strong_score'])
        ]
        data['signal'] = np.select(
            [signals[0], signals[1]], 
            [1, 0.5], 
            default=0
        )
        return data[['market_cap', 'f_score', 'signal', 'price']].assign(ticker=ticker)